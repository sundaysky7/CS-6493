"""模型加载与生成模块 / Model loading and generation helpers.

中文（版本1）：
    默认优先使用标准 CUDA 加载模型；仅在显式请求 4bit 量化时，才尝试使用
    bitsandbytes 的 4bit 路径。若 CUDA 不可用或显式要求 CPU，则回退为
    CPU 加载，并提供固定随机种子的文本生成接口。
    同时兼容历史模型别名，避免旧配置中的模型名失效。

English (Version 2):
    Prefer standard CUDA model loading by default. Only attempt the
    bitsandbytes-based 4-bit path when 4-bit quantization is explicitly
    requested. If CUDA is unavailable or CPU execution is explicitly requested,
    fall back to CPU loading while preserving deterministic generation helpers.
    Also supports legacy model aliases so older run configurations remain usable.
"""

from __future__ import annotations

import logging
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import is_bitsandbytes_available

LOGGER = logging.getLogger(__name__)

MODEL_ALIASES = {
    "deepseek-ai/DeepSeek-R1-Qwen-1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
}


def _log_runtime_placement(
    model: AutoModelForCausalLM,
    load_mode: str,
    resolved_model_name: str,
) -> None:
    """
    中文（版本1）：
        在模型加载完成后记录实际运行设备、dtype 与加载模式，便于确认是否真正走到
        CUDA / 4bit / CPU 路径。

    English (Version 2):
        Log the actual runtime device, dtype, and load mode after model loading
        so it is easy to confirm whether CUDA / 4-bit / CPU was actually used.
    """
    try:
        param = next(model.parameters())
        device = str(param.device)
        dtype = str(param.dtype)
    except StopIteration:
        device = "unknown"
        dtype = "unknown"

    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            device_index = param.device.index if param.device.index is not None else torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(device_index)
            LOGGER.info(
                "Model runtime placement | model=%s | mode=%s | device=%s | dtype=%s | gpu=%s",
                resolved_model_name,
                load_mode,
                device,
                dtype,
                gpu_name,
            )
            return
        except Exception:  # noqa: BLE001
            pass

    LOGGER.info(
        "Model runtime placement | model=%s | mode=%s | device=%s | dtype=%s",
        resolved_model_name,
        load_mode,
        device,
        dtype,
    )


def _set_seed(seed: int) -> None:
    """固定随机种子 / Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_model_name(model_name: str) -> str:
    """
    中文（版本1）：
        将历史模型名称映射为当前可访问的 Hugging Face 模型名称。

    English (Version 2):
        Map legacy model names to the current accessible Hugging Face model ids.
    """
    resolved_name = MODEL_ALIASES.get(model_name, model_name)
    if resolved_name != model_name:
        LOGGER.info("Resolved legacy model alias: %s -> %s", model_name, resolved_name)
    return resolved_name


def load_quantized_model(
    model_name: str,
    force_cpu: bool = False,
    enable_4bit: bool = False,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    中文（版本1）：
        默认优先使用标准 CUDA 加载模型。
        仅当 enable_4bit=True 时，才尝试使用 bitsandbytes 配置加载 4bit NF4
        量化模型，并使用 bfloat16 / float16 计算。
        若 CUDA 不可用或 force_cpu=True，则自动回退到 CPU 普通加载。
        当 tokenizer 缺少 pad_token 时，按要求回退到 eos_token。

    English (Version 2):
        Prefer standard CUDA loading by default.
        Only when enable_4bit=True, attempt loading with 4-bit NF4 quantization
        via bitsandbytes and bfloat16 / float16 compute. If CUDA is unavailable
        or force_cpu=True, automatically fall back to standard CPU loading.
        If no pad token is defined, fallback to EOS token.

    :param model_name: Hugging Face 模型名称 / HF model id.
    :param force_cpu: 是否强制使用 CPU / whether to force CPU execution.
    :param enable_4bit: 是否启用 4bit 量化 / whether to enable 4-bit quantization.
    :return: (model, tokenizer) 元组 / tuple.
    """
    resolved_model_name = _resolve_model_name(model_name)
    use_cpu = force_cpu or not torch.cuda.is_available()
    cuda_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(resolved_model_name, trust_remote_code=True)
    # Decoder-only models should use left padding for correct batched generation.
    tokenizer.padding_side = "left"

    load_mode = "cpu"

    if use_cpu:
        LOGGER.info("Loading model on CPU without 4-bit quantization: %s", resolved_model_name)
        model = AutoModelForCausalLM.from_pretrained(
            resolved_model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        model = model.to("cpu")
        load_mode = "cpu"
    else:
        try:
            if enable_4bit:
                try:
                    if not is_bitsandbytes_available():
                        raise RuntimeError("bitsandbytes is not available in the current environment.")

                    LOGGER.info("Loading model on CUDA with 4-bit quantization: %s", resolved_model_name)
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=cuda_dtype,
                        bnb_4bit_use_double_quant=True,
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        resolved_model_name,
                        quantization_config=bnb_config,
                        device_map="auto",
                        torch_dtype=cuda_dtype,
                        trust_remote_code=True,
                    )
                    load_mode = "cuda_4bit"
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning(
                        "Quantized CUDA loading failed for %s; falling back to standard CUDA loading. Reason: %s",
                        resolved_model_name,
                        exc,
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        resolved_model_name,
                        torch_dtype=cuda_dtype,
                        trust_remote_code=True,
                    )
                    model = model.to("cuda")
                    load_mode = "cuda_standard"
            else:
                LOGGER.info("Loading model on standard CUDA without 4-bit quantization: %s", resolved_model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    resolved_model_name,
                    torch_dtype=cuda_dtype,
                    trust_remote_code=True,
                )
                model = model.to("cuda")
                load_mode = "cuda_standard"
        except Exception as cuda_exc:  # noqa: BLE001
            LOGGER.warning(
                "CUDA loading failed for %s; falling back to CPU. Reason: %s",
                resolved_model_name,
                cuda_exc,
            )
            model = AutoModelForCausalLM.from_pretrained(
                resolved_model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            model = model.to("cpu")
            load_mode = "cpu"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    _log_runtime_placement(model, load_mode=load_mode, resolved_model_name=resolved_model_name)
    return model, tokenizer


def generate_model_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    seed: int = 42,
) -> str:
    """
    中文（版本1）：
        对给定 prompt 进行文本生成，返回“仅新增生成部分”的解码文本。
        通过固定 seed，减少同环境下重复实验波动。

    English (Version 2):
        Generate text for a prompt and return only the newly generated segment
        (excluding the prompt tokens). Seed is fixed for reproducibility.

    :param model: 量化后的模型 / quantized model.
    :param tokenizer: 对应 tokenizer.
    :param prompt: 输入提示词 / prompt text.
    :param max_new_tokens: 最大生成长度 / max generation length.
    :param temperature: 生成温度 / decoding temperature.
    :param seed: 随机种子 / random seed.
    :return: 模型生成响应文本 / generated response text.
    """
    _set_seed(seed)

    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def generate_model_responses_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_new_tokens: int = 512,
    temperature: float = 0.1,
) -> list[str]:
    """
    中文（版本1）：
        对一组 prompt 执行批量文本生成，返回与输入 prompts 对齐的新增生成文本列表。
        该接口用于提升吞吐，尽量减少逐样本调用 generate 带来的开销。

    English (Version 2):
        Run batched generation for a list of prompts and return newly generated
        texts aligned with the input order. This helper improves throughput by
        reducing per-sample generate overhead.
    """
    if not prompts:
        return []

    device = next(model.parameters()).device
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_lengths = inputs["attention_mask"].sum(dim=1)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    responses: list[str] = []
    for idx in range(output_ids.shape[0]):
        generated_ids = output_ids[idx][int(input_lengths[idx].item()) :]
        responses.append(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())
    return responses
