"""模型加载与生成模块 / Model loading and generation helpers.

中文（版本1）：
    负责优先按课程要求加载 4bit 量化模型；当 CUDA 不可用或显式要求 CPU 时，
    自动回退为普通精度 CPU 加载，并提供固定随机种子的文本生成接口。
    同时兼容历史模型别名，避免旧配置中的模型名失效。

English (Version 2):
    Prefer assignment-required 4-bit quantized loading on CUDA-capable systems.
    When CUDA is unavailable or CPU execution is explicitly requested, fall back
    to standard CPU loading while preserving deterministic generation helpers.
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
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    中文（版本1）：
        优先使用 bitsandbytes 配置加载 4bit NF4 量化模型，并使用 bfloat16 计算。
        若 CUDA 不可用或 force_cpu=True，则自动回退到 CPU 普通加载。
        当 tokenizer 缺少 pad_token 时，按要求回退到 eos_token。

    English (Version 2):
        Prefer loading a model/tokenizer pair with 4-bit NF4 quantization and
        bfloat16 compute on CUDA. If CUDA is unavailable or force_cpu=True,
        automatically fall back to standard CPU loading. If no pad token is
        defined, fallback to EOS token.

    :param model_name: Hugging Face 模型名称 / HF model id.
    :param force_cpu: 是否强制使用 CPU / whether to force CPU execution.
    :return: (model, tokenizer) 元组 / tuple.
    """
    resolved_model_name = _resolve_model_name(model_name)
    use_cpu = force_cpu or not torch.cuda.is_available()

    tokenizer = AutoTokenizer.from_pretrained(resolved_model_name, trust_remote_code=True)

    if use_cpu:
        LOGGER.info("Loading model on CPU without 4-bit quantization: %s", resolved_model_name)
        model = AutoModelForCausalLM.from_pretrained(
            resolved_model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        model = model.to("cpu")
    else:
        try:
            if not is_bitsandbytes_available():
                raise RuntimeError("bitsandbytes is not available in the current environment.")

            LOGGER.info("Loading model on CUDA with 4-bit quantization: %s", resolved_model_name)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                resolved_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Quantized CUDA loading failed for %s; falling back to CPU. Reason: %s",
                resolved_model_name,
                exc,
            )
            model = AutoModelForCausalLM.from_pretrained(
                resolved_model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            model = model.to("cpu")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
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
