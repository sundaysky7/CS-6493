"""Model loading and generation helpers.

中文：该模块提供 4bit 量化模型加载与固定随机种子推理。
English: This module provides 4-bit quantized model loading and deterministic generation.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _set_seed(seed: int) -> None:
    """Set all relevant random seeds.

    中文：同时固定 Python / NumPy / PyTorch（含 CUDA）随机种子。
    English: Set Python / NumPy / PyTorch (including CUDA) seeds.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_quantized_model(model_name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    加载 4bit 量化后的模型与 tokenizer。
    Load model/tokenizer with 4-bit quantization.

    中文：按照课程要求使用 NF4 + bfloat16。
    English: Uses NF4 quantization with bfloat16 compute per course requirements.

    :param model_name: Hugging Face 模型名 / Hugging Face model id
    :return: (model, tokenizer)
    """
    # 中文：bitsandbytes 量化配置。
    # English: BitsAndBytes quantization configuration.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # 中文：若无 pad_token，则按要求设置为 eos_token。
    # English: If pad token is missing, set it to EOS token as required.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    使用模型生成响应（固定 seed 保证可复现）。
    Generate a model response with a fixed seed for reproducibility.

    :param model: 量化模型 / quantized model
    :param tokenizer: 对应 tokenizer / paired tokenizer
    :param prompt: 输入提示词 / input prompt
    :param max_new_tokens: 最大生成 token 数 / max generated tokens
    :param temperature: 生成温度 / generation temperature
    :param seed: 随机种子 / random seed
    :return: 去除特殊 token 的生成文本 / decoded text without special tokens
    """
    _set_seed(seed)

    # 中文：将输入编码并移动到模型所在设备。
    # English: Tokenize input and move tensors to model device.
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

    # 中文：只解码“新增生成”的 token，避免把 prompt 原文算入输出。
    # English: Decode only newly generated tokens (exclude prompt tokens).
    generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
