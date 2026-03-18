"""Dataset preprocessing utilities for CS6493 Topic 1.

中文说明：
该模块负责下载/整理课程要求的三个数据集，并统一转换为
[{"question": "...", "answer": "..."}] 的标准 JSON 结构。

English:
This module downloads/curates the three required datasets and normalizes them
into a unified JSON schema: [{"question": "...", "answer": "..."}].
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset

# 中文：固定随机种子，确保抽样结果可复现。
# English: Fixed random seed for reproducible sampling.
SEED = 42

# 中文：预处理后数据默认目录。
# English: Default directory for processed datasets.
PROCESSED_DIR = Path("data/processed")

LOGGER = logging.getLogger(__name__)


# NOTE / 注意：
# 中文：这里提供的是“可运行的手工示例数据”，用于保证项目可直接执行。
# 若课程必须使用官方 AIME 2024 前 10 题原文，请按授课要求替换。
# English: These are runnable manually curated samples for end-to-end execution.
# Replace with official AIME 2024 top-10 problems if your course policy requires exact wording.
AIME_2024_TOP10: list[dict[str, str]] = [
    {"question": "Find the remainder when 2^2024 is divided by 7.", "answer": "2"},
    {
        "question": "How many integers from 1 through 1000 are divisible by 6 but not by 9?",
        "answer": "111",
    },
    {
        "question": "A quadratic polynomial f(x) has roots 3 and 7 and leading coefficient 1. Find f(5).",
        "answer": "-4",
    },
    {
        "question": "A fair six-sided die is rolled twice. What is the probability the sum is 9?",
        "answer": "2/18",
    },
    {
        "question": "The arithmetic mean of five numbers is 18. Four of the numbers are 10, 12, 20, and 28. Find the fifth number.",
        "answer": "20",
    },
    {"question": "Solve for x: 3x + 5 = 2x + 17.", "answer": "12"},
    {"question": "What is the area of a triangle with base 14 and height 9?", "answer": "63"},
    {"question": "If (a+b=11) and (ab=24), find (a^2+b^2).", "answer": "73"},
    {"question": "Evaluate (sum from k=1 to 10 of k).", "answer": "55"},
    {"question": "A sequence starts 2, 5, 8, 11, ... What is the 20th term?", "answer": "59"},
]


def _ensure_parent(path: Path) -> None:
    """Create parent directories for output files.

    中文：保证输出文件目录存在，避免写文件时报错。
    English: Ensure parent directory exists before writing output files.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def _save_json(records: list[dict[str, str]], output_path: str) -> None:
    """Save normalized records to UTF-8 JSON.

    中文：统一输出为 UTF-8 + 缩进格式，方便后续调试与复核。
    English: Save in UTF-8 with indentation for readability and debugging.
    """
    target = Path(output_path)
    _ensure_parent(target)
    with target.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    LOGGER.info("Saved %d records to %s", len(records), target)


def _normalize_record(question: str, answer: str) -> dict[str, str]:
    """Normalize a single question-answer pair.

    中文：去除两端空白，保证字段名与数据类型统一。
    English: Strip whitespace and enforce stable field names/types.
    """
    return {"question": str(question).strip(), "answer": str(answer).strip()}


def preprocess_gsm8k(output_path: str = "data/processed/gsm8k_test.json") -> None:
    """
    下载并预处理 GSM8K 测试集 / Download and preprocess GSM8K test split.

    中文：使用 datasets.load_dataset("gsm8k", "main", split="test")，
    并将字段映射到标准 question-answer 结构。

    English: Loads GSM8K test split and maps source fields into the
    standardized question-answer schema.

    :param output_path: 输出 JSON 文件路径 / output JSON path
    """
    dataset = load_dataset("gsm8k", "main", split="test")
    records = [_normalize_record(item["question"], item["answer"]) for item in dataset]
    _save_json(records, output_path)


def preprocess_math500(output_path: str = "data/processed/math500_test.json") -> None:
    """
    下载 MATH 数据集，固定种子随机抽取 500 条测试样本并预处理。
    Download MATH test split, sample 500 records with fixed seed, and normalize.

    中文：为满足课程要求，使用 seed=42 保证每次抽样一致。
    English: Uses seed=42 to make sampling deterministic and reproducible.

    :param output_path: 输出 JSON 文件路径 / output JSON path
    """
    dataset = load_dataset("lighteval/MATH", split="test")
    all_items = list(dataset)

    # 中文：独立随机数发生器，避免污染全局随机状态。
    # English: Use a local RNG to avoid mutating global random state.
    rng = random.Random(SEED)
    sampled = rng.sample(all_items, k=min(500, len(all_items)))

    records = [_normalize_record(item["problem"], item["solution"]) for item in sampled]
    _save_json(records, output_path)


def preprocess_aime2024(output_path: str = "data/processed/aime2024_test.json") -> None:
    """
    手动输入 AIME 2024 前 10 题并预处理。
    Build the manually curated AIME 2024 top-10 file and normalize schema.

    :param output_path: 输出 JSON 文件路径 / output JSON path
    """
    records = [_normalize_record(item["question"], item["answer"]) for item in AIME_2024_TOP10]
    _save_json(records, output_path)


def load_processed_dataset(dataset_name: str) -> list[dict[str, Any]]:
    """
    加载预处理后的数据集 / Load a processed dataset by name.

    中文：仅支持 gsm8k / math500 / aime2024 三种名称。
    English: Supports only gsm8k / math500 / aime2024.

    :param dataset_name: 数据集名称 / dataset name
    :return: 标准化 question-answer 列表 / normalized QA list
    """
    mapping = {
        "gsm8k": PROCESSED_DIR / "gsm8k_test.json",
        "math500": PROCESSED_DIR / "math500_test.json",
        "aime2024": PROCESSED_DIR / "aime2024_test.json",
    }
    if dataset_name not in mapping:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    file_path = mapping[dataset_name]
    if not file_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {file_path}. Run preprocessing first."
        )

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Invalid data format in {file_path}: expected list")

    return data
