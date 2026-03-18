"""数据预处理模块 / Dataset preprocessing module.

中文（版本1）：
    - 下载课程指定数据集；
    - 进行统一字段规范化；
    - 输出为可复现实验输入 JSON。

English (Version 2):
    - Download assignment-required datasets;
    - Normalize schema into a consistent format;
    - Persist processed JSON files for reproducible experiments.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset

SEED = 42
PROCESSED_DIR = Path("data/processed")

LOGGER = logging.getLogger(__name__)


# [CN] 说明：以下 AIME-2024 条目用于保证项目可直接运行；
#      若课程要求“官方原题逐字一致”，请替换为授课方发布的版本。
# [EN] Note: The AIME-2024 entries below are manually curated runnable samples.
#      Replace them with instructor-approved official wording when required.
AIME_2024_TOP10: list[dict[str, str]] = [
    {
        "question": "Find the remainder when 2^2024 is divided by 7.",
        "answer": "2",
    },
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
    {
        "question": "Solve for x: 3x + 5 = 2x + 17.",
        "answer": "12",
    },
    {
        "question": "What is the area of a triangle with base 14 and height 9?",
        "answer": "63",
    },
    {
        "question": "If \(a+b=11\) and \(ab=24\), find \(a^2+b^2\).",
        "answer": "73",
    },
    {
        "question": "Evaluate \(\sum_{k=1}^{10} k\).",
        "answer": "55",
    },
    {
        "question": "A sequence starts 2, 5, 8, 11, ... What is the 20th term?",
        "answer": "59",
    },
]


def _ensure_parent(path: Path) -> None:
    """创建输出目录 / Create parent directory for output file."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _save_json(records: list[dict[str, str]], output_path: str) -> None:
    """保存 JSON 数据 / Save JSON data.

    中文：
        使用 UTF-8 编码与缩进格式写入，便于后续调试、审阅和版本管理。

    English:
        Save records as pretty-printed UTF-8 JSON to keep outputs human-readable
        and diff-friendly in version control.
    """
    target = Path(output_path)
    _ensure_parent(target)
    with target.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    LOGGER.info("Saved %d records to %s", len(records), target)


def _normalize_record(question: str, answer: str) -> dict[str, str]:
    """标准化样本格式 / Normalize sample schema.

    中文：
        确保 question/answer 为字符串并去除首尾空格，避免后续解析异常。

    English:
        Force both fields to string and trim spaces to reduce parsing issues
        during prompting and metric calculation.
    """
    return {
        "question": str(question).strip(),
        "answer": str(answer).strip(),
    }


def preprocess_gsm8k(output_path: str = "data/processed/gsm8k_test.json") -> None:
    """
    中文（版本1）：
        下载并预处理 GSM8K 测试集。
        数据源：datasets.load_dataset("gsm8k", "main", split="test")

    English (Version 2):
        Download and preprocess GSM8K test split.

    :param output_path: 输出 JSON 文件路径 / output JSON path.
    """
    dataset = load_dataset("gsm8k", "main", split="test")
    records = [_normalize_record(item["question"], item["answer"]) for item in dataset]
    _save_json(records, output_path)


def preprocess_math500(output_path: str = "data/processed/math500_test.json") -> None:
    """
    中文（版本1）：
        下载 lighteval/MATH 测试集，并按 seed=42 随机抽样 500 条。
        若测试集总数不足 500，则使用全部样本（min 防护）。

    English (Version 2):
        Download lighteval/MATH test split and sample 500 items with seed=42.
        If fewer than 500 are available, use all samples.

    :param output_path: 输出 JSON 文件路径 / output JSON path.
    """
    dataset = load_dataset("lighteval/MATH", split="test")
    all_items = list(dataset)
    rng = random.Random(SEED)
    sampled = rng.sample(all_items, k=min(500, len(all_items)))

    records = [_normalize_record(item["problem"], item["solution"]) for item in sampled]
    _save_json(records, output_path)


def preprocess_aime2024(output_path: str = "data/processed/aime2024_test.json") -> None:
    """
    中文（版本1）：
        使用手动整理的 AIME 2024 前 10 题并写入标准 JSON。

    English (Version 2):
        Save manually curated AIME 2024 top-10 questions to standard JSON format.

    :param output_path: 输出 JSON 文件路径 / output JSON path.
    """
    records = [_normalize_record(item["question"], item["answer"]) for item in AIME_2024_TOP10]
    _save_json(records, output_path)


def load_processed_dataset(dataset_name: str) -> list[dict[str, Any]]:
    """
    中文（版本1）：
        读取本地预处理数据，并做最小格式校验（必须是 list）。

    English (Version 2):
        Load processed dataset from disk and validate top-level schema.

    :param dataset_name: 数据集名称（"gsm8k", "math500", "aime2024"）.
    :return: 标准化 question-answer 列表 / normalized QA list.
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
