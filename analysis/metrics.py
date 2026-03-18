"""评估指标模块 / Metrics module.

中文（版本1）：
    实现答案提取、正确性判定（含 ±1% 数值容差）、按组合聚合准确率与响应长度。

English (Version 2):
    Implements final-answer extraction, correctness judgment (with ±1% numeric
    tolerance), and grouped aggregation for accuracy and response length.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd


def extract_final_answer(model_response: str) -> str | None:
    """
    中文（版本1）：
        优先提取 `Final Answer:` 后内容；
        若不存在则回退到“最后一个数值 token”；
        若仍失败则回退到“最后一个非空行”。

    English (Version 2):
        Extraction priority:
        1) content after `Final Answer:`
        2) last numeric token
        3) last non-empty line

    :param model_response: 模型完整响应 / full model response.
    :return: 提取答案 / extracted answer, or None.
    """
    if not model_response:
        return None

    text = str(model_response).strip()

    # Preferred pattern: content after "Final Answer:"
    final_match = re.search(r"Final\s*Answer\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if final_match:
        candidate = final_match.group(1).strip().splitlines()[0].strip()
        if candidate:
            return candidate

    # Fallback: last numeric token
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:/[0-9]+)?", text)
    if numbers:
        return numbers[-1]

    # Final fallback: last non-empty line
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return lines[-1]

    return None


def _parse_number(text: str | None) -> float | None:
    """解析数字 / Parse numeric value (plain number or fraction)."""
    if text is None:
        return None

    value = text.strip()
    if not value:
        return None

    if "/" in value and re.fullmatch(r"[-+]?\d+\s*/\s*[-+]?\d+", value):
        num_str, den_str = value.replace(" ", "").split("/", maxsplit=1)
        den = float(den_str)
        if den == 0:
            return None
        return float(num_str) / den

    match = re.search(r"[-+]?\d*\.?\d+", value)
    if match:
        return float(match.group(0))

    return None


def is_answer_correct(model_response: str, true_answer: str, tolerance: float = 0.01) -> bool:
    """
    中文（版本1）：
        若预测与标准答案都能解析为数值，则按相对误差判断（默认 1%）；
        否则执行字符串不区分大小写精确匹配。

    English (Version 2):
        If both prediction and ground truth are numeric, use relative tolerance
        (default 1%); otherwise, perform case-insensitive exact text matching.

    :param model_response: 模型响应文本 / model response.
    :param true_answer: 标准答案 / ground truth answer.
    :param tolerance: 相对误差阈值 / relative tolerance.
    :return: 是否正确 / correctness flag.
    """
    extracted = extract_final_answer(model_response)
    if extracted is None:
        return False

    pred_num = _parse_number(extracted)
    true_num = _parse_number(true_answer)

    if pred_num is not None and true_num is not None:
        if true_num == 0:
            return math.isclose(pred_num, 0.0, abs_tol=tolerance)
        return abs(pred_num - true_num) / abs(true_num) <= tolerance

    return extracted.strip().lower() == str(true_answer).strip().lower()


def calculate_accuracy_and_length(
    raw_results_path: str = "results/raw_results.csv",
    accuracy_output_path: str = "results/accuracy.csv",
    length_output_path: str = "results/length.csv",
) -> None:
    """
    中文（版本1）：
        读取原始结果后按 (model, dataset, method) 分组：
        - accuracy: 正确率百分比（保留 1 位小数）
        - avg_length: 平均响应词数（保留 1 位小数）

    English (Version 2):
        Load raw results and aggregate by (model, dataset, method):
        - accuracy: percentage with one decimal place
        - avg_length: mean response length with one decimal place

    :param raw_results_path: 原始结果路径 / raw csv path.
    :param accuracy_output_path: 准确率输出路径 / accuracy csv path.
    :param length_output_path: 长度输出路径 / length csv path.
    """
    df = pd.read_csv(raw_results_path)
    if df.empty:
        raise ValueError("raw_results.csv is empty; cannot calculate metrics")

    df["is_correct"] = df.apply(
        lambda row: is_answer_correct(
            model_response=str(row["model_response"]),
            true_answer=str(row["true_answer"]),
        ),
        axis=1,
    )

    accuracy_df = (
        df.groupby(["model", "dataset", "method"], as_index=False)["is_correct"]
        .mean()
        .rename(columns={"is_correct": "accuracy"})
    )
    accuracy_df["accuracy"] = (accuracy_df["accuracy"] * 100).round(1)

    length_df = (
        df.groupby(["model", "dataset", "method"], as_index=False)["response_length"]
        .mean()
        .rename(columns={"response_length": "avg_length"})
    )
    length_df["avg_length"] = length_df["avg_length"].round(1)

    acc_path = Path(accuracy_output_path)
    len_path = Path(length_output_path)
    acc_path.parent.mkdir(parents=True, exist_ok=True)
    len_path.parent.mkdir(parents=True, exist_ok=True)

    accuracy_df.to_csv(acc_path, index=False)
    length_df.to_csv(len_path, index=False)
