"""Metrics calculation utilities.

中文：实现答案提取、答案判定与统计指标计算。
English: Implements answer extraction, correctness check, and aggregate metrics.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd


def extract_final_answer(model_response: str) -> str | None:
    """
    从模型响应中提取最终答案。
    Extract the final answer from model response.

    中文策略：
    1) 优先匹配 `Final Answer:` 后内容
    2) 兜底使用“最后一个数值”
    3) 再兜底使用“最后一行非空文本”

    English strategy:
    1) Prefer text after `Final Answer:`
    2) Fallback to the last numeric token
    3) Final fallback to the last non-empty line
    """
    if not model_response:
        return None

    text = str(model_response).strip()

    final_match = re.search(r"Final\s*Answer\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if final_match:
        candidate = final_match.group(1).strip().splitlines()[0].strip()
        if candidate:
            return candidate

    numbers = re.findall(r"[-+]?\d*\.?\d+(?:/[0-9]+)?", text)
    if numbers:
        return numbers[-1]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return lines[-1]

    return None


def _parse_number(text: str | None) -> float | None:
    """Parse decimal or fraction into float.

    中文：支持整数、小数、分数（如 2/3）。
    English: Supports integer, decimal, and fraction formats (e.g., 2/3).
    """
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
    判断模型答案是否正确。
    Check whether model answer is correct.

    中文：若可解析为数值，按相对误差 <= tolerance 判定（默认 ±1%）。
    English: For numeric answers, use relative error <= tolerance (default ±1%).
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
    计算准确率和平均响应长度并输出 CSV。
    Compute accuracy and average response length and export CSV files.

    中文：
    - accuracy.csv: model, dataset, method, accuracy(%)
    - length.csv:   model, dataset, method, avg_length

    English:
    - accuracy.csv: model, dataset, method, accuracy(%)
    - length.csv:   model, dataset, method, avg_length
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
