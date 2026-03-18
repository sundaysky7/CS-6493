"""Experiment runner for model × prompt × dataset combinations.

中文：负责执行核心实验循环，并输出 raw_results.csv。
English: Runs the core experiment loop and exports raw_results.csv.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from data.preprocess import load_processed_dataset
from models.loader import generate_model_response, load_quantized_model
from prompts.templates import generate_prompt

LOGGER = logging.getLogger(__name__)


def _word_count(text: str) -> int:
    """Count words by whitespace splitting.

    中文：按空格分词统计响应长度。
    English: Word count based on whitespace tokenization.
    """
    return len(str(text).split())


def run_full_experiment(
    models: list[str],
    prompt_methods: list[str],
    datasets: list[str],
    output_path: str = "results/raw_results.csv",
) -> None:
    """
    运行完整实验并保存原始结果。
    Run full experiments and save raw results.

    中文：
    循环组合 = 模型 × 数据集 × 提示方法。
    每条样本都记录 question / true_answer / model_response / response_length。

    English:
    Combination loop = models × datasets × prompt methods.
    For each sample, store question / true_answer / model_response / response_length.

    :param models: 模型名称列表 / model names
    :param prompt_methods: 提示方法列表 / prompting methods
    :param datasets: 数据集列表 / dataset names
    :param output_path: 原始结果 CSV 路径 / output CSV path
    """
    rows: list[dict[str, object]] = []

    for model_name in models:
        LOGGER.info("Loading model: %s", model_name)
        model, tokenizer = load_quantized_model(model_name)

        for dataset_name in datasets:
            data = load_processed_dataset(dataset_name)
            LOGGER.info("Running %s on %s (%d samples)", model_name, dataset_name, len(data))

            for method in prompt_methods:
                LOGGER.info("Method: %s", method)
                for idx, sample in enumerate(data):
                    question = sample["question"]
                    true_answer = sample["answer"]

                    try:
                        if method == "self_refine":
                            # 中文：Self-Refine 分两阶段：先草稿，再自我修正。
                            # English: Self-Refine runs in two stages: draft then refinement.
                            stage1_prompt = generate_prompt(question, "self_refine_stage1")
                            preliminary = generate_model_response(
                                model=model,
                                tokenizer=tokenizer,
                                prompt=stage1_prompt,
                            )
                            stage2_prompt = generate_prompt(
                                question,
                                f"self_refine_stage2::{preliminary}",
                            )
                            response = generate_model_response(
                                model=model,
                                tokenizer=tokenizer,
                                prompt=stage2_prompt,
                            )
                        else:
                            prompt = generate_prompt(question, method)
                            response = generate_model_response(
                                model=model,
                                tokenizer=tokenizer,
                                prompt=prompt,
                            )
                    except Exception as exc:  # noqa: BLE001
                        # 中文：容错策略：异常样本跳过并记录，确保长流程不中断。
                        # English: Fault tolerance: skip failed sample and continue loop.
                        LOGGER.exception(
                            "Failed on model=%s dataset=%s method=%s sample_idx=%d: %s",
                            model_name,
                            dataset_name,
                            method,
                            idx,
                            exc,
                        )
                        continue

                    rows.append(
                        {
                            "model": model_name,
                            "dataset": dataset_name,
                            "method": method,
                            "question": question,
                            "true_answer": true_answer,
                            "model_response": response,
                            "response_length": _word_count(response),
                        }
                    )

    df = pd.DataFrame(rows)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    LOGGER.info("Saved raw results to %s (%d rows)", target, len(df))
