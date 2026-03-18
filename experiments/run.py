"""实验运行模块 / Experiment runner.

中文（版本1）：
    负责遍历全部组合（模型 × 提示方法 × 数据集），调用模型生成并写出原始结果。
    对单条样本异常进行容错：记录日志并跳过，不中断整体长跑实验。

English (Version 2):
    Executes the complete Cartesian product of model × prompt method × dataset,
    collects raw responses, and writes them to CSV. Per-sample exceptions are
    logged and skipped so a long run can continue robustly.
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
    """按空格计词 / Count words via whitespace split."""
    return len(str(text).split())


def run_full_experiment(
    models: list[str],
    prompt_methods: list[str],
    datasets: list[str],
    output_path: str = "results/raw_results.csv",
) -> None:
    """
    中文（版本1）：
        运行完整实验并导出 raw_results.csv。
        Self-Refine 方法会执行两阶段推理：
            阶段1生成 preliminary answer，
            阶段2将其回填后生成 refined answer。

    English (Version 2):
        Run the full experiment and export raw_results.csv.
        For `self_refine`, a two-stage process is used:
            stage-1 -> preliminary answer,
            stage-2 -> refined answer using stage-1 output.

    :param models: 模型名称列表 / model ids.
    :param prompt_methods: 提示方法列表 / prompting methods.
    :param datasets: 数据集列表 / dataset names.
    :param output_path: 原始结果输出路径 / output csv path.
    CSV fields:
        model, dataset, method, question, true_answer, model_response, response_length
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
                        # [CN] 异常样本跳过并记录，保障批量实验鲁棒性。
                        # [EN] Skip and log failed samples to preserve long-run robustness.
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
