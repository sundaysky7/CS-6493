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
from pandas.errors import EmptyDataError

from data.preprocess import load_processed_dataset
from models.loader import (
    generate_model_response,
    generate_model_responses_batch,
    load_quantized_model,
)
from prompts.templates import generate_prompt

LOGGER = logging.getLogger(__name__)


def _word_count(text: str) -> int:
    """按空格计词 / Count words via whitespace split."""
    return len(str(text).split())


def _load_completed_row_keys(output_path: str) -> set[tuple[str, str, str, str, str]]:
    """
    中文（版本1）：
        从已有的 raw_results.csv 中读取已完成样本键，用于断点续跑时跳过。

    English (Version 2):
        Load completed sample keys from an existing raw_results.csv so resume
        runs can skip finished samples.
    """
    target = Path(output_path)
    if not target.exists():
        return set()

    try:
        df = pd.read_csv(target)
    except EmptyDataError:
        return set()

    required_columns = {"model", "dataset", "method", "question", "true_answer"}
    if df.empty or not required_columns.issubset(df.columns):
        return set()

    completed_keys: set[tuple[str, str, str, str, str]] = set()
    for _, row in df.iterrows():
        completed_keys.add(
            (
                str(row["model"]),
                str(row["dataset"]),
                str(row["method"]),
                str(row["question"]),
                str(row["true_answer"]),
            )
        )
    return completed_keys


def _append_result_rows(output_path: str, rows: list[dict[str, object]]) -> None:
    """
    中文（版本1）：
        将单条实验结果增量追加到 CSV。
        若目标文件不存在，则自动创建并写入表头。

    English (Version 2):
        Append a single experiment result row to CSV incrementally.
        If the target file does not exist yet, create it and write the header.
    """
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    row_df = pd.DataFrame(rows)
    write_header = not target.exists()
    row_df.to_csv(target, mode="a", header=write_header, index=False)


def _batched(items: list[dict[str, str]], batch_size: int) -> list[list[dict[str, str]]]:
    """按固定 batch size 切分样本列表 / Split samples by fixed batch size."""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _is_cuda_oom_error(exc: Exception) -> bool:
    """判断是否为 CUDA OOM 异常 / Detect CUDA OOM errors."""
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def _generate_with_oom_fallback(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    """
    批量生成并在 OOM 时自动二分降级，避免一次大批次直接失败。
    """
    if not prompts:
        return []

    try:
        return generate_model_responses_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    except Exception as exc:  # noqa: BLE001
        if not _is_cuda_oom_error(exc) or len(prompts) == 1:
            raise
        split = max(1, len(prompts) // 2)
        left = _generate_with_oom_fallback(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts[:split],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        right = _generate_with_oom_fallback(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts[split:],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return left + right


def run_full_experiment(
    models: list[str],
    prompt_methods: list[str],
    datasets: list[str],
    output_path: str = "results/raw_results.csv",
    max_samples_per_dataset: int | None = None,
    force_cpu: bool = False,
    enable_4bit: bool = False,
    resume: bool = False,
    batch_size: int = 16,
    flush_every: int = 100,
    max_new_tokens: int = 512,
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
    :param max_samples_per_dataset:
        每个数据集最多使用多少条样本；None 表示使用全部样本
        / maximum number of samples per dataset; None means use all samples.
    :param force_cpu: 是否强制使用 CPU 加载模型 / whether to force CPU loading.
    :param enable_4bit:
        是否启用 4bit 量化；若为 False，则优先使用标准 CUDA/CPU 路径
        / whether to enable 4-bit quantization; if False, prefer the standard
        CUDA/CPU loading path.
    :param resume: 是否启用断点续跑 / whether to resume from existing raw results.
    :param batch_size: 批量推理大小 / generation batch size.
    :param flush_every: 增量写盘阈值 / flush threshold for buffered rows.
    :param max_new_tokens: 每次生成最大新 token 数 / max new tokens per generation.
    CSV fields:
        model, dataset, method, question, true_answer, model_response, response_length
    """
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if resume:
        completed_keys = _load_completed_row_keys(output_path)
        LOGGER.info("Resume mode enabled: found %d completed rows in %s", len(completed_keys), target)
    else:
        if target.exists():
            target.unlink()
        completed_keys = set()

    row_count = 0
    skipped_count = 0
    row_buffer: list[dict[str, object]] = []

    for model_name in models:
        LOGGER.info(
            "Loading model: %s | force_cpu=%s | enable_4bit=%s",
            model_name,
            force_cpu,
            enable_4bit,
        )
        model, tokenizer = load_quantized_model(
            model_name,
            force_cpu=force_cpu,
            enable_4bit=enable_4bit,
        )

        for dataset_name in datasets:
            data = load_processed_dataset(dataset_name)
            if max_samples_per_dataset is not None:
                data = data[:max_samples_per_dataset]
            LOGGER.info(
                "Running %s on %s (%d samples) | batch_size=%d",
                model_name,
                dataset_name,
                len(data),
                batch_size,
            )

            for method in prompt_methods:
                LOGGER.info("Method: %s", method)
                pending_samples: list[dict[str, str]] = []
                for idx, sample in enumerate(data):
                    question = sample["question"]
                    true_answer = sample["answer"]
                    row_key = (
                        model_name,
                        dataset_name,
                        method,
                        str(question),
                        str(true_answer),
                    )

                    if row_key in completed_keys:
                        skipped_count += 1
                        LOGGER.info(
                            "Skipping completed sample | model=%s dataset=%s method=%s sample_idx=%d",
                            model_name,
                            dataset_name,
                            method,
                            idx,
                        )
                        continue

                    pending_samples.append(
                        {
                            "question": str(question),
                            "true_answer": str(true_answer),
                            "sample_idx": str(idx),
                        }
                    )

                for sample_batch in _batched(pending_samples, batch_size):
                    questions = [item["question"] for item in sample_batch]
                    answers = [item["true_answer"] for item in sample_batch]
                    indices = [int(item["sample_idx"]) for item in sample_batch]

                    try:
                        if method == "self_refine":
                            stage1_prompts = [
                                generate_prompt(question_text, "self_refine_stage1")
                                for question_text in questions
                            ]
                            preliminaries = _generate_with_oom_fallback(
                                model=model,
                                tokenizer=tokenizer,
                                prompts=stage1_prompts,
                                max_new_tokens=max_new_tokens,
                                temperature=0.1,
                            )
                            stage2_prompts = [
                                generate_prompt(
                                    question_text,
                                    f"self_refine_stage2::{preliminary}",
                                )
                                for question_text, preliminary in zip(questions, preliminaries)
                            ]
                            responses = _generate_with_oom_fallback(
                                model=model,
                                tokenizer=tokenizer,
                                prompts=stage2_prompts,
                                max_new_tokens=max_new_tokens,
                                temperature=0.1,
                            )
                        else:
                            prompts = [generate_prompt(question_text, method) for question_text in questions]
                            responses = _generate_with_oom_fallback(
                                model=model,
                                tokenizer=tokenizer,
                                prompts=prompts,
                                max_new_tokens=max_new_tokens,
                                temperature=0.1,
                            )
                    except Exception as exc:  # noqa: BLE001
                        # [CN] 批次失败时降级为逐样本，避免整批丢失。
                        # [EN] Fall back to per-sample generation when batch fails.
                        LOGGER.exception(
                            "Batch failed on model=%s dataset=%s method=%s (batch_size=%d): %s",
                            model_name,
                            dataset_name,
                            method,
                            len(sample_batch),
                            exc,
                        )
                        responses = []
                        for question_text in questions:
                            try:
                                if method == "self_refine":
                                    stage1_prompt = generate_prompt(question_text, "self_refine_stage1")
                                    preliminary = generate_model_response(
                                        model=model,
                                        tokenizer=tokenizer,
                                        prompt=stage1_prompt,
                                        max_new_tokens=max_new_tokens,
                                    )
                                    stage2_prompt = generate_prompt(
                                        question_text,
                                        f"self_refine_stage2::{preliminary}",
                                    )
                                    response_text = generate_model_response(
                                        model=model,
                                        tokenizer=tokenizer,
                                        prompt=stage2_prompt,
                                        max_new_tokens=max_new_tokens,
                                    )
                                else:
                                    prompt = generate_prompt(question_text, method)
                                    response_text = generate_model_response(
                                        model=model,
                                        tokenizer=tokenizer,
                                        prompt=prompt,
                                        max_new_tokens=max_new_tokens,
                                    )
                            except Exception as sample_exc:  # noqa: BLE001
                                LOGGER.exception(
                                    "Failed on model=%s dataset=%s method=%s sample_idx=%d: %s",
                                    model_name,
                                    dataset_name,
                                    method,
                                    indices[len(responses)],
                                    sample_exc,
                                )
                                response_text = ""
                            responses.append(response_text)

                    for idx, question, true_answer, response in zip(indices, questions, answers, responses):
                        if response == "":
                            continue
                        row = {
                            "model": model_name,
                            "dataset": dataset_name,
                            "method": method,
                            "question": question,
                            "true_answer": true_answer,
                            "model_response": response,
                            "response_length": _word_count(response),
                        }
                        row_buffer.append(row)
                        row_key = (
                            model_name,
                            dataset_name,
                            method,
                            str(question),
                            str(true_answer),
                        )
                        completed_keys.add(row_key)
                        row_count += 1

                        if len(row_buffer) >= flush_every:
                            _append_result_rows(output_path, row_buffer)
                            row_buffer.clear()

    if row_buffer:
        _append_result_rows(output_path, row_buffer)
        row_buffer.clear()

    LOGGER.info(
        "Saved raw results incrementally to %s | newly_saved=%d | skipped_completed=%d",
        target,
        row_count,
        skipped_count,
    )
