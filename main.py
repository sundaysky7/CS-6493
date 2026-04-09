"""CS6493 Topic 1 主入口 / Main entrypoint.

中文说明：
    本文件负责把所有模块按课程要求串成一条可复现实验流水线。
    执行顺序必须严格保持为：
    1) 数据预处理 -> 2) 全组合实验 -> 3) 指标统计 -> 4) 可视化输出。

English notes:
    This file orchestrates the full, reproducible pipeline required by the
    assignment. The execution order is intentionally fixed:
    1) preprocessing -> 2) full experiment -> 3) metrics -> 4) visualizations.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from analysis.metrics import calculate_accuracy_and_length
from analysis.visualize import (
    plot_accuracy_comparison,
    plot_accuracy_length_correlation,
)
from data.preprocess import (
    preprocess_aime2024,
    preprocess_gsm8k,
    preprocess_math500,
)
from experiments.run import run_full_experiment

MODEL_NAMES = [
    "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
]
PROMPT_METHODS = ["standard", "cot", "self_refine", "least_to_most"]
DATASETS = ["gsm8k", "math500", "aime2024"]


def _create_run_output_dir(force_cpu: bool, max_samples_per_dataset: int | None) -> Path:
    """
    中文（版本1）：
        为当前运行创建唯一结果目录，并返回该目录路径。

    English (Version 2):
        Create a unique output directory for the current run and return its path.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = "cpu" if force_cpu else "auto"
    sample_tag = (
        f"samples{max_samples_per_dataset}"
        if max_samples_per_dataset is not None
        else "samplesall"
    )
    run_dir = Path("results") / f"run_{timestamp}_{mode_tag}_{sample_tag}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _resolve_run_output_dir(
    force_cpu: bool,
    max_samples_per_dataset: int | None,
    resume_run_dir: str | None,
) -> tuple[Path, bool]:
    """
    中文（版本1）：
        若指定 resume_run_dir，则复用已有结果目录；
        否则创建一个新的运行目录。

    English (Version 2):
        Reuse an existing run directory when resume_run_dir is provided;
        otherwise create a new directory for the current run.
    """
    if resume_run_dir is not None:
        run_dir = Path(resume_run_dir)
        if not run_dir.exists() or not run_dir.is_dir():
            raise FileNotFoundError(f"--resume-run-dir does not exist or is not a directory: {run_dir}")
        return run_dir, True

    return _create_run_output_dir(
        force_cpu=force_cpu,
        max_samples_per_dataset=max_samples_per_dataset,
    ), False


def _normalize_model_names_for_resume(model_names: list[str]) -> list[str]:
    """
    中文（版本1）：
        规范化模型名称，用于断点续跑时兼容历史错误模型 ID。

    English (Version 2):
        Normalize model names so resumed runs remain compatible with a legacy
        incorrect model id.
    """
    alias_mapping = {
        "deepseek-ai/DeepSeek-R1-Qwen-1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    }
    return [alias_mapping.get(name, name) for name in model_names]


def _write_or_validate_run_config(
    run_dir: Path,
    run_metadata: dict[str, Any],
    resume: bool,
) -> None:
    """
    中文（版本1）：
        新运行时写入配置；断点续跑时校验关键配置是否一致，避免把不同实验混到同一目录。

    English (Version 2):
        Write config for new runs; validate key configuration on resumed runs to
        avoid mixing incompatible experiments in the same directory.
    """
    config_path = run_dir / "run_config.json"

    if resume and config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            existing_config = json.load(f)

        keys_to_validate = [
            "force_cpu",
            "max_samples_per_dataset",
            "models",
            "prompt_methods",
            "datasets",
        ]
        normalized_existing = {key: existing_config.get(key) for key in keys_to_validate}
        normalized_current = {key: run_metadata.get(key) for key in keys_to_validate}

        normalized_existing["models"] = _normalize_model_names_for_resume(
            list(normalized_existing.get("models", []))
        )
        normalized_current["models"] = _normalize_model_names_for_resume(
            list(normalized_current.get("models", []))
        )

        if normalized_existing != normalized_current:
            raise ValueError(
                "Resume configuration mismatch. Use the same runtime options and experiment "
                "settings as the original run, or start a new run directory."
            )

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(run_metadata, f, ensure_ascii=False, indent=2)


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    中文（版本1）：
        构建命令行参数解析器，用于支持 CPU 回退与小样本调试模式。

    English (Version 2):
        Build the CLI argument parser for CPU fallback and small-sample mode.
    """
    parser = argparse.ArgumentParser(
        description="Run the CS6493 math reasoning evaluation pipeline."
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU execution even when CUDA is available.",
    )
    parser.add_argument(
        "--max-samples-per-dataset",
        type=int,
        default=None,
        help="Use only the first N samples from each dataset for quick testing.",
    )
    parser.add_argument(
        "--resume-run-dir",
        type=str,
        default=None,
        help="Resume from an existing run directory instead of creating a new one.",
    )
    return parser


def main() -> None:
    """
    中文（版本1）：
        主函数入口，顺序执行完整实验。
        这里不做业务逻辑计算，只做流程编排与日志初始化，便于维护。

    English (Version 2):
        Main orchestration function.
        It only coordinates modules and logging configuration, keeping business
        logic inside dedicated modules for readability and maintainability.
    """
    args = _build_arg_parser().parse_args()

    if args.max_samples_per_dataset is not None and args.max_samples_per_dataset <= 0:
        raise ValueError("--max-samples-per-dataset must be a positive integer.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    logging.info(
        "Runtime options | force_cpu=%s | max_samples_per_dataset=%s | resume_run_dir=%s",
        args.force_cpu,
        args.max_samples_per_dataset,
        args.resume_run_dir,
    )

    run_dir, is_resumed_run = _resolve_run_output_dir(
        force_cpu=args.force_cpu,
        max_samples_per_dataset=args.max_samples_per_dataset,
        resume_run_dir=args.resume_run_dir,
    )
    if is_resumed_run:
        logging.info("Resuming existing run directory: %s", run_dir)
    else:
        logging.info("Created run output directory: %s", run_dir)

    run_metadata = {
        "force_cpu": args.force_cpu,
        "max_samples_per_dataset": args.max_samples_per_dataset,
        "models": MODEL_NAMES,
        "prompt_methods": PROMPT_METHODS,
        "datasets": DATASETS,
    }
    _write_or_validate_run_config(
        run_dir=run_dir,
        run_metadata=run_metadata,
        resume=is_resumed_run,
    )

    raw_results_path = str(run_dir / "raw_results.csv")
    accuracy_path = str(run_dir / "accuracy.csv")
    length_path = str(run_dir / "length.csv")
    figures_dir = str(run_dir / "figures")

    # [CN] 1) 数据预处理：统一输出为标准 question-answer JSON
    # [EN] 1) Data preprocessing: normalize all datasets into question-answer JSON.
    if is_resumed_run:
        logging.info("Resume mode: reuse existing processed datasets when available.")
        processed_paths = {
            "gsm8k": Path("data/processed/gsm8k_test.json"),
            "math500": Path("data/processed/math500_test.json"),
            "aime2024": Path("data/processed/aime2024_test.json"),
        }

        if processed_paths["gsm8k"].exists():
            logging.info("Skipping GSM8K preprocessing because processed file already exists.")
        else:
            preprocess_gsm8k()

        if processed_paths["math500"].exists():
            logging.info("Skipping MATH-500 preprocessing because processed file already exists.")
        else:
            preprocess_math500()

        if processed_paths["aime2024"].exists():
            logging.info("Skipping AIME2024 preprocessing because processed file already exists.")
        else:
            preprocess_aime2024()
    else:
        preprocess_gsm8k()
        preprocess_math500()
        preprocess_aime2024()

    # [CN] 2) 运行完整实验：遍历 model × method × dataset 并记录原始响应
    # [EN] 2) Run full experiment: iterate model × method × dataset combinations.
    run_full_experiment(
        models=MODEL_NAMES,
        prompt_methods=PROMPT_METHODS,
        datasets=DATASETS,
        output_path=raw_results_path,
        max_samples_per_dataset=args.max_samples_per_dataset,
        force_cpu=args.force_cpu,
        resume=is_resumed_run,
    )

    # [CN] 3) 指标统计：计算准确率与平均响应长度
    # [EN] 3) Metrics: compute grouped accuracy and average response length.
    calculate_accuracy_and_length(
        raw_results_path=raw_results_path,
        accuracy_output_path=accuracy_path,
        length_output_path=length_path,
    )

    # [CN] 4) 可视化：输出柱状图与相关性散点图
    # [EN] 4) Visualization: export comparison bar chart and correlation scatter plot.
    plot_accuracy_comparison(
        accuracy_path=accuracy_path,
        output_dir=figures_dir,
    )
    plot_accuracy_length_correlation(
        accuracy_path=accuracy_path,
        length_path=length_path,
        output_dir=figures_dir,
    )


if __name__ == "__main__":
    main()
