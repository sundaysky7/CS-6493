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

import logging

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
    "deepseek-ai/DeepSeek-R1-Qwen-1.5B",
]
PROMPT_METHODS = ["standard", "cot", "self_refine", "least_to_most"]
DATASETS = ["gsm8k", "math500", "aime2024"]


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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # [CN] 1) 数据预处理：统一输出为标准 question-answer JSON
    # [EN] 1) Data preprocessing: normalize all datasets into question-answer JSON.
    preprocess_gsm8k()
    preprocess_math500()
    preprocess_aime2024()

    # [CN] 2) 运行完整实验：遍历 model × method × dataset 并记录原始响应
    # [EN] 2) Run full experiment: iterate model × method × dataset combinations.
    run_full_experiment(
        models=MODEL_NAMES,
        prompt_methods=PROMPT_METHODS,
        datasets=DATASETS,
    )

    # [CN] 3) 指标统计：计算准确率与平均响应长度
    # [EN] 3) Metrics: compute grouped accuracy and average response length.
    calculate_accuracy_and_length()

    # [CN] 4) 可视化：输出柱状图与相关性散点图
    # [EN] 4) Visualization: export comparison bar chart and correlation scatter plot.
    plot_accuracy_comparison()
    plot_accuracy_length_correlation()


if __name__ == "__main__":
    main()
