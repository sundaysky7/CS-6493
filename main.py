"""Main entrypoint for CS6493 Topic 1 pipeline.

中文：按课程要求顺序执行“预处理 -> 实验 -> 指标 -> 可视化”。
English: Execute the required pipeline in order: preprocess -> experiment -> metrics -> visualization.
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

# 中文：课程指定模型。
# English: Course-required model list.
MODEL_NAMES = [
    "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "deepseek-ai/DeepSeek-R1-Qwen-1.5B",
]

# 中文：4 种提示方法（标准基线 + 3 种指定方法）。
# English: Four prompting methods (baseline + three required methods).
PROMPT_METHODS = ["standard", "cot", "self_refine", "least_to_most"]

# 中文：3 个评测数据集。
# English: Three evaluation datasets.
DATASETS = ["gsm8k", "math500", "aime2024"]


def main() -> None:
    """Run the full pipeline.

    中文：
    1) 数据预处理
    2) 全量实验
    3) 指标计算
    4) 图表生成

    English:
    1) Dataset preprocessing
    2) Full experiment run
    3) Metrics computation
    4) Figure generation
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    preprocess_gsm8k()
    preprocess_math500()
    preprocess_aime2024()

    run_full_experiment(
        models=MODEL_NAMES,
        prompt_methods=PROMPT_METHODS,
        datasets=DATASETS,
    )

    calculate_accuracy_and_length()

    plot_accuracy_comparison()
    plot_accuracy_length_correlation()


if __name__ == "__main__":
    main()
