"""可视化模块 / Visualization module.

中文（版本1）：
    生成两类图：
    1) 分组准确率柱状图；
    2) 准确率-响应长度散点图，并给出 Pearson 相关系数。

English (Version 2):
    Produces two figures:
    1) grouped accuracy bar comparison;
    2) accuracy-vs-length scatter plot with Pearson correlation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_accuracy_comparison(
    accuracy_path: str = "results/accuracy.csv",
    output_dir: str = "results/figures",
) -> None:
    """
    中文（版本1）：
        从 accuracy.csv 绘制分面柱状图（按 dataset 分列、model 着色）。

    English (Version 2):
        Build faceted bar charts from accuracy.csv (column facets by dataset,
        hue by model).

    :param accuracy_path: 准确率结果路径 / accuracy csv path.
    :param output_dir: 图表输出目录 / figure output directory.
    """
    sns.set_theme(style="whitegrid")
    df = pd.read_csv(accuracy_path)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    g = sns.catplot(
        data=df,
        x="method",
        y="accuracy",
        hue="model",
        col="dataset",
        kind="bar",
        height=4,
        aspect=1.2,
    )
    g.set_axis_labels("Prompt Method", "Accuracy (%)")
    g.set_titles("Dataset: {col_name}")
    g.figure.suptitle("Accuracy Comparison by Model/Method/Dataset", y=1.05)
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=30)

    figure_path = output / "accuracy_comparison.png"
    g.figure.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(g.figure)


def plot_accuracy_length_correlation(
    accuracy_path: str = "results/accuracy.csv",
    length_path: str = "results/length.csv",
    output_dir: str = "results/figures",
) -> None:
    """
    中文（版本1）：
        合并 accuracy 与 length 结果后绘制散点图，并计算 Pearson r。

    English (Version 2):
        Merge accuracy and length tables, then generate a scatter plot and
        compute Pearson correlation coefficient.

    :param accuracy_path: 准确率结果路径 / accuracy csv path.
    :param length_path: 响应长度结果路径 / length csv path.
    :param output_dir: 图表输出目录 / figure output directory.
    """
    sns.set_theme(style="ticks")
    acc_df = pd.read_csv(accuracy_path)
    len_df = pd.read_csv(length_path)

    merged = pd.merge(acc_df, len_df, on=["model", "dataset", "method"], how="inner")
    corr = merged["accuracy"].corr(merged["avg_length"], method="pearson")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 6))
    ax = sns.scatterplot(
        data=merged,
        x="avg_length",
        y="accuracy",
        hue="model",
        style="dataset",
        s=100,
    )
    ax.set_title(f"Accuracy vs Response Length (Pearson r = {corr:.3f})")
    ax.set_xlabel("Average Response Length (words)")
    ax.set_ylabel("Accuracy (%)")

    figure_path = output / "accuracy_length_correlation.png"
    plt.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close()
