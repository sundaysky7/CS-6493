"""Visualization utilities.

中文：生成准确率柱状图与准确率-长度相关性散点图。
English: Creates accuracy bar chart and accuracy-length correlation scatter plot.
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
    绘制准确率对比柱状图。
    Plot grouped bar chart for accuracy comparison.

    中文：按数据集分面（col），按模型分色（hue），按 method 分组。
    English: Facet by dataset, color by model, and group by method.
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
    绘制准确率-响应长度散点图并计算皮尔逊相关系数。
    Plot accuracy-vs-length scatter and compute Pearson correlation.

    中文：将 accuracy 与 avg_length 按 (model, dataset, method) 合并后绘图。
    English: Merge accuracy and avg_length on (model, dataset, method) before plotting.
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
