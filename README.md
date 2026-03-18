# CS6493 Topic 1: Prompt-based Math Reasoning Evaluation

本项目实现香港城市大学 CS6493 Topic 1 要求：

- 2 个指定模型
- 3 个指定数据集
- 4 种提示方法（Standard / CoT / Self-Refine / Least-to-Most）
- 自动化实验、指标统计与可视化

## 1. 环境要求

- Python 3.10+
- 建议 GPU: NVIDIA RTX 4090 (24GB)

## 2. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. 项目结构

```text
.
├── analysis/
│   ├── metrics.py
│   └── visualize.py
├── data/
│   ├── preprocess.py
│   └── processed/
├── experiments/
│   └── run.py
├── models/
│   └── loader.py
├── prompts/
│   └── templates.py
├── results/
│   └── figures/
├── main.py
└── requirements.txt
```

## 4. 运行方式

一键执行全流程：

```bash
python main.py
```

流程包括：
1. 预处理 GSM8K / MATH-500 / AIME2024
2. 运行模型 × 提示方法 × 数据集组合实验
3. 计算 accuracy 与 avg_length
4. 输出图表

## 5. 输出文件

- `results/raw_results.csv`
- `results/accuracy.csv`
- `results/length.csv`
- `results/figures/accuracy_comparison.png`
- `results/figures/accuracy_length_correlation.png`

## 6. 说明

- 所有随机操作使用 `seed=42`。
- 使用 4bit (`nf4`) + `bfloat16` 进行量化加载。
- 若模型无 `pad_token`，自动设置为 `eos_token`。
- `data/preprocess.py` 中提供了可运行的 AIME 样例数据，若课程要求严格使用官方原题，请替换为授课方指定版本后再运行。
- 源码注释采用双版本风格：每个核心函数均包含中文说明（版本1）与英文说明（版本2）。
