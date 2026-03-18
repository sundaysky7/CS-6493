# CS6493 Topic 1: Prompt-based Math Reasoning Evaluation

本项目实现香港城市大学 CS6493 Topic 1 的实验流程：2 个模型 × 4 种提示方法 × 3 个数据集。

This repository implements the CS6493 Topic 1 experiment pipeline:
2 models × 4 prompting methods × 3 datasets.

## 1. Environment / 环境

- Python 3.10+
- Recommended GPU / 建议显卡: RTX 4090 (24GB)

## 2. Install / 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Run / 运行

```bash
python main.py
```

Pipeline / 流程:
1. Preprocess datasets / 预处理数据
2. Run full experiment / 运行完整实验
3. Compute metrics / 计算指标
4. Generate figures / 生成图表

## 4. Output / 输出

- `results/raw_results.csv`
- `results/accuracy.csv`
- `results/length.csv`
- `results/figures/accuracy_comparison.png`
- `results/figures/accuracy_length_correlation.png`

## 5. Notes / 说明

- Seed is fixed to 42 for reproducibility / 固定 seed=42 保证可复现。
- Quantization uses 4bit NF4 + bfloat16 / 量化采用 4bit NF4 + bfloat16。
- If pad token is missing, it is set to eos token / 若无 pad_token 自动设为 eos_token。
- Source code now contains **both Chinese and English detailed comments** / 源码已包含中英文双语详细注释（版本1中文注释，版本2英文注释）。
