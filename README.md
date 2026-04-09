# CS6493 Topic 1: Prompt-based Math Reasoning Evaluation

本项目实现香港城市大学 CS6493 Topic 1 要求：

- 2 个指定模型
- 3 个指定数据集
- 4 种提示方法（Standard / CoT / Self-Refine / Least-to-Most）
- 自动化实验、指标统计与可视化

## 1. 环境要求

- Python 3.10+
- 建议 GPU: NVIDIA RTX 4090 (24GB)
- 支持 CPU fallback：当未检测到 CUDA，或显式使用 `--force-cpu` 时，将自动回退到 CPU 普通加载

## 2. 安装依赖

推荐在 **Windows + CPU** 环境下使用较短的项目路径（如 `D:\cs6493\proj`），以避免 `pip install` 时触发 Windows 路径过长问题。

### Windows PowerShell

```bash
python -m venv v
.\v\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

若 PowerShell 提示脚本执行被禁止，可先执行：

```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

然后重新执行：

```bash
.\v\Scripts\Activate.ps1
```

### Windows CMD

```bash
python -m venv v
v\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

如需快速验证流程，可使用小样本模式；如需在无 GPU 环境运行，可启用 CPU fallback。

当前 `requirements.txt` 已更适合 **Windows + CPU** 环境：
- Windows 下不会安装 `bitsandbytes`
- 非 Windows 环境仍可保留量化 GPU 依赖路径

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

常用运行方式：

```bash
# 自动模式：有 GPU 则走原 4bit 量化路径；无 GPU 则自动回退到 CPU
python main.py

# 显式强制使用 CPU
python main.py --force-cpu

# 小样本模式：每个数据集只取前 5 条
python main.py --max-samples-per-dataset 5

# CPU + 小样本模式，适合本地快速调试
python main.py --force-cpu --max-samples-per-dataset 5

# 断点续跑：继续写入某个历史运行目录
python main.py --force-cpu --max-samples-per-dataset 5 --resume-run-dir results/run_20260408_182221_cpu_samples5
```

流程包括：
1. 预处理 GSM8K / MATH-500 / AIME2024
2. 运行模型 × 提示方法 × 数据集组合实验
3. 计算 accuracy 与 avg_length
4. 输出图表

## 5. 输出文件

每次运行都会自动创建一个独立结果目录，例如：

- `results/run_20260408_182221_cpu_samples5/`

该目录下包含：

- `raw_results.csv`
- `accuracy.csv`
- `length.csv`
- `figures/accuracy_comparison.png`
- `figures/accuracy_length_correlation.png`
- `run_config.json`

如需断点续跑，可通过 `--resume-run-dir <历史运行目录>` 继续往同一个目录中写入结果。

## 6. 说明

- 所有随机操作使用 `seed=42`。
- 在 CUDA 可用且未指定 `--force-cpu` 时，优先尝试使用 4bit (`nf4`) + `bfloat16` 进行量化加载，保留原 GPU 路径不变。
- 当 CUDA 不可用，显式指定 `--force-cpu`，或量化依赖在当前环境不可用时，模型将自动回退到 CPU 普通加载。
- 若模型无 `pad_token`，自动设置为 `eos_token`。
- 可通过 `--max-samples-per-dataset N` 启用小样本模式，对每个数据集仅使用前 `N` 条样本，便于快速联调与验证流程。
- 原始实验结果采用增量保存：每成功完成一条样本推理，都会立即追加写入当前运行目录下的 `raw_results.csv`。
- 每次执行 `python main.py` 都会自动新建一个独立结果目录，避免覆盖历史实验结果，也便于中途中断后保留已完成部分。
- 可通过 `--resume-run-dir <历史运行目录>` 启用断点续跑：程序会继续使用该目录，并跳过已经写入 `raw_results.csv` 的样本。
- 断点续跑时，应保持与原运行一致的关键配置（如 `--force-cpu`、`--max-samples-per-dataset`），避免把不同实验设置混写到同一个结果目录中。
- `data/preprocess.py` 中提供了可运行的 AIME 样例数据，若课程要求严格使用官方原题，请替换为授课方指定版本后再运行。
- 源码注释采用双版本风格：每个核心函数均包含中文说明（版本1）与英文说明（版本2）。
