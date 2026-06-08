
---

# README.md

```md
# IEEE-CIS Fraud Detection Course Project

本仓库用于完成 IEEE-CIS Fraud Detection 数据集上的电商交易欺诈检测课程实验。

本项目目标不是复现 Kaggle 排行榜最优方案，也不是提出原创模型，而是建立一个可重复运行、步骤清楚、结果可解释的机器学习实验流程。

当前阶段：第 0 阶段项目骨架。  
当前尚未完成真实模型训练，尚未产生真实实验指标或结论。

## 1. Project Goal

This project builds a reproducible machine learning experiment for IEEE-CIS fraud detection.

Main goals:

1. Build a complete reproducible pipeline from raw data to model evaluation.
2. Compare baseline and non-linear models under class imbalance.
3. Analyze whether identity-related features improve fraud detection.
4. Analyze whether missingness patterns contain useful fraud signals.

## 2. Repository Structure

```text
ieee-cis-fraud/
├── .gitignore
├── AGENTS.md
├── PROJECT_CONTRACT.md
├── README.md
├── requirements.txt
├── requirements-optional.txt
│
├── configs/
│   ├── data_config.yaml
│   └── model_config.yaml
│
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_pipeline.py
│   ├── feature_groups.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── metrics.py
│   ├── train.py
│   ├── evaluate.py
│   └── run_experiments.py
│
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_data_pipeline.py
│   ├── test_metrics.py
│   └── test_cli.py
│
├── reports/
│   ├── stage_summary.md
│   ├── figures/
│   │   └── .gitkeep
│   └── tables/
│       └── .gitkeep
│
└── logs/
    └── .gitkeep