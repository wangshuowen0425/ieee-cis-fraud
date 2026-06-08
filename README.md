# IEEE-CIS Fraud Detection Course Project

This repository contains a reproducible course project for fraud detection on the IEEE-CIS Fraud Detection dataset. The project focuses on a clear experiment workflow, model comparison, and careful evaluation under class imbalance rather than Kaggle leaderboard optimization.

Current status: stage 0 only. Real model training, real evaluation metrics, figures, and experimental conclusions have not been produced yet.

## Project Structure

```bash
ieee-cis-fraud/
|-- configs/                 # YAML configuration files
|-- data/
|   |-- raw/                 # local raw CSV files, not committed
|   `-- processed/           # future processed datasets, not committed
|-- logs/                    # future run logs
|-- reports/
|   |-- figures/             # future figures
|   `-- tables/              # future result tables and metadata
|-- src/                     # project source modules
|-- tests/                   # automated tests
|-- AGENTS.md
|-- PROJECT_CONTRACT.md
|-- README.md
|-- requirements.txt
`-- requirements-optional.txt
```

## Raw Data Placement

Place the original CSV files here:

```bash
data/raw/train_transaction.csv
data/raw/train_identity.csv
```

Do not modify or overwrite the raw CSV files.

## Environment Setup

Create and activate a virtual environment in Windows Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

Install core dependencies:

```bash
python -m pip install -r requirements.txt
```

Install optional LightGBM dependency:

```bash
python -m pip install -r requirements-optional.txt
```

## Stage 0 Commands

Data check command placeholder:

```bash
python -m src.data_loader --help
```

Model configuration dry-run:

```bash
python -m src.run_experiments --config configs/model_config.yaml --dry-run
```

List registered models:

```bash
python -m src.run_experiments --config configs/model_config.yaml --list-models
```

Stage 0 does not train models, read the full dataset, generate final metrics, or make experimental claims.
