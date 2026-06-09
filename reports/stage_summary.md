# 实验报告概览

本报告汇总 IEEE-CIS Fraud Detection 项目从数据准备到模型评价、阈值优化、错误分析以及时间划分扩展实验的主要流程与结果。项目目标是建立一个可复现实验流程，比较不同模型与特征组在欺诈检测任务中的表现，并分析阈值选择和时间划分对最终评价的影响。

本项目统一以 `isFraud = 1` 为正类，主指标为 PR-AUC，辅助指标包括 ROC-AUC、Precision、Recall、F1、MCC、Accuracy 和混淆矩阵。核心结果表和图像分别保存在 `reports/tables/` 与 `reports/figures/`。

## Stage0 数据准备

### 方法

Stage0 主要完成项目骨架、配置文件、数据入口和可复现实验入口的准备工作。原始数据要求放置在：

- `data/raw/train_transaction.csv`
- `data/raw/train_identity.csv`

该阶段只进行文件存在性、配置结构和命令行入口检查，不训练模型，不生成实验结论，也不修改原始数据。

### 结果

Stage0 形成了后续实验所需的基础结构：

- 数据配置：`configs/data_config.yaml`
- 模型配置：`configs/model_config.yaml`
- 数据加载与处理入口：`src/data_loader.py`、`src/data_pipeline.py`
- 模型与评价入口：`src/models.py`、`src/metrics.py`、`src/run_experiments.py`
- 项目约束文档：`PROJECT_CONTRACT.md`

### 小结

Stage0 的主要价值是冻结项目边界和实验接口，保证后续阶段不会直接修改原始数据，也不会将未验证结果写成正式结论。

## Stage1 特征工程与探索性分析

### 方法

Stage1 将原始 transaction 与 identity 表按 `TransactionID` 合并，并构建后续建模所需的特征组。项目核心特征组包括：

- `transaction_basic`：基础交易字段，如金额、产品代码、银行卡字段、地址字段、距离字段、邮箱域名、C/D/M 系列字段。
- `transaction_identity`：在基础交易字段上加入部分 identity 和设备字段。
- `transaction_identity_missing`：在 identity 特征基础上加入缺失数量和缺失指示变量。

数据负责人输出处理后的训练、验证、测试数据，以及特征组定义和数据摘要。

### 结果

Stage1 的关键输出包括：

- `reports/tables/feature_groups.json`
- `reports/tables/data_summary.csv`
- `reports/tables/stage2_data_summary.csv`
- `reports/tables/stage2_feature_profile.csv`
- `reports/tables/stage2_data_quality.json`

相关图表和数据质量检查结果保存在 `reports/figures/` 与 `reports/tables/` 中。

### 小结

Stage1 明确了模型可使用的字段范围，并将 `TransactionID`、`isFraud` 和默认情况下的 `TransactionDT` 排除在模型特征之外，降低数据泄漏风险。缺失模式被作为可分析信号保留，但不对匿名字段做无依据语义解释。

## Stage2 模型训练与选择

### 方法

Stage2 在正式样本上进行模型训练、特征组消融和最终测试评价。核心模型包括：

- DummyClassifier
- LogisticRegression
- LightGBM

实验使用训练集拟合预处理器和模型，在验证集上比较模型与特征组表现，并基于验证集结果选择最终模型和特征组。测试集只用于最终评价，不参与模型选择或特征组选择。

### 结果

Stage2 的主要结果文件包括：

- `reports/tables/stage2_model_comparison_valid.csv`
- `reports/tables/stage2_ablation_valid.csv`
- `reports/tables/stage2_final_test.csv`

其中，验证集模型比较和特征组消融用于冻结后续 Stage3 的模型与特征组。最终选择沿用到 Stage3：`lightgbm` + `transaction_basic`。

### 小结

Stage2 表明非线性树模型在该欺诈检测任务中优于基础模型。特征组选择基于验证集完成，避免使用测试集重新挑选模型或特征。后续阈值优化和错误分析均建立在 Stage2 冻结选择之上。

## Stage3 错误分析与阈值优化

### 方法

Stage3 在 Stage2 选定模型和特征组的基础上进行阈值优化。流程如下：

1. 在训练集上重新拟合模型。
2. 在验证集上生成连续预测分数。
3. 在验证集上搜索成本敏感阈值。
4. 在测试集上比较默认阈值 `0.5` 与验证集选出的成本敏感阈值。

成本公式为：

> Cost = 10 * FN + 1 * FP

阈值搜索只使用验证集，测试集只用于最终比较。

### 结果

Stage3 的核心输出包括：

- `reports/tables/stage3_threshold_search_valid.csv`
- `reports/tables/stage3_selected_threshold.json`
- `reports/tables/stage3_threshold_comparison_test.csv`
- `reports/figures/stage3_threshold_cost_valid.png`
- `reports/figures/stage3_threshold_precision_valid.png`
- `reports/figures/stage3_threshold_recall_valid.png`
- `reports/figures/stage3_confusion_matrix_default_test.png`
- `reports/figures/stage3_confusion_matrix_cost_sensitive_test.png`
- `reports/figures/stage3_threshold_comparison_test.png`

Stage3 还基于真实测试集逐样本预测文件完成错误分析，主要输出包括：

- `reports/tables/stage3_error_count_comparison.csv`
- `reports/tables/stage3_error_amount_summary.csv`
- `reports/tables/stage3_error_category_summary.csv`
- `reports/tables/stage3_representative_errors.csv`
- `reports/figures/stage3_fp_fn_comparison.png`
- `reports/figures/stage3_error_amount_boxplot.png`
- `reports/figures/stage3_missing_count_by_error_type.png`
- `reports/figures/stage3_error_by_product_code.png`

### 小结

Stage3 的主要结论是：默认阈值更偏向高 precision，而成本敏感阈值显著提高 recall，并减少高成本的 FN。由于欺诈检测中漏报成本通常高于误报成本，成本敏感阈值更符合本项目设定的业务代价目标。错误分析进一步帮助定位 FP 和 FN 在金额、类别字段和缺失数量上的分布差异，但这些差异只表示相关性，不构成因果解释。

## Stage3 Extension 时间划分实验

### 方法

时间划分实验是核心实验完成后的扩展分析，用于比较随机划分与按 `TransactionDT` 时间顺序划分时的模型泛化差异。该实验不替代主实验，只作为更接近未来交易评估方式的补充。

实验将 `data/processed/stage2_formal` 中已有的 train、valid、test 重新合并为 120,000 条正式样本，按 `TransactionDT` 升序排序后重新切分：

| split | rows | fraud count | fraud rate | TransactionDT range |
| --- | ---: | ---: | ---: | --- |
| train | 72,000 | 2,427 | 3.37% | 86,499 - 8,724,304 |
| valid | 24,000 | 954 | 3.98% | 8,724,330 - 12,234,209 |
| test | 24,000 | 818 | 3.41% | 12,234,253 - 15,811,047 |

模型和特征组沿用 Stage3 冻结设置：`lightgbm` + `transaction_basic`。阈值仍然只在 validation 集上选择，成本公式继续使用 `10 * FN + 1 * FP`。

### 结果

时间划分 validation 集上选出的成本敏感阈值为 `0.06`。

time-split test 集结果如下：

| threshold | PR-AUC | ROC-AUC | precision | recall | F1 | FP | FN | cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| default 0.5 | 0.4905 | 0.8873 | 0.7719 | 0.3227 | 0.4552 | 78 | 554 | 5,618 |
| cost-sensitive 0.06 | 0.4905 | 0.8873 | 0.2850 | 0.6540 | 0.3970 | 1,342 | 283 | 4,172 |

与随机划分 Stage3 test 结果相比：

| split strategy | threshold | PR-AUC | ROC-AUC | cost | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| random_stage3 | default 0.5 | 0.6307 | 0.9181 | 5,158 | 48 | 511 |
| random_stage3 | cost-sensitive 0.07 | 0.6307 | 0.9181 | 3,307 | 1,047 | 226 |
| time_split | default 0.5 | 0.4905 | 0.8873 | 5,618 | 78 | 554 |
| time_split | cost-sensitive 0.06 | 0.4905 | 0.8873 | 4,172 | 1,342 | 283 |

相关输出文件包括：

- `reports/tables/time_split_data_summary.csv`
- `reports/tables/time_split_threshold_search_valid.csv`
- `reports/tables/time_split_selected_threshold.json`
- `reports/tables/time_split_threshold_comparison_test.csv`
- `reports/tables/time_split_vs_random_comparison.csv`
- `reports/figures/time_split_threshold_cost_valid.png`
- `reports/figures/time_split_threshold_precision_valid.png`
- `reports/figures/time_split_threshold_recall_valid.png`
- `reports/figures/time_split_confusion_matrix_default_test.png`
- `reports/figures/time_split_confusion_matrix_cost_sensitive_test.png`
- `reports/figures/time_split_vs_random_pr_auc.png`
- `reports/figures/time_split_vs_random_cost.png`

### 小结

时间划分下 PR-AUC 从随机划分的 `0.6307` 降至 `0.4905`，成本敏感阈值下 cost 从 `3,307` 增至 `4,172`。这说明在同一模型和特征组下，时间顺序评估更困难，模型在未来时间段上的泛化表现弱于随机划分结果。

需要注意的是，时间划分实验只能说明该样本和该切分方式下的泛化风险更高，不能直接声称代表真实银行部署效果。

## Stage3 Full Report 汇总

### 方法

最终汇总阶段整合 Stage3 阈值优化、错误分析和时间划分扩展结果，形成面向作业提交和答辩的结果说明。汇总重点包括：

- 选定模型与特征组是否稳定。
- 默认阈值与成本敏感阈值的取舍。
- FP/FN 的数量变化和错误分布。
- 随机划分与时间划分下的泛化差异。

### 结果

综合结果文件包括：

- Stage2 模型选择：`reports/tables/stage2_model_comparison_valid.csv`
- Stage2 特征组消融：`reports/tables/stage2_ablation_valid.csv`
- Stage3 阈值比较：`reports/tables/stage3_threshold_comparison_test.csv`
- Stage3 错误分析：`reports/tables/stage3_error_count_comparison.csv`
- 时间划分对比：`reports/tables/time_split_vs_random_comparison.csv`

图表可用于答辩展示：

- `reports/figures/stage3_threshold_comparison_test.png`
- `reports/figures/stage3_fp_fn_comparison.png`
- `reports/figures/time_split_vs_random_pr_auc.png`
- `reports/figures/time_split_vs_random_cost.png`

### 小结

本项目完成了从数据准备、特征工程、模型训练、模型选择、阈值优化、错误分析到时间划分扩展的完整实验流程。主实验结果支持使用 LightGBM 与 `transaction_basic` 特征组作为最终模型设置；成本敏感阈值能够减少 FN 并降低设定成本；时间划分实验显示随机划分可能高估未来时间段上的泛化表现。

最终结论应保持谨慎：本项目结果适合作为课程实验中的可复现实证分析，不应被解释为线上生产系统的直接性能保证。
