# 文件一：PROJECT_CONTRACT.md

# IEEE-CIS欺诈检测课程项目合作合同

版本：v0.1
状态：第0阶段冻结
随机种子：42

## 1. 项目目标

本项目使用IEEE-CIS Fraud Detection数据集，完成一个可重复运行的电商交易欺诈识别实验。

项目不追求Kaggle排行榜成绩，不复现竞赛第一名方案，不使用深度学习或复杂模型融合。

本项目的主要目标是：

1. 建立从原始数据到模型结果的完整可复现实验流程；
2. 比较基础模型与非线性模型在类别不平衡数据上的表现；
3. 分析identity信息是否提升欺诈检测效果；
4. 分析缺失模式本身是否包含欺诈信号。

## 2. 冻结的研究问题

### 主问题一

加入identity和设备相关信息后，模型的PR-AUC、Recall和Precision是否提高？

### 主问题二

加入missing_count和字段缺失指示变量后，模型性能是否进一步变化？

### 可选扩展

在核心实验全部完成后，比较随机划分与按TransactionDT进行时间划分的结果。

时间划分不属于前70%阶段成果的必做内容。

## 3. 原始数据

核心原始文件：

* data/raw/train_transaction.csv
* data/raw/train_identity.csv

连接键：

* TransactionID

目标变量：

* isFraud

时间字段：

* TransactionDT

原始数据不得被修改、覆盖或上传到Git仓库。

## 4. 数据规模

### 冒烟测试

使用最多50000行数据，目的仅是验证代码能否运行。

冒烟测试结果不得直接作为正式实验结论。

### 正式阶段数据

目标总样本量约为120000条。

正式抽样原则：

1. 尽量保留全部isFraud=1样本；
2. 使用随机种子42抽取非欺诈样本；
3. 抽样后重新检查欺诈比例；
4. 测试集不得重新采样或人为平衡；
5. 若内存不足，可将目标样本量降低至80000，但必须在报告中记录。

## 5. 数据划分

核心实验使用分层随机划分：

* 训练集：60%
* 验证集：20%
* 测试集：20%

随机种子统一为42。

划分必须在模型预处理之前完成。

缺失值填补、类别编码、缩放、特征选择等操作，只能根据训练集拟合。

验证集用于模型选择和阈值选择。

测试集只用于最终评价，不得用于调参或决定特征。

## 6. 特征范围

核心实验暂不使用V1至V339系列匿名特征。

只有完成核心实验后，才能将少量V系列字段作为扩展实验加入。

不得将以下字段直接作为模型特征：

* TransactionID
* isFraud

TransactionDT在核心随机划分实验中默认不作为模型特征，仅保留用于追踪和可选时间划分。

## 7. 特征组

### transaction_basic

仅使用transaction表中的基础交易字段，包括：

* TransactionAmt
* ProductCD
* card系列
* addr系列
* dist系列
* P_emaildomain
* R_emaildomain
* C系列
* D系列
* M系列

最终字段以实际数据存在情况为准。

### transaction_identity

在transaction_basic基础上加入选定的identity字段，包括：

* id_01至id_15
* DeviceType
* DeviceInfo

若字段不存在或类型异常，应记录并跳过，不得静默报错。

### transaction_identity_missing

在transaction_identity基础上加入：

* missing_count
* 主要字段的is_missing指示变量

缺失指示变量的具体字段列表必须保存到feature_groups.json。

## 8. 模型范围

核心模型固定为：

1. DummyClassifier
2. LogisticRegression
3. LightGBM

如果LightGBM在30分钟内仍无法安装或稳定运行，则替换为：

* HistGradientBoostingClassifier，或
* RandomForestClassifier

不得因安装LightGBM阻塞整个项目。

不进行大规模GridSearch。

每个模型只使用一组经过说明的合理参数。

## 9. 评价指标

主指标：

* PR-AUC

辅助指标：

* ROC-AUC
* Precision
* Recall
* F1
* MCC
* Confusion Matrix
* Fraud Support

Accuracy可以记录，但不得作为核心结论依据。

所有指标统一以欺诈类别isFraud=1作为正类。

## 10. 数据接口

数据负责人最终输出：

* data/processed/train.parquet
* data/processed/valid.parquet
* data/processed/test.parquet
* data/processed/metadata.json
* reports/tables/feature_groups.json
* reports/tables/data_summary.csv

Parquet不可用时，可以使用CSV，但必须同步修改metadata.json和README。

处理后数据应保留原始数值和类别值。

编码、填补和缩放由模型Pipeline负责，不在数据输出前永久写死。

每份数据必须包含：

* isFraud
* 该阶段允许使用的特征
* TransactionID，用于样本追踪
* TransactionDT，用于可选时间分析

模型训练前必须主动排除TransactionID和默认情况下的TransactionDT。

## 11. 文件责任

### 数据负责人A

主要负责：

* src/data_loader.py
* src/data_pipeline.py
* src/feature_groups.py
* src/eda.py
* configs/data_config.yaml
* tests/test_data_loader.py
* tests/test_data_pipeline.py

### 模型与集成负责人B

主要负责：

* src/preprocessing.py
* src/models.py
* src/metrics.py
* src/train.py
* src/evaluate.py
* src/run_experiments.py
* configs/model_config.yaml
* tests/test_metrics.py
* tests/test_cli.py
* README.md
* requirements.txt
* requirements-optional.txt

任何人不得直接修改对方负责的文件。

发现接口问题时，应先记录问题，再由对应负责人修改。

## 12. Git规则

主分支：

* main

开发分支：

* feature/data-pipeline
* feature/modeling-evaluation

禁止直接在main上进行实验性修改。

每次提交应只包含一个清晰任务。

建议提交信息：

* chore: 初始化项目
* feat(data): 添加原始数据检查
* feat(data): 添加分层抽样与划分
* feat(model): 添加模型注册与配置
* feat(eval): 添加统一评价指标
* test: 添加数据管线测试
* docs: 更新实验说明

未经人工检查，Codex不得自行提交、推送、合并或删除分支。

## 13. 可复现要求

所有随机过程统一使用random_state=42或等价设置。

所有路径使用pathlib和相对路径。

不得在代码中出现个人电脑的绝对路径。

运行日志写入logs目录。

结果表写入reports/tables目录。

图表写入reports/figures目录。

关键配置写入configs目录，不得全部硬编码在Python文件中。

## 14. Codex使用规则

Codex必须先阅读：

1. AGENTS.md
2. PROJECT_CONTRACT.md
3. 当前任务指定的文件

Codex每次只能完成一个范围明确的小任务。

每次修改后必须：

1. 运行相关命令；
2. 运行相关测试；
3. 报告修改文件；
4. 报告测试结果；
5. 报告尚未解决的问题。

Codex不得：

* 下载数据；
* 修改原始数据；
* 自动扩大项目范围；
* 未经授权重构整个目录；
* 一次生成大量未经运行验证的代码；
* 使用测试集决定模型参数；
* 提交虚构的实验结果；
* 将占位数字写成真实结果。

## 15. 范围变更规则

需要修改以下事项时，两人必须共同同意，并更新本文件：

* 样本规模
* 数据划分
* 特征组定义
* 主要模型
* 主要评价指标
* 数据接口
* 文件责任

临时口头决定不视为正式变更。

## 16. 第0阶段完成标准

第0阶段完成时，应满足：

1. main包含完整目录骨架；
2. AGENTS.md和PROJECT_CONTRACT.md存在；
3. 原始数据未被Git追踪；
4. 两个开发分支已经创建并推送；
5. 两人Python环境可以运行；
6. 数据分支能够检查原始文件是否存在；
7. 模型分支能够显示实验CLI帮助信息；
8. 基础测试通过；
9. 两边均完成一次独立提交；
10. 两人的git status均为clean。

---

# 文件二：AGENTS.md

# Codex Project Instructions

## Required reading

Before making changes, read:

1. PROJECT_CONTRACT.md
2. The task prompt
3. The files assigned to the current role

The contract is the source of truth for project scope, interfaces and ownership.

## Role boundaries

The prompt will identify the current role as either:

* data owner
* modeling and integration owner

Only modify files explicitly assigned to that role.

Do not modify files owned by the other role.

When a required change falls outside the assigned files, stop and report:

* the file that needs modification
* the reason
* the expected interface change

Do not make the cross-boundary change yourself.

## Stage 0 restrictions

During stage 0:

* do not train models
* do not read the complete dataset into memory
* do not download data
* do not modify files under data/raw
* do not generate final experimental claims
* do not run expensive operations
* do not commit, push, merge or delete branches
* do not refactor the repository structure

## Data safety

Treat data/raw as read-only.

Never commit:

* raw datasets
* processed datasets
* serialized models
* local environment files
* secrets or credentials

Never overwrite the original CSV files.

## Coding conventions

Use:

* Python 3.10 or later
* pathlib for file paths
* type hints for public functions
* docstrings for modules and public functions
* logging instead of scattered print statements
* small testable functions
* deterministic random seeds
* UTF-8 text files
* English names for variables, functions and files

User-facing reports may be written in Chinese.

Avoid:

* absolute paths
* hidden global state
* broad exception handling
* silent fallback behavior
* duplicated preprocessing logic
* unnecessary dependencies
* notebook-only logic

## Configuration

Store adjustable values in YAML configuration files.

Do not duplicate configuration values across multiple files.

Use random seed 42 unless the contract is formally updated.

## Execution

Before editing, run:

* git branch --show-current
* git status --short

After editing:

1. run the smallest relevant command
2. run the relevant tests
3. inspect git diff
4. confirm no unrelated files changed

## Final response format

At the end of each task, report:

### Files changed

List every changed file.

### Commands run

List the commands that were executed.

### Test results

State which tests passed or failed.

### Decisions made

Explain any implementation choice not already fixed by the contract.

### Remaining issues

List unresolved problems, risks or required cross-role changes.

### Suggested commit message

Provide one concise commit message.

Do not claim success unless the relevant commands actually ran successfully.
