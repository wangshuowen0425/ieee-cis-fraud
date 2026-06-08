# Codex 项目执行规则

## 1. 必须先阅读的文件

每次开始修改前，必须先阅读：

1. PROJECT_CONTRACT.md
2. 当前任务提示词
3. 当前角色被允许修改的文件

PROJECT_CONTRACT.md 是本项目的最高约束文件，项目范围、数据接口、文件责任、模型范围、评价指标和 Git 规则都以它为准。

如果当前任务提示词与 PROJECT_CONTRACT.md 冲突，应立即停止并报告冲突，不要自行猜测。

## 2. 角色边界

当前任务会指定角色：

- 数据负责人
- 模型、评价与集成负责人

只能修改当前角色被明确允许修改的文件。

不得修改另一位同学负责的文件。

如果发现必须修改对方负责的文件，应停止并报告：

- 需要修改的文件名
- 为什么需要修改
- 希望对方提供或调整什么接口

不得自行跨角色修改。

## 3. 第 0 阶段限制

第 0 阶段只允许建立项目骨架、配置检查、原始文件存在性检查、模型注册骨架、指标名称验证和 CLI dry-run。

第 0 阶段禁止：

- 训练模型
- 读取完整数据集
- 下载数据
- 修改 data/raw 下的文件
- 生成最终实验结论
- 运行高成本操作
- 创建 Notebook
- 重构项目目录
- 自动提交、推送、合并或删除分支

## 4. 数据安全规则

data/raw 视为只读目录。

不得提交以下内容：

- 原始数据集
- 处理后的数据集
- 训练好的模型文件
- 本地虚拟环境
- 本地缓存文件
- 密钥、token 或任何个人凭证

不得覆盖原始 CSV 文件。

原始文件应放置在：

- data/raw/train_transaction.csv
- data/raw/train_identity.csv

后续生成的处理数据应放入：

- data/processed/

结果表、图像和日志应分别放入：

- reports/tables/
- reports/figures/
- logs/

## 5. 代码风格

应使用：

- Python 3.10 或更高版本
- pathlib 处理路径
- 类型标注
- 模块和公开函数的 docstring
- logging，而不是到处 print
- 小而可测试的函数
- 确定性的随机种子
- UTF-8 文本文件
- 英文变量名、函数名和文件名

面向用户或报告的文字可以使用中文。

避免：

- 个人电脑绝对路径
- 隐藏的全局状态
- 过宽泛的异常捕获
- 静默 fallback
- 重复的预处理逻辑
- 不必要的依赖
- 只能靠 Notebook 手动运行的逻辑

## 6. 配置规则

可调整参数应写在 YAML 配置文件中。

不要在多个文件中重复写同一个配置值。

除非 PROJECT_CONTRACT.md 被正式更新，否则统一使用随机种子：

- 42

实验设置不要全部硬编码在 Python 文件中，应该优先放在 configs/ 目录下。

## 7. 模型与评价规则

核心模型为：

- DummyClassifier
- LogisticRegression
- LightGBM

如果 LightGBM 无法安装或稳定运行，可使用合同中允许的替代模型：

- HistGradientBoostingClassifier
- RandomForestClassifier

不得把替代模型伪装成 LightGBM。

主指标为：

- PR-AUC

辅助指标包括：

- ROC-AUC
- Precision
- Recall
- F1
- MCC
- Confusion Matrix
- Fraud Support
- Accuracy

Accuracy 可以记录，但不得作为核心结论依据。

正类永远是：

- isFraud = 1

## 8. 防止数据泄漏

训练集、验证集、测试集的划分必须发生在模型预处理之前。

以下操作只能在训练集上 fit：

- 缺失值填补
- 类别编码
- 数值缩放
- 特征选择
- 阈值选择

验证集可以用于模型选择或阈值选择。

测试集只能用于最终评价，不得用于调参、选特征或选阈值。

## 9. 执行要求

修改文件前，必须运行：

```bash
git branch --show-current
git status --short