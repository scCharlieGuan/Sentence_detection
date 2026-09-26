# 项目结构与论文复现说明

本项目依据 `UCL Dissertaiton/md files` 中第 3–6 章和附录 A–C 组织。研究目标是预测从专家复制片段投射得到的 toxicity 句子弱标签。factual 和 medical 保留为扩展接口，不能当作论文已经验证的任务。

## 1. 分层与阅读顺序

| 层次 | 文件 | 职责 |
| --- | --- | --- |
| 兼容入口 | `run_pipeline.py` | 启动 CLI，保留历史 `prepare_data`、`load_prepared` 导入 |
| 参数层 | `src/cli.py` | 解析参数，检查模式必需参数，默认选择论文可靠性协议 |
| 配置层 | `src/config.py` | 读取 YAML、校验配置、复制并验证 CLI 覆盖值 |
| 流程层 | `src/workflows.py` | 一个模式一个处理函数；统一日志与输出路径 |
| 数据层 | `src/data_loader.py`、`src/preprocessing.py` | 原始数据读取、清洗、句子切分、片段对齐与支持计数 |
| 标签层 | `src/weak_labels.py` | 派生不同支持规则，保留 `label_original` |
| 划分与持久化 | `src/splitting.py`、`src/data_pipeline.py` | 生成分组划分，保存／加载快照并检查行身份与指纹 |
| 输入与模型 | `src/input_features.py`、`src/models.py`、`src/dataset.py` | 构造句子／上下文输入与模型对象 |
| 训练层 | `src/training.py`、`src/classical_training.py`、`src/transformer_training.py` | 分派训练，分别实现传统模型与 ALBERT |
| 实验层 | `src/baselines.py`、`src/experiments.py` | 词法基线与标签规则敏感性实验 |
| 评估与审计 | `src/evaluate.py`、`src/audit.py`、`src/weak_label_analysis.py` | 指标、bootstrap、数据与预测审计 |
| 推理层 | `src/predict.py` | 加载保存模型及阈值，生成句子预测 |

建议沿“入口 → CLI → 流程 → 数据／训练模块”阅读。研究脚本直接调用业务模块，不需要经过命令行。`legacy/`、`notebooks/` 和历史产物不属于主运行路径。

## 2. 论文与实验对应关系

| 论文部分 | 实现与运行入口 |
| --- | --- |
| 第 3 章、附录 A：弱标签构造 | `preprocessing.py`、`weak_labels.py`；`--mode prepare` |
| E1：对齐诊断 | `weak_label_analysis.py`；`--mode weak-label-audit` |
| E2：标签规则敏感性 | `experiments.py`；`--mode label-sensitivity` |
| E3：六种划分、种子 42–44 | `scripts/run_split_sensitivity.py` |
| E4：词法、冻结 BGE、ALBERT | `baselines.py` 及两类训练后端 |
| E5：配对差异与人工复核队列 | `evaluate.py`、`audit.py`；`--mode compare`、`--mode audit` |
| 第 5 章、附录 C：结果整理 | `scripts/build_research_report.py` 与已有结果快照 |
| 第 6 章 | 解释边界与未来扩展，不作为新训练目标 |

## 3. 协议与兼容性

- 默认配置是 `configs/reliability_protocol.yaml`。`config.yaml` 和 `thesis_protocol.yaml` 是其他／历史配置，按需显式传入。
- 精确匹配优先，fuzzy 使用 `best_window`；主阈值 0.78、回退阈值 0.68，最大连续窗口 3 句。
- 按 questionID + responder 汇总不同标注者；同一标注者不会重复投票。
- 主划分使用 questionID 近重复连通分量，字符相似度阈值 0.90、最短 fuzzy 文本 30 字符、500 次分配尝试。
- 数据指纹沿用附录 A 的字段与序列化方式。加载时核对指纹与划分表行身份；新生成的 manifest 额外保存 `split_file_sha256`。旧 manifest 没有该字段时仍可读取。
- 指纹不覆盖所有派生标签，也不证明标签本身正确。旧快照没有 split hash，无法仅凭旧 manifest 检出所有有效的重新分配。
- JSON 中的 `pr_auc` 实际是 Average Precision（AP），保留键名以兼容已有报告。
- 阈值按验证集 F1 选择；测试集不得用于选参。BGE 的内部 CV 仍按 questionID 分组，保持论文附录 B 披露的协议与局限。
- `--mode train --model-name tfidf_lr` 使用通用传统模型训练路径；论文 E4 的主词法模型应使用 `--mode baselines`，E2 固定模型应使用 `--mode label-sensitivity`，三者不是同一个调参协议。
- ALBERT 通用配置的默认 loss 不代表论文 weighted-CE 条件；直接训练论文条件时显式传入 `--loss weighted_cross_entropy`。标签敏感性流程会自动设置该 loss。

## 4. 常用命令

在项目根目录、已安装 requirements 的 Python 环境中运行：

```powershell
python run_pipeline.py --help
python run_pipeline.py --mode prepare
python run_pipeline.py --mode diagnose
python run_pipeline.py --mode weak-label-audit
python run_pipeline.py --mode label-sensitivity --sensitivity-models tfidf,albert
python run_pipeline.py --mode baselines --seeds 42
python -m scripts.run_split_sensitivity --config configs/reliability_protocol.yaml --seeds 42,43,44
python run_pipeline.py --mode train --model-name sbert_lr --run-name bge_target
python run_pipeline.py --mode train --model-name sbert_lr --input-mode target_next --run-name bge_target_next
python run_pipeline.py --mode train --model-name albert --loss weighted_cross_entropy --run-name albert_weighted_ce
python -m pytest -q
```

这些是复现操作说明，不表示本次重构已经重新训练所有模型。重新 prepare、训练和审计会写入配置指定目录；如需独立新实验，请复制配置并修改 `processed_dir`、`model_dir`、`report_dir`。原始配置保留以便追溯论文已存结果。

`--seed` 对已有 prepared 数据不会重新划分；E3 的 split seed 由专用脚本控制。`--mode compare` 只需要两个预测文件，无须加载原始或 prepared 数据。

## 5. 本次检查范围

修改前已有 14 项测试通过。新增测试覆盖 CLI 默认协议及覆盖校验、非法划分比例与阈值网格、预测文件命名冲突、快照指纹错配，以及 CSV 行顺序变化后的划分复用。

最终 **43 项测试全部通过**；命令行帮助、Python 编译检查通过。使用临时输出目录跑通 TF-IDF 训练、评估、导出和两句新文本推理，确认导出概率一致。BGE／ALBERT 没有重新训练，已有模型及结果未被覆盖。

本地论文主快照读取确认：2,907 行；train / val / test 为 2,035 / 429 / 443；数据 SHA-256 为 `cbb3cabf8249d351cdd3c86bb867e9e2059320009c36ed666ae505bec6c10aba`，与附录 B 一致。
