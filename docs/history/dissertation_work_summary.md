# Sentence Error Detection 项目工作总结

## 项目工作概述

本项目的主要研究目标，是在心理健康问答数据中识别包含潜在问题的具体句子，从而支持对生成式或人工心理健康建议的安全性、可靠性与可解释性评估。项目配置层面支持 `toxicity`、`factual` 和 `medical` 三类二元检测任务；但从当前可核实的处理数据、训练日志和结果文件来看，完整的论文级实验目前只在 `toxicity` 任务上执行。因此，本报告对 toxicity 结果作实证总结，对 factual consistency 和 inappropriate medical advice 仅说明已有代码能力及尚待完成的工作，不将其描述为已完成实验。

已经形成的整体流程为：首先对 CounselBench 原始标注数据进行探索与结构检查；随后将专家复制的错误文本跨度对齐到完整回答中的句子，构建带标注者支持数的句子级弱标签；再以 `questionID` 为分组单位固定训练、验证和测试集，并检查重复、近重复、来源偏差和标签冲突；在统一协议下比较 TF-IDF、固定 Sentence-BERT embedding 分类器和 ALBERT 微调；最后通过多随机种子、paired bootstrap、McNemar test、模型分歧样本导出以及专门的弱标签质量审计，判断模型差异及当前性能上限。

各处理任务之间具有清晰的依赖关系：原始数据检查为 span 对齐规则提供依据；弱标签和固定划分构成所有模型的共同输入；数据审计决定哪些性能结果可以被视为可信；强词汇基线用于判断任务中的 lexical signal；SBERT 与 ALBERT 实验检验预训练语义表示、上下文和微调策略的增益；统计检验与弱标签审计则用于确定观察到的模型差异是否超过随机波动，以及继续调参是否仍是最有效的研究投入。

当前已经形成的主要成果包括：一套模块化且可配置的 sentence-level detection 流水线；2,848 条 toxicity 句子级弱标签及固定的 question-grouped split；8 类 TF-IDF 强基线、多个固定 embedding 与上下文消融、5 类 ALBERT 训练策略；模型、训练历史、验证/测试预测和统一指标文件；多随机种子与配对显著性检验；530 行模型错误审计表；以及包含 306 条高风险弱标签句子和 44 条未对齐 span 的专项质量审计。需要特别强调的是，当前没有人工裁决后的句子级 gold labels，因而弱标签的真实 precision、recall 和噪声率仍无法计算。

### 证据表述约定

- “已经执行”或“已经生成”表示当前工作区中存在相应数据、日志、模型、预测或结果文件，并可核实其内容。
- “代码已经实现”表示源代码中存在该功能，但当前没有足够的结果文件证明已完成正式实验。
- “代码预期生成”或“需要进一步验证”表示仅能确认接口、配置或输出定义，不能确认实际运行结果。

---

## 任务 1：原始 CounselBench 数据结构与标注模式探索

### 1. 为什么需要进行该任务

项目的目标不是对完整回答作单一分类，而是将专家指出的问题定位到具体句子。原始数据以重复评价记录为单位，包含完整回答、评分、原因和 annotator copied span，并不直接提供句子标签。因此，在建立模型前必须确认数据规模、重复结构、字段缺失和标注列的实际形式，否则容易把重复标注误当成独立回答，或错误解释 `*_copy` 字段。

该任务也直接关系到可持续发展语境下的负责任 AI 评估：心理健康建议属于高敏感度应用，如果数据来源、标注覆盖或重复模式不清楚，模型性能可能只反映数据收集流程，而不是对有害内容的真实识别能力。

### 2. 该任务包括什么

输入为 `data/raw/CounselBench.csv`。探索工作检查了行列规模、字段类型、缺失值、唯一值、重复回答，以及 `toxicity_copy`、`toxicity_reason` 和 `medical_copy` 等字段的示例关系。

#### 具体处理步骤

1. 读取原始 CSV 并检查字段结构，确认数据包含 question、response、topic、responder、`survey_id` 以及三类任务对应的 score/copy/reason 字段。
2. 统计问题、回答、回答来源和主题的唯一值，识别同一回答在不同评价记录中的重复。
3. 单独检查 `toxicity_copy` 的非空覆盖情况，以及“存在 toxicity reason 但 copied span 为空”的记录。
4. 检查不同 question/responder 组合中完全相同的回答，发现通用模板回答可能跨问题复用；这一发现后来成为分组划分和弱标签聚合审计的重要依据。

#### 涉及的文件

- `data/raw/CounselBench.csv`：包含 2,000 条原始评价记录和 20 个字段，是所有后续处理的输入。
- `notebooks/row-data_eda.ipynb`：原始数据探索 notebook，当前保存有执行输出。
- `src/data_loader.py`：正式流水线的数据读取与必需字段检查模块。
- `data/README.md`：说明原始数据路径及处理后数据目录。

### 3. 该任务的处理结果

当前文件可以验证以下结果：

- 原始数据共有 2,000 行、20 列、100 个 `questionID`、372 个唯一 response、20 个 topic、4 类 responder 和 100 个 `survey_id`。
- `questionText` 有 220 行缺失；完整 `response` 无缺失。
- 有 506 行 `toxicity_copy` 非空，但非空单元格不等于最终可用 span 数，因为其中包含无效标记、重复文本或需要解析的格式。
- notebook 先按 `questionID` 与 `responder` 去重得到 400 个 question–responder 组合，其中有 31 条记录属于重复 response。
- 当前 notebook 保存了对 `medical_copy` 异常取值和“有 toxicity reason、无 toxicity copy”记录的探索样例，但未输出经过系统整理的缺失率报告。

### 4. 处理结果说明了什么

原始数据的 2,000 行并非 2,000 个独立回答，而是围绕 372 个回答形成的重复专家评价。模型训练和数据划分必须在回答或更高层级进行分组，不能按原始行随机切分。31 条跨 question–responder 组合的重复回答也表明，仅按回答文本或仅按问题分组各有局限，需要同时审计 question leakage、response leakage 和模板复用。

这些探索结果为下一项任务提供了两个直接要求：一是建立模块化配置与可复现协议，避免 notebook 中的探索逻辑成为唯一数据来源；二是把 copied span 系统地转换成可审计的句子标签。

---

## 任务 2：建立模块化、可配置和可复现的实验框架

### 1. 为什么需要进行该任务

早期实现集中在单一 toxicity 脚本中，路径、任务字段、训练与评估逻辑相互耦合，不利于比较不同模型，也容易在不同运行中混用数据划分、阈值和输出目录。硕士 dissertation 需要清楚说明每个模型是否在完全一致的数据协议下比较，并能够保存配置、预测和训练历史以供复核，因此需要将单体代码重构为职责明确的研究流水线。

### 2. 该任务包括什么

该任务将配置读取、数据处理、输入构造、模型建立、训练、评估、审计和推理拆分到 `src/` 模块，并通过 `run_pipeline.py` 提供统一命令行入口。正式 thesis 协议与旧探索协议使用不同的数据、模型和报告目录，防止结果相互覆盖。

#### 具体处理步骤

1. 使用 YAML 集中管理任务、数据路径、分组方式、模型、训练参数、阈值网格和输出位置。
2. 对任务名与 target column、split 比例、模型名、输入模式和模型选择指标进行配置校验。
3. 在 `run_pipeline.py` 中组织 `prepare`、`diagnose`、`baselines`、`train`、`evaluate`、`audit`、`compare`、`export` 和 `predict` 模式。
4. 统一固定 Python、NumPy 和 PyTorch 随机种子，并在 CUDA 可用时选择 GPU，否则使用 CPU。
5. 将原始单体实现保存在 `legacy/`，保留方法演化记录；正式代码则通过模块间调用构成可复现流水线。
6. 为关键配置、预处理、输入边界、指标、阈值和统计函数建立轻量测试。

#### 涉及的文件

- `legacy/toxicity_detection_v2.py`：保留的早期单体实现，用于追溯原始方法。
- `run_pipeline.py`：统一 CLI 入口；当前工作区修改已将默认配置指向 `configs/thesis_protocol.yaml`。
- `configs/config.yaml`：旧的 response-level 探索配置及更广的调参定义。
- `configs/thesis_protocol.yaml`：论文主协议，固定 question-grouped split 和隔离输出目录。
- `src/config.py`：配置加载、路径解析和一致性校验。
- `src/utils.py`：随机种子、设备、目录、日志和 JSON 输出工具。
- `src/data_loader.py`、`src/preprocessing.py`、`src/input_features.py`、`src/models.py`、`src/train.py`、`src/evaluate.py`、`src/audit.py`、`src/predict.py`：流水线各功能模块。
- `tests/test_config.py`、`tests/test_preprocessing.py`、`tests/test_evaluate.py`、`tests/test_input_features.py`：轻量回归测试。
- `README.md`、`REFACTORING_REPORT.md`：项目使用方法、结构和早期重构说明。

### 3. 该任务的处理结果

代码已经形成明确调用关系：`run_pipeline.py` 先通过 `src/config.py` 载入协议，再调用数据读取与预处理；已准备数据可进入 `src/audit.py` 或 `src/train.py`；训练阶段调用 `src/input_features.py`、`src/models.py`、`src/dataset.py` 和 `src/evaluate.py`；推理则通过 `src/predict.py` 读取已保存模型和 threshold。

`data/processed_question_grouped`、`models/thesis` 和 `reports/thesis` 已实际生成，说明隔离的 thesis 流程已经执行。`protocol_manifest.json` 保存了任务、seed、数据量、split 大小和数据集 SHA-256 指纹，并明确标记 test set 禁止参与模型选择。当前检查期间在 `MentalFlow` 环境执行了 `pytest`，结果为 7 项测试全部通过。

需要区分的是：测试覆盖了关键轻量函数，不等同于对所有模型训练路径和外部模型下载的端到端测试。当前 Git 历史只有两个较大版本提交；未提交工作区中可确认的实质新增内容主要是弱标签质量脚本及报告，另有默认 thesis 配置入口修改。`src/preprocessing.py` 的当前未提交 diff 主要为注释附近的空白调整，不应被描述为新的算法修改。

### 4. 处理结果说明了什么

该框架使数据、阈值和模型结果具备可追溯性，并减少不同模型各自使用不同 split 或评估标准的风险。它已经达到支撑系统实验的工程基础，但还需要在 dissertation 中记录软件版本、硬件、运行命令和完整环境依赖；当前项目根目录未发现可核实的锁定环境文件，现有文件中也未说明全部训练运行的 GPU 型号与耗时。

在统一框架建立后，下一项任务是将原始 copied spans 转换为可供所有模型共同使用的句子级监督信号。

---

## 任务 3：专家 copied span 对齐与句子级弱标签构建

### 1. 为什么需要进行该任务

CounselBench 的 `toxicity_copy` 标注是专家从回答中复制的文本片段，而研究目标要求对每个句子作判断。copy 可能与原回答完全一致、跨越多个句子、包含格式差异、代表整个回答，或者无法在回答中直接找到。因此，必须建立透明且可审计的 span-to-sentence alignment 规则，并保留标注者支持数，才能把原始证据转换为句子级弱标签。

### 2. 该任务包括什么

任务输入为原始 response、`toxicity_copy` 和 `survey_id`。核心操作包括文本规范化、spaCy 句子切分、无效 copy 过滤、结构化单元格解析、exact character overlap、fuzzy sentence/window matching、整段标记处理，以及按独立 annotator 聚合句子标签。输出为 sentence dataset 和 alignment log。

#### 具体处理步骤

1. 规范化换行和多余空格，但保留标点语义；为空值及 `n/a`、`none`、`no toxicity` 等标准无效标记建立过滤规则。
2. 对列表、字典或多行形式的 copied text 提取候选 span，并在单元格内部去重。
3. 使用 spaCy 对规范化回答切句，保存每句文本及其 `start`、`end` 字符偏移；若指定模型不可用，则代码会回退到 blank English sentencizer。
4. 优先在完整回答中查找 exact span，并按字符覆盖率将其分配给句子；若 exact alignment 失败，再比较单句和最多三个连续句子的 fuzzy similarity。
5. 对表示整个回答的值使用 `all_response` 方法；低于 fallback threshold 的 copy 记为 `unmatched`，不强制生成正类。
6. 同一 annotator 提供多个重叠 fragment 时只对同一句投一票，避免将一个人的多个片段误计为多人共识。
7. 按配置中的 `min_positive_annotators` 将 `match_count` 转换成二元 `label`，同时保留 `positive_annotators`、`total_annotators`、完整 response、question、topic 和 responder 元数据。

#### 涉及的文件

- `src/preprocessing.py`：实现文本规范化、copy 解析、句子切分、exact/fuzzy 对齐、annotator 计票和弱标签构建。
- `configs/thesis_protocol.yaml`：配置 alignment thresholds、最大 fuzzy window、whole-response 标记和最小标注者数。
- `data/processed_question_grouped/toxicity/sentences.csv`：正式 toxicity 句子数据。
- `data/processed_question_grouped/toxicity/alignment_log.csv`：每个可用 span 的方法、分数和命中句索引。
- `notebooks/processed_sen-dataset_eda.ipynb`：处理后句子数、正类率和 token length 的已执行探索。

### 3. 该任务的处理结果

正式输出包含 2,848 个句子，来自 372 个唯一回答和 100 个问题。其中正类 603 句，整体正类率为 21.17%。sentence 文件保留 15 个字段，包括句子偏移、标签、标注者支持数、完整回答和来源信息。

alignment log 中共有 405 个可用 copied spans：224 个采用 `exact_char_overlap`，128 个采用 `fuzzy_sentence_or_window`，9 个采用 `all_response`，44 个未匹配。361 个 span 至少匹配到一句，对齐成功率为 89.14%；但该成功率只是算法找到候选位置的比例，并非与人工边界真值比较得到的 alignment accuracy。

### 4. 处理结果说明了什么

项目已经把原始的回答级重复评价转化成结构化句子监督，并保留了足够的 provenance 供误差追踪。这使 TF-IDF、SBERT 和 ALBERT 可以共享完全相同的标签定义和文本行。

同时，对齐结果暴露了重要限制：10.86% 的 span 完全未匹配，31.60% 依赖 fuzzy 方法，而 fuzzy span 往往同时传播到多个句子。弱标签因此不应被直接视为 gold standard。下一项任务必须先固定不会产生明显 question leakage 的 split，并量化重复、来源和标签冲突，才能对模型表现作可信解释。

---

## 任务 4：Question-grouped 数据划分、泄漏排查与数据偏差诊断

### 1. 为什么需要进行该任务

同一用户问题对应 human、Gemini、GPT-4 和 Llama 3 等不同来源回答。如果这些回答被拆分到训练和测试集，模型可能利用问题主题、通用措辞或回答模板，而不是学习目标错误。旧的 response-level split 虽能避免完全相同 response 跨集合，却不能阻止同一 question 的不同回答跨集合。因此，论文主实验需要更严格的 question-grouped protocol，并对 exact duplicate、near duplicate、标签冲突和来源偏差进行独立审计。

### 2. 该任务包括什么

该任务以 `questionID` 为分组单位进行 70%/15%/15% 划分，对 response group 的最高句子标签进行分层近似，并持久化每个句子的 split。之后检查 row、response 和 question overlap，计算规范化重复句、character TF-IDF 近重复、surface feature 差异、responder/topic prevalence 及 alignment 统计。

#### 具体处理步骤

1. 在 `configs/thesis_protocol.yaml` 中将 `split_group_column` 固定为 `questionID`。
2. 先划分 train 与临时集合，再将临时集合划分为 validation 和 test；在类别条件允许时进行分层，并固定 seed 42。
3. 将 split assignment 保存为独立 CSV，并生成带 SHA-256 数据指纹的 protocol manifest。
4. 核对三组 row、`response_id` 和 `questionID` 的交叉情况。
5. 通过规范化字符串检查 exact duplicate 与冲突标签；使用 character 3–5 gram TF-IDF cosine similarity 检查近重复。
6. 比较正负类的长度、token、标点、negation 和 `should` 比例，并按 responder 与 topic 统计正类率。

#### 涉及的文件

- `src/preprocessing.py`：grouped split 实现。
- `src/audit.py`：split integrity、重复、近重复、surface pattern、来源和标签审计。
- `configs/thesis_protocol.yaml`：question-grouped 主协议。
- `data/processed_question_grouped/toxicity/splits.csv`：2,848 行固定 split assignment。
- `data/processed_question_grouped/toxicity/protocol_manifest.json`：数据指纹与协议声明。
- `reports/thesis/analysis/toxicity/dataset_audit.json`：数据审计结果。
- `reports/thesis/ANALYSIS_REPORT.md`：严格协议与旧协议的综合诊断记录。

### 3. 该任务的处理结果

固定 split 的数据量如下：

| Split | 句子数 | 唯一回答数 | 正类数 | 正类率 |
|---|---:|---:|---:|---:|
| Train | 1,969 | 258 | 440 | 22.35% |
| Validation | 462 | 56 | 84 | 18.18% |
| Test | 417 | 58 | 79 | 18.94% |

审计确认三组之间不存在 row、`response_id` 或 `questionID` overlap，且 manifest 明确标记 test set 不得用于模型选择。现有 thesis 报告还记录：旧 response-only split 中有 49 个 questionID 同时出现在 train 和 test；改成 question-grouped 后，canonical TF-IDF PR-AUC 从旧协议约 0.51 降到严格协议约 0.36，说明旧协议明显高估了泛化表现。

严格协议仍发现 10 组规范化后相同句子跨 split，7 组相同句子存在冲突标签；train–test 比较中有 8 个测试句与某个训练句的 character TF-IDF similarity 不低于 0.95。来源分布也高度不均：human 句子正类率 47.27%，Gemini 20.51%，GPT-4 10.60%，Llama 3 9.54%；topic 正类率从 `family-conflict` 的 49.38% 到 `professional-ethics` 的 7.48%。

### 4. 处理结果说明了什么

Question-grouped split 修复了一个会直接破坏模型比较有效性的严重问题，当前 thesis 结果比旧结果更接近对新问题的泛化能力。不过，模板句和近重复仍可能带来残余 leakage，而 responder/topic 与标签强相关，模型也可能把来源风格当成 toxicity signal。

因此，下一项任务不能只使用一个弱 TF-IDF 作为对照；需要建立包含 word、character、class weighting 和 calibrated SVM 的强词汇基线，并分析其最高权重特征，判断任务究竟依赖语义还是数据表面模式。

---

## 任务 5：构建强 TF-IDF 基线并分析词汇与模板信号

### 1. 为什么需要进行该任务

如果 toxicity 弱标签主要由特定词汇、固定建议模板、来源风格或标注过程中的表面线索决定，TF-IDF 已可能捕获大部分可学习信息。在这种情况下，预训练语言模型没有明显优势并不一定表示微调失败，而可能表示任务本身或标签主要是 lexical。建立具有竞争力且经过统一调参的传统基线，是判断深度模型增益的必要前提。

### 2. 该任务包括什么

在固定 train/validation/test 上比较 word unigram、word unigram+bigram、character 3–5 gram 和 word+character 特征；分类器包括 Logistic Regression 与 calibrated Linear SVM，并分别设置若干 class-weight 方案。regularization 参数只用训练或验证阶段选择，decision threshold 只在 validation 上按 F1 选择，test 仅用于最终报告。

#### 具体处理步骤

1. 运行 8 个声明式 baseline specifications，覆盖四类 representation、LR/SVM 和 unweighted/balanced 组合。
2. 对每个实验在 `C ∈ {0.01, 0.1, 1.0, 10.0}` 中按 validation PR-AUC 选择 regularization。
3. 保持 train-only fitted model，不在 threshold 选择后重新用 train+validation 拟合，以避免概率尺度改变。
4. 在 validation threshold grid 上选择 F1 最优阈值，并冻结后计算 test metrics。
5. 对 word unigram+bigram unweighted LR 导出最高权重的正/负 unigram 和 bigram。
6. 为 seeds 42–46 保存独立预测和 aggregate summary；对 deterministic LR 而言，同一固定 split 下多 seed 结果相同，这是算法确定性的表现，不代表跨数据划分稳定性。

#### 涉及的文件

- `src/models.py`：定义 TF-IDF LR、word、character、word+character 和 calibrated Linear SVM 模型。
- `src/audit.py`：8 个强基线、验证集模型选择、阈值选择、预测和特征权重导出。
- `src/evaluate.py`：统一指标与阈值函数。
- `reports/thesis/analysis/toxicity/baselines/baseline_results.csv`：40 行结果，即 8 个实验 × 5 seeds。
- `reports/thesis/analysis/toxicity/baselines/*_predictions.csv`：各实验各 seed 的测试预测。
- `reports/thesis/analysis/toxicity/baselines/*_seed_summary.json`：多 seed 汇总。
- `reports/thesis/analysis/toxicity/baselines/tfidf_top_features.json`：word LR 特征权重。
- `reports/thesis/metrics/toxicity/tfidf_lr/`：canonical TF-IDF 模型、调参、验证/测试预测和指标。

### 3. 该任务的处理结果

在 seed 42 的严格测试集上，canonical `tfidf_lr` 得到 PR-AUC 0.360、ROC-AUC 0.718 和 F1 0.432。更强的 character TF-IDF LR 将 PR-AUC 提高到 0.440、ROC-AUC 提高到 0.743；word+character LR 的 PR-AUC 为 0.429，且在 8 个基线中得到最高 F1 0.467。说明单一 word TF-IDF 不是充分的强基线，character pattern 是当前任务的重要信号。

最高权重特征并未形成清晰的 toxicity 语义词表。正类 unigram 包括 `his`、`is`、`if`、`she`、`relationship`、`the`、`good` 和 `own`；正类 bigram 包括 `with his`、`your own`、`his mother`、`your boyfriend`、`these feelings`。负类 unigram 包括 `can`、`remember`、`health`、`client`、`grief`、`mental` 和 `care`；负类 bigram 包括 `can help`、`mental health`、`so sorry`、`the client` 和 `your doctor`。

### 4. 处理结果说明了什么

强基线的表现及其权重共同说明，模型很可能利用 pronoun、relationship wording、帮助模板、专业心理健康措辞和 responder writing style，而不只是识别明确 toxic language。character 模型优于 canonical word 模型，也符合固定表达、形态和模板信号较强的解释。

这些结果为预训练模型设定了更合理的比较标准：深度模型不仅要超过 PR-AUC 约 0.36 的 canonical TF-IDF，还应与 PR-AUC 0.44 的 character baseline 比较。下一项任务因此检验通用 sentence embedding、embedding normalization、分类头和上下文结构能否提供超出表面词汇的增益。

---

## 任务 6：固定 Sentence-BERT embedding、分类头与上下文输入消融

### 1. 为什么需要进行该任务

固定 sentence embedding 可以低成本测试预训练语义表示是否适合当前句子分类任务，但它并未使用任务标签更新 encoder。如果当前 SBERT 仅作为通用 feature extractor，再接简单分类器，其上限可能受 domain mismatch、target signal dilution 和冻结表示限制。因此需要分别比较 encoder、normalization、classifier 和 input context，而不是把所有变化混成一次实验。

### 2. 该任务包括什么

已执行实验使用 BAAI `bge-base-en-v1.5` 和 MiniLM embedding。embedding 由冻结的 SentenceTransformer 生成，再输入 Logistic Regression、calibrated Linear SVM 或单隐藏层 MLP。输入形式覆盖 target sentence、previous+target、target+next、previous+target+next、target+full response、question+target 和 question+window，并使用 `[QUESTION]`、`[PREVIOUS]`、`[TARGET]`、`[NEXT]` 和 `[RESPONSE]` 标记目标位置。

#### 具体处理步骤

1. 在相同行顺序和固定 split 上构建七种 target-aware input，不允许邻句跨越 `response_id` 边界。
2. 对 BGE 比较 raw、L2-normalized 和 standardized embedding。
3. 对固定 BGE embedding 比较 LR、calibrated Linear SVM 和浅层 MLP。
4. 比较 BGE 与 MiniLM，以判断 encoder 选择是否比分类头更重要。
5. 使用 training-only group-aware cross-validation 调 classifier 参数；threshold 仍只用 validation label 选择。
6. 保存 `model.joblib`、embedding model name、normalization configuration、validation/test predictions 和 metrics。

#### 涉及的文件

- `src/input_features.py`：七种带 target marker 的输入构造与 response 边界控制。
- `src/models.py`：SBERT LR、Linear SVM、MLP，以及代码层面的 LightGBM/ExtraTrees 支持。
- `src/train.py`：冻结 SentenceTransformer encoding、normalization、group-aware tuning 和结果保存。
- `src/predict.py`：读取 embedding 配置和分类器进行单回答或数据集推理。
- `reports/thesis/metrics/toxicity/sbert_*`：各 embedding、分类器和 context 消融结果。
- `models/thesis/toxicity/sbert_*`：已经保存的分类器与 embedding 元数据。

### 3. 该任务的处理结果

严格测试集上的主要结果如下：

| 实验 | PR-AUC | ROC-AUC | F1 |
|---|---:|---:|---:|
| BGE-LR，sentence | 0.340 | 0.686 | 0.329 |
| MiniLM-LR，sentence | 0.333 | 0.675 | 0.389 |
| BGE calibrated SVM | 0.337 | 0.676 | 0.374 |
| BGE MLP | 0.319 | 0.657 | 0.354 |
| BGE-LR，target+next | **0.364** | 0.729 | 0.409 |
| BGE-LR，window | 0.354 | 0.717 | 0.375 |
| BGE-LR，full response | 0.353 | 0.729 | 0.303 |
| BGE-LR，question+target | 0.199 | 0.502 | 0.278 |

BGE raw 与 L2 的 test results 完全相同，额外 standardization 将 PR-AUC 降到 0.311。加深分类头没有带来改善，MLP 反而低于 LR。`target+next` 是当前固定 embedding 中最好的上下文形式，但相对 sentence-only 的 PR-AUC 差仅 0.024；paired bootstrap 95% CI 为 [-0.030, 0.075]，p=0.347，不能证明上下文带来稳定提升。

代码还支持 `sbert_lgbm` 和 `sbert_extra_trees`，但当前严格 thesis 目录中没有可核实的正式结果，因此不能描述为已完成实验。当前代码也没有实现 supervised SBERT fine-tuning 或 contrastive fine-tuning；现有所有 SBERT 结果都是冻结的通用 embedding 实验。

### 4. 处理结果说明了什么

固定通用句向量没有超过强 character TF-IDF，说明语义 embedding 本身并不足以消除标签噪声、来源偏差或模板信号。`question+target` 的大幅下降更可能表明单向量压缩使 target 被长问题稀释，而不能据此断言 question context 没有研究价值。对于 factual 或 medical 任务，question 仍可能是理论上必要的输入。

下一项任务进一步检验 end-to-end fine-tuning 是否能通过任务监督更新表示，并系统比较 ALBERT 的 class imbalance、loss 与冻结策略。

---

## 任务 7：ALBERT 微调、类别不平衡和训练策略优化

### 1. 为什么需要进行该任务

与固定 embedding 不同，ALBERT fine-tuning 可以直接根据 toxicity 弱标签调整 encoder。不过，小样本、约 19% 的测试正类率、弱标签噪声和 ALBERT 参数共享都可能造成快速过拟合或冻结策略失效。需要在不盲目扩大搜索空间的前提下，检查 token truncation、dynamic padding、encoder/head learning rate、loss、sampler、early stopping 和多个 seeds。

### 2. 该任务包括什么

ALBERT 使用匹配的 `AlbertTokenizerFast` 和 `AlbertForSequenceClassification`。数据集只做 truncation，不做固定 max-length padding；batch 内由 `DataCollatorWithPadding` 动态补齐。训练支持全编码器微调、只训练分类头和最后参数组策略，支持普通 CE、weighted CE、focal loss 和 weighted sampler，并分别设置 encoder 与 classification head learning rate。

#### 具体处理步骤

1. 在训练前统计全部 input 的未 padding token length 和 truncation risk。
2. 使用 batch size 16、max length 128、encoder LR `1e-5`、head LR `5e-5`、weight decay 0.01、warmup ratio 0.10、gradient clipping 1.0，并在 CUDA 可用时启用 mixed precision。
3. 以 validation PR-AUC 选择最佳 checkpoint，early stopping patience 为 2；checkpoint 确定后才在 validation 上按 F1 选择 probability threshold。
4. 分开运行普通 CE、weighted CE、focal loss 和 weighted sampler，避免同时使用 class weight 与 sampler 造成重复补偿。
5. 比较 full fine-tuning 与 classifier-only；代码也支持 `last_groups`，但当前没有对应正式结果。
6. 对 weighted CE 运行 seeds 42–46，保存各自模型、训练历史、验证/测试预测、token summary 和 aggregate seed summary。

#### 涉及的文件

- `src/dataset.py`：Transformer dataset 与 truncation，padding 交由 batch collator。
- `src/models.py`：ALBERT tokenizer/model factory。
- `src/train.py`：冻结、分层 learning rate、loss、sampler、AMP、gradient accumulation、early stopping、checkpoint 和 history。
- `src/input_features.py`：token length audit。
- `configs/thesis_protocol.yaml`：正式训练参数。
- `reports/thesis/metrics/toxicity/albert*`：各策略日志、history、token summary、预测与指标。
- `models/thesis/toxicity/albert*`：实际保存的 tokenizer、config、Safetensors checkpoint 与 inference threshold。

### 3. 该任务的处理结果

ALBERT 单句 token length 的 median 为 24，P90 为 39，P95 为 45，P99 为 59，最大值为 107；在 `max_length=128` 下 2,848 句均未被截断。因此，当前单句 ALBERT 的瓶颈不是 max length，dynamic padding 主要带来显存和计算效率改善。

seed 42 结果显示：普通 CE 的 PR-AUC 为 0.354、F1 为 0.389；weighted CE 的 PR-AUC 为 0.389、F1 为 0.491；focal loss 的 PR-AUC 为 0.365、F1 为 0.375；weighted sampler 的 PR-AUC 为 0.338、F1 为 0.421。classifier-only weighted CE 的 PR-AUC 仅 0.204，说明完全冻结 encoder 明显欠拟合。训练历史显示普通 CE 最佳 checkpoint 在 epoch 1，weighted CE 在 epoch 2，后续 validation loss/selection metric 没有持续改善，与小数据上的快速过拟合一致。

weighted CE 的 seeds 42–46 已完整运行：PR-AUC 为 `[0.389, 0.360, 0.339, 0.363, 0.351]`，均值 `0.360 ± 0.018`；ROC-AUC 为 `0.704 ± 0.017`；precision 为 `0.383 ± 0.048`；recall 为 `0.509 ± 0.034`；F1 为 `0.435 ± 0.033`。validation-selected threshold 从 0.40 到 0.75，均值为 `0.53 ± 0.16`，显示概率尺度和阈值对 seed 较敏感。

### 4. 处理结果说明了什么

weighted CE 改善了 seed 42 的阈值后 F1，但该 seed 同时是五次运行中 PR-AUC 最高的一次。五 seed 平均 PR-AUC 与 canonical TF-IDF 的 0.360 基本相同，仍低于 character TF-IDF 的 0.440。因而现有证据支持“类别权重对某些 operating point 有帮助”，但不支持“ALBERT 的排序能力已明显超过 TF-IDF”。

ALBERT 共享参数的结构也意味着传统 BERT 式逐层 unfreezing 的解释并不直接适用；当前没有证据表明增加 layer-wise complexity 会比改进标签更有效。下一项任务通过统一指标、配对统计检验和样本级错误审计，判断上述差异是否具有统计和研究意义。

---

## 任务 8：统一评估、多随机种子、显著性检验与模型错误审计

### 1. 为什么需要进行该任务

在类别不平衡任务中，单一 accuracy 或单次 F1 容易误导。模型排序能力、验证集 threshold、seed 波动和特定 operating point 必须分开报告。此外，TF-IDF、SBERT 和 ALBERT 对同一测试句的互补错误，可以帮助判断模型是真正学习了不同信息，还是共同受错误标签影响。

### 2. 该任务包括什么

所有模型统一计算 accuracy、precision、recall、F1、F0.5、F2、macro-F1、ROC-AUC、PR-AUC、prevalence、PR-AUC lift、positive prediction rate 和 confusion matrix。核心模型使用相同的 417 条 test rows 进行 paired bootstrap 和 exact McNemar test。多 seed 结果汇总 mean、sample standard deviation、min、max 和每次原始值，并导出置信度排序的模型分歧样本。

#### 具体处理步骤

1. 在 validation 上从固定 grid 选择 F1/F0.5/F2 threshold；同分时选择更接近 0.5 的阈值，保证确定性。
2. 对冻结 threshold 的 test probabilities 计算统一指标并保存 row-level predictions。
3. 对两个模型在完全相同测试行上的 PR-AUC 和 ROC-AUC 差异进行 2,000 次 paired bootstrap；丢弃单类 bootstrap sample。
4. 对 hard prediction correctness 的不一致单元运行 exact McNemar test；该检验只回答两个分类器错误率是否不同，不衡量 ranking quality。
5. 按 high-confidence TP/FP/FN、boundary、两模型互相正确/错误、全部模型失败和模型分歧等类别导出样本，保留完整文本、label、概率、预测和来源元数据。

#### 涉及的文件

- `src/evaluate.py`：threshold、统一指标、多 seed aggregate、paired bootstrap 和 McNemar。
- `src/audit.py`：prediction merge、统计比较和 error audit export。
- `run_pipeline.py`：`audit`、`compare` 和 `export` CLI。
- `reports/thesis/analysis/toxicity/model_comparison_*.json`：模型配对比较。
- `reports/thesis/analysis/toxicity/manual_error_audit.csv`：530 行样本级审计表。
- `reports/thesis/metrics/toxicity/*/test_predictions.csv`：统计检验和人工复核的共同输入。
- `reports/thesis/ANALYSIS_REPORT.md`：已执行实验的集中解释。

### 3. 该任务的处理结果

seed 42 weighted-CE ALBERT 相对 canonical TF-IDF 的 PR-AUC 差为 0.028，95% CI `[-0.048, 0.113]`，paired-bootstrap p=0.474；ROC-AUC p=0.604。相对强 character TF-IDF，ALBERT PR-AUC 差为 -0.052，95% CI `[-0.139, 0.047]`，p=0.316；hard predictions 的 McNemar p=0.545。当前任何统计口径都不支持 ALBERT 在排序能力或总体错误率上优于强 character baseline。

`target+next` BGE 相对 sentence-only BGE 的 ROC-AUC 差接近但未达到常用显著性门槛（p=0.053），PR-AUC 差不显著（p=0.347）。因此，上下文实验应表述为“有进一步验证价值”，而不是已证实的提升。

`manual_error_audit.csv` 实际包含 530 行，覆盖 12 类审计标签：50 条 high-confidence TP、50 条 FP、28 条 FN、50 条 boundary、50 条 models disagree、46 条 all models wrong，以及 TF-IDF、SBERT、ALBERT 两两互补错误。全部模型失败的若干正类句子，如一般性的 self-care、couples therapy 或 needs identification 建议，表面上并无明确 toxicity，提示标签定义、跨句传播或弱标签噪声可能比模型容量更关键。

### 4. 处理结果说明了什么

单次 seed 42 的 weighted ALBERT F1 较高，但其排序增益不显著，而且五 seed 平均值回落到 canonical TF-IDF 水平。论文可以支持“模型错误模式不同”和“某些 threshold 下 ALBERT 减少特定错误”，但不能声称 ALBERT 或上下文模型具有稳定显著优势。

模型分歧也不能作为标签真假的最终证据，因为三个模型都由同一弱标签训练。它的合理用途是形成高价值人工审计队列。下一项任务因此把 alignment provenance、annotator consensus、模型一致性和 label threshold sensitivity 合并，专门评估弱标签质量。

---

## 任务 9：弱标签质量、标注共识与标签阈值敏感性分析

### 1. 为什么需要进行该任务

当前标签由专家 copied span 自动对齐到句子，属于 weak supervision。如果 span 边界、fuzzy matching、entire-response 标记或 annotator aggregation 产生系统误差，继续调参的收益可能低于人工复核和标签规则改进。需要把“模型表现不高”分解为模型能力问题与监督信号上限问题。

### 2. 该任务包括什么

专项脚本整合 sentence labels、alignment log、三个模型的共同 test predictions，以及 threshold 1/2 的传统基线结果。它统计 label consensus、alignment method、fuzzy 跨句扩散、正类 provenance、responder/topic prevalence、模型投票与标签支持数关系，并导出高风险句子和 unmatched spans。

#### 具体处理步骤

1. 核对当前二元标签与 `match_count >= 1` 规则是否一致，并统计每个支持数对应的句子量。
2. 将 threshold 1 与至少两位 annotator 的 threshold 2 比较，量化正类数量与 prevalence 的变化。
3. 分析 exact、fuzzy、all-response 和 unmatched span 的数量、句子传播范围与 similarity score。
4. 标记 `fuzzy_only`、single-annotator fuzzy、all-response propagation、相同句冲突和非标准 annotator pool 等风险组。
5. 合并 TF-IDF、BGE target+next 和 weighted ALBERT 的 417 条共同测试预测，比较不同支持数和 provenance 下的 unanimous hit/miss。
6. 导出高风险句子、未匹配 span、JSON 全量统计和可直接阅读的 Markdown 报告。

#### 涉及的文件

- `scripts/analyze_toxic_weak_labels.py`：当前工作区新增的专项弱标签分析脚本。
- `reports/thesis/analysis/toxicity/weak_label_quality/weak_label_quality_summary.json`：机器可读完整统计。
- `reports/thesis/analysis/toxicity/weak_label_quality/weak_label_quality_report.md`：质量结论与建议。
- `reports/thesis/analysis/toxicity/weak_label_quality/high_risk_sentences.csv`：306 条高风险句子。
- `reports/thesis/analysis/toxicity/weak_label_quality/unmatched_spans.csv`：44 条未对齐 span。
- `reports/thesis/analysis/toxicity/label_threshold_2/`：threshold 2 的 8 个基线与预测结果。

### 3. 该任务的处理结果

603 个 threshold-1 正类中，433 个仅有一位 annotator 支持，占 71.81%；改成至少两位支持后只剩 170 个正类，整体 prevalence 从 21.17% 降到 5.97%。match count 与当前 label rule 完全一致，但这只能证明实现一致，不能证明标签语义正确。

128 个 fuzzy spans 中有 125 个命中多个句子，比例为 97.66%，单个 fuzzy span 最多传播到 16 句；119 个 fuzzy span 的 score 恰好为 1.0，说明当前相似度分数明显饱和，不能直接解释为高置信度。603 个正类中有 309 个只由 fuzzy alignment 支持，占 51.24%；245 个同时属于 single-annotator 与 fuzzy-only，占全部正类 40.63%。

在共同测试集上，annotator 支持数越高，三个模型的一致命中率越高：单人支持正类的 unanimous hit 为 33.3%，两人支持为 60.0%，三人支持为 62.5%。15 个由 all-response propagation 得到的测试正类中，8 个被全部模型判负，且没有一个被全部模型判正。这是“回答级问题不应自动传播为每句正类”的强诊断信号，但仍需人工裁决确认。

threshold sensitivity 结果显示：threshold 1 下最佳 PR-AUC 为 0.440、最佳 ROC-AUC 为 0.743、最佳 F1 为 0.467；threshold 2 下最佳 PR-AUC 为 0.293、最佳 ROC-AUC 为 0.834、最佳 F1 为 0.276。更高共识标签的 ROC ranking 更容易，但正类过少，导致 threshold-dependent 指标不稳定。专项报告将当前弱标签总体判断为“中等偏低质量”，适合作为 high-recall candidate labels 或带置信度监督，不适合作为未经复核的 gold labels。

分析还发现至少 3 个 response group 的 `total_annotators` 明显偏离通常的 5 人规模，例如 63、45 和 10。结合原始 EDA 中通用回答跨不同问题复用的现象，这提示当前按完全相同 response text 聚合时可能把不同问题下的评价合并，并只保留第一条 question metadata。该问题在 dissertation 主实验定稿前需要优先修复或做敏感性分析。

### 4. 处理结果说明了什么

弱标签中确实存在可学习信号，因为模型表现高于 prevalence/random ranking；但单标注者、fuzzy 跨句扩散、all-response 传播、未匹配 span 和异常 annotator pool 共同构成了明显监督上限。现有结果支持将“标签质量”视为当前最重要的性能瓶颈之一，也说明继续扩大模型或大规模调参的预期收益可能低于改进 label construction 和人工审计。

下一阶段研究应以 `high_risk_sentences.csv` 和 `unmatched_spans.csv` 为优先队列建立人工裁决子集，并据此估计弱标签 precision/recall、重新设计 response+question 聚合键和 confidence-aware training。只有在标签与聚合规则稳定后，才适合继续开展 supervised sentence-encoder fine-tuning、target-aware cross-encoder 和其他预训练模型比较。

---

## 整体成果总结

当前项目已经形成一条可追溯的 toxicity sentence detection 研究流程：从 2,000 条 CounselBench 原始评价记录中识别 372 个唯一回答，经 span parsing、句子切分、exact/fuzzy alignment 和 annotator-level aggregation，生成 2,848 条带 provenance 的句子弱标签；再按 100 个 questionID 固定 train/validation/test，保存 split 与数据指纹；随后在统一阈值和 test isolation 原则下运行 TF-IDF、固定 SBERT embedding 和 ALBERT 微调，并保存模型、训练历史、验证/测试预测和多指标结果；最后通过重复与来源审计、特征权重、上下文消融、类别不平衡消融、五 seed 复现、paired bootstrap、McNemar、样本级错误导出和弱标签专项分析解释性能上限。

从现有文件能够支持的核心研究结论是：

1. 旧 response-only split 会因同一 question 跨集合而高估泛化表现，question-grouped protocol 是论文主实验的必要修复。
2. character TF-IDF 是比 canonical word TF-IDF 更强的传统基线，其 PR-AUC 0.440 高于所有已运行深度模型的五 seed 平均表现。
3. 当前 SBERT 是冻结的通用 embedding，而不是 supervised fine-tuning；更换分类头或简单拼接长上下文没有产生稳定优势。
4. weighted CE 可改善 ALBERT 某些 seed 的 threshold-dependent F1，但 weighted ALBERT 五 seed 平均 PR-AUC 为 `0.360 ± 0.018`，与 canonical TF-IDF 基本相同，且未显著超过强 character baseline。
5. 标签和数据生成过程存在实质风险：71.8% 正类仅一位 annotator 支持，51.2% 正类仅由 fuzzy alignment 支持，10.9% span 未对齐，并存在 all-response 传播、冲突标签、来源偏差和异常 annotator aggregation。
6. 现阶段最有价值的论文贡献不只是追求单次最高分，而是证明 split、weak-label quality、source bias 与 context representation 如何共同限制 sentence-level safety detection。

这些成果能够支撑 dissertation 的 data、methods、experimental protocol、results 和 limitations 部分，也为可持续发展主题提供了负责任 AI 角度的实证基础：在心理健康场景中，可靠的标注、泛化协议和不确定性报告比单纯增加模型参数更重要。

## 当前限制与建议补充内容

### 尚未完成或无法确认的步骤

- factual consistency 和 inappropriate medical advice 虽已在配置和 schema 中支持，但当前没有对应的 question-grouped processed dataset、模型、指标或误差审计；不能将 toxicity 结论推广到这两类任务。
- 当前 SBERT 没有 supervised classification fine-tuning、partial/full encoder unfreezing 或 contrastive learning 的已执行结果。
- ALBERT 的 `last_groups` 策略在代码中存在，但没有正式结果；gradual unfreezing 和 layer-wise learning-rate decay 未形成可核实实验。
- `sbert_lgbm` 和 `sbert_extra_trees` 有代码接口，但 strict thesis output 中没有已验证结果。
- 当前没有 target-aware ALBERT context、cross-encoder 或 hierarchical model 的正式运行结果。
- `reports/figures/` 只有占位文件，当前未形成可核实的正式论文图表；notebook 的探索输出不能替代最终图表和 caption。

### 可能影响结果可靠性的限制

- 没有人工裁决后的句子级 gold subset，无法计算真实标签 precision、recall、inter-annotator agreement 或 alignment accuracy。
- 相同 response text 可能跨不同问题复用；当前按 response text 建立 `response_id` 并聚合 annotator 的方法可能合并不同语境，异常的 63/45 annotator pool 已提供风险证据。
- 严格 split 中仍有 10 组 exact normalized duplicate 和 8 条高相似 train–test 近重复测试句，尚未完成去重敏感性实验。
- responder 与 topic prevalence 差异很大，但现有结果文件没有完整的 responder-stratified、topic-stratified 性能及公平性报告。
- 只有 weighted-CE ALBERT 完成 5 个训练 seeds；SBERT context 与其他 ALBERT loss 消融大多为单 seed。
- baseline 的多个 seed 在固定 split 下对 deterministic LR 给出相同结果，不能替代 repeated grouped split 或 nested resampling。
- paired bootstrap 以句子为 resampling unit，而同一回答内句子并非完全独立。最终论文可考虑按 question 或 response cluster bootstrap，以更符合数据相关结构。
- threshold 在较小 validation set 上选择，weighted ALBERT 的 threshold 在 0.40–0.75 间波动，需报告 threshold stability 或 calibration。
- 当前项目文件中未完整记录全部实验的软件版本、GPU 型号、训练时长、能耗或碳排放；对于 “AI for Sustainable Development”，建议补充计算资源与效率报告。

### Dissertation 中建议进一步解释、验证或报告的内容

1. 人工复核 `high_risk_sentences.csv`、`unmatched_spans.csv`、冲突标签和全部模型失败样本，建立分层抽样的 adjudicated gold subset。
2. 将聚合键改为至少包含 `questionID + response`，重新生成标签并比较当前结果；将 all-response 标记改成 response-level 或 uncertain label，而非自动传播到每句。
3. 限制 fuzzy alignment 只保留最佳非重叠句/窗口，并同时考虑 copy coverage 与 sentence coverage；报告规则变更前后的标签数量和人工 precision。
4. 完成去 exact/near duplicate sensitivity test，并按 responder、topic、标注支持数和 alignment provenance 报告性能。
5. 对最终候选模型至少运行 5 seeds；统计上采用同一测试样本的 paired comparison，并考虑 cluster bootstrap。McNemar 仅用于 hard classification correctness，不能替代 PR-AUC comparison。
6. 标签清理后先运行 supervised classification fine-tuning，再考虑对比学习；如果构造 pair，必须保证同一 question/response 的相关句子不跨 split。
7. 对 factual/medical 独立设计输入：toxicity 可先用 target/local context，factual 通常需要 question 和外部事实，medical advice 通常需要 question 中的对象与病情。不要直接复制 toxicity 的最优结构和超参数。
8. 在模型性能表中同时报告 prevalence、PR-AUC、ROC-AUC、precision、recall、F1、F0.5、F2、threshold、positive prediction rate 和 mean±SD；明确所有 threshold 和超参数均未使用 test labels。
9. 补充正式流程图、标签质量图、precision–recall curve、seed distribution 和分组错误分析图；现有文件中这些图尚未生成。
10. 将研究结论限制为当前数据与 weak-label definition 下的 sentence detection，不把模型输出表述为临床诊断、安全认证或真实世界医疗决策工具。
