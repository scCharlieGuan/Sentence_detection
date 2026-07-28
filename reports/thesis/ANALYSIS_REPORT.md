# Sentence Error Detection：论文级诊断与实验报告

本报告只记录本次在 `MentalFlow` 环境实际运行的结果。严格主协议使用
`configs/thesis_protocol.yaml`，按 `questionID` 分组；旧的 response-grouped
结果只作为探索性证据，不进入论文主表。

## 1. 当前性能瓶颈诊断

### 数据问题

- 数据只有 372 个唯一回答、2,848 个句子；严格测试集为 417 句、79 个正类。
- 旧 split 虽无 response_id 交叉，却有 49 个 questionID 同时出现在 train 和
  test。改为 question-grouped 后该值为 0，TF-IDF PR-AUC 从旧协议约
  0.51 降到严格协议的 0.36，证明旧协议受问题/模板重叠影响。
- 即使 question-grouped，仍有 10 组规范化后相同句子跨集合，8 个测试句与
  训练句的字符 TF-IDF 相似度至少 0.95。论文需再报告“去近重复”敏感性结果。
- 来源偏差很强：human 回答的正类率为 47.3%，gemini 为 20.5%，gpt4 为
  10.6%，llama3 为 9.5%。模型可能学习 responder 风格而非目标错误。
- topic 正类率从 family-conflict 的 49.4% 到 professional-ethics 的 7.5%，
  需要按 topic/responder 分组报告。

### 标签问题

- 405 个 copied spans 中 44 个（10.9%）完全无法对齐，128 个（31.6%）
  依赖 fuzzy 对齐。
- 71.8% 的正类句只有一位标注者支持；正类的标注者数中位数为 1。
- 7 组相同句子存在冲突标签。
- 多个被标为正类、但三个模型均判负的句子表面上并不含 toxicity，例如
  “Practice self-care...” 和 “Consider couples therapy...”。这可能来自跨句
  span、整段标注或宽泛的任务定义，是高优先级人工复核对象。
- 将标签门槛从 1 位改为 2 位标注者后，严格测试正类率从 18.9% 降为 6.7%。
  排序变容易（最佳 ROC-AUC 0.834），但阈值后 F1 很不稳定，说明高共识标签
  更干净但样本太少。继续调参的预期收益可能低于标签复核。

### 特征问题

- TF-IDF 高权重正类 unigram 包括 `is/the/you/we/will/own/good/control/no`，
  bigram 包括 `good luck/your own/these feelings/you will`；负类包括
  `remember/mental/therapy/professional`、`mental health/can help/remember that`。
- 大量 stopwords、写作风格和模板短语位居最高权重，未直接表达 toxicity。
  结合 responder 正类率差异，强基线很可能利用来源与文体信号。
- unigram、bigram、word+char 的差距较小，说明词汇信号已接近饱和；增加
  n-gram 并未带来稳定的大幅提升。

### 输入上下文问题

- 单句 BGE-LR PR-AUC 为 0.340；`target + next` 为 0.364，是最佳固定向量
  输入，但 PR-AUC 差值的 95% CI 为 [-0.030, 0.075]，p=0.347，不显著。
- `question + target` PR-AUC 只有 0.199。通用 embedding 把长问题与目标句
  压成一个向量时会稀释目标信号；这不代表问题上下文本身无用，而是说明
  当前固定向量融合方式不合适。
- `target + full response` 的 ROC-AUC 为 0.729，但验证阈值迁移后 recall 很低。
  后续应使用 target-aware cross-encoder 或分开编码，而不是简单平均整段语义。

不同任务的理论判断：

| 任务 | 单句是否通常足够 | 推荐输入 |
|---|---|---|
| toxicity | 常可依赖局部词汇，但当前标签明显比纯 toxicity 更宽 | target；再比较 target+邻句 |
| factual | 通常不足，还可能需要外部事实 | question + target + 局部回答；必要时检索 |
| medical advice | 经常不足，建议对象和病情来自问题 | question 与 target 分隔输入 |

### 训练问题

- 旧 ALBERT 日志显示训练 loss 从约 0.63 降到 0.33/0.03，而验证 loss 上升，
  是明确的快速过拟合。
- 旧代码用每轮“阈值调优后的 F1”选择 checkpoint，混合了排序能力与阈值；
  新代码按验证 PR-AUC 选 checkpoint，之后才在该 checkpoint 上选阈值。
- 旧 classical 流程在验证集选阈值后又用 train+val 重训，改变了概率尺度却
  沿用旧阈值。严格流程禁止该组合。
- ALBERT 单句 token 长度：median 24、P90 39、P95 45、P99 59、max 107；
  `max_length=128` 的截断率为 0。因此单句实验的瓶颈不是截断，动态 padding
  更节省显存。
- ALBERT 最佳 epoch 很早：普通 CE 为 epoch 1，weighted CE 为 epoch 2。
- ALBERT-base 只有一个 physical hidden group 且跨 logical layers 共享参数，
  “只解冻最后若干逻辑层”不像 BERT 那样自然。classifier-only 的 PR-AUC
  仅 0.204，完全冻结编码器明显欠拟合；full fine-tuning 更合理。

### 评估问题

- 旧结果混用 F1 与 average precision 作为调参目标；严格协议统一用
  train-only group CV 的 average precision。
- 固定测试集上的 5 个 LR seed 结果完全一致，这是确定性模型的预期结果，
  不能冒充“跨数据划分稳定性”。
- weighted-CE ALBERT 已运行 seeds 42–46；其他深度消融仍只有 seed 42。

### 模型选择问题

- 当前 SBERT 是固定 BGE embedding + 简单分类器，不是任务监督微调。
- BGE raw 与 L2 结果完全相同（模型输出模块本身已归一化）；额外标准化使
  PR-AUC 从 0.340 降到 0.311。
- MiniLM-LR 为 0.333；BGE-LR 为 0.340。SVM 0.337、MLP 0.319，分类头容量
  不是主要瓶颈。
- 现阶段没有证据支持直接换更大模型。标签、split、来源偏差和 target-aware
  上下文的优先级更高。

## 2. 必须优先修复的问题

1. 论文主实验必须使用 question-grouped split，不能使用旧 response-only split。
2. 人工审计单标注者正类、fuzzy/unmatched span、冲突标签和疑似安全的正类。
3. 去除或分组跨 split 的模板/近重复句，并做敏感性对照。
4. 完成 5 seeds 的深度模型训练；当前单 seed 不足以形成最终结论。
5. 对 responder/topic 做分层指标，判断模型是否只学习来源。
6. factual/medical 必须重新做上下文输入实验，不能照搬 toxicity 的最佳结构。

## 3. 已实施的代码修改

| 文件 | 修改 | 接口影响 |
|---|---|---|
| `src/preprocessing.py` | 按 annotator 而非 span 计票；保存 question/source 元数据；支持配置 split group | 保留原函数签名 |
| `src/input_features.py` | 七种 target-aware 输入；token 长度审计 | 新模块 |
| `src/models.py` | word/char/combined TF-IDF；SBERT LR/SVM/MLP | 扩展模型名 |
| `src/dataset.py` | 移除固定 max padding，交由动态 collator | 保留 Dataset 接口 |
| `src/train.py` | group CV、统一阈值、预测导出；SBERT 归一化；ALBERT 分层 LR、冻结、三种 loss、sampler、AMP、历史和 early stop | 保留 `train_model(...)` |
| `src/evaluate.py` | F0.5/F2/PPR、paired bootstrap、McNemar、多 seed 聚合 | 扩展接口 |
| `src/audit.py` | split/近重复/来源/标签审计、8 个强基线、错误样本导出 | 新模块 |
| `src/predict.py` | 上下文推理、保存模型的批量预测 | 保留单回答接口 |
| `run_pipeline.py` | diagnose/baselines/audit/compare/export；seed/run-name/消融覆盖参数 | CLI 扩展 |
| `configs/thesis_protocol.yaml` | 独立 question-grouped 论文协议 | 不覆盖旧配置 |
| `tests/*` | 指标、阈值、McNemar、上下文边界测试 | 新测试 |

## 4. 推荐实验矩阵

| ID | 模型 | 输入 | loss | 不平衡 | LR | 阈值 | seed | 目的/状态 |
|---|---|---|---|---|---|---|---|---|
| T01 | word TF-IDF LR | target | log loss | none | C via group CV | val F1 | 42–46 | 已运行 |
| T02 | char TF-IDF LR | target | log loss | none | C via val suite | val F1 | 42–46 | 已运行 |
| T03 | word+char SVM | target | hinge+calibration | balanced | C via val suite | val F1 | 42–46 | 已运行 |
| S01 | BGE-LR | target | log loss | balanced | C via group CV | val F1 | 42 | 已运行 |
| S02 | MiniLM-LR | target | log loss | balanced | C via group CV | val F1 | 42 | 已运行 |
| S03 | BGE-LR | target+next | log loss | balanced | C via group CV | val F1 | 42 | 已运行 |
| S04 | BGE-LR | question+target | log loss | balanced | C via group CV | val F1 | 42 | 已运行 |
| S05 | BGE-LR | target+full response | log loss | balanced | C via group CV | val F1 | 42 | 已运行 |
| A01 | ALBERT full | target | CE | none | 1e-5/5e-5 | val F1 | 42 | 已运行 |
| A02 | ALBERT full | target | weighted CE | class weight | 1e-5/5e-5 | val F1 | 42–46 | 已运行 |
| A03 | ALBERT full | target | focal | none | 1e-5/5e-5 | val F1 | 42 | 已运行 |
| A04 | ALBERT full | target | CE | weighted sampler | 1e-5/5e-5 | val F1 | 42 | 已运行 |
| A05 | ALBERT head-only | target | weighted CE | class weight | head 5e-5 | val F1 | 42 | 已运行 |
| A06 | ALBERT full | target+next | weighted CE | class weight | 1e-5/5e-5 | val F1 | 42–46 | 高优先级待运行 |
| A07 | task-tuned encoder | target-aware context | CE/weighted CE | 分开比较 | 小范围搜索 | val F1 | 42–46 | 标签审计后运行 |

注意：每一种不平衡方法单独实验；不同时使用 weighted loss 与 sampler。

## 5. 严格协议模型结果

下表除最后一行外均为 seed 42 单次结果。

| 模型 | PR-AUC | ROC-AUC | Precision | Recall | F1 | F0.5 | F2 | threshold | PPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF LR | 0.360 | 0.718 | 0.325 | 0.646 | 0.432 | 0.361 | 0.539 | 0.50 | 0.377 |
| char TF-IDF LR（强基线） | **0.440** | **0.743** | 0.471 | 0.405 | 0.435 | 0.456 | 0.417 | 0.25 | 0.163 |
| word+char TF-IDF LR | 0.429 | 0.740 | 0.410 | 0.544 | 0.467 | 0.431 | 0.511 | 0.25 | 0.252 |
| BGE-LR | 0.340 | 0.686 | 0.308 | 0.354 | 0.329 | 0.316 | 0.344 | 0.55 | 0.218 |
| BGE-LR target+next | 0.364 | 0.729 | 0.287 | 0.709 | 0.409 | 0.326 | 0.548 | 0.45 | 0.468 |
| ALBERT CE | 0.354 | 0.685 | 0.268 | 0.709 | 0.389 | 0.306 | 0.533 | 0.15 | 0.501 |
| ALBERT weighted CE | **0.389** | **0.729** | **0.448** | 0.544 | **0.491** | **0.464** | 0.522 | 0.65 | 0.230 |
| ALBERT focal | 0.365 | 0.677 | 0.257 | 0.696 | 0.375 | 0.294 | 0.519 | 0.35 | 0.513 |
| ALBERT weighted sampler | 0.338 | 0.691 | 0.353 | 0.519 | 0.421 | 0.378 | 0.475 | 0.80 | 0.278 |
| ALBERT weighted CE, 5-seed mean±std | **0.360±0.018** | **0.704±0.017** | **0.383±0.048** | **0.509±0.034** | **0.435±0.033** | **0.402±0.042** | **0.476±0.028** | 0.53±0.16 | 0.255±0.037 |

统计比较：

- weighted-CE ALBERT vs TF-IDF：PR-AUC 差 0.028，95% CI
  [-0.048, 0.113]，paired-bootstrap p=0.474；ROC-AUC p=0.604。该比较使用
  seed 42，而 seed 42 恰好是五个 ALBERT seeds 中 PR-AUC 最高的一次。
- weighted-CE ALBERT 五次 PR-AUC 为
  [0.389, 0.360, 0.339, 0.363, 0.351]；均值 0.360 与 TF-IDF 的 0.360
  基本相同，模型差异小于 ALBERT 的 seed 波动。
- seed-42 weighted ALBERT 相对强 char TF-IDF 的 PR-AUC 差为 -0.052，
  95% CI [-0.139, 0.047]，p=0.316；hard predictions 的 McNemar p=0.545。
  当前任何统计口径都不支持 ALBERT 优于强 TF-IDF 基线。
- 不能声称 ALBERT 的排序性能显著优于 TF-IDF。
- 两者 hard predictions 的 McNemar p=2.52e-6，weighted ALBERT 的总体错误更少。
  该结论依赖各自在验证集选择的阈值，不能替代 PR-AUC 结论。

## 6. 错误分析

完整 530 行人工审计表位于 `reports/thesis/analysis/toxicity/manual_error_audit.csv`。

TF-IDF 正确、weighted ALBERT 错误的例子：

- 正类：“Rebuilding trust will require effort, patience, and understanding...”
- 正类：“Her use of marijuana during pregnancy could potentially affect...”
- 这些句子是否真属于 toxicity 很可疑，应优先核标签。

weighted ALBERT 正确、TF-IDF 错误的例子：

- 负类：“Other challenges may arise from clients who are mandated to attend therapy...”
- 负类：“Approach the conversation with empathy, but also set clear boundaries...”
- TF-IDF 的错误与 `therapy/relationship/boundaries` 等来源相关词共现有关。

全部模型失败的例子：

- 正类：“Practice self-care, engage in activities that bring you joy...”
- 正类：“Consider couples therapy to work through your feelings...”
- 正类：“Start by identifying your needs and preferences...”

这些例子更像标签噪声或需要整段标注语境，而非三个模型共同缺失一个明显的
toxicity 语义模式。人工复核应查看 copied span、annotator reason 和完整回答。

## 7. 论文级结论

目前可以支持：

- 旧 response-only split 会高估泛化性能。
- 当前数据存在明显弱标签噪声、来源偏差和模板信号。
- 固定通用 embedding 并不天然优于 TF-IDF；分类头加深也没有解决问题。
- weighted CE 改善了 ALBERT 的单次阈值后表现，完整微调明显优于只训练头；
  但五 seed 的平均排序性能没有超过 TF-IDF。
- `target+next` 值得进入后续上下文实验。

目前不能支持：

- ALBERT 或任一深度模型在排序能力上显著优于 TF-IDF。
- 上下文一定提升性能。
- 某个 seed 42 的模型就是最终最佳模型。
- toxicity 上的输入结构和超参数可直接迁移到 factual/medical。
- 更大或更新的预训练模型会解决当前瓶颈。

## 8. 下一步优先级

高优先级：

1. 人工复核单标注者、fuzzy/unmatched、冲突标签和全部模型失败样本。
2. 运行 target+next ALBERT 的 seeds 42–46，并复核 weighted-CE 的阈值稳定性。
3. 去近重复、按 responder/topic 分层评估。
4. 为 factual/medical 生成各自 question-grouped 数据并做独立上下文实验。

中优先级：

1. 对 label threshold=1/2 做完整敏感性分析。
2. supervised sentence-encoder classification fine-tuning；先做 classification loss，
   再考虑基于同一回答内正负句构造的对比学习。
3. 使用 target/context 分开编码或 cross-encoder，避免长问题稀释 target。
4. 对最终两模型做 5-seed paired bootstrap 与 McNemar。

低优先级：

1. 在上述问题处理后再比较 RoBERTa/DeBERTa/domain-adapted model。
2. layer-wise LR decay；ALBERT 参数共享使其论文价值有限。
3. 更深 MLP、同时叠加多种不平衡方法或大规模 Optuna 搜索。
