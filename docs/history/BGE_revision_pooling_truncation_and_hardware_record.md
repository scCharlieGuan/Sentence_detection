# BGE revision、pooling、truncation metadata 与硬件记录

记录日期：2026-08-24（Europe/London）  
项目：`sentence_error_detection_project`  
对应实验：`sbert_bge_target` 与 `sbert_bge_target_next`

## 1. 结论摘要

| 项目 | 核实结果 | 证据强度 |
|---|---|---|
| BGE checkpoint | `BAAI/bge-base-en-v1.5` | 项目产物直接记录 |
| Hugging Face revision / commit | `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` | 从运行主机缓存恢复；项目产物未直接绑定 commit |
| embedding 维度 | 768 | checkpoint pooling 配置与模型配置一致 |
| pooling | CLS-token pooling；mean、max、mean-sqrt-length pooling 均关闭 | checkpoint `1_Pooling/config.json` 直接记录 |
| checkpoint 内部后处理 | `Normalize` 模块 | checkpoint `modules.json` 直接记录 |
| 实验输出归一化 | `SentenceTransformer.encode(..., normalize_embeddings=True)`，随后保存为 `float32` | 训练代码与 `embedding_config.json` 直接记录 |
| 有效最大序列长度 | 512 token（包含特殊 token） | checkpoint 与运行环境加载结果一致 |
| 截断方向 / 策略 | 右侧截断；Sentence-Transformers Transformer 模块使用 tokenizer 的 `longest_first` 截断 | tokenizer 配置及已安装运行库行为 |
| target-only 实际截断 | 0 / 2,907；最大 105 token | 用保存的数据、输入构造代码和同一 tokenizer 复算 |
| target+next 实际截断 | 0 / 2,907；最大 137 token | 用保存的数据、输入构造代码和同一 tokenizer 复算 |
| 运行环境 | Python 3.11.15；PyTorch 2.13.0+cu130；Transformers 5.9.0；Sentence-Transformers 5.5.1 | 与 protocol manifest 对应的 `MentalFlow` 环境当前实测 |
| GPU | NVIDIA GeForce RTX 2060，6,144 MiB，compute capability 7.5 | 当前主机实测；历史日志未保存 GPU 标识 |
| CPU / RAM | Intel Core i7-10875H @ 2.30 GHz，16 logical processors；15.79 GiB 可见物理内存 | 当前主机实测；历史日志未保存 CPU/RAM 标识 |

## 2. BGE revision

两个已保存的实验均在 2026-08-16 加载 `BAAI/bge-base-en-v1.5`：

- target-only：`results/reliability_study/metrics/toxicity/sbert_bge_target/run.log`，时间为 15:26:35；
- target+next：`results/reliability_study/metrics/toxicity/sbert_bge_target_next/run.log`，时间为 15:27:38。

项目模型目录中的 `sentence_transformer_name.txt` 和 `embedding_config.json` 也记录了相同 checkpoint 名称，但没有直接保存 revision。运行主机的 Hugging Face 缓存中只有一个该模型的 snapshot，`refs/main` 指向：

```text
a5beb1e3e68b9ab74eb54cfd186867f64f240e1a
```

该 snapshot 创建于 2026-06-07，早于上述实验运行，且缓存中未发现第二个 revision。因此，这个 commit 是现有同机证据所支持的实验 revision。严格而言，实验产物没有保存“输出—commit”绑定或模型文件哈希，所以论文中宜表述为“从运行主机缓存恢复的 revision”，不宜声称它由实验 manifest 原生记录。

## 3. Pooling 与 embedding 后处理

缓存 snapshot 的 `modules.json` 定义了以下模块顺序：

1. `sentence_transformers.models.Transformer`
2. `sentence_transformers.models.Pooling`
3. `sentence_transformers.models.Normalize`

`1_Pooling/config.json` 的有效设置为：

```json
{
  "word_embedding_dimension": 768,
  "pooling_mode_cls_token": true,
  "pooling_mode_mean_tokens": false,
  "pooling_mode_max_tokens": false,
  "pooling_mode_mean_sqrt_len_tokens": false
}
```

因此，句向量取最后一层 `[CLS]` token 的 768 维表示，不采用 mean pooling 或 max pooling。checkpoint 本身还带有 `Normalize` 模块。项目训练代码 `src/classical_training.py` 又依据 `embedding_normalization: l2` 调用 `encode(..., normalize_embeddings=True)`；重复 L2 归一化在数值意义上是幂等的。代码最终将特征显式转换为 NumPy `float32`，BGE encoder 保持冻结，不参与 Logistic Regression 训练。

## 4. Truncation metadata 与实际截断审计

以下三个来源一致给出最大长度 512：

- `sentence_bert_config.json`：`max_seq_length = 512`；
- `tokenizer_config.json`：`model_max_length = 512`；
- 在 `MentalFlow` 环境加载同一 snapshot：`SentenceTransformer.max_seq_length = 512`。

tokenizer 的 `truncation_side` 和 `padding_side` 均为 `right`。Sentence-Transformers 的 Transformer 模块在编码时使用 tokenizer 的 `truncation="longest_first"` 和 `max_length=512`。对于本项目的单文本输入，这等价于超过限制时从右侧移除末尾 token。长度统计包含 `[CLS]`、`[SEP]` 等特殊 token。

使用 `results/reliability_study/data/toxicity/sentences.csv` 的全部 2,907 行、`src/input_features.py` 的原输入构造逻辑及该 revision 的 tokenizer 重新计算得到：

| 输入 | median | p90 | p95 | p99 | max | 超过 512 | 截断率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| target-only | 24 | 38 | 44 | 57 | 105 | 0 | 0.00% |
| target+next | 51 | 73 | 81 | 97 | 137 | 0 | 0.00% |

因此，配置层面存在 512-token 右侧截断，但两个已报告 BGE 实验的实际数据均未触发截断。`configs/reliability_protocol.yaml` 中的 `max_length: 128` 属于 ALBERT 路径；BGE 的 `SentenceTransformer.encode` 调用没有读取该字段，不能把 128 误报为 BGE 的截断上限。

## 5. 硬件与软件环境记录

### 5.1 当前运行主机实测

| 类别 | 记录 |
|---|---|
| 主机 | ASUSTeK ROG Strix `G512LV_G512LV` |
| CPU | Intel(R) Core(TM) i7-10875H CPU @ 2.30 GHz |
| CPU 并行度 | 16 logical processors |
| 可见物理内存 | 16,953,593,856 bytes（15.79 GiB） |
| GPU | NVIDIA GeForce RTX 2060 |
| GPU 显存 | 6,144 MiB |
| GPU compute capability | 7.5 |
| 当前 NVIDIA driver | 610.62 |
| 操作系统 | Microsoft Windows build 10.0.26200，x64 |

### 5.2 与实验 manifest 对应的 Conda 环境

`results/reliability_study/data/toxicity/protocol_manifest.json` 记录 Python 3.11.15；本机 `MentalFlow` 环境与之吻合，并保留了以下依赖：

| 软件 | 版本 |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.13.0+cu130 |
| CUDA runtime（PyTorch build） | 13.0 |
| cuDNN runtime version identifier | `92000`（保留 PyTorch 原始返回值，避免对版本编码作额外推断） |
| Transformers | 5.9.0 |
| Sentence-Transformers | 5.5.1 |
| Tokenizers | 0.22.2 |

当前该环境检测到 CUDA 可用、1 张 GPU，设备名为 NVIDIA GeForce RTX 2060。训练代码在 `torch.cuda.is_available()` 为真时选择 `cuda`，否则选择 `cpu`。

### 5.3 历史可证明范围

硬件表和软件版本是 2026-08-24 在保存项目、模型缓存与匹配 Conda 环境的同一主机上采集的。BGE 运行日志没有持久化 hostname、GPU UUID、CPU、RAM、driver、CUDA device 或能耗，所以这些字段不能仅凭实验产物证明为 2026-08-16 运行时的精确状态。现有证据支持“实验很可能在上述主机及 `MentalFlow` 环境中运行”，但论文应保留这一 provenance 限制。当前 driver 版本尤其不应当作历史 driver 版本报告。

checkpoint 自带的 `config_sentence_transformers.json` 还记录了其导出时的软件版本（Sentence-Transformers 2.2.2、Transformers 4.28.1、PyTorch 1.13.0+cu117）；这些是模型发布/导出 metadata，不是本项目实验运行环境版本。

## 6. 可直接用于论文的方法描述

> Frozen embeddings were produced with `BAAI/bge-base-en-v1.5` (revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`, recovered from the run host's sole cached snapshot). The Sentence-Transformers pipeline used 768-dimensional CLS-token pooling followed by L2 normalisation. Inputs were tokenised with a 512-token maximum and right-side truncation; however, no input was truncated (maximum 105 tokens for target-only inputs and 137 tokens for target-plus-next inputs). Embeddings were stored as float32 features and the encoder was frozen. The matched environment contained Python 3.11.15, PyTorch 2.13.0+cu130, Transformers 5.9.0, and Sentence-Transformers 5.5.1. The run host currently records an Intel i7-10875H CPU, 15.79 GiB RAM, and an NVIDIA GeForce RTX 2060 with 6 GiB VRAM; because the original run logs did not persist hardware identifiers, these hardware values are recovered host metadata rather than an immutable per-run record.

## 7. 证据位置

- `results/reliability_study/models/toxicity/sbert_bge_target/embedding_config.json`
- `results/reliability_study/models/toxicity/sbert_bge_target_next/embedding_config.json`
- `results/reliability_study/metrics/toxicity/sbert_bge_target/run.log`
- `results/reliability_study/metrics/toxicity/sbert_bge_target_next/run.log`
- `results/reliability_study/data/toxicity/protocol_manifest.json`
- `results/reliability_study/data/toxicity/sentences.csv`
- `src/classical_training.py`
- `src/input_features.py`
- `configs/reliability_protocol.yaml`
- 主机缓存：`C:/Users/15509/.cache/huggingface/hub/models--BAAI--bge-base-en-v1.5/`
