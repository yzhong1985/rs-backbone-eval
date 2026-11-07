# T11 OlmoEarth v1.3 预计算 Embedding 对齐与比较方案

- 日期：2026-11-06
- 状态：**64 维结果作废，重跑中（2026-11-07 第二轮评审）**。旧提取器的像元角点/
  中心混淆导致 1,087/2,433 个 patch 列偏移 1 像元；已在 `2483162` 修复并重提，
  旧 6 个 run 全部 `superseded`。旧数字（0.5759 / 0.5756 / +0.09）不再引用；
  128 维镜像已含、特征已按新规则重提，待 64 维定稿后启动。
  实施细节见 `docs/evaluation/OlmoEarth_v1.3_镜像下载与提取_2026-11-07.md`
- 范围：T11 PASTIS-HD 上新增一个“预计算 embedding 产品”对照：**OlmoEarth v1.3 64 维**，与现有 AEF 64 维并列
- 相关文档：
  - `docs/evaluation/AEF_PASTIS对齐抽查_2026-11-03.md`
  - `docs/evaluation/AEF_PASTIS-HD多时相对齐与FLAIR-HUB评估_2026-11-03.md`
  - `docs/models/OlmoEarth接入进展.md`（当前 repo 是 v1.2 Tiny/Base，不是 v1.3）

## 0. 一页结论

1. **OlmoEarth v1.3 已经有开源、全球预计算的 10 m 年度 embedding**，64/128 维、int8、-128=NODATA，形式上与 AEF 非常接近。
2. **可以做，而且成本可控**：2,433 个 PASTIS patch 的 64 维特征约 **2.6 GB**（int8），提取和训练都比重新跑 foundation model 便宜。
3. **比较的是“年度 embedding 产品”**，不是 backbone：AEF 与 OlmoEarth v1.3 都融合了 S1/S2（OlmoEarth 还加 Landsat），输入预算和多模态来源必须写清。
4. **主比较用 64 维**，与 AEF 对齐；128 维作为可选扩展，单独一栏，不与 64 维混排。
5. **不要和 repo 里的 OlmoEarth v1.2 Tiny/Base 混为一个模型**：v1.3 是新模型、新产品，应使用新的 model/task/protocol id。
6. **最大前置风险是 2019 年法国覆盖**：v1.3 全球覆盖是 2023–2025，2017–2022 只覆盖部分国家。必须先查 coverage layer；若 2019 法国未覆盖，需要走 OlmoEarth Studio 或本地推理，成本和排期会变化。

## 0.1 对上一轮 P2 评审的响应

| P2 意见 | 本版处理 |
|---|---|
| 覆盖检查用错依据：`coverage.tif` 是预期范围，不能证明 2019 已生成 | 改为直接读 `embeddings.zarr` 的 2019 切片，在 24 个 PASTIS patch 上统计 valid share；`coverage.tif` 只作预期范围参考；当前实际 2019 可用性**尚未确认**，列为 Phase 0 第一项 |
| 与 AEF 训练协议不一致：AEF 是固定第 8 轮，MLP 不是 LiteSpatialHead | 已改为：`aef_worker.heads.MLPHead` / `UPerNetHead`，MLP lr 3e-3、UPerNet lr 1e-3，**固定第 8 轮直接评测，不做 validation 选模**；不再写 LiteSpatialHead |
| 无效像元口径未落实；AEF 置零特征但标签全范围参与 loss/mIoU | 明确主口径 A：与 AEF 一致（无效特征置零、标签全范围），可直接沿用旧 AEF 分数；另加 valid-mask 诊断指标。若评审要求 valid-mask 作主表，则 AEF 必须重跑，不能沿用旧分数 |
| 新 task 注册/准入清单不完整 | 增加 §5.5 清单：`input_kind: embedding_features`、`OFFICIAL_EPOCHS/TEST/VALIDATION["T11O"]`、worker 协议映射、`BOE/BOEM` 独立轨、`TASK_NAMES`/`OUT_OF_SUITE`、registry/scorecard/portal/report 测试；并说明为什么推荐 T11O、复用 T11 需要 model-level input override |

## 1. 背景

T11 目前已有两组“预计算 embedding”结果：

| 输入 | 头 | test mIoU |
|---|---|---:|
| AEF 64 维（2019 年度） | MLP | 0.4849 ± 0.0081 |
| AEF 64 维（2019 年度） | UPerNet | **0.4879 ± 0.0099** |

OlmoEarth v1.3 于 2026-10-22 发布，同时开放：

- 全球年度预计算 embedding：10 m、64/128 维、int8；
- 模型权重、推理代码；
- OlmoEarth Studio（可按时间/区域/分辨率自定义导出）；
- 论文中的 task-level embedding dump（`allenai/olmoearth-paper-embeddings`）。

官方入口：

- 文档：<https://docs.olmoearth.allenai.org/embeddings/>
- 数据：<https://source.coop/olmoearth/olmoearth-v1-3-embeddings>
- Hugging Face 模型：<https://huggingface.co/allenai/OlmoEarth-v1_3-Base>

> 注：上面的 docs 域名以官方实际为准；方案评审时以 Source Cooperative 页面和官方 docs 链接为准。

## 2. 目标与非目标

### 2.1 目标

在 **完全相同的 T11 划分、标签、头和种子**下，回答：

> 对 PASTIS-HD 地块/作物分割，OlmoEarth v1.3 的年度 embedding 是否比 AEF 更好？

具体：

- 64 维 OlmoEarth v1.3 vs 64 维 AEF；
- 同一 MLP / UPerNet 头；
- 同一 Fold 1–3 / 4 / 5 划分；
- seed 17/42/73；
- 8 epoch（与 AEF probe 一致）；
- 报告 test mIoU、validation mIoU、每类 IoU/混淆矩阵。

### 2.2 非目标

- 不把 OlmoEarth v1.3 embedding 与 OlmoEarth v1.2 Tiny/Base 的 **监督 20-bin 结果**混为一个模型；
- 不把 64 维和 128 维混在同一个 cell；
- 不把“embedding 产品比较”写成“backbone 排名”；
- 不并入 Prithvi-suite D1；
- 不做高分辨率目标提取（10 m embedding 不适用于 iSAID/FAIR1M 级别的小目标，见 `docs/evaluation/AEF_PASTIS对齐抽查_2026-11-03.md` 的讨论）。

## 3. 数据规格与前置检查

### 3.1 规格

| 项目 | AEF | OlmoEarth v1.3 |
|---|---|---|
| 分辨率 | 10 m | 10 m |
| 维度 | 64 | 128（前 64 维可单独使用） |
| 时间 | 年度 | 年度 |
| 数值 | int8，-128=NODATA | int8，-127..127，-128=NODATA |
| 覆盖 | 2017–2025 | 2023–2025 全球；2017–2022 部分国家 |
| 输入源 | S1/S2 等 | S1 + S2 + Landsat-8/9 |
| 访问 | GCS COG | Source Cooperative Zarr（S3 兼容） |

### 3.2 必须先查的三件事

1. **2019 年法国实际窗口是否有数据**
   - **不能用 `coverage.tif` 判断某年是否已生成**：官方说明它表示“预期覆盖范围”，不是逐年完成情况；
   - 必须直接读 `embeddings.zarr/utmNN/embeddings` 的 2019 切片，在 PASTIS 的 10 m 窗口上统计有效像元（非 -128）；
   - 抽样至少 24 个 patch（与 AEF QC 同一批），记录每个 patch 的 valid share；
   - 若 2019 窗口全为 -128，才走 §3.3 的备选路径。
2. **Zarr 的坐标/分块/维度顺序**
   - 数组是 `(time, band, y, x)`，int8，`-128` 为 NODATA；
   - 需要确认 `utmNN` 的 `spatial:transform`、`proj:code`、`time` 轴；
   - 用 `zarr>=3` + `obstore` 直接读小窗口，不要整年下载。
2b. **64 维的官方依据**
   - 官方 README 说前 64 维是 Matryoshka 子嵌入、可单独使用；但这需要 Phase 0 核实：64 维是独立数组/独立产品，还是 128 维的前 64 维；
   - 如果只是 128 维截断，不能把它当作独立训练的 64 维产品；报告要写清“128 维的前 64 维”；
   - 128 维与 64 维不能混在同一个 cell。
2c. **外部事实核验**
   - 发布日期、2023–2025 全球覆盖、2017–2022 部分国家、Zarr 维度顺序等，均以 Source Cooperative README/官方论文为准；
   - 评审文档中的外部事实不能只依赖二手报道，Phase 0 要逐条核对。
3. **官方 dequantization**
   - OlmoEarth 官方公式：`(abs(v)/127.5)² · sign(v)`，`-128` 先转 NaN；
   - **该公式与 AEF 的 `sign(x)·(x/127.5)²` 数学上相同**，不是两套不同公式；真正的差异是 OlmoEarth 要求 `-128 → NaN` 并显式并入 `valid` mask，AEF worker 则先解码再把 invalid 置零；
   - 训练前在 QC 里对比解码前后的均值/方差/分位数，并记录 `invalid_pixel_policy`。

> **2026-11-07 更正**：24 个 QC patch 已全部提取，valid share 全为 1.0；逐 patch 四联图已生成。NDVI 上的 8/24 `(0,+1)` 并非随机不确定性，而是提取器把 Zarr 角点原点当作像元中心、少减半个像元导致的系统性列偏移（影响 1,087/2,433 个 patch）。已改为 `floor((x-c)/a)`（最近像元中心）并重提、重跑；原“接受当前网格、不做平移修正”的决定作废。

### 3.3 2019 年覆盖不足时的备选

| 方案 | 说明 | 成本/风险 |
|---|---|---|
| A. OlmoEarth Studio 自定义导出 | 按 2019、法国、10 m、S1+S2(+Landsat) 导出 COG | 需要 API key；按官方计费；最省工程 |
| B. 本地运行 v1.3 Base | 下载权重，对 PASTIS 的 S1/S2/Landsat 窗口推理 | 工程量大；需要复现 rslearn 输入管线 |
| C. 论文 task-level embedding dump | `allenai/olmoearth-paper-embeddings` 里可能有 PASTIS | 只能做快速探针，分辨率和样本顺序不一定匹配 T11，不能当正式结果 |

**建议**：先做 A 的可行性检查；如果 Studio 能按 2019/法国导出，优先 A。B 只在 Studio 不可行且必须本地复现时考虑。C 只用于 sanity check。

### 3.4 数据泄漏检查（必须先做）

OlmoEarth 论文的评测任务里包含 PASTIS；需要确认 v1.3 的预训练/蒸馏没有使用 PASTIS 或法国 2019 的作物标签：

- 查 v1.3 技术报告的预训练数据与评测数据清单；
- 查 Hugging Face 模型卡/Source Cooperative provenance，确认没有 PASTIS/French 2019 crop labels；
- AEF 也要补同样的说明（AEF 的预训练/产品是否接触过 PASTIS）；
- 如果任一方使用了 PASTIS 训练/蒸馏，则该对比是 **in-domain / 污染** 结果，必须单独标注，不能作为“未见任务泛化”的证据。

### 3.5 Manifest 与样本对齐

T11O 必须有自己的 manifest：

- `build-manifest --task T11O`，aux path 指向 OlmoEarth `.npz`，encoding 写 `olmoearth_int8_as_float16`；
- 验收 test 496 / validation 482 / train 1455，且 sample_id 集与 T11 完全一致；
- 固定 `RS_EVAL_MANIFEST`，记录 manifest sha256；
- 如果 sample_id 集不一致，AEF 与 OlmoEarth 的比较不在同一批样本上，不能直接对比。

## 4. 对齐协议

完全复用 AEF 的对齐与 QC 逻辑，只替换读取层。

### 4.1 采样网格

1. 以 SPOT 的 1 m 仿射为准，不用 patch bbox；
2. 在 1 m 网格上取 10 m 单元中心：`5 + 10k`，得到 128×128 个采样点；
3. 转到 OlmoEarth Zarr 的 UTM zone；
4. 按最近像元读取 embedding；
5. 跨 tile/分块边界时用 `boundless` 或窗口裁剪 + 零填充，记录 `valid`；
6. `-128` 作为 NODATA，不参与统计。

### 4.2 明确不做

- 不做逐 patch 平移修正（会按 NDVI/标签“拟合”对齐）；
- 不做全局偏移修正（除非 QC 发现系统性方向；AEF QC 没有）；
- 不跳过 OlmoEarth 的 `-128 → NaN` / `valid` mask 处理；公式本身与 AEF 相同，差异在 NODATA 语义；
- 不静默插值/补值。

### 4.3 QC 子集与逐 patch 可视化

- 取 24 个 patch，覆盖 4 个 MGRS tile、Fold 1–5；
- **逐 patch 出图，风格与 AEF 对齐 QC 保持一致**，方便并排看：
  - `olmoearth_pastis_alignment.png`：三行/四行图，S2 RGB / SPOT 1 m / OlmoEarth PCA 伪彩 / 地块边界叠加；
  - `olmoearth_pastis_zoom_<patch>.png`：选取代表性 patch（10000、10003、20020，以及高信号/边界 patch）做 512×512 放大，叠加 `ParcelIDs` 边界；
  - `olmoearth_pastis_tiles.png`：4 个 MGRS tile 各取一个 patch 的三联图（S2 / OlmoEarth PCA / 标签）；
  - `olmoearth_alignment_batch.csv`：tile、fold、年份、Zarr window、valid share、PC1-NDVI 相关、最佳平移；
- 出图细节：
  - OlmoEarth 先按官方公式 dequantize，PCA 只用 `valid` 像元拟合/统计；
  - 伪彩、拉伸、图例尽量沿用 AEF QC 的参数，保证两张图可以直接对比；
  - 图上标注 patch id、tile、year、valid share、最佳平移；
  - 输出目录建议 `/home/yzhong/rs-eval/audits/olmoearth-<date>/`，与 AEF 的 `aef-20261103/` 并列；
- 验收：
  - 高信号 patch 的最佳平移在 `(0,0)`；
  - 无系统性方向；
  - valid share 与 AEF 同量级；
  - 地块边界、道路、田块纹理在视觉上对得上；
  - 若边界 patch 出现裁剪，必须用窗口/`boundless` 修复后复测。

## 5. 实验设计

### 5.1 新的 task / model / protocol

为避免和 AEF、OlmoEarth v1.2 混淆，建议**新建独立 task**：

| 项 | 建议 id | 说明 |
|---|---|---|
| task | `T11O` | PASTIS-HD OlmoEarth v1.3 embedding；与 T11 同数据/同划分 |
| model | `olmoearth_v13_embedding` | 预计算产品，不是 v1.2 Tiny/Base |
| protocol | `olmoearth_v13_probe_mlp` | 与 `aef_probe_mlp` 同头 |
| protocol | `olmoearth_v13_probe_upernet` | 与 `aef_probe_upernet` 同头 |

任务 YAML 建议：

- `dataset_id: PASTIS-HD`
- `sensor: olmoearth_v1_3_embedding`
- `input_kind: embedding_features`
- `bands: [OLE_00 … OLE_63]`（64 维主实验）
- `time_steps: 1`
- 同 T11 的 `num_outputs: 20`、`ignore_index: 19`、`primary_metric: mean_iou`
- 同 Fold 1–3 / 4 / 5 划分

> 128 维实验单独建 `T11O128`（或 `T11O_128`）与对应 model/protocol，不塞进 64 维 cell。

### 5.2 训练口径

- 头：**必须复用 AEF worker 的实际头实现**，不是泛称：
  - MLP：`aef_worker.heads.MLPHead`（`Conv2d(64,128,1)` + `GroupNorm(8)` + `GELU` + `Conv2d(128,20,1)`），AEF 用的 lr = 3e-3；
  - UPerNet：`aef_worker.heads.UPerNetHead`（width 128、PPM sizes `(1,2,3,6)`、合成 1/2/4/8 金字塔），AEF 用的 lr = 1e-3；
  - 不要写成 `LiteSpatialHead`，那是 LIM/其他任务的头，不是 AEF 的 MLP。
- epoch：**固定 8 轮，取第 8 轮权重直接评测**；
- **不做 validation 选模**：AEF 的现有数字是固定第 8 轮，不是 best-epoch；新方案必须一致，否则不能和 AEF 比；
- seed：17/42/73；
- 指标：test mIoU（0–18，void=19 忽略）、foreground mIoU、每类 IoU；validation 只报告，不参与选模；
- 设备：与 AEF 相同（CPU 即可；若用 GPU，记录 device，不与 CPU 混 cell）；
- 每个 run 记录：
  - task/model/protocol/seed；
  - embedding 来源（Source Cooperative / Studio / local）；
  - 维度、年份、coverage、dequantization、valid share；
  - git commit、manifest sha256、host。

### 5.3 无效像元口径（必须先定）

当前 AEF worker 的行为是：

- `_decode` 先对特征做 dequantize；
- 把 `valid=False` 的特征位置**置零**；
- 但对应标签**仍然参与 loss 和 mIoU**，零值也进入统计。

因此新方案必须二选一，并写清楚：

| 口径 | 做法 | 与现有 AEF 分数 |
|---|---|---|
| **A. 全范围（推荐主口径）** | 与 AEF 完全一致：无效特征置零，标签全范围参与 loss/mIoU | 可直接沿用现有 AEF 0.4849 / 0.4879 |
| B. 共同有效像元 | 训练/评测都只用 `valid=True` 的像元，AEF 与 OlmoEarth 用同一 mask | **必须重跑 AEF**；旧分数不能直接沿用 |

建议：

- 主表用 **A**，保证和已有 AEF 结果可比；
- 另加一个 **B 的诊断指标**（只在 valid 像元上算 mIoU），不作为 headline；
- 如果评审要求把 B 作为主口径，则 AEF 需要按同一 mask 重跑，报告中不能混用旧分数；
- 无论选哪个，都要在 run tags 写 `invalid_pixel_policy`（`zero-fill-full-label` / `valid-mask`）和 `valid_share`。

### 5.4 对照表

最终报告建议放三行：

| 输入 | 维度 | 头 | test mIoU |
|---|---:|---|---:|
| AEF 2019 | 64 | MLP | 0.4849 ± 0.0081 |
| AEF 2019 | 64 | UPerNet | 0.4879 ± 0.0099 |
| OlmoEarth v1.3 2019 | 64 | MLP | 待跑 |
| OlmoEarth v1.3 2019 | 64 | UPerNet | 待跑 |
| OlmoEarth v1.3 2019 | 128（可选） | UPerNet | 待跑 |

解读边界：

- 这是“年度 embedding 产品”对比，不是 backbone 对比；
- 64 维 vs 128 维是输入预算差异；
- OlmoEarth 使用 Landsat，AEF 的源不完全相同；
- 不与 T11 的 BBF/BBT 混排，也不进 Prithvi-suite D1；
- 若 OlmoEarth 64 维显著更好，只能说明“该产品在 PASTIS-HD 上更适合”，不能直接外推到其他任务/传感器。

### 5.5 注册与准入清单（P2 重点）

仅加 YAML 和参数化通道不够；要进入正式结果，至少需要以下改动：

1. **TaskSpec**
   - `input_kind` 目前只接受 `raster` / `aef_features`，新增 `embedding_features`；
   - 新 task `T11O` 的 YAML：`sensor: olmoearth_v1_3_embedding`、`input_kind: embedding_features`、`bands: OLE_00..OLE_63`、`time_steps: 1`、`num_outputs: 20`、`ignore_index: 19`。
1b. **Manifest**
   - `build-manifest --task T11O`；aux path 指向 OlmoEarth `.npz`，encoding 写 `olmoearth_int8`（实施时从草案的 `olmoearth_int8_as_float16` 修正：直接存 int8，不转 float16，dequantize 公式不变，见 `OlmoEarth_v1.3_镜像下载与提取_2026-11-07.md`）；
   - 验收 sample_id 集：test 496 / validation 482 / train 1455，且与 T11 完全一致；
   - 固定 `RS_EVAL_MANIFEST` 并记录 sha256；否则两边的比较不在同一批样本上。
2. **正式预算与样本锚**
   - `OFFICIAL_EPOCHS["T11O"] = 8`，否则 8 轮会被 `is_reduced_run` 当成短跑过滤；
   - `OFFICIAL_TEST_SAMPLES["T11O"] = 496`；
   - `OFFICIAL_VALIDATION_SAMPLES["T11O"] = 482`；
   - 否则 `is_shrunken_eval` / worker 的样本锚无法通过。
3. **Worker 身份**
   - 新 model `olmoearth_v13_embedding`：`worker_id: aef`，`checkpoint_env` 可保留一个未使用的环境名；
   - 新 protocols `olmoearth_v13_probe_mlp` / `_upernet`：`regime: frozen_probe`、`workers: [aef]`、`tasks: [T11O]`；
   - `aef_worker.runner` 的协议→头映射增加这两个 id；
   - `_decode` 接受 `embedding_int8_as_float16`（或统一成通用 encoding），并保持官方 dequantize + valid 置零。
4. **轨道与报告**
   - 新 track：`BOE`（UPerNet）/ `BOEM`（MLP），**不要复用 AEF 的 BA/BAM**，也避免 `BO` 与 `B0` 视觉混淆；
   - `PROTOCOL_TRACKS`、`TRACK_ORDER`、`TRACK_NAMES` 增加；
   - `matrix` / `report` / `portal` 的 `TASK_NAMES` 增加 `T11O`；
   - `OUT_OF_SUITE` 增加 `T11O`（T11 本身已在 Prithvi-suite 之外）；
   - 报告 T11 解读段增加 OlmoEarth embedding 说明。
5. **测试**
   - `tests/test_registry.py`：任务/协议精确集合；
   - `tests/test_scorecard.py` / `test_review_fixes.py`：新 track 不混 cell；
   - `tests/test_portal.py`：任务列表包含 `T11O`；
   - `tests/test_report_sections.py`：T11O 面板存在。
6. **T11 vs T11O**
   - 推荐 **T11O**：输入产品、波段名、sensor、dequantization 都与 AEF 不同，独立 task 最不容易混淆；
   - 若坚持复用 T11，则必须在 `ModelSpec`/`RunWriter` 增加 model-level 的 input override（sensor/bands/input_kind），否则 `run.json.input` 会写成 AEF，兼容性矩阵会把两个产品混在一行；这比新增 T11O 更侵入。

## 6. 实现步骤

### Phase 0：可行性（0.5–1 天）

1. 直接读 2019 的 Zarr 窗口，统计 24 个 PASTIS patch 的 valid share；`coverage.tif` 只用于预期范围参考，不能作为“2019 已生成”的证据；
2. 读 1 个 20 km tile 的 Zarr 小窗口，确认维度顺序、CRS、NODATA、dequantization；
3. 用 3 个 patch 做对齐 QC，并出第一版逐 patch 可视化（S2 / SPOT 1 m / OlmoEarth PCA / 标签）；
4. 如果 2019 未覆盖，完成 Studio/本地推理的决策和成本评估。

### Phase 1：全量提取（0.5–1 天）

1. 2,433 个 patch，按 128×128、10 m 网格提取 64 维；
2. 存 `.npz`：`features`（int8 或 float16）、`valid`；
3. sidecar JSON：tile、window、transform、year、source、valid share、sha256、dequantization；
4. 抽 5% patch 复算 QC；对 24 个 patch 出批量图、对代表性 patch 出放大图，确认没有新的边界问题；
5. 估算磁盘：64 维约 2.6 GB（int8），128 维约 5.2 GB。

### Phase 2：训练与评测（数小时）

1. 注册 `T11O` task、model、protocols；
2. 复用 AEF worker 的 MLP/UPerNet 头，参数化通道数/band 名；
3. 3 seed × 2 头，8 epoch；
4. audit、报告、异常检查。

### Phase 3：报告（0.5 天）

- 更新 T11 embedding 面板；
- 写清 64/128、AEF/OlmoEarth、数据来源和 license；
- 给出 test/validation 均值 ± sd、每类 IoU；
- 结论只限定在 PASTIS-HD / 10 m 年度 embedding。

## 7. 成本估算

| 项目 | 估算 |
|---|---|
| 提取 64 维 | 2,433 patch，约 5 GB，CPU/IO 为主 |
| 提取 128 维（可选） | 约 10 GB |
| 训练 | 2 头 × 3 seed × 8 epoch，CPU/GPU 数小时 |
| 工程 | Zarr 读取 + 对齐 QC + 注册，约 2–3 天 |
| 存储 | 4090 当前 `/geodata` 约 34 GB 可用；64 维 5 GB 足够，128 维需先确认空间 |

## 8. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 2019 法国未生成 | Phase 0 直接读 2019 Zarr 窗口；全 -128 才走 Studio/本地推理，不拿 coverage.tif 当已生成证据 |
| Zarr 读取/网络不稳定 | 先下载覆盖 PASTIS 的 tile 到本地，再提取；记录 source URL 和 sha256 |
| dequantization 用错 | 只用官方 `embedding_transforms.py`；在 QC 里对比解码前后统计 |
| 64/128 混排 | 128 维单独 task/cell，不并入 64 维 |
| 与 OlmoEarth v1.2 混淆 | 新 model id `olmoearth_v13_embedding`，报告写明版本 |
| License | OlmoEarth Artifact License，有用途限制；公司使用前法务确认 |
| 对齐边界 | 复用 AEF 的 `boundless`/多 tile 逻辑，24 patch QC + 5% 全量抽检 |
| 结果过解读 | 只主张 PASTIS-HD 年度 embedding 产品对比，不主张 backbone 排名/跨任务泛化 |

## 9. 验收标准

- [ ] 2019 年法国 coverage 已确认，或 Studio/本地路径已批准；
- [ ] 24 patch QC 通过：高信号 patch 在 `(0,0)`、无系统性方向、valid share 合理；
- [ ] 2,433 patch 全量提取完成，sidecar 完整，5% 抽检通过；
- [ ] `T11O` 64 维 MLP/UPerNet × seed 17/42/73 全部 `completed`；
- [ ] `audit_runs.py` 通过，run.json 有来源/维度/valid share/dequantization tags；
- [ ] 报告新增 embedding 产品对照面板，AEF 与 OlmoEarth 64 维同表，128 维单独；
- [ ] 不进入 Prithvi-suite D1，不与 BBF/BBT 混排。

## 10. 需要审批的决策点

- [ ] 是否新增独立 task `T11O`（而不是复用 T11 的 AEF task）；
- [ ] 主实验是否只做 **64 维**，128 维作为可选 Phase 2；
- [ ] 2019 法国未覆盖时，是否允许使用 OlmoEarth Studio（API key/计费）；
- [ ] 是否接受 OlmoEarth Artifact License 的用途限制（建议法务确认）；
- [ ] 是否批准约 5 GB（64 维）本地存储和 2–3 天工程排期；
- [ ] 是否同意“embedding 产品比较”单独成面板，不并入 backbone D1。

## 11. 参考

- OlmoEarth Embeddings 文档：<https://docs.olmoearth.allenai.org/embeddings/>
- OlmoEarth v1.3 预计算数据：<https://source.coop/olmoearth/olmoearth-v1-3-embeddings>
- OlmoEarth v1.3 模型：<https://huggingface.co/allenai/OlmoEarth-v1_3-Base>
- OlmoEarth v1.3 发布说明：<https://allenai.org/blog/olmoearth-v1-3>
- 论文 embeddings 数据：<https://huggingface.co/datasets/allenai/olmoearth-paper-embeddings>
- 本仓库 AEF 对齐 QC：`docs/evaluation/AEF_PASTIS对齐抽查_2026-11-03.md`
- 本仓库 AEF 多时相对齐：`docs/evaluation/AEF_PASTIS-HD多时相对齐与FLAIR-HUB评估_2026-11-03.md`
