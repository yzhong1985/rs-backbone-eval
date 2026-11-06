# T11 OlmoEarth v1.3 预计算 Embedding 对齐与比较方案

- 日期：2026-11-06
- 状态：待审批（尚未开始提取/训练）
- 范围：T11 PASTIS-HD 上新增一个“预计算 embedding 产品”对照：**OlmoEarth v1.3 64 维**，与现有 AEF 64 维并列
- 相关文档：
  - `docs/evaluation/AEF_PASTIS对齐抽查_2026-11-03.md`
  - `docs/evaluation/AEF_PASTIS-HD多时相对齐与FLAIR-HUB评估_2026-11-03.md`
  - `docs/models/OlmoEarth接入进展.md`（当前 repo 是 v1.2 Tiny/Base，不是 v1.3）

## 0. 一页结论

1. **OlmoEarth v1.3 已经有开源、全球预计算的 10 m 年度 embedding**，64/128 维、int8、-128=NODATA，形式上与 AEF 非常接近。
2. **可以做，而且成本可控**：2,433 个 PASTIS patch 的 64 维特征约 **5 GB**（float16），提取和训练都比重新跑 foundation model 便宜。
3. **比较的是“年度 embedding 产品”**，不是 backbone：AEF 与 OlmoEarth v1.3 都融合了 S1/S2（OlmoEarth 还加 Landsat），输入预算和多模态来源必须写清。
4. **主比较用 64 维**，与 AEF 对齐；128 维作为可选扩展，单独一栏，不与 64 维混排。
5. **不要和 repo 里的 OlmoEarth v1.2 Tiny/Base 混为一个模型**：v1.3 是新模型、新产品，应使用新的 model/task/protocol id。
6. **最大前置风险是 2019 年法国覆盖**：v1.3 全球覆盖是 2023–2025，2017–2022 只覆盖部分国家。必须先查 coverage layer；若 2019 法国未覆盖，需要走 OlmoEarth Studio 或本地推理，成本和排期会变化。

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

1. **2019 年法国是否在 coverage layer 内**
   - 若在：直接读 Zarr；
   - 若不在：走 §3.3 的备选路径。
2. **Zarr 的坐标/分块/维度顺序**
   - 需要确认 `(time, band, y, x)` 或 `(band, y, x)` 的实际布局；
   - 用官方示例和 coverage layer 做一次小窗口读取。
3. **官方 dequantization**
   - AEF 用的是 `sign(x)·(x/127.5)²`；
   - OlmoEarth 有 `embedding_transforms.py`，**不能直接套 AEF 的公式**，必须用官方量化/反量化。

### 3.3 2019 年覆盖不足时的备选

| 方案 | 说明 | 成本/风险 |
|---|---|---|
| A. OlmoEarth Studio 自定义导出 | 按 2019、法国、10 m、S1+S2(+Landsat) 导出 COG | 需要 API key；按官方计费；最省工程 |
| B. 本地运行 v1.3 Base | 下载权重，对 PASTIS 的 S1/S2/Landsat 窗口推理 | 工程量大；需要复现 rslearn 输入管线 |
| C. 论文 task-level embedding dump | `allenai/olmoearth-paper-embeddings` 里可能有 PASTIS | 只能做快速探针，分辨率和样本顺序不一定匹配 T11，不能当正式结果 |

**建议**：先做 A 的可行性检查；如果 Studio 能按 2019/法国导出，优先 A。B 只在 Studio 不可行且必须本地复现时考虑。C 只用于 sanity check。

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
- 不把 AEF 的 `sign(x)·(x/127.5)²` 直接套到 OlmoEarth；
- 不静默插值/补值。

### 4.3 QC 子集

- 取 24 个 patch，覆盖 4 个 MGRS tile、Fold 1–5；
- 输出：
  - `olmoearth_pastis_alignment.png`（S2 RGB / SPOT 1 m / OlmoEarth PCA / 标签）；
  - `olmoearth_pastis_zoom_<patch>.png`；
  - `olmoearth_alignment_batch.csv`（tile、fold、年份、tile/window、valid share、PC1-NDVI 相关、最佳平移）；
- 验收：
  - 高信号 patch 的最佳平移在 `(0,0)`；
  - 无系统性方向；
  - valid share 与 AEF 同量级；
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

- 头：复用 AEF worker 的 MLP / UPerNet（`LiteSpatialHead` / 共享 UPerNet 头）；
- epoch：8（与 AEF probe 一致）；
- seed：17/42/73；
- 选模：validation mIoU；
- 指标：test mIoU（0–18，void=19 忽略）、foreground mIoU、每类 IoU；
- 设备：与 AEF 相同（CPU 即可；若用 GPU，记录 device，不与 CPU 混 cell）；
- 每个 run 记录：
  - task/model/protocol/seed；
  - embedding 来源（Source Cooperative / Studio / local）；
  - 维度、年份、coverage、dequantization、valid share；
  - git commit、manifest sha256、host。

### 5.3 对照表

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

## 6. 实现步骤

### Phase 0：可行性（0.5–1 天）

1. 查 Source Cooperative coverage layer，确认 2019 法国；
2. 读 1 个 20 km tile 的 Zarr 小窗口，确认维度顺序、CRS、NODATA、dequantization；
3. 用 3 个 patch 做对齐 QC；
4. 如果 2019 未覆盖，完成 Studio/本地推理的决策和成本评估。

### Phase 1：全量提取（0.5–1 天）

1. 2,433 个 patch，按 128×128、10 m 网格提取 64 维；
2. 存 `.npz`：`features`（int8 或 float16）、`valid`；
3. sidecar JSON：tile、window、transform、year、source、valid share、sha256、dequantization；
4. 抽 5% patch 复算 QC，确认没有新的边界问题；
5. 估算磁盘：64 维约 5 GB（float16），128 维约 10 GB。

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
| 2019 法国未覆盖 | Phase 0 先查 coverage；未覆盖则用 Studio，或降级为 paper embeddings sanity check，不硬跑 |
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
