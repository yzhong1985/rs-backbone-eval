# T11O：OlmoEarth v1.3 64 维年度 embedding 与 AEF 对照（2026-11-07 定稿）

- 任务：T11O（PASTIS-HD，与 T11 同一批 patch、标签、Fold 1–3/4/5）
- 输入：OlmoEarth v1.3 2019 年度 embedding，64 维（128 维的 Matryoshka 前缀），10 m
- 头与预算：AEF worker 的 `MLPHead` / `UPerNetHead`，固定第 8 轮，不做 validation 选模
- seed：17 / 42 / 73；设备：CPU
- manifest：`ef75efa9d93a5dca7cc71773b1d43d008a35d4453b86e571e50f71b5595b5de4`
  （2,433；train 1,455 / validation 482 / test 496，sample_id 与 T11 完全一致）
- 代码：`364b948`（干净树）；`scripts/audit_runs.py` 判定 T11O 6/6 满足正式预算
- 数据修正：旧提取器把 Zarr 像元角点当作中心，1,087/2,433 个 patch 列偏移 1 像元
  （review 2026-11-07 P1）。已改为 `floor((x-c)/a)` 并重提，旧 v1/v2 run 全部
  `superseded`；本文数字全部来自修复后的 `-v3` run。

## 1. 结论

**在完全相同的数据、划分、头和 8-epoch 预算下，OlmoEarth v1.3 64 维在 PASTIS-HD
上比 AEF 2019 64 维高约 +0.09 mIoU，三个种子方向一致：**

| 输入（64 维） | 头 | test mIoU（mean ± sd, n=3） |
|---|---|---:|
| AEF 2019 | MLP | 0.4849 ± 0.0081 |
| AEF 2019 | UPerNet | 0.4879 ± 0.0099 |
| **OlmoEarth v1.3 2019** | **MLP** | **0.5788 ± 0.0107** |
| **OlmoEarth v1.3 2019** | **UPerNet** | **0.5799 ± 0.0105** |

配对差值（同 seed，OlmoEarth − AEF）：

- MLP：+0.0904 / +0.0925 / +0.0989，均值 **+0.0939**
- UPerNet：+0.1018 / +0.0811 / +0.0933，均值 **+0.0921**

三个种子全部为正，两种头给出的差值一致（约 +0.09），不是单一种子或单头驱动。

这是**年度 embedding 产品**的对比，不是骨干排名；“OlmoEarth 架构比 LIM/Prithvi
强”不能由此得出。

## 2. 逐 seed

| 输入 | 头 | s17 | s42 | s73 | mean ± sd |
|---|---|---:|---:|---:|---:|
| AEF | MLP | 0.4760 | 0.4921 | 0.4866 | 0.4849 ± 0.0081 |
| AEF | UPerNet | 0.4777 | 0.4886 | 0.4974 | 0.4879 ± 0.0099 |
| OlmoEarth v1.3 | MLP | 0.5664 | 0.5846 | 0.5854 | 0.5788 ± 0.0107 |
| OlmoEarth v1.3 | UPerNet | 0.5794 | 0.5697 | 0.5907 | 0.5799 ± 0.0105 |

AEF 的 6 条 canonical run 全部在同一 manifest `cf092238…` 上（s17-v4、s42-v2、
s73-v4；旧 `29717adc…` 的 6 条已 `superseded`）。

## 3. 训练与准入

| 项 | 值 |
|---|---|
| 编码 | `olmoearth_int8`，`sign(v)*(|v|/127.5)^2`，-128=NODATA |
| 无效像元口径 | `zero-fill-full-label`（与 AEF 一致：特征置零，标签全范围参与 loss/mIoU） |
| preprocessing | `olmoearth_signed_square_imagenetfree` |
| 头 / lr | MLP 3e-3；UPerNet 1e-3；AdamW，weight decay 1e-4 |
| 统计量 | 训练集 tile-balanced 80 patch，逐 run 计算 |
| 评测样本 | test 496/496，validation 482/482（硬锚点） |
| 耗时 | MLP 2.8–3.3 min；UPerNet 28.7–37.4 min |
| worker 校验 | 启动时逐个核对 payload sha256 与 manifest checksum（`payload.ok`） |
| run id | `olmoearth_v13_embedding-T11O-olmoearth_v13_probe_{mlp,upernet}-train-{17,42,73}-v3` |

## 4. 采样修复的影响

修复前后同一 seed（s17）的 MLP：0.5622 → **0.5664**（+0.0042）；AEF 不受影响
（AEF 的抽取本来就是 rasterio `rowcol` 的 floor 语义）。修复后差距从 +0.0861
变为 +0.0904（s17 MLP），方向和量级不变，但旧数字不可再用。

修复前的 20ep 诊断（0.596109）与 20ep UPerNet 因特征被原地覆盖，已分别
`superseded` / 隔离。`-v3` 已重跑完成（s17，`panel=diagnostic`）：MLP **0.5976**、
UPerNet **0.6037**，比同 seed 的 8ep 分别高 +0.031 / +0.024。

## 5. 解读边界

1. **产品对比**：AEF 与 OlmoEarth v1.3 都是年度合成 embedding；两边都是多模态
   （AEF：S2、Landsat、S1、GEDI、DEM、ERA5-Land、PALSAR-2、GRACE、文本；
   OlmoEarth：S1/S2/Landsat），差值不能归因到单一传感器或模型代际。
2. **64 维是 Matryoshka 前缀**：128 维单独成 cell，不与本表混排。
3. **对齐**：当前使用最近像元中心（floor 索引），构造上为亚像元级；旧 +1 px
   系统偏移已修复。
4. **不并入 D1**：T11O 不在 Prithvi-suite D1，也不与 T11 的 BBF/BBT 全量微调混排。
5. **LPIS 标签年**沿用 T11 的 2019 技术假设，未获数据发布说明确认。
6. 弱类（Grapevine/Beet/Orchard）仍接近 0；类别级差异需另出表与混淆矩阵。

## 6. 后续（已完成）

- 20ep 诊断：`-v3` 已跑完（MLP 0.5976 / UPerNet 0.6037，seed 17，`panel=diagnostic`）。
- 128 维：`T11O128` 6 个 run 已在 `37883a4` 完成，两个头都优于 64 维
  （MLP +0.0349、UPerNet +0.0228）；见
  `docs/reports/T11O_OlmoEarth_v1.3_128维对照_2026-11-08.md`。
