# T11O 实验设置说明：OlmoEarth v1.3 年度 embedding 对照

> **2026-11-07 23:15 更新：64 维重跑完成。** 旧提取器的像元角点/中心混淆已修复
> （1,087/2,433 个 patch 列偏移），旧 v1/v2 run 全部 `superseded`；下表为修复后
> `-v3`/`-v4` run 的定稿数字。

- 日期：2026-11-07
- 状态：64 维定稿；20 epoch 诊断待重跑；128 维待启动
- 用途：提交上级评审当前实验设置、结果口径与下一步计划
- 相关文档：
  - `docs/evaluation/T11_OlmoEarth_v1.3_embedding对齐与比较方案_2026-11-06.md`（方案与评审响应）
  - `docs/evaluation/OlmoEarth_v1.3_镜像下载与提取_2026-11-07.md`（数据获取与编码）
  - `docs/reports/T11O_OlmoEarth_v1.3_64维对照_2026-11-07.md`（结果快照）

## 0. 一页摘要

在 PASTIS-HD 的同一批 2,433 个 patch、同一 Fold 划分、同一个轻量头、同样固定
8 epoch 的设置下，把输入从 **AEF 2019 年度 embedding（64 维）** 换成
**OlmoEarth v1.3 2019 年度 embedding（64 维）**：

| 输入（64 维） | 头 | test mIoU（mean ± sd, n=3） |
|---|---|---:|
| AEF 2019 | MLP | 0.4849 ± 0.0081 |
| AEF 2019 | UPerNet | 0.4879 ± 0.0099 |
| **OlmoEarth v1.3 2019** | **MLP** | **0.5788 ± 0.0107** |
| **OlmoEarth v1.3 2019** | **UPerNet** | **0.5799 ± 0.0105** |

配对差值（同 seed，OlmoEarth − AEF）：MLP +0.0904/+0.0925/+0.0989（均值
**+0.0939**）；UPerNet +0.1018/+0.0811/+0.0933（均值 **+0.0921**）；三个种子
全部为正。

**这是“年度 embedding 产品”的对比，不是骨干模型排名。** 结果不能外推成
“OlmoEarth 架构比 LIM/Prithvi 强”，也不能外推到其他数据集或传感器。

## 1. 目标与定位

T11（PASTIS-HD）原本已有 AEF 年度 embedding 的冻结头结果（BA/BAM 轨）。
本实验新增第二个“年度 embedding 产品”作为对照，回答：

1. 在相同数据和训练协议下，两个产品的下游分割效果差多少；
2. 差异是否在多个种子、两种头容量下稳定；
3. 64 维是否足够，还是需要 128 维。

T11O 是独立 task（不是复用 T11），因为输入产品、波段名（`OLE_00..63` vs
`AEF_00..63`）、sensor、反量化方式都不同；复用 T11 会让 `run.json.input`
写成 AEF，兼容性矩阵把两个产品混在一行。

## 2. 数据与样本

| 项 | 值 |
|---|---|
| 数据集 | PASTIS-HD（法国，18 类作物 + 背景 + void） |
| patch 数 | 2,433（单 patch 即 group，跨 Fold 无组泄漏） |
| 划分 | Fold 1–3 train 1,455 / Fold 4 validation 482 / Fold 5 test 496 |
| 与 T11 的关系 | 同一批 patch、同一份标签、同一划分；sample_id 集合与 split 逐一相等 |
| manifest | `ef75efa9d93a5dca7cc71773b1d43d008a35d4453b86e571e50f71b5595b5de4`（旧 `d8f46dcf…` 已作废：其 checksum 对应被改写的旧缓存） |
| 覆盖 | 全部 2,433 patch 的 `valid_share = 1.0`（2019 年法国无缺测） |
| 标签年 | 2019（沿用 T11 的技术验证假设，未获 LPIS 年份说明确认） |

对齐（2026-11-07 更正）：Zarr 仿射原点是像元**角点**，旧提取器用
`round((x - c)/a)` 直接索引，在 1,087/2,433 个 patch 上系统性偏移 1 个像元（列方向）。
已改为 `floor((x - c)/a)`（包含像元的索引，即最近像元中心），并重提 64/128 维；
旧结果已标记 superseded。当前对齐在构造上是亚像元级，不再有系统性平移。

## 3. 两个输入产品

| 项 | AEF 2019（T11） | OlmoEarth v1.3 2019（T11O） |
|---|---|---|
| 来源 | Google AlphaEarth Foundations | Source Cooperative 全球 Zarr |
| 维度 | 64 | 64（128 维的 Matryoshka 前缀） |
| 存储/编码 | int8 存为 float16，`raw_aef_int8_as_float16` | int8，`olmoearth_int8` |
| 反量化 | `sign(v)*(|v|/127.5)^2`，-128=NODATA | 同一公式，-128=NODATA |
| 时间 | 2019 年度合成，1 个时间步 | 2019 年度合成，1 个时间步 |
| 分辨率/网格 | 10 m（SPOT 像元中心采样网格） | 10 m UTM 30/31/32 |
| 预训练输入 | **多模态**：Sentinel-2、Landsat 8/9、Sentinel-1、GEDI、GLO-30 DEM、ERA5-Land、PALSAR-2、GRACE、文本（[Earth Engine 目录](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL)） | Sentinel-1 + Sentinel-2 + Landsat |

**注意**：两者都是“把一年影像压成一张 64 维图”的产品；差异包含传感器组合、
时序聚合方式、模型代际，不能归因到单一因素。

## 4. 训练协议（与 AEF 完全一致）

| 项 | 设置 |
|---|---|
| 头 | `aef_worker.heads.MLPHead`（Conv1×1 64→128 + GroupNorm + GELU + Conv1×1）<br>`aef_worker.heads.UPerNetHead`（width 128，PPM (1,2,3,6)，1/2/4/8 金字塔） |
| 学习率 | MLP 3e-3；UPerNet 1e-3；AdamW，weight decay 1e-4 |
| batch | 8（有效 batch 与 AEF 一致） |
| epoch | **固定 8 轮，取第 8 轮权重直接评测**；不做 validation 选模 |
| seed | 17 / 42 / 73 |
| 特征标准化 | 训练集 tile-balanced 80 patch 统计 mean/std，逐 run 计算 |
| 无效像元 | 主口径 A：特征置零，标签全范围参与 loss/mIoU（`zero-fill-full-label`） |
| 设备 | CPU（与 AEF 的现有数字同设备；不是资源限制） |
| 头参数量 | MLP 11,156；UPerNet 3,770,644（与 AEF 完全相同的实现） |

8 epoch 是 T11 预声明的正式预算（probe 设计文档 2026-11-03 第 2 节），
目的是与 AEF 已有结果可比，不是认为 8 是最优预算。预算敏感性见第 7 节。

## 5. 轨道与准入

- 轨道：`BOE`（UPerNet）、`BOEM`（MLP），与 AEF 的 `BA`/`BAM` 分开。
- 正式锚点：`OFFICIAL_EPOCHS["T11O"] = 8`、test 496、validation 482；
  worker 在样本数不符时拒绝写正式指标。
- `scripts/audit_runs.py`：T11O 6/6 满足正式预算，0 条不一致。
- 不进入 Prithvi-suite D1；不与 T11 的 BBF/BBT（原始 S2 全量微调）混排。
- 同一 seed 的 8/20 epoch 结果不合并：20 epoch 是独立协议、独立轨道。

## 6. 64 维正式结果

| 输入 | 头 | s17 | s42 | s73 | mean ± sd |
|---|---|---:|---:|---:|---:|
| AEF | MLP | 0.4760 | 0.4921 | 0.4866 | 0.4849 ± 0.0081 |
| AEF | UPerNet | 0.4777 | 0.4886 | 0.4974 | 0.4879 ± 0.0099 |
| OlmoEarth v1.3 | MLP | 0.5664 | 0.5846 | 0.5854 | 0.5788 ± 0.0107 |
| OlmoEarth v1.3 | UPerNet | 0.5794 | 0.5697 | 0.5907 | 0.5799 ± 0.0105 |

- 配对差值：MLP +0.0904 / +0.0925 / +0.0989（均值 +0.0939）；
  UPerNet +0.1018 / +0.0811 / +0.0933（均值 +0.0921）。
- 耗时：MLP 2.8–3.3 min/run；UPerNet 28.7–37.4 min/run。
- AEF canonical run 全部在 `cf092238…`（s17-v4 / s42-v2 / s73-v4）；
  旧 `29717adc…` 的 6 条已 superseded。

## 7. 预算敏感性诊断（20 epoch，单种子）

观察到 8 轮结束时训练 loss 仍在下降（MLP 0.4569、UPerNet 0.4493），因此增加一个
**单种子（s17）20 epoch** 诊断，独立协议与轨道（`BOE20`/`BOEM20`），不替换
8-epoch 主 cell，也不进入 headline。

- MLP 20ep：旧值 0.5961 作废（旧特征 + 未标诊断）；用 `-v3` 重跑，结果出来前
  **不能**再声称“8 epoch 偏短”，这句结论待 20ep v3 验证。
- UPerNet 20ep：旧 run 因特征在原目录被覆盖、前后半程数据不一致，已隔离并作废；
  用 `-v3` 重跑，结论同样待验证。

解读：**当前 8 epoch 对 OlmoEarth 而言确实偏短，MLP 还有明显提升空间。**
这带来一个口径问题：如果要比较“产品 + 预算”，AEF 也必须在相同预算下重跑
（同 seed、同头、20 epoch），否则不能把 OlmoEarth 的优势全部归因于产品。
是否补 AEF 20ep 取决于 UPerNet 的结果和评审意见（约 75 分钟/seed）。

## 8. 128 维计划

- OlmoEarth 的 64 维是 128 维的 Matryoshka 前缀；128 维单独建 task
  `T11O128` 与独立轨道，不并入 64 维 cell。
- 本地镜像的 shard 对象已包含 128 维，**无需再下载**；128 维特征已按新采样规则
  重提完成（2,433 patch，4.9 GB，用时 13.2 s）。
- 训练设置与 64 维完全一致（8 epoch、seed 17/42/73、两种头），预计 1.5–2 h。
- 预期：若 128 维继续提升，说明 64 维截断了有用信息；若持平，说明 64 维已够用。
  两种结果都会写进报告。

## 9. 数据获取与可复现性

- shard 集合：`scripts/olmoearth_shard_list.py` 按 patch bbox 精确计算，
  2019 年共 **135 个 shard**（59.86 GB）；脚本可复现该集合。
- 下载：2 MiB Range + 高并发 + 段内断点续传（源站单连接只有几 KB/s），
  下载后校验大小并原子拼接。
- 提取：`scripts/olmoearth_extract_pastis.py --zarr-root ...`；本地读取与
  2026-11-06 的远程提取结果逐元素一致。
- 完整流程与命令见 `docs/evaluation/OlmoEarth_v1.3_镜像下载与提取_2026-11-07.md`。

## 10. 解读边界与不足

1. **产品对比，不是骨干排名**：没有在 T11 上微调 OlmoEarth v1.3 Base；
   T11 的 BB 轨是原始 S2 时序全量微调，输入和协议都不同。
2. **单数据集、单年份、单国家**：PASTIS-HD 法国农田；不能外推到其他区域/传感器。
3. **两边都是多模态**：AEF 的官方输入包括 Sentinel-2、Landsat 8/9、Sentinel-1、
   GEDI、DEM、ERA5-Land、PALSAR-2、GRACE 与文本；OlmoEarth 使用 S1+S2+Landsat。
   本地记录的 115,222 景只是 AEF 的 S2 部分，不能当作完整输入清单。差值不能归因到
   单一传感器或模型代际。
4. **对齐**：旧提取器的列方向系统性 +1 px 已修复并重跑；修复后构造上为最近像元中心，
   不再有平移修正问题。
5. **类别级指标缺失**：当前 worker 未持久化 per-group 混淆矩阵，无法给
   bootstrap 置信区间；弱类（Grapevine/Beet/Orchard）接近 0，需要单独出表。
6. **LPIS 标签年**：按 2019 处理是技术验证决定，仍需数据发布说明确认。
7. **预算不对等风险**：20ep 诊断的旧值（0.5961）已作废，需用 `-v3` 重跑；在 AEF 也未跑 20ep 之前，“产品+预算”的结论不成立。

## 11. 复现与记录

```bash
# 注册与准入（已提交）
uv run rs-eval list-tasks | grep T11O
uv run python scripts/audit_runs.py /geodata/rs-eval/runs .

# 固定 manifest 跑 6 个正式 run
export RS_EVAL_MANIFEST=/geodata/rs-eval/data/manifests/T11O/ef75efa9d93a5dca7cc71773b1d43d008a35d4453b86e571e50f71b5595b5de4.jsonl
RS_EVAL_T11O_SUFFIX=v3 bash scripts/rerun_baselines.sh --detached T11O

# 20 epoch 诊断（seed 17）
RS_EVAL_T11O20_SUFFIX=v3 bash scripts/rerun_baselines.sh --detached T11O20
```

| 项 | 值 |
|---|---|
| 正式 run id | T11O `…-train-{17,42,73}-v3`；AEF `…-train-{17,73}-v4` + s42-v2（v1/v2/v3 旧记录已作废） |
| 代码 | 采样修复 `2483162`；panel/准入与本轮修复见后续提交 |
| 训练设备 | 4090 主机的 CPU（32 cores） |
| 数据 | 本地镜像 59.86 GB（仓库外） |

## 12. 请评审确认的点

1. 64 维、8 epoch、固定第 8 轮的主口径是否认可；
2. 是否补 AEF 20ep（以及是否扩到 3 seed），把“产品差异”和“预算差异”分开；
3. 128 维是否现在就跑，还是等 64 维结论定稿；
4. 是否需要类别级 IoU / 混淆矩阵（需要给 worker 加持久化并重跑两边）；
5. 是否要在第二个数据集（如 PLANTEUR 或其他作物 benchmark）验证外推。
