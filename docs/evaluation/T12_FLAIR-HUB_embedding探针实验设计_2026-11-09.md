# T12：FLAIR-HUB 作物分割（AEF / OlmoEarth embedding 探针）实验设计

- 日期：2026-11-09
- 状态：**设计稿**，未注册 task、未下载数据、未改 `configs/`
- 目的：给 T11O 的"年度 embedding 产品对照"补一个**跨区域、跨年份**的外部验证，
  回答"OlmoEarth 领先 AEF 约 +0.09 是否只在 PASTIS-HD 的 2,433 个 patch 上成立"
- 关联：`docs/evaluation/PASTIS-HD_模型成绩与复现指南.md`、
  `docs/evaluation/AEF_PASTIS-HD多时相对齐与FLAIR-HUB评估_2026-11-03.md`、
  `docs/evaluation/T11O_OlmoEarth_v1.3_实验设置_2026-11-07.md`

## 0. 审批要点（一页）

**申请什么**

| 项 | 数量 | 说明 |
|---|---|---|
| 下载 | **9.6 GB**（2019 年 S2 时序 9.29 + LPIS 标签 0.26） | 仅整包 726 GB 的 **1.3%** |
| 工作盘 | **40 GB**（`/home`，`/geodata` 只剩 32 GB） | 解压后 20–25 GB + 特征/标签 1–3 GB + 余量 |
| 人力 | 4–5 天（主要是 0.2 m→10 m 标签聚合） | 训练只需约 2 h CPU |
| GPU | **不需要** | MLP 探针，CPU 6–10 min/run |

**不下载什么（占全量 86%）**：0.2 m 航拍 241 GB、DEM 383 GB。这两个只有做 FLAIR
官方 0.2 m 评测才需要；我们在 10 m embedding 网格上评测，用不到。S1（7 GB）和
SPOT（6 GB）在本次设计里也不是必需，留作后续"1 m 输入值多少"的扩展。

**批准后第一步不是全量下载**：先做 M0（2019 各 domain 的 UTM tile 与 AEF/OlmEarth
覆盖核查）和 M1（单个 ROI 端到端）。两者只需要 1 个 domain 的文件（<1 GB），通过
后再拉剩下的 8.6 GB。

**成功标准**：Panel B（跨域留出 2 个 domain）上 AEF 与 OlmoEarth 的配对差值与
PASTIS 的方向是否一致；一致 → "产品差距可跨区域复现"；不一致 → 说明 PASTIS 的
+0.09 有区域特异性，这本身也是要报的结果。

## 1. 为什么做这个任务

现在的证据结构有一个洞：T11/T11O 的全部数字都是**同分布留出**（PASTIS-HD 官方
Fold 1–3 / 4 / 5），而且 T1/T2/T5 的官方 split 本身还有 patch 级重叠。仓库里唯一
独立的泛化证据是 T1 的 Bolivia holdout。

FLAIR-HUB 能补上的是：

- **跨区域**：法国本土 + 海外省（66 个 domain，2,528 km²），不是同一批农田；
- **跨年份**：2017–2022，每个 ROI 的标签年与该年航拍一致；
- **同样的输入对象**：AEF 与 OlmoEarth v1.3 都是全球产品，法国在覆盖范围内，
  不需要任何模型权重；
- **成本可控**：只需要 S2 时序和 LPIS 标签，2019 年合计约 **9.6 GB**（见 §3.3）。

它**不**用来回答"谁在 FLAIR 榜单上更高"：我们的评测在 10 m 网格上，官方在 0.2 m，
两者不可比。

## 2. 数据事实（2026-11-09 实测）

### 2.1 规模与模态（IGNF/FLAIR-HUB，etalab-2.0）

| 项 | 值 |
|---|---|
| 总量 | 约 726 GB、885 个 zip（HF 仓库实测） |
| 空间 | 66 个 domain（D004–D091，不连续）、2,822 个 ROI、2,528 km² |
| 基本单元 | 241,100 个 512×512 patch；**0.2 m 下约 102.4 m 见方** |
| 标签 | LPIS 作物（3 级：23/31/46 类，背景 22，mixed 21 需忽略）；CoSIA 土地覆盖 19 类 |
| 输入 | 0.2 m 航拍 RGBI、1.6 m SPOT、10.24 m S2 时序、S1 ASC/DESC 时序、DEM |

**关键几何事实**：一个 512×512 patch 只有 102 m 见方，在 10 m embedding 网格上只有
约 10×10 像素。所以**样本单位必须是 ROI（约 100 个 patch，≈1 km²），不是单 patch**。
一个 ROI 在 10 m 上约 100×100 像素，与 PASTIS 的 128×128 同量级。

### 2.2 年份与划分（来自 IGNF/MAESTRO_FLAIR-HUB_base 公开 CSV，实测）

| 年份 | train patch | val patch |
|---|---:|---:|
| 2017 | 100 | 0 |
| 2018 | 9,125 | 2,475 |
| **2019** | **40,200** | **8,775** |
| 2020 | 50,450 | 11,150 |
| 2021 | 50,350 | 14,975 |
| 2022 | 2,000 | 800 |

2019 子集：**371 个 train ROI / 87 个 val ROI**，patch/ROI 中位数 100（25–400），
train 与 val 的 ROI **零重叠**；13 个 domain 里 train/val **共享全部 domain**（ROI 级
划分，同域不同块），所以它只能算同分布留出，不能算跨域。

→ 跨域结论需要我们自己再划一刀（§4 Panel B）。

### 2.3 下载体积（按模态实测，zip 大小）

| 年份 | S2 时序 | LPIS 标签 | S1 ASC | S1 DESC | S2 云掩膜 |
|---|---:|---:|---:|---:|---:|
| 2019（16 个 domain-year） | 9.29 GB | 0.26 GB | 3.09 GB | 4.04 GB | 0.26 GB |
| 2020（25） | 14.61 GB | 0.34 GB | 6.48 GB | 6.34 GB | 0.42 GB |
| 2021（25） | 12.12 GB | 0.27 GB | 5.33 GB | 5.75 GB | 0.38 GB |

全量 726 GB 的构成（实测）：**0.2 m 航拍 241 GB、DEM 383 GB**，这两个我们不用；
S2 时序 38.6 GB（三年）、S1 32.8 GB、SPOT 6.4 GB、CoSIA 标签 1.8 GB、LPIS 0.9 GB。

| 口径 | 下载 | 解压后（估 2–3×） | 特征+标签 | 建议预留 |
|---|---:|---:|---:|---:|
| **2019 S2+LPIS（本设计）** | **9.6 GB** | 20–25 GB | 1–3 GB | **40 GB** |
| 2019 加 S1 | 16.7 GB | 35–40 GB | 3 GB | 60 GB |
| 2019–2021 S2+LPIS | 36.9 GB | 80–100 GB | 4–9 GB | 120 GB |

下载命令（只取 2019 的两个模态；HF 分片文件名是
`data/<domain>-<year>_<MODALITY>.zip`）：

```bash
huggingface-cli download IGNF/FLAIR-HUB --repo-type dataset \
  --include "data/D*-2019_SENTINEL2_TS.zip" "data/D*-2019_ALL_LABEL-LPIS.zip" \
  --local-dir /home/yzhong/rs-eval/data-flair
```

**磁盘布局**：原始 zip 与解压文件放 `/home/yzhong/rs-eval/data-flair/`（`/geodata`
只剩 32 GB，放不下 40 GB 工作集）；最终 10 m 的 npz 与 manifest 放
`/geodata/rs-eval/data/derived/flair-hub/`，与现有 `olmoearth-2019-128` 的软链接
做法一致。

## 3. 任务定义（拟注册为 T12）

| 字段 | 值 |
|---|---|
| task_id | `T12` |
| dataset_id | FLAIR-HUB（LPIS 作物） |
| application | 农作物品类语义分割（FLAIR-HUB，年度 embedding 跨区域验证） |
| input_kind | `embedding_features`（AEF / OlmoEarth v1.3） |
| sensor | `aef` / `olmoearth_v1_3_embedding`（同一 task 两个 model） |
| num_outputs | 23（LPIS level-1：0–20 作物，21 mixed，22 background） |
| ignore_index | 21（mixed，官方也从 loss 中排除） |
| background_class | 22 |
| primary_metric | mean_iou（GT 中出现的类做 macro；void/ignore 不计） |
| 官方锚点 | 2019 子集：train 371 ROI / val 87 ROI / test 见 §4 |

### 3.1 样本构造

1. 以 ROI 为单位，把该 ROI 的 512×512 patch 按坐标拼成连续栅格（0.2 m）。
2. 用 ROI 的几何把 AEF/OlmoEarth 的 10 m UTM 网格窗口切出来（128×128，越界补
   NODATA 并写进 `valid` 掩膜）。
3. 标签从 0.2 m 聚合到同一 10 m 网格：
   - 每个 10 m 像元对应 50×50 = 2,500 个 0.2 m 标签像元，取**面积多数**；
   - 该像元内有效标签面积 < 50% → 记为 ignore；
   - 实现上先做 50×50 分块计数再取众数，避免浮点误差。
4. 只保留标签覆盖率 ≥ 30% 的窗口；窗口数写进 manifest。

### 3.2 输入

| 型号 | 输入 | 维度 |
|---|---|---|
| AEF 2019 | 年度 embedding | 64 |
| OlmoEarth v1.3 2019 | 年度 embedding | 64 / 128（两个 task，不混排） |

预处理沿用 T11O：`sign(x)*(|x|/127.5)^2`，-128 = NODATA → 0，标签全范围参与 loss。

### 3.3 年份选择

- **主口径：2019。** 理由：单年样本最大（40,200/8,775 patch），AEF 与 OlmoEarth 在
  法国本土的 2019 覆盖已经在 PASTIS 上验证过，下载只需 9.6 GB。
- **扩展口径（可选）：2019–2021。** 样本约 3 倍，下载 +26 GB；只有当主口径的
  test 样本不足以给出可分辨的差值时才启用，且必须单列，不与 2019 混排。

## 4. 划分（两套 panel，必须分开报告）

### Panel A：同分布（in-domain）

- train = MAESTRO `train.csv` 的 2019 ROI（371）
- val/test = `val.csv` 的 87 个 ROI 按 domain 分层对半切（44/43）
- ROI 之间零重叠（已核对），但同 domain → **这是同分布留出，不是泛化证据**

### Panel B：跨域（cross-domain，headline）

- 从 2019 的 13 个 domain 里**整域留出 2 个**做 test（选择依据：地理位置/生态区
  差异最大，在 build 时固定并写进 manifest；候选 D063、D072，仅在 val 中出现）
- train/val = 其余 11 个 domain 的 ROI，按 80/20 划分
- 这一 panel 回答的是"换一个区域还成不成立"，是本次实验的主要结论

两个 panel 的 test 集合完全不同，禁止合并成一个平均。

## 5. 训练与评测协议

| 项 | 值 |
|---|---|
| 头 | **MLP**（2026-11-09 决定：embedding 探针默认只用 MLP；UPerNet 不再默认跑） |
| 学习率 | 3e-3，AdamW，weight decay 1e-4 |
| epoch | 8，固定取第 8 轮（不做 validation 选模） |
| seed | 17 / 42 / 73 |
| 设备 | CPU（与 T11O 同口径，MLP 约 6–10 min/run） |
| 指标 | macro mIoU（GT 中出现类），另报背景类与 21 个作物类的分项 |

## 6. 准入与账本

- 新轨道 `BFA`（AEF）/ `BFO`（OlmoEarth 64）/ `BFO128`（OlmoEarth 128），
  与 T11/T11O 分开，**不并入 Prithvi-suite D1**。
- 记录：manifest sha256、年份、domain 列表、类别映射表、标签聚合规则、
  `tags.panel`（A/B）、`tags.year`。
- 审计：新增 `T12` 的官方 epoch（8）与 test 样本锚点；与 T11O 一样要求
  `payload.ok`。

## 7. 成本

| 阶段 | 工作 | 估计 |
|---|---|---|
| 下载 | 2019 S2 TS + LPIS（+S1 可选） | **9.6 GB**，1–2 h |
| 标签聚合 | 0.2 m → 10 m，ROI 拼接 | **2–3 天**（主要工作量，需逐 ROI 验证） |
| 特征提取 | AEF/OlmoEarth 2019，371+87 个 ROI | 半天（复用现有提取器） |
| 训练 | 3 个 model × 2 panel × 3 seed | 每 run 6–10 min，合计约 2 h |
| 合计 | | **约 4–5 天 + 半个 GPU/CPU 天** |

## 8. 风险与限制（写进 task caveats）

1. **10 m 标签降采样改变了任务**：官方 FLAIR 分数在 0.2 m 上，我们的数字不能与
   leaderboard 比；0.2 m 的细碎类别（沟渠、田埂）在 10 m 上会被多数投票吞掉。
2. **背景类 22 的定义**：LPIS 只登记申报地块，葡萄园等常缺报，背景类会吸收它们；
   mIoU 里保留背景（与 FLAIR 官方 23 类口径一致），但必须单独报"不含背景"的版本。
3. **LPIS 是申报用途，不是真实作物**：与 PASTIS-HD 有同样的问题，跨数据集比较要
   承认这一点。
4. **OlmoEarth 年份覆盖**：v1.3 的全球年度 embedding 是 2023–2025，2017–2022 是
   "select countries"；法国已在 PASTIS 2019 上验证，但 FLAIR 的海外 domain 可能
   没有覆盖 → build 时按 UTM tile 过滤，只保留本土 30/31/32，被排除的 ROI 计数要
   记录。
5. **domain 级样本量小**：2019 只有 13 个 domain，Panel B 留出 2 个后，test 的
   ROI 数可能只有几十个；不显著时只能报方向，不能报显著性。
6. **标签年**：LPIS 年份跟随航拍年份（官方说明），所以 2019 patch 用 2019 LPIS 和
   2019 embedding 是对齐的；这一点比 PASTIS-HD 的"2019 假设"更干净。

## 9. 里程碑

| 阶段 | 交付 | 通过标准 |
|---|---|---|
| M0 覆盖核查 | 列出 2019 各 domain 的 UTM tile 与 AEF/OlmoEarth 覆盖 | 本土 ROI 覆盖率 ≥ 90% |
| M1 单 ROI 端到端 | 1 个 ROI 的 0.2 m → 10 m 标签、embedding 对齐、探针可跑 | 标签可视化与原图一致；`payload.ok` |
| M2 Panel A | 371/44/43 ROI 的 AEF vs OlmoEarth 64/128 | 3 seed 完成，审计通过 |
| M3 Panel B | 2 个留出 domain | 与 Panel A 分开报告，方向一致或明确说明不一致 |

## 10. 待确认

1. 是否采用 MAESTRO 的 CSV 作为划分来源（公开、可复现，但不是官方 `split_1`）；
   若要严格官方口径，需要先拿到 `split_1` 的 GeoPackage。
2. Panel B 的留出 domain 选择（D063/D072 只是候选，需要看地理分布再定）。
3. 是否加 S1（+7 GB）：AEF 与 OlmoEarth 的 embedding 本身已含 S1，加 S1 只对
   "原始时序"轨有意义，本设计不需要。
4. 是否同时跑 128 维（建议跑：跨域条件下 128 维是否仍然领先 64 维，是本次实验的
   次要结论）。
