# T12：FLAIR-HUB 作物分割（AEF / OlmoEarth embedding 探针）实验设计

- 日期：2026-11-09（第 2 版，按评审修正 5 处）
- 状态：**设计稿**，未注册 task、未下载数据、未改 `configs/`
- 目的：给 T11O 的"年度 embedding 产品对照"补一个**跨区域（整域留出）**的外部验证，
  回答"OlmoEarth 领先 AEF 约 +0.09 是否只在 PASTIS-HD 的 2,433 个 patch 上成立"
- 关联：`docs/evaluation/PASTIS-HD_模型成绩与复现指南.md`、
  `docs/evaluation/AEF_PASTIS-HD多时相对齐与FLAIR-HUB评估_2026-11-03.md`、
  `docs/evaluation/T11O_OlmoEarth_v1.3_实验设置_2026-11-07.md`、
  `docs/reviews/审查响应_2026-11-09_T12-FLAIR-HUB_第1轮.md`

## 0. 审批要点（一页）

| 项 | 数量 | 说明 |
|---|---|---|
| FLAIR 下载 | **9.6 GB**（2019 年 S2 时序 9.29 + LPIS 标签 0.26） | 整包 726 GB 的 1.3% |
| embedding 访问 | **按窗口流式读取约 2–3 GB**（本地 npz 缓存 1–2 GB） | 不做整 shard 镜像（见 §2.4） |
| 工作盘 | **40 GB**（`/home`；`/geodata` 只剩 32 GB） | 解压后 20–25 GB + 缓存 2 GB + 余量 |
| 人力 | 4–5 天（主要是 0.2 m→10 m 标签聚合与逐 ROI 校验） | 训练约 2 h CPU |
| GPU | **不需要** | MLP 探针，CPU 6–10 min/run |

**不下载什么（占全量 86%）**：0.2 m 航拍 241 GB、DEM 383 GB——只有 FLAIR 官方
0.2 m 评测才需要。S1（7 GB/年）与 SPOT（6.4 GB）本次也不需要。

**批准后先做 M0/M1（<1 GB）**：M0 用元数据核对覆盖面，M1 跑通单个 ROI 的
0.2 m→10 m 标签、embedding 对齐与前向；通过后再下载 2019 的 9.6 GB。

**成功标准**：官方 test（2019 年是 3 个整域留出）上 AEF 与 OlmoEarth 的配对差值
是否与 PASTIS 同方向。一致 → 产品差距可跨区域复现；不一致 → 说明 +0.09 有区域
特异性，本身也是要报的结果。

## 1. 为什么做这个任务

T11/T11O 的全部数字都是**同分布留出**（PASTIS-HD 官方 Fold 1–3 / 4 / 5），仓库里唯一
独立的泛化证据是 T1 的 Bolivia holdout。

FLAIR-HUB 的价值：

- **官方 split 的 test 是整域留出**（2019：D012/D076/D085，与 train 的 13 个 domain
  完全不重叠，见 §2.2），所以它天然是跨区域验证；
- **同样的输入对象**：AEF 与 OlmoEarth v1.3 都是全球产品，法国在覆盖范围内，
  不需要任何模型权重；
- 成本可控：FLAIR 数据 9.6 GB + embedding 流式读取。

它**不**回答"谁在 FLAIR 榜单上更高"：我们在 10 m 网格评测，官方在 0.2 m。

## 2. 数据事实（2026-11-09 实测）

### 2.1 规模与模态（IGNF/FLAIR-HUB，etalab-2.0）

| 项 | 值 |
|---|---|
| 总量 | 约 726 GB、885 个 zip（HF 仓库实测） |
| 空间 | 66 个 domain（D004–D091，不连续）、2,822 个 ROI、2,528 km² |
| 基本单元 | 241,100 个 512×512 patch；**0.2 m 下约 102.4 m 见方** |
| 标签 | LPIS 作物（3 级：23/31/46 类）与 CoSIA 土地覆盖（19 类） |
| 输入 | 0.2 m 航拍 RGBI、1.6 m SPOT、10.24 m S2 时序、S1 ASC/DESC、DEM |

**关键几何事实**：一个 512×512 patch 只有 102 m 见方，在 10 m embedding 网格上只有
约 10×10 像素。**样本单位必须是 ROI（约 100 个 patch，≈1 km²，10 m 上约 100×100）。**

### 2.2 官方划分（从 `GLOBAL_ALL_MTD_SPLIT.gpkg` 的 `split_1` 列直读，2026-11-09）

| 年份 | train patch / ROI / domain | valid patch / ROI / domain | **test patch / ROI / domain** |
|---|---|---|---|
| 2019 | 40,200 / 371 / 13 | 8,775 / 87 / 13 | **9,375 / 127 / 3**（D012, D076, D085） |
| 2020 | 50,450 / 604 / 18 | 11,150 / 132 / 18 | 26,875 / 218 / 7 |
| 2021 | 50,350 / 642 / 20 | 14,975 / 178 / 20 | 13,200 / 193 / 5 |

- 全量：train 152,225 / valid 38,175 / test 50,700，与 IGNF/MAESTRO 发布的
  `train.csv` / `val.csv` / `test.csv` **逐条一致**（已核对行数）。
- **test 在每个年份都是整域留出**：2019 的 3 个 test domain 不出现在该年 train/valid
  的任何 patch 里；ROI 在三份划分之间零重叠。
- 第 1 版设计里"D063/D072 仅在 val、可作跨域留出"是**错的**：这两个 domain
  同时出现在 train 和 val；官方 split 已经有跨域 test，不需要自己再划。

### 2.3 FLAIR 下载体积（zip 大小实测）

| 年份 | S2 时序 | LPIS | S1 ASC | S1 DESC | S2 云掩膜 |
|---|---:|---:|---:|---:|---:|
| 2019（16 个 domain-year） | 9.29 GB | 0.26 GB | 3.09 GB | 4.04 GB | 0.26 GB |
| 2020（25） | 14.61 GB | 0.34 GB | 6.48 GB | 6.34 GB | 0.42 GB |
| 2021（25） | 12.12 GB | 0.27 GB | 5.33 GB | 5.75 GB | 0.38 GB |

全量 726 GB 构成：航拍 241 GB、DEM 383 GB、S2 38.6 GB（三年）、S1 32.8 GB、
SPOT 6.4 GB、CoSIA 1.8 GB、LPIS 0.9 GB。

```bash
huggingface-cli download IGNF/FLAIR-HUB --repo-type dataset \\
  --include "data/D*-2019_SENTINEL2_TS.zip" "data/D*-2019_ALL_LABEL-LPIS.zip" \\
  --local-dir /home/yzhong/rs-eval/data-flair
```

### 2.4 embedding 获取（第 1 版漏算，这里是实测估算）

FLAIR 的 ROI 分散在全法，按存储单元整块下载非常浪费：

| 产品 | 存储单元 | 2019 触及的单元数 | 整块下载 |
|---|---|---:|---:|
| OlmoEarth v1.3 | shard `(1,128,2048,2048)` = 20.48 km 见方，537 MB | **279 个** | **146 GB** |
| AEF | COG tile 8192×8192 px = 81.92 km 见方，约 2.65 GB | **54 个** | **143 GB** |

（按 2019 全部 58,350 个 patch 的 bbox 换算到 UTM 30/31/32 后统计；PASTIS 当年是
135 个 shard / 59.5 GB。）

**结论：不做整块镜像。** 只按 ROI 窗口做 range 读取，缺什么读什么：

- 单个 ROI：AEF 128×128×64 = 1.0 MB；OlmoEarth 128×128×128 = 2.1 MB；
- 585 个 ROI（2019 三份划分）：约 **1.8 GB**，加 COG/zarr 分块开销按 2–3 GB 估；
- 读取速度按实测代理 2.4–2.6 MB/s（640 并发）→ **20–30 分钟**一次全量提取；
- 提取结果落 npz 缓存（约 1–2 GB），训练不再重复读远端。

M0 会先用小样本验证 range 读取对 sharded zarr 与 COG 都可行；若不可行，退回到
"按 domain 分批镜像"（单个 domain 约 5–15 GB），并在报告里记实际用量。

## 3. 任务定义（拟注册为 T12）

| 字段 | 值 |
|---|---|
| task_id | `T12` |
| dataset_id | FLAIR-HUB（LPIS 作物，level-1） |
| application | 农作物品类语义分割（FLAIR-HUB，年度 embedding 跨区域验证） |
| input_kind | `embedding_features`（AEF / OlmoEarth v1.3） |
| sensor | `aef` / `olmoearth_v1_3_embedding` |
| num_outputs | **23**（LPIS level-1，0–22） |
| ignore_index | **无**（23 类全部参与 loss；见 §3.2） |
| background_class | 22 |
| primary_metric | mean_iou（GT 中出现的类做 macro） |
| 锚点 | 2019：train 371 ROI / valid 87 / test 127 |

### 3.1 标签对齐（共同 CRS、原点、像元边界）

1. **共同网格**：直接用 AEF/OlmoEarth 的 10 m UTM 网格。仿射变换取 AEF sidecar 的
   `window_transform`（`a=10, e=-10`，原点是像元**角点**），与 T11O 的
   `grid_bbox` 完全同一个定义；像元索引用 `floor((x-c)/a)`（最近像元中心）。
2. **ROI 拼接**：每个 patch 有 `patch_xmin/xmax/ymin/ymax`（EPSG:2154，来自
   `GLOBAL_ALL_MTD_GEOM.gpkg`）；按 bbox 把 patch 拼成 0.2 m 连续栅格，CRS 统一到
   EPSG:2154，像元边界由 bbox 决定（不做任何整数缩放）。
3. **重投影**：把 0.2 m 的 LPIS 第 1 通道（`label_channel_nomenclature: 1`，
   23 类）用 `rasterio.warp.reproject(..., dst_transform=<10 m 仿射>,
   resampling=Resampling.mode)` 投到上述网格——**不是**在 0.2 m 网格上做固定
   50×50 分块再贴上去；mode 重采样等价于"落在该 10 m 像元内的 0.2 m 标签取众数"。
4. **有效掩膜**：另建 0/1 有效栅格（未标注/ignore=0），用 `Resampling.average`
   投到 10 m；有效比例 < 0.5 的像元记 ignore（不参与 loss 与 mIoU）。
5. **窗口**：ROI 的 10 m 外包框扩到 128×128（居中补 NODATA，写 `valid`），
   标签覆盖率 < 30% 的窗口丢弃，数量写进 manifest。
6. **验收（两轮评审后重写）**。两个教训：10 m 上的众数标签**不保面积**（不能用类别
   占比当门槛）；而"众数标签 vs 面积分数"的平移剖面**共享同一条坐标链路**，一起错位
   时最优点仍在 0（评审合成验证：两者共同偏移 20 m，最优平移仍为零）。所以：

   本机合成验证（0.2 m、500×500、完全对齐、零错位）：

   | 情形 | 0.2 m 占比 | 10 m `mode` | `average`（面积分数） |
   |---|---|---|---|
   | 类 A 每 5 列占 2 列 | 40% | **0%** | 40% |
   | 10 m 宽道路（每 10 列 1 列） | 10% | **0%** | 10% |

   - **A. 绝对几何（唯一门槛，必须用独立参考）**：把 0.2 m 标签与 ROI 投到 10 m 后，
     与 **Sentinel-2 L2A 2019 影像**做相位相关（LPIS 田块边界 ↔ 影像边界，两者是独立
     来源），要求 **|偏移| ≤ 1 像元（10 m）且峰值在 0 位移**；逐 ROI 报偏移、可用窗口
     比例与失败率。另抽 3 个 ROI 手工量 2–3 个控制点（道路交叉口）作为绝对锚点，
     报米级偏移。**标签链路整体偏移（例如仿射原点错半个像元）只有这一步能发现。**
   - **B. 内部一致性（不是门槛，是回归检查）**：10 m 众数标签与"0.2 m 面积分数"
     （one-hot 走 `Resampling.average` 的覆盖率向量）的分布距离随 −3…+3 像元平移的
     曲线，最小值应在 0。两者来自**同一条链路**，共同错位时它照样通过，所以它只能
     抓"两次重采样之间不一致"的实现 bug，**不能**当作对齐验收。
   - **C. bbox 角点（实现自检，不是门槛）**：角点落格误差 ≤0.5 像元。它同样只用我们
     自己的 gpkg + 网格定义；坐标约定整体偏了它也会通过，保留只为抓取整/取整错误。
   - **D. 类别保真度（报告项，不是门槛）**：逐类比"10 m 众数保留的像元数"与"由面积
     分数求出的期望像元数"；把被抹掉的细长类（宽度 <10 m 的道路、沟渠、田埂）列清单，
     这些类在 10 m GT 里不再出现、自然不进 §3.2 的 macro，但**必须列出来**，否则会被
     误读成模型在这些类上得 0；原因写"10 m 分辨率不表达"，不写"对齐失败"。
   - **E. 人工核对**：抽 5 个 ROI，把 10 m 标签叠在 **S2 影像**上目视检查田块边界
     是否一致——**不是**叠回我们自己拼的 0.2 m 栅格（同一链路，错了也看不出来）。

### 3.2 类别与忽略项（按官方配置修正）

官方 `configs/train/config_supervision.yaml` 的 `ALL_LABEL-LPIS` 段：

```yaml
value_weights:
    default: 1
    default_exceptions:        # 空：没有任何类被降权或忽略
```

所以 **21（mixed crops）与 22（background）都是真实类别，都参与训练与 mIoU**。
第 1 版把 21 当 ignore 是错的，会改变训练目标和类集。

- 主口径：23 类 macro mIoU（GT 中出现类平均）。
- 辅助口径：22 类（去掉 background）单列，便于和只报作物类的文献对照。
- 注意：FLAIR 论文正文有一句"mixed 类被排除出 loss"，与当前发布的配置不一致；
  我们**以配置为准**，并在报告里注明这处版本差异。

### 3.3 年份

- **主口径：2019。** 单年样本最大（371/87/127 ROI），AEF 与 OlmoEarth 在法国本土的
  2019 覆盖已在 PASTIS 上验证，FLAIR 只需 9.6 GB。
- **不使用"跨年份"作为本次结论**：ROI 不跨年重复（实测 0 个），只有 4 个 domain 有
  两个年份；因此 2019 只能验证**跨区域**，不能验证跨年。
- **可选第二年份：2020**（+14.9 GB FLAIR + 流式 embedding），它是另一组整域留出的
  跨区域测试，不是同一批 ROI 的跨年对照；只有主口径显著时才做，且单列。

## 4. 划分（只用官方 split，不再自造 panel）

| 划分 | 2019 | 用途 |
|---|---|---|
| train | 371 ROI / 13 domain | 训练 |
| valid | 87 ROI / 13 domain | 仅诊断（不选模） |
| test | **127 ROI / 3 个整域（D012, D076, D085）** | 主结论：跨区域 |

- 三份之间 ROI 零重叠、test 与 train 的 domain 零重叠（已逐条核对）。
- 训练固定取第 8 轮（与 T11O 同口径），不做 validation 选模。
- 第 1 版设想的"Panel A 同分布 / Panel B 自选跨域"取消：官方 test 本身就是跨域，
  再划一套只会多一套不可比的数字。

## 5. 训练与评测协议

| 项 | 值 |
|---|---|
| 头 | **MLP**（2026-11-09 决定：embedding 探针默认只用 MLP） |
| 学习率 | 3e-3，AdamW，weight decay 1e-4 |
| epoch | 8，固定第 8 轮 |
| seed | 17 / 42 / 73 |
| 设备 | CPU（与 T11O 同口径，6–10 min/run） |
| 指标 | macro mIoU（23 类 / 22 类两版）、每类 IoU、背景类单列 |

**账本标签**（按准入规则检查过 `matrix.build_matrix`：只接受
`tags.panel == "primary"`，所以不能把 panel 写成 A/B，否则两组都不入矩阵）：

- `tags.panel` 留空（默认 primary）；
- `tags.t12_year = "2019"`、`tags.t12_split = "official_split_1"` 只作检索；
- 年份/划分信息同时写进 manifest 文件名与 `notes`，报告以 manifest 为准。

## 6. 准入与账本

- 新轨道 `BFA`（AEF）/`BFO`（OlmoEarth 64）/`BFO128`（OlmoEarth 128），与 T11/T11O
  分开，**不并入 Prithvi-suite D1**。
- 记录：manifest sha256、年份、domain 列表、类别映射、标签聚合规则、`payload.ok`。
- 审计：新增 T12 的官方 epoch（8）与 test 样本锚点（2019 年 127 个 ROI）。

## 7. 成本

| 阶段 | 工作 | 估计 |
|---|---|---|
| 下载 | 2019 S2 TS + LPIS | 9.6 GB，1–2 h |
| 元数据/覆盖核查（M0） | GEOM + SPLIT gpkg（已实测 2 GB 包内 138 MB） | 2 h |
| 标签聚合（M1） | 0.2 m→10 m，逐 ROI 校验 | **2–3 天**（主要工作量） |
| 特征提取 | AEF/OlmoEarth 按窗口流式读取 + npz 缓存 | 2–3 GB，20–30 min |
| 训练 | 3 model × 3 seed | 每 run 6–10 min，合计约 1.5 h |
| 合计 | | **4–5 天 + 半个 CPU 天** |

## 8. 风险与限制

1. **10 m 标签降采样改变任务**：官方分数在 0.2 m；0.2 m 的细碎类别在 10 m 上会被
   mode 重采样吞掉。数字不能进 FLAIR 榜单，只用于 AEF/OlmoEarth 的同口径对照。
2. **背景类 22 与 mixed 21 都是真实类**（官方配置），会把"未申报地块"吸收进
   background；主表报 23 类，另报 22 类版本。
3. **LPIS 是申报数据**，葡萄园等常缺报；与 PASTIS 同样的问题，跨数据集比较要承认。
4. **跨区域样本量小**：2019 test 只有 127 个 ROI / 3 个 domain；差值不显著时只能报
   方向，不能报显著性。
5. **OlmoEarth 年份/区域覆盖**：v1.3 的全球年度 embedding 是 2023–2025，2017–2022
   是 "select countries"；M0 必须按 UTM tile 核对，海外 domain 可能没有覆盖，
   被排除的 ROI 要计数并单独说明。
6. **流式读取的可行性**：sharded zarr 与 COG 的 range 读取需要在 M0 验证；慢的话
   退回到按 domain 分批镜像（单个 domain 5–15 GB）。
7. **本设计不验证跨年份**（ROI 不跨年重复）；跨年验证需要另设任务。

## 9. 里程碑

| 阶段 | 交付 | 通过标准 |
|---|---|---|
| M0 覆盖核查 | 2019 各 domain 的 UTM tile、AEF/OlmoEarth 覆盖、流式读取可行性 | 本土 ROI 覆盖率 ≥ 90%，单 ROI 读取 < 5 s |
| M1 单 ROI 端到端 | 0.2 m→10 m 标签、embedding 对齐、前向 | 与 S2 2019 互相关 ≤1 像元（唯一门槛）+ 3 个 ROI 手工控制点；内部一致性曲线最小值在 0；5 个 ROI 叠 S2 目视通过；细长类丢失清单 |
| M2 全量 | 371/87/127 ROI 的 AEF vs OlmoEarth 64/128 | 3 seed 完成，审计通过 |
| M3 可选 | 2020 年同协议 | 与 2019 分开报告 |

## 10. 待确认

1. 主口径是否接受"官方 split + 10 m 标签网格"这套新协议（与 FLAIR 榜单不可比）。
2. 是否加 2020 年（+14.9 GB）——我建议先看 2019 的差值是否显著。
3. 128 维是否一起跑（建议跑：跨域下 128 是否仍领先是次要结论）。
4. 是否需要在 M1 之后先给一版 5 个 ROI 的标签可视化，再决定全量。
