# T11 PASTIS-HD 首轮 AEF probe（技术验证，2026-11-03）

状态：**技术验证**。标签年按 2019 处理（正式结论仍需 LPIS 发布文档确认，见 `AEF_PASTIS-HD多时相对齐与FLAIR-HUB评估_2026-11-03.md` §5）；
单 seed（17）、无调参扫描；T11 不并入 Prithvi-suite D1，不与文献分数直接比。
本文档与 probe 脚本（`scripts/t11_build_manifest.py`、`scripts/t11_aef_probe.py`）
同一次提交；审计 JSON 在 `/home/yzhong/rs-eval/audits/aef-20261103/`。

## 1. 数据与划分

- 数据根：`/home/yzhong/rs-eval/data-aef`（不在 `.env` 的 `data_root` 下；注册时需要
  `RS_EVAL_AEF_DATA_ROOT` 或软链，manifest 生成器显式使用该根）；
- PASTIS-HD 官方包已全部下载（20,543 / 20,543 文件）：
  S2 2,468、S1A/S1D 各 2,468（含 35 个**无标注** patch，官方 metadata 只有 2,433，
  manifest 只登记有标注的 patch）、SPOT 1 m 2,433、ANNOTATIONS 4866、
  INSTANCE_ANNOTATIONS 5,829、文档 3 件；共 61,649 文件 / 81 GB；
- AEF 2019 COG：17 个 tile，8192²×64 int8（zstd，nodata −128），10 m，45 GB；
- AEF 特征：2,433 / 2,433 个 patch 已提取（`features/2019/<pid>.npz`，float16
  64×128×128 + `valid`，4.9 GB，39 s）；跨 COG 边界 patch 用多 COG 拼接，
  `valid_share` 全部为 1.0（单 COG 版本曾有 58 个 patch 覆盖不全，已修复并重跑）；
- 总占用约 130 GB（`/geodata` 94% 已用，本数据全部在根盘）。

### 划分（事先声明，非调参产物）

官方 5-fold：Fold 1–3 → train 1,455；Fold 4 → validation 482；Fold 5 → test 496。
单 patch 是 group（patch 之间无共享地块），按 Fold 切分无组泄漏。
manifest：`/home/yzhong/rs-eval/data-aef/manifests/T11/pastis_hd_2019.jsonl`
（2,433 行，`scripts/t11_build_manifest.py` 生成）。

### 标签口径

- `TARGET_<pid>.npy` 形状 `(3,128,128)`：通道 0 = 语义（0 背景、1–18 作物、19 void），
  通道 1 = patch 内实例号，通道 2 = 逐实例属性（每 patch 18–107 个取值，README 未说明
  含义）；本轮只用通道 0，ignore 19；通道 1/2 不给语义比赛用。
- 18 类作物名来自 `documentation/label_names.json`（Meadow、Soft winter wheat、
  Corn、Winter barley、Winter rapeseed、Spring barley、Sunflower、Grapevine、
  Beet、Winter triticale、Winter durum wheat、Fruits/vegetables/flowers、Potatoes、
  Leguminous fodder、Soybeans、Orchard、Mixed cereal、Sorghum）。

## 2. 实验设计（头阶梯，固定预算）

冻结 AEF 特征（唯一 input），同一批数据与 budget 下比较不同头部容量；
目的回答"增益来自 AEF 特征还是来自头容量"，并确定正式轨道的头部候选。

- **budget 固定**：train 300（Fold 1–3 按 sample_id 排序取前 300）、val 100、
  test 150、8 epoch、batch 8、AdamW wd 1e-4、CPU、seed 17；
- **归一化**：`sign(x)·(x/127.5)²（x 为 int8 AEF 等级，−128 = nodata → valid=0 → 特征置 0）；
  逐通道 mean/std 只用 train 前 80 个 patch 的 valid 像素统计（固定，不随 seed 变）；
- **头部**（都是从头训练的可训练参数）：

| 头 | 结构 | 参数量 | lr |
|---|---|---:|---:|
| linear | 1×1 卷积 64→20 | 1.3 K | 1e-2 |
| mlp | 1×1 MLP 64→128→20（GN+GELU） | 11.2 K | 3e-3 |
| conv | 3 层 3×3 空洞卷积 d=1/2/4（w=64，GN+GELU）+1×1 | 112 K | 1e-3 |
| upernet | UPerNet-lite：合成 4 级金字塔（stride 1/2/4/8）+ PPM + 2 级 FPN | 3.77 M | 1e-3 |

- UPerNet-lite 的金字塔是**合成的**（AEF 只有单尺度 10 m 图），不是原生 4 阶段；
  接它只是把 T9 头家族的容量量级搬过来对照，不声明 AEF 有金字塔信息。
- 每个头的 lr 是预注册的单一值（经验性选择，无扫描）；
  头部选择本身不构成对 head-ladder 的"调参"，因为 4 个头都报。

## 3. 指标口径（已对齐官方协议）

- 主口径：**test mIoU，类集 {0,…,18}，void 19 忽略，只平均 GT 中出现的类**。
  与 PASTIS-HD 语义分割的公开协议一致：
  `maestro/conf/dataset/pastis_hd.py` 里 `pastis_seg = TargetRasterConfig(
  num_classes=19, missing_val=19)`（类集 0–18，19 是 missing），
  `maestro/train/metric.py` 的 `average_iou` 在 GT 出现的类上平均
  `tp/(tp+fp+fn)`，不排除背景；
- 辅口径：前景 1–18 mIoU（派生诊断列，防止"背景简单"抬分/压分造成误读）；
  不进主表排名；
- 注意：本 probe 的 val/test 是 **Fold 4/5 的前 100/150 个 patch（按 sample_id 排序，
  非随机）**，GT 类覆盖比完整 fold 少，类平均分母与正式口径可能不同 ——
  所以本表数字只做头间横向比较，不与正式结果横向比。

## 4. 结果

### 4.1 固定预算（300/100/150、8 epoch）

| 头 | 参数量 | lr | val 0–18 | **test 0–18（主口径）** | val 1–18 | test 1–18 |
|---|---:|---:|---:|---:|---:|---:|
| linear | 1.3 K | 1e-2 | 0.4274 | 0.3598 | 0.4172 | 0.3462 |
| mlp | 11.2 K | 3e-3 | 0.4432 | **0.3685** | 0.4333 | 0.3554 |
| conv | 112 K | 1e-3 | 0.4223 | 0.3111 | 0.4107 | 0.2946 |
| upernet | 3.77 M | 1e-3 | 0.4079 | 0.3303 | 0.3953 | 0.3150 |

训练 loss（8 epoch 末）：linear 0.660 / mlp 0.561 / conv 0.582 / upernet 0.591。
耗时：11.8 s / 34 s / 54 s / 330 s（CPU）。

### 4.2 全量复跑（1,455/482/496、8 epoch）

| 头 | 参数量 | lr | val 0–18 | **test 0–18（主口径）** | val 1–18 | test 1–18 |
|---|---:|---:|---:|---:|---:|---:|
| mlp | 11.2 K | 3e-3 | 0.4649 | **0.4476** | 0.4511 | 0.4331 |
| upernet | 3.77 M | 1e-3 | 0.4027 | 0.3817 | 0.3855 | 0.3635 |

（完整 Fold 1–3 / 4 / 5，8 epoch；linear/conv 只跑了预算小样，未在全量上重跑。
mlp 全量耗时约 55 s，upernet 约 27 min，CPU。）

## 5. 解读（边界内）

1. **mlp 是该预算下的最佳头**，比 linear 高约 +0.9 pt（0.3598→0.3685）：
   AEF 特征的非线性合成有正收益，且代价只有 ~11 K 参数；
2. **两个卷积头都不如 linear**：这不是"空间上下文无用"，而是小样本+短预算养不动
   ——conv 的训练 loss 最低（0.582）但 test 掉到 0.3111，明显过拟合/未到收敛；
   头容量的结论要等全量复跑；
3. linear→mlp 的收益为正说明 AEF 64 维特征不是单纯线性可分的表征，
   但正规化后能拉开多大仍需全量验证；
4. 本轮所有数字都在"子集+短预算"口径上，**不能作为 AEF 在 PASTIS 上的能力结论**，
   只能说明：特征与 2,433 个 patch 的数据管线已打通，且方向为正。

## 6. 局限与风险（正式 run 之前必须消除）

1. **单 seed 17**；正式轨 seed 17/42/73；
2. **类别不平衡未加权**：弱类（Grapevine、Beet、Orchard 等）IoU 接近 0，
   正式轨道需要类别权重或着重采样（决策放到 probe 之后）；
3. **lr 未扫**：每头一个预注册 lr；头间比较对 lr 敏感度未量化；
4. **子集偏差**：val/test 是前缀切片；正式数字必须落完整 fold；
5. **AEF 特征归一化**：`(x/127.5)²（不保号）变体未测； SpareDelta对 (x/127.5)
   线性缩放与现口径的差异待敏感性检查（不进正式轨道）；
6. **无云过滤**：PASTIS 包内无云掩膜，S2 时序原样进 AEF 场景云的影响是 AEF 产生
   端的问题，属 E2a/E2b 敏感性，不在本轮范围；
7. 文献可比性：AEF 是**年度合成特征**，与 U-TAE/OmniSat/MAESTRO 的（多时相）
   协议不同窗口，同数字不可比；本轮只做内部头间比较。

## 7. 复现

```bash
# manifest
uv run --with numpy python scripts/t11_build_manifest.py \
    --data-root /home/yzhong/rs-eval/data-aef

# 固定预算（本轮全部数字）
uv run --project workers/lim python scripts/t11_aef_probe.py --head linear \
    --train-patches 300 --val-patches 100 --test-patches 150 --epochs 8 \
    --out $AUDIT/t11_linear.json
uv run --project workers/lim python scripts/t11_aef_probe.py --head mlp ... --out $AUDIT/t11_mlp.json
uv run --project workers/lim python scripts/t11_aef_probe.py --head conv ... --out $AUDIT/t11_conv.json
uv run --project workers/lim python scripts/t11_aef_probe.py --head upernet ... --out $AUDIT/t11_upernet.json
```

结果 JSON：`/home/yzhong/rs-eval/audits/aef-20261103/`
（`t11_linear.json` / `t11_mlp.json` / `t11_conv.json` / `t11_upernet.json`，
含 params/lr/history/test_per_class_iou/note）。

## 8. 下一步

1. 全量 1,455 复跑 mlp 与 upernet（已排，结果回填 §4.2）；
2. T11 注册：`input_kind` schema + `configs/tasks/T11_pastis_hd.yaml` +
   `tests/test_registry.py`；主口径写进 `primary_metric` 定义；
3. T9 v4 结束后跑 LIM/SPOT 单景 T11 对照（A4 轨道）；
4. 正式 run（三 seed、完整 fold、类别加权决策）仍等 LPIS 标签年确认。
