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

manifest sha256（2026-11-03 修订版，含 `tile`/`aef_encoding` 字段）：
`2be347dc1215762d7d73578eb6c960cd815efbc85d1e67dcc99d715bdcc3eec4`。

**两份 manifest 的口径要分清（round-6 审查后声明）**：

| | 正式 manifest（`rs-eval build-manifest` 进账本） | probe 工作清单（`t11_build_manifest.py`） |
|---|---|---|
| `timestamps` | `["2019"]`（年度合成单时间戳） | S2 全部采样日的日期列表（2018-09→2019-10） |
| `bands` | `AEF_00..AEF_63`（64 维特征通道名） | SPOT RGB + S2 六波段子集（源模态描述） |
| `sensor` | `aef` | `spot-6/7+s2` |
| 输入 | `image_paths[0]` = AEF `.npz`（input_kind=aef_features） | `aef_path` 字段（另有 SPOT/S2/Parcel 路径） |
| 用途 | dispatch/formal track（审计可见） | probe/seed 复现与 E2a/E2b 单景检索（不进审计） |
字段口径：`sensor`/`bands` 描述样本携带的**源栅格**模态（SPOT 1 m RGB +
S2 六波段子集），AEF 输入单独由 `aef_path`/`aef_bands`(64)/`aef_grid`/
`aef_encoding` 声明——语义分割 worker 只读 `aef_path` 与 `label_path`；
正式注册后这些字段由 `input_kind` contract 校验，不再靠 prose。

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

- **budget 固定**：train 300 / val 100 / test 150，**按 tile 分层抽样**
  （比例配额 + 轮转交替，保证四个 tile 都有；修订前的前缀切片曾把 297/300 个
  train patch 和全部 val 都缩到一个 tile——审查 2026-11-03 #2）、
  逐通道 mean/std 只用 train 的 tile 分层 80-patch 统计量（固定，不随 seed 变）、
  8 epoch、batch 8、AdamW wd 1e-4、CPU、seed 17；
  **训练循环逐 epoch 用固定种子 shuffle**（修订前按 sample_id 顺序喂，
  等于逐 tile 训练、每 epoch 结尾全是 t32ulu，大容量头受影响更大——审查 #3）；
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
- 注意：本 probe 的 val/test 是 **Fold 4/5 的 tile 分层子集（100/150）**——参加
  全部四个 tile，但每个 fold 内仍是子集；GT 类覆盖比完整 fold 少，类平均分母与
  正式口径可能不同 —— 所以本表数字只做头间横向比较，不与正式结果横向比
  （旧口径"前 100/150"的结果连同 json 保留为 `*_prefixbias.json`，不引用）。

## 4. 结果

> 修订（审查 2026-11-03 #2/#3）：本表为 **tile 分层抽样 + 逐 epoch shuffle** 复跑后的数字；
> 前缀切片的旧结果保留在 `audits/aef-20261103/*_prefixbias.json`，不再引用。
> 单-tile 前缀子集曾把 conv 压到 0.31、upernet 压到 0.33——翻转后这两个头回升 0.10+
> （审查怀疑成立：顺序喂 + 单 tile 就是大头容量被低估的主因之一）。

### 4.1 固定预算（300/100/150，tile 分层、8 epoch）

| 头 | 参数量 | lr | val 0–18 | **test 0–18（主口径）** | test 前景 |
|---|---:|---:|---:|---:|---:|
| linear | 1.3 K | 1e-2 | 0.4182 | 0.4154 | 0.4012 |
| **mlp** | 11.2 K | 3e-3 | 0.4628 | **0.4457** | 0.4332 |
| conv | 112 K | 1e-3 | 0.4356 | 0.4223 | 0.4083 |
| upernet | 3.77 M | 1e-3 | 0.4276 | 0.4071 | 0.3923 |

**tile 覆盖**：train 61/75/93/71（t30uxv/t31tfj/t31tfm/t32ulu）、val 23/25/27/25、
test 37/43/42/28；统计量样本 16/20/26/18。
耗时：11.8 s / 34 s / 54 s / 330 s（CPU）。

### 4.2 全量复跑（1,455/482/496、8 epoch、逐 epoch shuffle）

| 头 | 参数量 | lr | val 0–18 | **test 0–18（主口径）** | test 前景 |
|---|---:|---:|---:|---:|---:|
| mlp | 11.2 K | 3e-3 | 0.5046 | 0.4769 | 0.4637 |
| **upernet** | 3.77 M | 1e-3 | 0.5082 | **0.4884** | 0.4754 |

（linear/conv 未在全量上重跑，正式轨道只保留 mlp/upernet 两个候选；
mlp 全量约 55 s，upernet 约 27 min，CPU。）

## 5. 解读（边界内）

1. **固定预算：mlp 仍是最佳单头**（test 0.4457，比 linear +3.0 pt）：
   非线性合成有正收益，代价只有 ~11 K 参数；
2. **采样修正后头排序完全变盘**：前缀切片曾把 conv/upernet 压低 0.10+；
   分层 + shuffle 后 conv 0.4223 > upernet 0.4071 > linear——**本轮最重要的发现
   是采样偏差足以倒置结论**，此前"卷积头不收敛"的解释撤回；
3. **全量数据反超**：upernet 0.4884 > mlp 0.4769（+1.2 pt），与文献期望一致——
   空间上下文头需要足量数据才能兑现；正式轨道候选：upernet（主）、mlp（对照）；
4. 与同事 A 档并排（同口径 test 0–18，但他们剔云 472 / 我们全量 496，是**两个不同
   的样本集**）：AEF 冻结特征 + 3.77M 头 8 epoch = 0.4884；其 A 档 8 行
   （DOFA-B 26.9 → DOFA-L 32.6、Prithvi 39.3，Clay 41.9，MaRS 42.3，AnySat 47.5）
   属于不同预算（80/50 epoch）、不同输入（20-bin 时序 vs 年度合成）和不同 seed 数
   （3 vs 1）。**本节只并列数字，不下"高于/低于"结论**：单项 seed、差一个量级的
   预算和不相等的样本集压不进同一张 rank 表；三 seed 同预算复跑后才进入比较
   （见 §9/§10 的正式对照计划）；
5. 本轮仍是 single-seed 技术验证；正式 run 需 17/42/73 + 完整 fold + 类别加权决策。

## 6. 局限与风险（正式 run 之前必须消除）

1. **单 seed 17**；正式轨 seed 17/42/73；
2. **类别不平衡未加权**：弱类（Grapevine、Beet、Orchard 等）IoU 接近 0，
   正式轨道需要类别权重或着重采样（决策放到 probe 之后）；
3. **lr 未扫**：每头一个预注册 lr；头间比较对 lr 敏感度未量化；
4. **子集偏差已修复**：固定预算层现在按 tile 分层抽样 + 逐 epoch shuffle（旧前缀切片
   结果归档为 `*_prefixbias.json`）；正式数字仍须落完整 fold（§4.2 已是完整 fold）；
5. **AEF 特征值域变换**：本轮统一用保号的 `sign(x)·(x/127.5)²`；未测的变体是
   线性缩放 `x/127.5` 和不含 sign 的 `(x/127.5)²`。单点敏感性检查放在
   正式轨道之前（只确认量级差异，不进正式结果）。
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

## 9. V1 冒烟：LIM-S2 20-bin（3080 已验证）

`scripts/t11_lim_smoke.py`（commit `2e9fc07b`）在 20 GB host 上以真实权重
`mit_huge_600.pth` 完成全前向（bundle 同步 `d2bb869 → 2e9fc07b`）：

| 项 | 值 |
|---|---|
| 输入 | batch 8 × **20 bin** × 3（B04/B03/B02，DN/10000×255 + LIM ImageNet 统计量）× 128² |
| 金字塔 | stride 4/8/16/32（128/64/32/... 输出：32²·128c / 16²·512c / 8²·1024c / 4²·2048c）|
| 吞吐（forward-only） | **7 样本/s**（4 batch/4.5 s）|
| 峰值显存 | **2174.6 MiB**（batch 8，encoder 与 backward 解耦后余量充足）|
| 审计 JSON | `audits/aef-20261103/t11_lim_smoke_3080.json` |

含义：
1. 冻结轨的显存压力主要在头部缓存与 LTAE2d（千万级 token 决定），编码器不再穿透
   反向图——**batch 8 的显存余量宽裕，可以稳住同事 A 档的 batch 口径**；
2. 1,455 个 train patch 的 20-bin 编码器前向 = 29,100 帧 ≈ **3.5 分钟**（可整折缓存）；
3. LTAE2d 头还未实现（当前只有 20-bin→mean 的 proxy），正式 V1 还需要：
   `workers/lim` 增加 T11 条目 + LTAE2d 头 + T11 真实 20-bin 缓存 → 冒烟再进 worker 化。

## 10. 下一步

1. T11 注册：`input_kind` schema + `configs/tasks/T11_pastis_hd.yaml` +
   `tests/test_registry.py`；主口径写进 `primary_metric` 定义；
2. LIM V1 正式对照（3080，20-bin 缓存没问题后接 LTAE2d+UPerNet、三 seed），
   随后 SPOT 单景对照（A4 轨道）；
3. 三 seed 正式 run（完整 fold、类别加权决策）仍等 LPIS 标签年确认；
4. 与同事对齐的并表材料：向同事要 20-bin 缓存口径与 84 个剔云 patch 清单。
