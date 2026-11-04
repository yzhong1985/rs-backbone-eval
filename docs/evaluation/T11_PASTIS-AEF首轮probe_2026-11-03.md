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

manifest sha256（2026-11-03 修订版，含 `tile`/`aef_encoding`/`s2_path` 字段，
LIM single/median 变体也读这份 [T11 contract]）：
当前 `cf092238a3de7fe979fdda2325f4d9631413c5142628fe49861cc9a65…`；
六连 AEF run 用的旧 sha `29717adc…` 已归档在
`manifests/T11/archive/29717adc….jsonl`，账本里已完成的 run 的 manifest sha
不受重生成影响（按 run 里存的 sha 复核）。

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
  等于逐 tile 训练、每 epoch 结尾全是 t32ulu，大容量头受影响更大——审查 #3）。
- **归一化**：`sign(x)·(x/127.5)²`（x 为 int8 AEF 等级，−128 = nodata → valid=0 → 特征置 0）；
  逐通道 mean/std 用上一行的 **tile 分层 80-patch** 统计量（固定，不随 seed 变；旧的
  "前 80 个 patch"表述与分层口径矛盾，reviewer 2026-11-04 #5 已更正）；
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
- 三 seed 正式结论（§11）：**0.4879 ± 0.0099（upernet）/ 0.4849 ± 0.0081（mlp）**
  ——两头的均值差在 1 个 sd 内，**"upernet 反超 mlp"不成立，探针结论撤回**；
  正式口径下不再按头分流（单 seed 排序是采样/初始化伪影）；
- 注意：本 probe 的 val/test 是 **Fold 4/5 的 tile 分层子集（100/150）**——参加
  全部四个 tile，但每个 fold 内仍是子集；GT 类覆盖比完整 fold 少，类平均分母与
  正式口径可能不同 —— 所以 §4.1 数字只做头间横向比较，不与正式结果横向比
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
3. **全量数据（样研线）**：upernet 0.4884 > mlp 0.4769（+1.2 pt）——空间上下文头
   需要足量数据才能兑现（正式口径按 §11 撤回：三 seed 后两头均值差 <1 sd，
   "upernet 正式候选"不再成立，两头并列表述，不按头分流）；
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

## 11. 正式三 seed 结果与口径声明（round-6/7 审查后）

2026-11-03 的六连跑全部 completed，dispatch 日志已按 §11 末段镜像进各 run
目录（`train.log`）。BA 轨道在 report 与 Portal 同步验证通过
（`实验报告_2026-11-03.md` 的 2.0 矩阵含两行 T11；`/api/tasks/T11` 与
`/api/matrix` 的 `BA` cells 均显示 3 seeds 聚合，T11 无泄漏、无排除项）：

| 头 | 参数量 | test mIoU (mean ± sd, n=3) | s17 | s42 | s73 | 墙钟/seed |
|---|---:|---:|---:|---:|---:|---:|
| upernet | 3.77 M | **0.4879 ± 0.0099** | 0.4777 | 0.4886 | 0.4974 | ~28 min |
| mlp | 11.2 K | 0.4849 ± 0.0081 | 0.4760 | 0.4921 | 0.4866 | ~2.6 min |

两点读法：
1. **均值差在 1 个 sd 内**，且 seed 间互相换位 —— "upernet 反超 mlp"的 probe
   结论撤回；正式结果以三 seed 均值±sd 表述，不按头分流；
2. **同一头内**不同 seed 的离散可达 2 pt（upernet s17 最低 0.4777、s73 最高
   0.4974）——mlp/upernet 头间均值差 <1 sd，seed 间换位更大：正式表述两头
   并列、不按头排序（reviewer 2026-11-04 #5 指出旧文字把"同一头内 seed 离散"
   误写成"头变体间"，已更正）。

这批 run 是**本轮技术验证口径**——

1. 标签年按 2019 处理（LPIS 年未确认）；
2. 类别不加权（类别权重方案未定）；
3. 训练预算 = 8 epoch 全量 Fold 1–3（worker 内锁死，无环境旁路）。

正式结论仍以下一节的下一步清单为准。若标签年确认或类别权重方案落地导致口径变化，
这批 run 将用 `tags.superseded` 标记保留，不入 headline（AGENTS 的 run-ledger
准入规则照常执行）；届时重走 `scripts/rerun_baselines.sh --detached T11BA`
（`RS_EVAL_T11BA_SUFFIX=v2`，禁止复用无后缀 id）。

决定记录：**这批 run 让它跑完**（round-6 reviewer 询问是否中途停止），原因：
- 重复的 `WorkerFailure` 类定义是纯代码卫生问题，运行时语义（抛出/捕获用
  同一个类）不受影响；
- git_commit 是 `454064d` 无 `-dirty`，审计按祖先关系转录不排除；
- 重跑成本（~1.5 h CPU）远大于接受这批已完成 run 的收益。

## 12. 下一步

1. ~~T11 注册~~（完成：`input_kind` schema + `configs/tasks/T11_pastis_hd.yaml`
   + `tests/test_registry.py`；轨道 `BA` 单列在矩阵末尾，不并入 Prithvi-suite D1；
   commit `7c4c7ae` / `f51251a` / `454064d`）；
2. ~~三 seed 正式 dispatch~~（完成：mean±sd 见 §11，全量 496 口径）；
3. LIM single/median 两变体：`lim_worker/t11.py` + `BB` 轨道协议已实现并
   dispatch（single 三 seed 完成 0.1920±0.0059；median 进行中）——同时
   reviewer 2026-11-04 发现运行时双归一化 + CPU 计算两个 P1，已在
   `t11_common` 修复（tile 统计 z-score 为最终口径，ImageNet 仿射抵消即
   声明例外）；修复后 single/median 待 CSI 重新 dispatch，旧 run 标
   superseded；
4. LIM-S2 20-bin（2026-11-04 实现就绪，等 dispatch）：
   LTAE2d 官方源确认在 `VSainteuf/utae-paps` @ `987874e`（MIT，`ltae-src` 仓库
   只有 1d 版），vendor 进 `workers/lim/src/lim_worker/vendor/ltae/`（唯一改动是
   一行 import 路径，见 `PROVENANCE.md`）；`lim_worker/t11_ltae.py` per-level
   LTAE2d（n_head 16 / d_k 8 / d_model 256 / T 366 DoY 位置编码 / dropout 0.2，
   逐像素时序注意力，随后仍接公共 UPerNetHead fpn 256——20 帧各自过冻结
   MiT-Huge，叠加 4 层金字塔按 level 折成年度 embedding）+ 20-bin 确定性
   linspace 采样（无随机性，train/val/test 同一选择）；协议
   `lim_s2_20bin`、轨道 BB、`tests/test_lim_t11_ltae.py` 4 用例全过；
   **正式三 seed 走 Colab A100 40GB**（上级决策 2026-11-04）：40GB 放得下
   batch 8（level-0 attention buffer ≈ 3.4 GB fp32），整 cell 声明
   `host=colab-a100-40gb` 单 GPU class，run notes 里写明；
   数据不上传：`DATA_S2` 在 Colab 侧从官方 `IGNF/PASTIS-HD`（HF）直下，
   本部署连不通 HF 已实测（2×curl 000）；上传件只有
   `scripts/pack_t11_drive.py` 产出的 `~/rs-eval/drive-t11-pack/`（LIM 权重
   6.2 GB + annotations/sidecars zip + 合同 manifest + git bundle，合计
   ~6.5 GB），`PROVENANCE.json` 内置 5 个样本的 DATA_S2 sha256 信任锚，
   notebook 下载后逐文件复核（本机无 HF 网络无法直接 md5 对照）；
5. **上级决策（2026-11-04）：取消 472 剔云子集**。T11 的正式口径固定为
   全量 fold（test 496），不做剔云复制；与同事 A 档对照时只需注明"样本集
   不同（他们 472 / 我们 496），不换算"，472 那一列不再追加；
6. 正式结论等 LPIS 标签年确认；类别加权决策：正式轨之前过一轮类别权重方案。

## 13. 2026-11-04 审查修订（round-11，两路 reviewer）

完整问题清单与逐条处置见 `docs/reviews/审查响应_2026-11-04.md`。要点：

- **归一化口径**（双方 P1 #1）：代码实际输出 = DN/10000×255 后的
  **train-fold tile 统计 z-score**（z-score 对仿射变换不变，ImageNet 项
  完全抵消），已存数字不变；`preprocessing_id` 改为 `*_tilestats`，
  notes/tags 如实声明其 final 口径，NORM_DECISION a808302 对 T11 声明为
  **例外**。不需要 required_fixes、不需要重跑已保存的数字——但 official
  single/median 会按修订后的代码在新 host 重跑一次并以新 run 替代
  （旧 run 标 superseded，审计轨迹保留）。
- **设备**（P1 #2）：`lim_worker/t11_common.training_device()` 统一
  cuda-if-available，device/host(`torch.cuda.get_device_name`)/torch 版本
  写进 tags，notes 不再写死 host 字符串；评估样本覆盖
  `test_manifest_samples` vs `test_samples_evaluated` 双 tag，不符
  直接失败（matrix `is_shrunken_eval` 联动排除）。
- **头家族声明**：BA↔BB 为"同头设计（UPerNet/PPM bins/fpn 256）、同预算、
  同评估代码"的可控对比——但 AEF 头实现在 `aef_worker/heads.py`
  （3.77M，输入是 64 维年度单图），LIM 头走公共 `rs_backbone_eval/heads.py`
  （6.96M，输入是 MiT 四级金字塔）；_bb 只跑 upernet 的理由是头部消融在
  BA 已完成且结论为"次要轴"，20-bin Cell 不重复。
- **Colab S2 数据完整性**：`s2_sha256.json` 全量 2,433 文件哈希随包传递，
  HF 下载后逐文件校验；Colab 运行归账规则写进 §12 第 4 条。

## 14. Colab 侧运行归账与可比性声明（round-11 附加）

**归账流程**（正式 single/median/20-bin 三 seed 全部在 Colab A100 出数）：

1. bundle 在 pin commit 上解包，worktree 干净 → run.json 的 `git_commit`
   与本机 HEAD 一致，audit 的祖先链校验成立；
2. `data_root` 指向 Drive 挂载（`derived/aef-2019/` 布局按 manifest 相对路径），
   `LIM_CHECKPOINT` 指向包内权重；dispatch 照常在 Colab 内执行；
3. run 目录回传后放入 `/geodata/rs-eval/runs/`，`tags.host` 为
   `torch.cuda.get_device_name` 运行时值（预期 `nvidia-a100-sxm4-40gb`），
   同一 cell 的三个 seed 必须同 host，与 4090/3080 其余 lane 不混 cell；
4. 正式数字以新 run 为准，旧的 CPU 版 single/median run 在新 run 完成后标
   `tags.superseded`（原账本记录保留）。

**头家族可比性**（BA ↔ BB 合表口径）：BA/BB 同头设计（UPerNet + PPM bins
+ fpn 256）、同 seed 集、同预算（8 epoch / AdamW 1e-3 / 有效 batch 8）、同
评估代码与同一合同 manifest —— 但头是**两份实现**：AEF 头在
`aef_worker/heads.py`（3.77M，输入 64 维年度单图），LIM 头走公共
`rs_backbone_eval/heads.py`（6.96M，输入 MiT 四级金字塔）；参数量差异记录在
各 run 的 `training.trainable_parameters`。表述规则：可以写"同头家族可控
对比"，不可写"头完全相同"；BB 只跑 upernet 的理由是头部消融在 BA 已完成
且两头均值差 <1 sd（次要轴），BB 内部的头不再作为变量。
