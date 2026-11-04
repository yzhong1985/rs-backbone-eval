# T11 PASTIS-HD 首轮 AEF probe（技术验证，2026-11-03）

状态：**技术验证**。标签年按 2019 处理（正式结论仍需 LPIS 发布文档确认，见 `AEF_PASTIS-HD多时相对齐与FLAIR-HUB评估_2026-11-03.md` §5）；
单 seed（17）、无调参扫描；T11 不并入 Prithvi-suite D1，不与文献分数直接比。
本文档与 probe 脚本（`scripts/t11_build_manifest.py`、`scripts/t11_aef_probe.py`）
同一次提交；审计 JSON 在 `/home/yzhong/rs-eval/audits/aef-20261103/`。

> **口径变更（2026-11-04，§22）**：T11 BB 的正式 cell 改为**本机 4090 单 class**
> （`-4090` run id），Colab A100 退为备选；下文凡称"A100 是正式结果来源"的
> 段落均为历史记录，以 §22 为准。

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
  基本抵消；std 上加的 +1e-3 ε 回原始量纲约 ~0.058/255 色，数值上
  **近似等效**而非严格等于——reviewer round-2 措辞更正）。已存数字在
  Δε 量级不受影响；`preprocessing_id` 改为 `*_tilestats`，
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

## 15. BB 轨三 seed 收尾（旧代码 CPU 版，2026-11-04）

| 协议 | test mIoU（3 seed） | 备注 |
|---|---|---|
| `lim_s2_single`（年中单帧） | **0.1920 ± 0.0062** | 已三 seed · 旧代码 CPU |
| `lim_s2_median`（时序中位数） | **0.1496 ± 0.0027** | 已三 seed · 旧代码 CPU |
| `lim_s2_20bin` + LTAE2d | 待跑 | Colab A100（§14 归账） |

三个读法（先声明：这些数字按 §14 计划会在新代码/新 host 复跑后替代，
本节结论都是"指向性"而非正式版）：

1. **可控的"时序聚合谱系"前两点已落位**：AEF 年度合成 0.488 ≫ 单帧
   0.192 ≫ 中位数合成 0.150。惊喜处不在"单帧低"而在**中位数低于单帧**
   ——无学习的像素级时序压缩抹掉了"真实拍出来的图"的分布特征（MiT-Huge
   是航拍 RGB 预训练，对分布外伪图更敏感），这类信号支持"关键在那个
   learning 的时序聚合（LTAE2d / AEF）而非帧数本身"的叙事，等 20-bin
   数字出来后整条谱系闭合；
2. median 的 seed 间 sd 很小（0.0027），single 反而大（0.0059/0.0062）——
   采样框架参数越多、或者说随机性越少，输出越稳定，但绝对值最低；
3. 上级口径提醒：**这批 CPU 版 run 进不了正式表**，等 Colab 复跑后按
   `tags.host` 分 cell，旧记录标 superseded 保留审计轨迹。

## 16. 20-bin LTAE2d 一种子诊断（REDUCED-BUDGET，2026-11-04）

run `lim_s2_20bin-train-17-diag2`：1 epoch（override tag）、batch 2（与
T9 lane 共卡），全链路端到端通过。

| 指标 | 值 | 说明 |
|---|---:|---|
| test mIoU（1 epoch） | 0.1779 | foreground 0.1506；validation 0.1762 |
| wall time | 241 s/epoch | train 728 步 + 978 评估样本；共享卡、batch 2 |
| trainable 参数 | 7.63 M | 头家族（LTAE2d×4 + UPerNet fpn 256） |
| 显存 | +3.6 GB | batch 2 于 bf16~fp32 混合 |

三个读法：

1. **信号健康**：1 epoch 已到 0.178，超过 medians 三 seed 均值（0.150）、
   接近单帧（0.192）——时序注意力学得非常快，正式 8 epoch 大概率显著
   高于 single/median，谱系最后一格值得排期；
2. **性能换算**：共享 4090 上 728 步/epoch ≈ 4 min；A100 40GB 单卡独占、
   batch 8 提升吞吐 ~3-4×，全 confident **正式 8-epoch 单 run 预估
   25–40 min GPU 时长**（远低于原来"1–2 h"的写法），三 seed + 评估排队
   在 Colab 上 2–3 个会话内可以全部收完；
3. **batch 8 显存**：batch 2 时 LTAE2d 段 <4 GB（总 9.5 GB 与 T9 共存），
   level-0 attention buffer 随 batch 线性，batch 8 预估 ~12–14 GB，
   A100 40GB 无需 accum；4090 24GB 与 T9 共卡时不能直接 batch 8。

> 附注（review 2026-11-04 round 3）：上面 §16 的诊断 run 与 T9 lane 共卡
> 执行，违反"lane 期间不另起 GPU 任务"；该 run 为 REDUCED-BUDGET 不产出
> 正式数，但 T9 s42 受到显存/吞吐干扰。后续诊断一律改到 3080 或等 lane
> 空闲窗口，并把共卡记录纳入 T9 异常排查的嫌疑清单。

## 17. BB 轨扩展：AnySat 适配（2026-11-04）

AnySat 的 S2 入口与 PASTIS-HD 完全对齐：官方 10 波段顺序
（B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12）与 PASTIS 一致（**无需谱段
适配器**），且该入口本身就是逐像素的 LTAE2d 式时序注意力（T=367 位置
编码）；`output="dense"` 返回 **(B,128,128,1536)**（2×768），接共享
`LiteSpatialHead(1536)`。已用真实权重在 CPU 上验证：

| 变体 | dense 形状 | 单 patch 耗时（CPU） |
|---|---|---|
| single（T=1） | (1,128,128,1536) | 2.1 s |
| 20-bin（T=20） | (1,128,128,1536) | 4.9 s |

落地：`workers/anysat/src/anysat_worker/t11.py` + 三个协议
`anysat_s2_{single,median,20bin}`（track BB，`OFFICIAL_*_SAMPLES` 硬锚
496/482）；token 间距 80 m（16×16 tile tokens，`ANYSAT_T11_PATCH_M`
可覆盖）；输入 DN/10000 后按训练折 tile 统计逐波段 z-score；冻结骨干 +
`LiteSpatialHead`，8 epoch / AdamW 1e-3 / 有效 batch 8（2×4 累积，
`ANYSAT_T11_BATCH/ACCUM` 可覆盖）。测试：`tests/test_anysat_t11.py`
4 用例（三协议端到端 + 解码）在 anysat 环境全过。

排期：等 4090（T9 s73 → T10B2）或 3080（Prithvi → Clay）空闲窗口各跑
三 seed；CPU 不适合（每 epoch 1455 patch × 2–5 s）。

### 17.1 变体收敛决定（上级 2026-11-04）

后续主干对照只做 **single**（+ 20-bin，如 AnySat 这类原生时序入口零成本时
保留）：**median 变体停止新增**——逐像素中位数合成出来的是一种真实世界
不存在的伪场景，很多预训练主干没在分布内见过它；LIM 已有的 median 三
seed 结果保留为历史观测（0.1496 ± 0.0027，低于单帧，正好佐证这一判断）。
已撤掉 `anysat_s2_median` 协议（代码层保留解码分支，不注册不派发）；
Clay / OlmoEarth 只实现 single。

## 18. BB 轨扩展：Clay 适配（2026-11-04，single-only）

Clay v1.5 的 `sentinel-2-l2a` 入口与 PASTIS-HD 对齐：10 波段顺序相同
（blue..swir22），像素用 Clay 官方 metadata 的 sensor mean/std 标准化
（原始反射率尺度 0–10000），gsd=10 m，10 个波长直接取自
`configs/metadata.yaml`。分割沿用 T9 配方：`SegmentEncoder` 输出 stride-8
token 网格，共享 `UPerNetHead` 接收平均池化出的 4 级特征。真实权重 CPU
前向验证：logits **(1,20,128,128)**，全 finite，0.2 s。协议
`clay_s2_single`（track BB，496/482 硬锚，single-only）；测试
`tests/test_clay_t11.py` 2 用例在 clay 环境通过。

## 19. BB 轨扩展：OlmoEarth v1.2 适配（2026-11-04，single-only）

PASTIS-HD 的 10 个 S2 L2A 波段正好覆盖 OlmoEarth `SENTINEL2_L2A` 的前两个
band-set：set0 `[B02,B03,B04,B08]`、set1 `[B05,B06,B07,B8A,B11,B12]` 全部有真值；
set2 `[B01,B09]`（海岸/水汽）缺失，按 `MaskValue.MISSING=3` 整组置缺失，
不做常数填充。这与 T1 的六波段 constant-fill 适配器不同，因此 T11 可以诚实地记为
`frozen_probe`（`frozen_backbone_protocol` 新增 `CLEAN_ADAPTER_PROTOCOLS` 白名单，
T1 路径仍强制 `frozen_adapter`）。输入为 year-mid 单帧，真实获取日期以
`(day, month-1, year)` 传入；像素转反射率（×1e-4）后按训练 fold 统计量 z-score，
再双线性上采样到 224。真实 Tiny 权重 CPU 前向：输入 `(1,224,224,1,12)`，mask
`[0,3]`，特征 **(1,192,14,14)**、全 finite。协议 `olmoearth_s2_single`（track
BB，496/482 硬锚，single-only）；测试 `tests/test_olmoearth_t11.py` 2 用例在
olmoearth 环境通过。

## 20. 第 7 轮审查修复：时间编码、头统一与 route C 校验（2026-11-04）

- **时间位置**：PASTIS 序列跨 2018-09 至 2019-10，day-of-year 会出现重复
  （2018-09-24 与 2019-09-24 同为 267）并在 1 月从 322 跳回 17。LIM 20-bin
  与 AnySat 20-bin 改用 `days_since_reference()`（参考日 2018-09-01，与
  AnySat PASTIS loader、U-TAE 一致）；LIM 的 LTAE `T` 改为上游默认 1000。
  run tags 记 `temporal_positions` / `ltae_T`。旧的 diag2 按旧编码跑，
  保持 REDUCED-BUDGET 历史记录、不重跑。
- **头统一**：AnySat 的 `LiteSpatialHead(1536)` 换成 `UPerNetHead([1536],
  fpn_dim=256)`，dense 输出池化到 stride-8 后作为单尺度金字塔；BB 轨四个
  主干（LIM/AnySat/Clay/OlmoEarth）现在共用同一解码器系列。AnySat dense 的
  全分辨率特征先降到 stride-8，和 Clay T9 的做法一致。
- **步数**：AnySat/Clay 的 `optimizer_steps` 改为统计真实
  `optimiser.step()` 次数（此前按 micro-batch 累加，多记 4 倍）。
- **缓存**：AnySat 归一化缓存 key 加入 manifest sha、patch 数、版本，写入
  改为原子替换。
- **route C**：包里记录 checkpoint sha256（不拷 6.2 GB 文件），notebook 在
  dispatch 前验文件、回传时验 `run.json.model_checkpoint_sha256`；下载判断
  同时覆盖 DATA_S2 与 ANNOTATIONS；实际 `HF_ENDPOINT` 记入
  `hf_endpoint.json` 与 run tags。
- **协议范围**：AnySat/notebook 不再派发 median；已有 median 结果保留为
  历史观测。

## 21. 20-bin 8-epoch 预览与 s73 暂停/续训记录（2026-11-04）

- 时间编码修正提交前启动的旧 watcher 在 14:57 命中 s73 的 epoch-0 边界
  `latest.pt` 后暂停了 s73，并按当时的 dirty 工作树跑完 20-bin 预览
  （14:57–15:06，4090，官方 8 epoch / batch 8，未降 batch）。
- 预览结果（`lim_s2_20bin-train-17-preview8ep`，commit `be0e39e-dirty`）：
  **test mIoU 0.4069**、foreground 0.3907；validation 0.4181 / 0.4023。
  run tags 已记录 `temporal_positions=days since 2018-09-01`、`ltae_T=1000`，
  说明它用的就是修正后的时间编码。
- 该 run **标 `tags.superseded`**：树是 dirty 的、host 是 4090，不能当正式
  20-bin 结果；正式 cell 改为 4090 单 class（2026-11-04 上级决策，见 §22）。
  数字仅作**方向参考**：它和 `lim_s2_single` 的 0.1920（旧 CPU 三 seed）在
  设备、代码版本、时间位置上都不是同一口径，不能据此判断差距大小，正式
  对比只看 4090 同 host、同 commit 的 new cell。
- s73 暂停记录（`...-73-v4`，epoch 1 / 2,976 steps）已标 `superseded`，
  从同一 `latest.pt` 续训为 `...-73-v4-resume`（commit `468df4c`，与 s17/s42
  的 v4 代码版本一致，identity 校验通过）。续训在新 worktree 中执行，避免
  切换主仓库代码影响运行中的进程。


## 22. 第 8 轮审查修复与 4090 正式 cell 决策（2026-11-04）

**决策**：T11 BB 正式 cell 由 Colab A100 改为 **本机 4090 单 class**
（全部新 run 带 `-4090` 后缀、同一 host），因此不再需要上传 6.2 GB 的 LIM
权重；`drive-t11-pack-c-r8` 保留为备选。旧的 CPU single 三 seed 在新 GPU
run 完成后标 `superseded`。

**OlmoEarth T11 补齐（上一轮漏改）**：

- `optimizer_steps` 改为统计真实 `optimiser.step()`（4×2 累积，此前记成 2 倍）；
- 归一化缓存 key 加入 manifest sha、patch 数、版本，写入改原子替换；
- 解码头 `LiteSpatialHead` → **`UPerNetHead(fpn 256)`**，`head_id` 改为
  `olmoearth_s2_single_upernet_fpn256`，与 LIM/AnySat/Clay 一致；
- `input_res` 修正：128 px 双线性放大到 224 px，编码器实际看到 5.7 m 像素，
  因此改传 `10*128/224 ≈ 5.714 m`（原传 10 m 会让空间位置编码按错误 GSD
  计算）；tags 记录 `encoder_input_px=224`、`encoder_input_res_m=5.714`；
- 写入 `tags.hf_endpoint`（存在时）；删除死代码 `_doy()`。

**GPU 标准化设备冲突（AnySat/Clay/OlmoEarth）**：

- 三个 worker 都是 `mean/std.to(device)` 后与 `torch.from_numpy(...)`（CPU）
  做减法，CPU 测试路径不暴露，GPU 上会直接报 device mismatch；
- 统一改为 **numpy 侧标准化**（`(frames - mean_np) / std_np`）再 `torch.stack`
  搬到 GPU，结构上消除这类错误。

**AnySat token 间距**：删除 `ANYSAT_T11_PATCH_M` 覆写，固定 80 m（16×16
token）；若将来要试别的间距，必须声明新协议，避免 160 m 诊断混进 80 m cell。

**HF mirror 生效时机**：`huggingface_hub` 在 import 时读取 `HF_ENDPOINT`，
notebook 原来在 import 之后才设置，45 GB 请求仍打到主站。现在第一个 cell 在
import 前设置；镜像地址仍写入 `hf_endpoint.json` 与 run tags。
