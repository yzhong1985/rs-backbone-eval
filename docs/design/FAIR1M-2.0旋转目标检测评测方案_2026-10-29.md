# FAIR1M-2.0 旋转目标检测评测方案

日期：2026-10-29  
版本：v4，按第二轮评审简化 G0，补充 worker 预算口径、新模型接入顺序，并记录
2026-10-29 的数据实测与 G0 骨架落地情况。  
状态：**数据已取得并验收；契约、几何、COCO AP、检测器与 T10 数据层已提交；GPU smoke
与正式预算尚未完成。**  
所属项目：`rs-backbone-eval`；参考代码：`/home/yzhong/projects/rs_segmentation`。  
目标机器：4090 评测节点，RTX 4090 24 GB。  
关联：[第二轮审查响应](../reviews/审查响应_2026-10-29_FAIR1M与新主干_第2轮.md)；[第一轮审查响应](../reviews/审查响应_2026-10-29_FAIR1M方案_第1轮.md)；[MaRS / Galileo 独立可行性说明](../models/MaRS与Galileo接入评测可行性_2026-10-29.md)。

## 1. 评审摘要与建议决策

建议新增 FAIR1M-2.0 的 **37 类旋转框检测**任务，采用统一的 **FPN + Rotated Faster R-CNN** 检测器，先比较冻结主干的表征能力，再根据显存和排期决定全量微调。重点主干为 LIM，即已上传的 `mit_huge_600.pth`；现有 AnySat、Clay、Prithvi、OlmoEarth Tiny/Base 按相同检测协议逐个接入。

用户已明确要求比较“主干 + Faster R-CNN head”。本版采用旋转框版本，保留 FAIR1M OBB 任务定义；Oriented R-CNN 不再作为默认方案。ResNet-50 是单独的管线复现锚点，不与 LIM 拼接，也不把评测范围缩减为两个模型。MaRS 和 Galileo 是用户已明确转达的上级新增候选，先通过 iSAID B2U 接入、污染审计及资源验证，再在 G0/G1 通过后加入同一检测协议；候选身份已确认，正式训练规模仍分阶段排期。

本方案建议评审以下事项：

| 决策 | 建议 | 边界 |
| --- | --- | --- |
| 任务 | FAIR1M-2.0，完整 37 类、旋转框 | 34 类旧实验或镜像子集只作单列调试 |
| 统一检测器 | Rotated Faster R-CNN + FPN | 全部主干锁定同一实现、配置和后处理 |
| 第一轮 | 冻结主干，训练 FPN、RPN、RoI 检测头及必要适配器 | 与全量微调分轨；显式报告适配器参数 |
| 工程验证与正式范围 | R50 复现锚点 + LIM 工程验证；正式覆盖原有六个模型配置 | R50 标准训练不混入冻结主表；新增模型通过独立接入门槛 |
| 正式预算 | 初步建议 12 个完整训练 epoch、种子 17/42/73 | **项目拟定预算，不是 FAIR1M 官方统一预算**；先测算资源再锁定 |
| 计算优先级 | 建议 iSAID 首批 B2U 结果后进入检测 P1 | CPU 调研/数据工作可先行；不能由分割低分推断检测必然无价值 |
| 数据落盘 | `/home` 存完整原始数据，`/geodata` 按需放热点数据 | 当前 `/geodata` 仅约 199 GB 可用，不能直接灌入全集 |
| 结果口径 | 内部 train/dev；37 类、五大类 mAP 与尺寸分层召回必报 | 不冒称官方 test，不据此宣布通用主干优劣，不并入现有 D1 |

### 1.1 已确认决策（2026-10-29，用户书面同意）

| # | 决策 | 落地位置 |
| --- | --- | --- |
| 1 | 数据集命名：用镜像里的 **1.0 train** 作正式训练集（官方称与 2.0 train 一致），内部切 dev；**不称 2.0 全集** | §4.2.1，run 记录 `dataset.version` |
| 2 | manifest 扩展：`label_path`（栅格）与 `annotation_path`（矢量）**二选一**，另记每块实例数与忽略数 | `contracts.ManifestSample` |
| 3 | 轨道命名：检测用 **B2D / B1D**（与分割的 B2U/B1U 区分） | `configs/protocols/rotated_{frozen,finetune}.yaml`、`matrix.PROTOCOL_TRACKS` |
| 4 | 预算先按 **optimizer step** 定，实测吞吐后再折成 epoch | 本轮未注册 T10 预算，见 §9.3 |
| 5 | mAP 口径 **COCO 101 点**（AP/AP50/AP75 + 每类 + 尺寸分层），需要可比 | `detection/ap.py` |

## 2. 数据集说明与评测边界

FAIR1M 面向高分辨率遥感细粒度目标检测，包含 5 大类、37 个细类，空间分辨率约 0.3–0.8 m，原图约 1,000–10,000 像素。2.0 与 1.0 训练集一致，增加验证集并扩充测试集。依据为[官方 benchmark 介绍](https://www.gaofen-challenge.com/benchmark)。本文不把镜像页面的样本数或容量当作 2.0 全集清单。

原始数据通常为三通道 TIFF 影像及 XML 四边形标注，具体字段以实际取得的 2.0 发布包验收。格式可参考 [TorchGeo 的 FAIR1M 说明](https://docs.torchgeo.org/en/stable/api/datasets/fair1m.html)。

| 大类 | 细类数 | 细类名称（展示名，正式 ID 顺序需单独锁定） |
| --- | ---: | --- |
| 飞机 | 11 | Boeing737、Boeing747、Boeing777、Boeing787、ARJ21、C919、A220、A321、A330、A350、other-airplane |
| 船舶 | 9 | Passenger Ship、Motorboat、Fishing Boat、Tugboat、Engineering Ship、Liquid Cargo Ship、Dry Cargo Ship、Warship、other-ship |
| 车辆 | 10 | Small Car、Bus、Cargo Truck、Dump Truck、Van、Trailer、Tractor、Truck Tractor、Excavator、other-vehicle |
| 球场 | 4 | Baseball Field、Basketball Court、Football Field、Tennis Court |
| 道路设施 | 3 | Roundabout、Intersection、Bridge |

### 2.1 为什么采用旋转框检测器

普通 Faster R-CNN 输出水平框。将 FAIR1M 四边形转成外接水平框后可以训练，但任务定义、IoU 和密集目标重叠关系都发生变化，所得成绩必须标为 HBB，不能直接对比标准 OBB 检测成绩。

正式路线保留方向信息，采用 `le90` 旋转框约定，统一框转换、旋转增强、IoU、RoIAlign 和 NMS 的角度方向。MMRotate 中这些操作的约定并不完全相同，必须通过可视化和算子测试核对，见[官方基础说明](https://mmrotate.readthedocs.io/en/latest/intro.html)。

### 2.2 能回答和不能回答的问题

- 能回答：同一高清 RGB 输入、同一检测头、同一划分和预算下，各主干的冻结检测表征及可选微调能力。
- 不能直接回答：多光谱/SAR/时序能力、跨传感器泛化、所有下游任务上的整体架构排名。
- LIM 在高清 RGB 输入及多尺度输出上具有任务适配优势；现有卫星主干可能存在输入波段、GSD 和特征步长差异，必须逐项披露。
- LIM 的预训练源与 FAIR1M 都可能包含 Google Earth 影像；预训练污染及跨 split 地理重叠尚未排除，不能声称严格未见区域泛化。

## 3. 已核查的部署事实

2026-10-29 通过远端只读检查和 `uv run --frozen rs-eval doctor` 核对：

| 项目 | 实际情况 |
| --- | --- |
| 评测项目 | `/home/yzhong/projects/rs-backbone-eval`，本轮核查 HEAD 为 `90a98a9`；活动开发仍在继续 |
| 参考仓库 | `/home/yzhong/projects/rs_segmentation`，HEAD 为 `9e09d47048f9300f495fa18df8fc749701d30024` |
| GPU | NVIDIA RTX 4090，24,564 MiB |
| 数据根目录 | `/geodata/rs-eval/data` |
| 权重 / 账本根目录 | `/geodata/rs-eval/weights`、`/geodata/rs-eval/runs` |
| 缓存根目录 | `/home/yzhong/rs-eval-cache` |
| 存储余量 | `/geodata` 约 199 GB；`/home` 所在卷约 1.4 TB；均为时点快照 |
| 主要权重 | `/home/yzhong/models/mit_huge_600.pth`，6,621,891,160 字节 |
| 权重 SHA-256 | `3c2e1b2e9257d8a74330a507145b82d4e6dd5bea65705aec73c8ae0caa8a6985`，远端已复核 |
| 数据现状 | 已检查的项目 raw 根目录及 home 浅层目录未发现 FAIR1M；**不代表全盘扫描证明不存在** |

`docs/design/iSAID与LIM评测方案_2026-10-29.md` 已记录 LIM 编码器约 524.4 M 参数，完整预训练模型另含重建解码器；实际快照记为 `epoch=150 / args.epochs=300`。检测接入时应再次导出 checkpoint metadata，**不能根据文件名中的 `600` 认定它训练了 600 epoch**。

当前 LIM / iSAID 接入代码已陆续入库，但检测任务、协议和 worker 链路仍待实现。已有分割支持不能被视为检测支持；实现时再次检查 registry 的实时状态。

## 4. 数据取得、下载与验收

### 4.1 下载渠道及核验状态

| 渠道 | 用途 | 本次核验情况 |
| --- | --- | --- |
| [官方 benchmark](https://www.gaofen-challenge.com/benchmark) | 取得 FAIR1M-2.0 发布清单、下载入口、划分和评测说明 | 已取得 HTTP 200 的 JavaScript 页面壳；网页抽取工具另有超时，**尚未核实一个可用的全集直链** |
| [官方论文](https://arxiv.org/abs/2103.05569) | 核对任务定义与引用 | 不是可直接下载 2.0 全集的接口 |
| [JDet FAIR 数据说明](https://github.com/Jittor/JDet/blob/master/docs/fair.md) | 参考 XML 转换、大图切片及结果合并流程 | 可读取；旧入口和参数不能替代当前 2.0 发布说明 |
| [Hugging Face `blanchon/FAIR1M`](https://huggingface.co/datasets/blanchon/FAIR1M) | 备选工程 smoke 数据 | 数据卡明确描述部分训练数据；**不可称作完整 FAIR1M-2.0**，也不能假定拥有官方 2.0 val/test |
| `hf-mirror.com` 镜像 | 本次实际数据来源 | **2026-10-29 实测**：本机到 `huggingface.co` 15 s 超时不可达，`hf-mirror.com` 可达（约 10 MB/s）；`LittleCollections/FAIR1M1` = 1.0 train（与 2.0 train 相同），`FAIR1M2` 仓库名实际是 **1.0 的 val**，不是 2.0 |

不采用未经核验的网盘提取码或把 FAIR1M-1.0 / 1.5 链接改名成 2.0。ModelScope 等其他镜像只有在提供者、版本、文件列表及 split 身份核对后才可加入正式来源表。

### 4.2 完整数据的取得步骤

1. 数据负责人打开官方入口，确认申请/下载要求、使用条款及 **FAIR1M-2.0** 的发布说明，保存来源页面和日期。
2. 优先取得经核实的官方训练影像与标注、类别表和发布说明。默认从 train 按原图分组划内部 dev；官方 val/test 作为后续独立轨道，逐项核实可用标注及提交条件，不以测试标签或评测服务器可用作为默认前提。
3. 保存每个压缩包的文件名、split、URL 或资源编号、声明容量和上游校验值。没有上游 hash 时生成本地 SHA-256，但明确其只能标识本地副本，不能独立证明内容为官方版本。
4. 若入口不可用，由数据负责人通过官方页面公布的联系渠道申请有效资源；本文不代发邮件。拿到确切链接以前，完整数据取得是 P0 的阻断项，不能承诺一条命令下载全集。
5. 优先让 4090 机器直接从授权数据源下载。此前本地到该机 SSH 中转仅约 0.3–0.6 MB/s，不适合作为大规模数据搬运的默认方案。

默认正式内部比较命名为 `FAIR1M2-train-holdout`：经核实的官方 train 按原图/场景固定划分 train/dev，先划分再切片。建议 dev 约 12%、种子 17，P0 检查类别覆盖后一次性锁定。报告为内部开发集成绩；若需要无调参污染的最终测试，应另外保留独立 holdout 或取得合适的官方 split。

### 4.2.1 2026-10-29 实际取得与验收结果

| 项 | 实测值 |
| --- | --- |
| 下载渠道 | `hf-mirror.com/datasets/LittleCollections/FAIR1M1`，4 个包共 **30.0 GB**（`images.zip` 5.1 G、`images-1.zip` 16 G、`images-2.zip` 8.9 G、`labelXmls.zip` 17 M） |
| 校验 | 本地 SHA-256 记录在 `downloads/sha256sums.txt`，并登记进 `configs/downloads/catalog.yaml`（`data_fair1m_train`）；`unzip -t` 全部通过 |
| 影像 | **train 14,756 张**（FAIR1M1，与自身 XML 300/300 尺寸一致）；**val 1,732 张**（改用 `FAIR1M2` 仓库自己的 images，1,732/1,732 尺寸一致）；FAIR1M1 里另 1,732 张**无标注**影像移入 `images/unlabeled/`，不进任何 split |
| 标注 | **14,756 个 XML / 311,682 个实例 / 37 类齐全**；每个多边形恰好 **5 点闭合**（首点重复），解析零失败 |
| 尺寸 | 抽查 300 张：train 98% ≤1024（主力 1000×1000、800×600、600×800），少量 4000 px；**说明该镜像给的是已裁好的图块，不是 1,000–10,000 px 的原始大图** |
| 配对校验 | 每张影像尺寸必须与 XML `<size>` 一致。曾经把 FAIR1M1 的无标注影像按同名配上 FAIR1M2 的 val 标注，**仅 458/1,732 一致**（等效地，重建框会整体错位）；现已改为"影像和标注取自同一仓库"，并把这条例行校验写进 `scripts/fetch_fair1m.sh`（低于 99% 直接失败）。详见[第三轮审查响应](../reviews/审查响应_2026-10-29_FAIR1M第3轮.md) |
| 磁盘 | 最终落在 `/geodata/rs-eval/data/raw/fair1m`：影像 49 GB + 标注 185 MB + 原始压缩包 29.3 GB = 78 GB，剩余 95 GB。**最初用 `/home` + 符号链接的布局被放弃**：下载器的路径守卫 `_target()` 会 resolve 链接并判定"逃出 data root"抛错，进而让 `/api/downloads` 返回 500；改为真实目录，并把该守卫的失败降级为单条 `invalid_target` 状态（不拖垮整个 Portal），已加回归测试 |

**这是一条必须写进结论的 provenance 差异**：官方 FAIR1M 是大场景，本镜像已裁成约 1000 px
的图块。协议因此退化为"每图 1 块、必要时 pad"，裁块重叠与边界规则仍然生效（少量 4000/5000 px
的图会真正切块），但不能声称复现了官方的整图推理流程。

### 4.2.2 manifest 产物（已生成并审计）

| manifest | 训练块 | 验证 | 测试 | 说明 |
| --- | ---: | ---: | ---: | --- |
| `26ff87ea…`（1024/824） | 19,228 | — | 2,445 | 主协议；`audit-data` 零错误、零组泄漏 |
| `holdout_1024_s824_seed17` | 15,389 | 3,839 | 2,445 | 从 train 按原图划 20%、seed 17 |
| `64718b2c…`（512/256，备用） | 141,095 | — | 18,744 | 诊断用，见下 |

旧 manifest（`2395e0f1…`、`c462d443…`、`98b425ec…`、`c695b3d7…`）已被取代：前两个用的是错配的
val 影像，后两个的派生标注里还留着替换前影像的尺寸。全部移入 `manifests/T10/superseded/`。
派生标注现在与影像尺寸强一致（builder 发现缓存尺寸与影像不符时会重写并计数）。

1024 档：75% 的块存在 padding、**43% 的块有效像素不足一半**（800×600 → 1024 的黑边）；
512 档：padding 归零，但空块占 39%、块数涨到 7.3 倍、同像素覆盖约 2.6 倍。

**维持 1024/824 的理由**（注意检测的有效 batch 是 **2**，不是 T9 的 8，下面的步数按 2 折算）：

| 口径 | 1024/824 | 512/256 |
| --- | ---: | ---: |
| 每 epoch 训练块 | 19,228 | 141,095 |
| 12 epoch 折成 optimizer step（÷2） | 115,368 | 846,570 |
| 25k step 覆盖 | 50,000 块 ≈ **2.6 epoch** | 50,000 块 ≈ **0.35 epoch** |
| 每 step 见到的像素（2 块） | ≈2.1 Mpx（有效 ≈1.6） | ≈0.52 Mpx |
| padding | 75% 有 pad，43% 有效像素不足一半 | 0 |
| 空块（无目标） | 11.6% | 38.7% |

1. **分辨率完全不变**：两档都是从原图上按原像素裁切，没有重采样，目标始终 20 px 中位数；
   "512 更精细"在这里不成立。
2. 预算单位是 optimizer step（决策 4）。同样 25k step，1024 看到的像素是 512 的 4 倍，
   512 只走 0.35 个 epoch；要 12 epoch 就得 847k step，是 1024 的 7.3 倍。
3. 大目标是硬约束：抽样 4,000 张图的 84,202 个实例中，宽度中位数 20 px、p99 164 px、最大
   932 px（Bridge）。**超过 512 px 的占 0.02%**——这些 Bridge 在 512 块里任何一块都放不下，
   会全部落入"<0.7 可见→忽略"，既训不到也评不出；1024 能装下 932 px。
4. 与 iSAID（T9）的 1024 保持同一裁块，两个主场任务的"主场"结论口径一致。

代价是 43% 的块有效像素不足一半，主要是 800×600 被 pad 到 1024。缓解已实现：**落在 padding
上的 anchor 与 proposal 一律不参与采样**（既不当目标也不当背景），真实像素矩形由标注里的
原图尺寸算出、并和 manifest 的 `valid_fraction` 交叉校验，翻转后矩形跟着换角
（`data.tiling.valid_rect` + `GroundTruth.valid_region`）。512 的 manifest 保留在盘上，作为
"若将来拿到官方 5,000–10,000 px 大图"时的候选。

### 4.3 存放路径与下载管理

完整 payload 存在仓库外。拟用下列部署布局，实施前检查路径不存在或归属一致，禁止覆盖已有数据：

```text
/home/yzhong/rs-eval-data/fair1m2/       # 容量更充足的卷
  downloads/                          # 原始压缩包、下载记录
  raw/{train,val,test}/                # 原始影像和标注，不改写
  derived/<preprocess_hash>/           # 转换标注、切片、tile manifest
  provenance/                         # 来源、许可、包清单、SHA-256

/geodata/rs-eval/data/
  raw/fair1m2 -> /home/yzhong/rs-eval-data/fair1m2/raw
  derived/fair1m2 -> /home/yzhong/rs-eval-data/fair1m2/derived
  manifests/<新检测任务ID>/...
```

这些符号链接是**拟实施布局**，本文没有创建。消费者需使用 `RuntimePaths` 和 registry，不把另一部署的路径写死进 worker。若下载器禁止跨根链接，则在下载器配置评审中增加受控外部卷支持，不能绕过路径校验。

正式来源应登记 `configs/downloads/catalog.yaml`，使用项目已有的下载方法和 schema；遇到网盘/人工申请型来源时，先核对下载器是否支持人工导入，必要时补设计，不能编造一个未支持的 `method`。

直接 HTTP 下载模板如下，**只在取得授权直链后**使用。URL 放在会话变量，不把登录 token、临时签名或 Cookie 提交到文档/仓库：

```bash
FAIR1M_DOWNLOAD_DIR=/home/yzhong/rs-eval-data/fair1m2/downloads
mkdir -p "$FAIR1M_DOWNLOAD_DIR"
# FAIR1M_ARCHIVE_URL、FAIR1M_ARCHIVE_NAME 从已核准的包清单取得。
curl --fail --location --continue-at - --retry 5 \
  --output "$FAIR1M_DOWNLOAD_DIR/${FAIR1M_ARCHIVE_NAME:?请设置包文件名}" \
  "${FAIR1M_ARCHIVE_URL:?请设置授权直链}"
sha256sum "$FAIR1M_DOWNLOAD_DIR/$FAIR1M_ARCHIVE_NAME"
```

下载中断后续传应先确认资源 URL 仍指向相同内容。若服务端不支持 Range，按下载器失败策略处理，不把不完整包标记完成。

### 4.4 可选 smoke 镜像下载示例

以下为 **HF 部分数据镜像** 的独立工具用法，不是全集下载，也不是现有 registry 已登记资源。下载前必须先选定完整 revision、读取文件清单并确认容量；命令采用 [HF 官方 CLI 参数](https://huggingface.co/docs/huggingface_hub/guides/cli)。

```bash
FAIR1M_SMOKE_DIR=/home/yzhong/rs-eval-data/fair1m-smoke-hf
export HF_HOME=/home/yzhong/rs-eval-cache/huggingface
export HF_HUB_DOWNLOAD_TIMEOUT=60
# 本次已从 HF API 核对的镜像 revision；下载前仍需做 dry-run。
export FAIR1M_SMOKE_REVISION=465fe09861ef08aa88d1b08680f121d00abf2bd2
/home/yzhong/.local/bin/uvx --from huggingface-hub hf download \
  blanchon/FAIR1M --repo-type dataset \
  --revision "${FAIR1M_SMOKE_REVISION:?请设置已核对的revision}" \
  --local-dir "$FAIR1M_SMOKE_DIR" --dry-run
# 只有清单/容量确认后，才去掉 --dry-run 执行下载。
```

正式集和 smoke 集使用不同 `dataset_id`、manifest 和 run 标记。镜像可能为 Parquet / 处理后格式，必须先验证其是否保留原始四点框、类别、原图 ID 和 split；无法恢复时只用于相应低级检查，不承担正式链路验收。

本次 API 文件清单核查：上述 revision 有 **1,732 个 TIFF + 1,732 个 XML**，位于 `data/images/` 和 `data/labelXmls/`，没有显式命名的 train/val/test 顶层目录。它可用于检查原始文件读入；划分身份、标签内容及全集覆盖仍需另验收，不能从文件成对存在推断为官方 2.0 完整划分。

### 4.5 数据验收门槛

- 压缩包全部通过 hash/解压完整性检查，记录实际包容量、解压容量、影像数和实例数；不预填未经实测的“全集多少 GB”。
- 原始 XML 和影像一一配对；孤立文件、坏图、坐标越界、无效多边形、重复 ID 和未知类均列明，不能静默跳过。
- 完整保留 37 类及三个 `other-*`，生成唯一的 `classes.json`、`class_aliases.json` 和各自 hash。空格/连字符可显式映射，但不能把不同类合并。
- 训练/验证/测试原图 ID 固定。检查文件 hash 及可用的来源/地理信息；官方 split 若发现重叠先披露，不暗中改成另一个划分。
- 先抽取各类及各尺寸图做可视化，再转换全量。正式任务使用原始标注和原始 RGB，不采用不可追溯的重压缩镜像。
- 查看实际数据使用条款并保存许可文本；未确认的镜像许可不得替代原始发布方条款。

## 5. 预处理、大图切片和防泄漏

### 5.1 转换及空间约定

1. 原始 XML 四边形保留一份无损记录；若以重复首点闭合，核对后去掉重复点，保留四个有效顶点。
2. 转换为 DOTA 风格四点标注或实现原生 XML loader；项目 canonical 类别 ID 顺序独立锁定，并保存原始名称到 canonical 名称的映射。
3. 使用统一的 `poly2obb` / `obb2poly` 实现转换为 `le90`。对明显非矩形四边形，记录拟合误差；最终本地评测保留原始多边形 GT，不把最小外接矩形默认为官方原始标注。
4. RGB/BGR、图像值域和各主干标准化分别写入预处理配置；不同主干可使用其必要的标准化，但不能获得额外真实波段。

### 5.2 切片建议

正式首版锁定 **1024×1024、重叠 200 像素、单尺度**，即内部步长 824；这是本项目建议，不冒称 FAIR1M 唯一官方设置。512/768 只用于 P0 显存 smoke。默认某模型在 1024 下 OOM 则标记资源阻断，不因其精度或 OOM 自动修改主协议。如另立全组 512 比较，须在该比较任何正式指标可见之前锁定触发条件、全模型范围、切片/重叠、预算及协议版本；不得看到主表结果后反向声称预注册。已有 iSAID 1024 轨道不追溯改写。

必须覆盖原图右/下边缘；小于 tile 的图整图 padding，不得丢图。单轴起点可以采用：

```python
def tile_starts(length: int, size: int = 1024, stride: int = 824) -> list[int]:
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts
```

裁片本身**不改变 GSD 或目标的像素大小**。不得把整张上万像素大图缩成 1024 来代替切片，否则会严重压缩小目标。padding 区域有独立 valid region，损失和推理都要正确屏蔽。

跨边界目标首版建议以原始多边形落入 tile 的面积比例判定：保留比例不少于 0.7 的目标；不足者进入明确的忽略区域，不能当作背景。需验证 RPN 和 RoI assigner 都处理相应 ignored GT；仅写一个 `difficulty=1` 并不保证 loader 会忽略它。阈值、忽略规则和退化框处理写入预处理 hash，并统计影响实例数。

### 5.3 划分与空样本

- 默认从经核实的官方 train 按原图/场景分组划 train/dev，再切片；所有同源 tile 必须在同一个 split。建议 dev 约 12%、种子 17，检查类别覆盖并记录按原图/tile/实例的实际比例。
- dev 用于已声明的选模/调参，成绩明确标为内部 dev。官方 val/test 若后续取得，单独锁定其身份与用途，不把 dev 当成盲测；测试集不参与调参。
- 初版保留合法空 tile，所有主干使用完全相同的 tile 列表。若需负样本下采样，须预先声明采样规则和种子，生成新的 manifest。
- 报告原图数、tile 数、空 tile 数、每类实例数、边界忽略数和像素覆盖率；覆盖率必须 100%，不能只看脚本正常退出。

每个 tile manifest 至少记录原图 ID、split、源文件 hash、tile 起点/尺寸、valid region、GSD 及来源、标注 hash、预处理 hash。当前项目的 manifest schema 能否容纳这些字段需在检测接入时扩展验收。

## 6. 模型路线与权重接入

### 6.1 统一检测结构

所有主干向检测器提供四级 NCHW 特征，经固定结构的 FPN 输出 256 通道，再进入同一个 RPN、普通 RoIAlign 和旋转框回归检测头。参考 Rotated Faster R-CNN 使用水平 proposal，不等同于 Oriented RPN，也不必使用 RoIAlignRotated。统一 anchor、采样、损失、RoIAlign 坐标约定、两个全连接层和后处理；原始输入通道不同导致 FPN 入口权重形状不同，应报告参数量而非宣称全部张量一致。`num_classes=37` 指 37 个前景类；softmax 内部背景维由框架管理，不手动把配置写成 38 类。

LIM 原生四级输出步长为 4/8/16/32，Huge 对应通道 `[256,512,1024,2048]`、深度 `[3,4,18,3]`、注意力头数 `[4,8,16,32]`。其他主干若只有单一 token 网格，需声明取层和构建金字塔的方式；上采样不会恢复已丢失的小目标细节。

| 模型 | 接入重点 | 首轮训练范围 |
| --- | --- | --- |
| LIM / MiT-Huge | 原生 RGB 和四级特征；提取预训练编码器 | 冻结编码器，仅训练统一检测器 |
| Prithvi | RGB 到必要输入波段的显式适配；固定取层与网格变换 | 检测器 + 必要输入适配器 |
| Clay | 固定 RGB 波长和 GSD 策略；抽取特征层、构建尺度 | 检测器；结构确需适配时另记 |
| AnySat | RGB 输入路径、光谱适配和 token 排列核对 | 检测器 + 必要输入适配器 |
| OlmoEarth Tiny / Base | 分别注册模型；锁定 RGB 映射和缺失波段策略 | 检测器；按真实实现报告适配器 |
| ResNet-50 | 必需的参考管线校准；可另加冻结主表基线 | 标准训练复现独立；进入冻结表须另跑冻结协议 |
| MaRS / Galileo | 见独立可行性文档，通过接入验收后扩展矩阵 | 各自输入和特征适配必须披露 |

不同模型必要的入口/金字塔适配器不一定同容量，因此“统一头”不等于完全相同的可训练参数。每个模型都要报告 backbone、input adapter、feature adapter、FPN、RPN、RoI head 的参数量和冻结状态。

多光谱主干的输入策略可参考已有 iSAID 设计，但必须重新核对实际 worker。若缺乏可靠逐图 GSD，使用预先锁定的声明值并记录 `gsd_source=declared`；不能把范围中点当测量值。某模型接入未完成时标“未实现”，不能记零分或宣称已支持。

### 6.2 LIM 权重提取与验证

- 从 checkpoint 的 `model` / `state_dict` 中识别编码器参数；排除预训练重建 decoder、mask token、优化器和 scaler。
- 已核实上游 `models_mit.py` 在阶段 1 加 `absolute_pos_embed`；当前 LIM worker 已按此实现双三次插值并相加。检测迁移保留并验证该行为，不再把“是否使用位置编码”列为开放问题。
- 固定通道、深度、注意力头数、LayerNorm epsilon 等结构参数；不能仅依赖 `strict=False` 的“未报错”。
- 输出权重加载报告：应加载参数总数、已加载数量、按元素数的覆盖率、缺失/多余键和形状不匹配列表；预期外的编码器缺失或形状不匹配必须阻断接入。
- 做四级特征尺寸验证、冻结前后参数/梯度检查、同输入重复 forward 一致性检查和单 batch 训练验证。
- 原始 6.62 GB 文件保留不改；编码器派生权重放 `weights_root` 下，以源 hash、转换脚本 commit 和输出 hash 关联。可通过受控链接复用 `/home/yzhong/models/` 文件，避免重复占用 `/geodata`。

### 6.3 参考代码中已发现的问题

以下路径均相对 `/home/yzhong/projects/rs_segmentation`；只作为待修正清单，本文没有修改该仓库。

| 位置 | 实际问题 | 检测接入应如何处理 |
| --- | --- | --- |
| `downstream/mmrotate/configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py:19` | 文件名叫 r50，实际是 MiT，通道 `[192,384,768,1536]` | 新建准确命名的 Huge 配置，连同 FPN 改为 Huge 通道 |
| 同文件 `:71` | `num_classes=34` | 恢复完整 37 类 |
| `downstream/mmrotate/mmrotate/datasets/dior.py:53` | 后一个 CLASSES 覆盖前一个，注释掉三个 `other-*` | 新建 FAIR1M dataset，显式类别映射，不继续借 DIOR 名称掩盖任务 |
| 同 dataset 的 loader | 读取 `.txt` 并拼接 `.jpg` | 需要经过验证的原始 XML/TIFF 转换，不能直接读取原始发布包 |
| `downstream/mmrotate/mmrotate/models/backbones/mit.py:298` | 硬编码加载旧 `mit_large.pth`，直接访问特定位置编码键 | 配置化路径、结构校验和加载覆盖率报告 |
| `downstream/mmrotate` 与 `downstream/mmdetection` | 前者 MMRotate 0.3.4 / MMDet 2.25.1，后者 MMDet 3.3.0 | 使用隔离 worker 环境，禁止通过 PYTHONPATH 偶然混装 |

## 7. 环境、4090 显存与算力预算

### 7.1 G0：torchvision 训练算子 + 无梯度 CPU 旋转几何

**首选架构不再要求各 worker 编译 MMCV。** 固定的 Rotated Faster R-CNN 配置使用水平 proposal、水平 IoU 匹配、普通 RoIAlign 和水平 NMS；RoI 的回归目标仍为旋转框，回归损失使用 Smooth L1。参考源码会先将 OBB GT 转为 HBB 做 proposal 匹配，再取原 OBB 作为回归目标。[配置](https://github.com/open-mmlab/mmrotate/blob/main/configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py)、[训练路径](https://github.com/open-mmlab/mmrotate/blob/main/mmrotate/models/roi_heads/rotate_standard_roi_head.py)

| 环节 | 首选实现 | 验收重点 |
| --- | --- | --- |
| 共享 FPN/RPN/RoI、旋转框编码及损失 | 普通 torch 逻辑，固定一份源代码 | anchor、采样、角度边界、target std、GT 处理和 loss 对齐 |
| 水平 IoU、NMS、普通 RoIAlign | 与 worker torch 配套的 torchvision.ops 预编译包 | 安装/CPU-CUDA 算子可用性、RoIAlign 前向及特征梯度 |
| 旋转 NMS、整图合并及多边形 IoU/AP | 首版 CPU Shapely 多边形几何；纯 torch 几何可作经验证替代 | 角度到顶点转换、退化框、阈值边界、同分排序、原始多边形 GT |
| 公开参考复现 | 独立 MMRotate/MMCV 环境 | 只用于 G1 校准，不装入所有正式 worker |

CPU 几何不在反传路径上；旋转框编解码和 Smooth L1 仍要正确实现。忽略区域、切片裁剪等标签操作也可能需要几何计算，属于无梯度数据/匹配路径，不能因简化依赖而省掉。Shapely 依赖 GEOS，通常可以使用 wheel；本方案避免自编译 torch 扩展，并不声称完全没有原生二进制依赖。[Shapely 几何接口](https://shapely.readthedocs.io/en/stable/reference/shapely.intersection.html)

仍需保留一个小而明确的 G0：

1. 检查每个 worker 的 torch/torchvision/CUDA 配套版本。torchvision 自带原生算子，只有 torch 已安装不等于这些算子可用。第二轮元数据快照：Prithvi 有 torchvision 0.29.1、Clay 有 0.19.0；AnySat、OlmoEarth、LIM 的独立环境未发现其 dist-info，不能提前写“全部可直接运行”。
2. 显式锁定 RoIAlign 的 `aligned`、`sampling_ratio`、FPN 层级分配和坐标语义，对齐参考实现，不直接照搬 torchvision 默认值；测试前向/特征梯度、AMP 下局部 FP32 规则和空 GT。[RoIAlign 文档](https://docs.pytorch.org/vision/stable/generated/torchvision.ops.roi_align.html)
3. 对 CPU 旋转几何与参考结果做一致性验证。水平 NMS 的同分选择在 CPU/GPU 间可能不同，须预声明 tie-break 与容差，不凭单例完全一致就验收。[NMS 文档](https://docs.pytorch.org/vision/stable/generated/torchvision.ops.nms.html)
4. 测大图密集预测的 CPU 合并耗时和内存；可用空间索引/AABB 不相交筛除加速，但不能把水平 NMS 当成旋转 NMS 或擅自减 max detections。将验证/合并耗时计入总预算，不能预设“慢一点一定能接受”。

共享检测代码仍与 MMRotate/MMDet registry、runner 解耦，根 CLI 不导入 torch。只有上述首选方案出现具体阻断时，才重新评审其他后端；不把逐 worker 编译 MMCV 列为默认任务。G1 精度锚点保留，因为减少依赖并不证明移植后的检测器正确。

### 7.1.1 G1：检测管线精度锚点

先用公开 R50 + Rotated Faster R-CNN checkpoint 对齐参考实现的推理，再按对应标准 1x 配置复现训练。优先采用能获得影像和检测标注的 DOTA-v1.0；已有影像不代表标注已具备。参考配置中的冻结层、BN、优化器、batch 和调度保持原样，不能用后文的冻结 probe 配方替代。

MMRotate 公布的 R50、1024/200、le90、1x 参考 mAP 为 73.40，仅适用于其匹配的 DOTA 协议。取得完全相同的评测条件时，可将训练复现差距 1–2 AP 百分点作为验收参考；它不是 FAIR1M 或本地 val 的目标。官方测试不可评时，锁定同一内部 split 的参考实现/checkpoint 成绩用于校准，明确局限。相同 checkpoint 的推理差异需要逐项定位，不以 1–2 AP 容差掩盖代码错误。[参考模型表](https://github.com/open-mmlab/mmrotate/blob/main/configs/rotated_faster_rcnn/README.md)、[参考配置](https://github.com/open-mmlab/mmrotate/blob/main/configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py)

R50 标准训练锚点独立于冻结主表；若希望主表有 R50，也应另跑冻结 R50 + 统一检测器。G0/G1 通过前，不把自研检测器低 mAP 归因于某主干能力弱。

### 7.2 首轮训练设置（建议，待 P0 锁定）

| 项 | 第一轮建议 |
| --- | --- |
| 协议 | 冻结主干的检测 probe；所有模型保持相同训练模块定义 |
| micro batch / 梯度累积 | 1 × 2，有效 batch = 2；对检测任务作硬校验 |
| 输入 | 正式 1024 单尺度 tile，固定 RGB 与预处理 |
| 精度 | 经算子验证的 AMP；必要时旋转算子局部 FP32，所有模型同规则 |
| 主干 | `requires_grad=False` 且 `eval()`，关闭训练随机性与运行统计更新 |
| FPN / RPN / RoI | 随机初始化后训练，首版不额外引入 BN；需归一化时统一采用已声明实现 |
| 初始化对照 | 固定头初始化种子；基线预训练来源与数据单列 |
| 优化器候选 | AdamW，初始 LR 1e-4、weight decay 0.05，warmup 500 个 optimizer step |
| 学习率预算 | 有效 batch 固定，按 optimizer step 记录；建议 12 epoch，8/11 epoch 衰减，仅作为待验证的初始配方 |
| 选模 | 固定开发/验证集的主指标，不能在测试集选 checkpoint |
| 种子 | 17 / 42 / 73，先声明后运行，不挑最好种子 |

有效 batch 2 是独立的检测配方建议，参考经典检测的单卡设置和资源约束，与 iSAID 的有效 batch 8 不是同一任务协议。1×2 梯度累积也不自动等价于真实 micro batch 2；R50 标准复现遵循其原配置。

候选超参不是实测最优。P0 可用工程 smoke 排除不稳定配置；正式阶段每个模型使用相同数量的验证搜索机会、相同 split 和可比预算，最终配置及选择规则在查看测试结果前锁定。

全量微调放到第二阶段，单独协议/结果表。Huge 的 24 GB 显存可行性目前未实测；若 OOM，先验证 AMP、激活检查点、micro batch 1，再评估部分解冻或更大显存。梯度累积不会降低单个 micro batch 的激活显存，部分解冻也不能冒称全量微调。

必要输入适配器有梯度时，冻结主干仍可能需要保留主干对输入的反向计算图；不得统一套 `no_grad()` 切断适配器梯度。因此每个主干都需单独测冻结配置的显存和吞吐，不能按 LIM 的冻结成本推算全组。

### 7.3 资源门槛与耗时计算

1. 对每个主干测至少 100 个预热 step 后的 200 个稳定训练 step，分别测验证推理，记录峰值 allocated/reserved 显存、每个 optimizer step 时间、原图/tile 吞吐和数据等待。
2. 以 4090 独占运行测量，当前剩余显存只是快照；不要与已有 iSAID 队列并发抢卡。
3. 训练时间估算：`ceil(N_train_tiles / effective_batch) × epochs × 秒/optimizer_step`，再加实测验证、合并、checkpoint I/O 和失败重试成本。
4. 正式排期同时计算全模型 × 三种子成本；没有实际 tile 数和 step 测量前，不承诺“几小时/几天跑完”。
5. 从首批原图测解压比、切片膨胀率和 checkpoint 大小，再估算全量存储；下载、原始包、解压图、切片、缓存、多个模型多种子 checkpoint 均计入。每个卷预留至少 50 GB 且不低于该卷容量 10% 的运行余量。
6. 冻结特征缓存先不默认启用。1024 的稠密多尺度缓存可能很大；启用时单独设计键、容量和增强一致性，并计入预计算成本。

## 8. 实验矩阵与公平性

| 阶段 | 实验 | 准入 |
| --- | --- | --- |
| P0 | G0 后端架构、G1 参考推理及标准训练复现；数据审计、各主干 smoke | 参考校准单列；smoke 不作正式精度结论 |
| P1 | LIM/R50 冻结开发验证，扩展六个既有模型配置；建议在 iSAID 首批 B2U 后排 GPU | G0/G1 已通过；未执行完整预算标 `REDUCED-BUDGET` |
| P2 | 接入现有所有选定主干，固定 37 类、正式数据/预算/三种子 | 满足新任务准入规则后进入冻结检测表 |
| P3 | 可选全量微调、同架构预训练/随机初始化、少标注 | 独立 panel，另评审算力 |

随机初始化对照必须与同架构预训练成对比较。冻结随机特征、随机初始化后全量训练、预训练后全量微调回答的问题不同，不混成“预训练收益”一个数字。

必须固定：划分、样本列表、类别顺序、切片/合并方式、有效 batch、监督量、统一检测头、种子和验证搜索预算。模型特有的 RGB 标准化、波段填充、GSD 声明、特征抽层及金字塔适配要显式记录，并报告实际输入信息量及可训练参数。

LIM 的原生高分辨率金字塔与 ViT 单尺度 token 的派生金字塔存在结构差异；统一 FPN 只能控制检测器的一部分变量，不能消除所有空间细节差异。不要把插值出的 /4 网格表述为原生 /4 信息。

### 8.1 新候选与单卡预算边界

既有配置为 LIM、AnySat、Clay、Prithvi、OlmoEarth Tiny/Base 共 6 个；加 MaRS/Galileo 后最多 8 个正式主干配置。全部通过门槛后的检测三种子主表为 **8×3=24 个 run**，另计 R50 参考推理/标准训练校准。只有另行决定增加冻结 R50 主表基线，才再加 3 个 run 达到 27 个。R50 锚点不自动复制为三个种子的主表实验，iSAID 接入试验、smoke 和后处理成本也不能漏计。

顺序为新候选来源/输入审计 → iSAID B2U smoke → 预声明种子 17 的完整预算首轮 → 资源排期 → 检测接入及三种子展开。小规模阶段只能报告单种子结果，不能挑最好 seed；Galileo 资源门槛不通过则保留候选并标阻断，不能替换成 Nano 冒充 Base。参见[新模型说明 §6–8](../models/MaRS与Galileo接入评测可行性_2026-10-29.md)，其中统一列出全部主干的 GSD / 步长及 home/away。

## 9. 指标、合并与结果准入

### 9.1 本地主指标与官方分数

本地主指标已锁定为 **COCO 风格 37 类旋转框 AP**（评审决策 5：COCO 101 点，需要可比），
实现在 `src/rs_backbone_eval/detection/ap.py`：`map`（IoU 0.5:0.05:0.95 平均）、`map_50`、
`map_75`、每类 `ap50/<类名>`，以及按 GT 面积分档的 `recall_small/medium/large`
（边界沿用 COCO 的 32²/96² px²）。匹配按分数贪心、用**多边形** IoU；被裁切的忽略目标既不算
命中也不算误报；无 GT 的类别从均值中剔除并显式报告 `classes_with_targets`，不静默改分母。
与 pycocotools 逐位一致尚未验证，因此结果仍标“本地验证指标”，不宣称等价官方分数。

**同时必报五大类 mAP 和尺寸分层召回**，以及每类 AP/GT 数、峰值显存、训练/推理吞吐、端到端时间、可训练参数和失败次数；AP75 可选。五大类指标由同一份 37 类预测和 GT 映射大类、按锁定规则重新 NMS 并重新计算，不能平均细类 AP 代替。类别不可知是一类检测的另一指标，不叫五大类。尺寸按原图 GT 多边形像素面积分档，预先锁定边界、IoU 和 max detections，报告每档支持数。另训练五类 head 属于独立实验。GT 为零的类别必须按锁定评价器的明确规则处理，报告缺失类别，禁止静默更换 mAP 分母来冒充完整 37 类结果。

### 9.2 原图级评测

验证/测试先对 tile 推理，再按 manifest 的原图 ID 和偏移恢复全图坐标、过滤 padding 区域、用统一的旋转/多边形 NMS 合并重复检测。主结果在**原始整图 GT**上评测，不能仅报告切片级 AP，因为重叠切片会重复计算目标。

合并阈值、单 tile 和整图最大检测数、score threshold、类别无关/相关 NMS、极大图处理及 TTA 全部锁定；正式首版关闭多尺度/旋转 TTA。测试集没有公开 GT 时生成该版本要求的官方提交格式；在取得提交反馈前不填“官方测试 mAP”。

### 9.3 账本与复现字段

只通过项目 `dispatch` 创建正式 run，worker 使用 `RunWriter` 管理生命周期，不手写结果账本。除现有通用字段外，实施阶段需评审检测任务所需的元数据/产物字段：

- dataset 版本、完整/子集身份、原图及 tile manifest hash、类别映射和预处理 hash；
- checkpoint 源 hash、实际训练轮次 metadata、编码器提取/加载报告；
- backbone / adapter / FPN / RPN / RoI 的训练状态及参数量；
- `head_id`、后端版本、协议、有效 batch、精度、输入路径、GSD 来源和特征步长；
- 原图级预测及 evaluator 版本/hash、合并配置、各类 AP/GT 数和评测日志；
- 预处理、缓存、训练、验证、合并、测试提交成本分别计量。

三种子完整报告均值和标准差，逐种子原始值保留。同一 seed 重跑不是新观测；只有 `completed`、完整项目锁定预算且未被排除的运行进入正式检测表。smoke/短跑不进入 headline。预算准入必须在正式比较前实现：task/protocol 声明 epoch 或 optimizer-step 预算；未知、零值、未注册预算默认不准入，epoch 制核对完成量与 step 证据，step 制核对真实 optimizer steps。不能仅靠 notes 中 SMOKE 字样。第一轮只检查控制层，漏查 LIM worker 内部 `OFFICIAL_EPOCHS=50` 会把完成 25,000 step 的运行写成 REDUCED-BUDGET。共享预算函数已在 `94ac560` 提交，`9b9826c` 记录隔离与 v2 重跑。随后 `11bafd4` 增加 `OFFICIAL_STEPS={T9:25000}` 权威准入，拒绝 400/400 自报短跑、零/缺失完成量及未知任务；worker 的停止预算只作审计，不能自行认定正式标准。该部分代码回归已通过；检测任务自身预算注册仍待实施。详见第二轮审查响应。新检测任务与现有分割任务/D1 分开显示，不用检测 AP 和分割 IoU 直接做无说明的总平均。

**T10 预算当前故意不注册**（决策 4）：`matrix.OFFICIAL_STEPS` 里没有 T10，于是
`stopped_short_of_budget` 对任何 T10 run 都返回"未达预算"，检测结果无法进入 headline。
这是有意的 fail-closed：检测吞吐尚未实测，先写一个数字等于用猜测验收自己。实测后按
"每 step 秒数 + 目标像素量"折算成正式 step 预算再注册，并在同一提交里加回归。

## 10. 接入 rs-backbone-eval 的实施清单

下表在 v3 时全部是待实现项；2026-10-29 已落地第一列，**第二列写明仍未完成的部分**。
task ID 已注册为 `T10`（`configs/tasks/T10_fair1m.yaml`），轨道为 `B2D`/`B1D`。

| 层 | 已落地（2026-10-29） | 仍未完成 |
| --- | --- | --- |
| 数据/下载 | `data_fair1m_train` catalog 条目（含本地 SHA-256）；`scripts/fetch_fair1m.sh` 复现下载 + 压缩包完整性 + 1:1 配对检查；`data/datasets/fair1m.py` builder | 官方 2.0 val/test 渠道仍是阻断（本协议不需要）；人工导入型来源未设计 |
| task schema | `TaskSpec.family` 增加 `detection`；`ManifestSample` 改为 `label_path` / `annotation_path` 二选一，并记录 `instance_count` / `ignored_instance_count` | 报告/Portal 的检测专属区块 |
| model 注册 | 复用已注册模型；`workers/lim` 已接 T10（`t10.py` + runner 路由 + 每任务协议白名单） | TorchVision 环境同步与 GPU smoke；其余 5 个主干的检测适配 |
| protocol | `rotated_frozen`（B2D）/ `rotated_finetune`（B1D），`matrix` 轨道与 variant 分组已加 | T10 的**预算未注册**（fail-closed，见 §9.3） |
| 头与适配 | `detection/{fpn,anchors,detector}.py`：FPN + Rotated Faster R-CNN，水平 proposal/RoIAlign + 旋转回归，无 MMCV | 各主干特征契约适配（Clay/Galileo 需派生金字塔） |
| 训练与评测 | `Fair1mTiles` 数据集、忽略目标同时作用于 RPN 与 RoI 采样、`detection/{geometry,ap,nms}.py`、整图合并 | GPU 吞吐/显存实测；G1 的 R50 校准 |
| 聚合/Portal | mAP 走既有 `MetricRecord`（方向 maximize），per-class 用 `ap50/<类名>` | Portal 的检测视图、五大类与尺寸召回的展示 |
| 队列 | — | `rerun_baselines.sh` 新增检测 lane |
| 回归 | `tests/test_{fair1m_manifest,detection_geometry,detection_ap,detection_model}.py`；registry/catalog/Portal 计数同步 | 检测 smoke 后的真实数值回归 |

LIM/UPerNet 相关改动已陆续提交，现场仍有活动 LIM 开发。本次只写文档，不迁移或中断正在使用 checkout 的 lane，不修改 run.json。后续检测开发使用独立 worktree，正式 run 使用干净且固定的代码版本；指标含义变更时同步审计准入。诊断区已有 dirty run 保留原始证据，不直接推断污染正式表。

建议新增配置名 `fair1m_obb_frozen` / `fair1m_obb_finetune` 及 `rotated_faster_rcnn_fpn37_v1_b2` 头身份，均属提案。由于 variant 分组未必包含 batch/backend/preprocess，必须硬校验这些协议常量；一旦改变，更新协议/头版本和分组规则，不让不同设置被平均。

## 11. 分阶段交付与评审验收

### P0：先解除阻断

- [x] **G0 后端架构已落地**：torchvision 算子 + CPU 旋转几何，检测器在 CPU 上完成一步训练
  （loss 有限、梯度只落在检测头）；MMRotate 只保留为 G1 参考，不装进任何 worker。
- [ ] **通过 G1 参考推理及 R50 标准训练校准**；同 split 对照，不能挪用公开测试分数。
- [x] 取得 train 数据并按 §4.2.1 验收：14,756 图 / 311,682 实例 / 37 类；内部 dev 已按
  原图分组划出（seed 17，0 泄漏）。**注**：来源为镜像，且影像是已裁好的图块，见 §4.2.2。
- [x] 双卷已落实：原始 48.6 GB 在 `/home`，`/geodata` 仅放符号链接和 manifest。
- [x] CPU 几何精度（多边形裁剪/IoU/le90 往返）与旋转 NMS 已通过单测；torchvision
  `roi_align` 路径在 worker 环境通过 CPU 前向。**待办**：密集整图合并耗时实测。
- [ ] 导出 LIM checkpoint metadata/编码器加载报告，完成四级特征契约检查（T9 已有 573 键全中，
  T10 复用同一加载器，待 GPU smoke 记录）。
- [x] XML/TIFF 转换与 37 类映射通过；1024/824 切块、忽略 GT 生效（3,073 次忽略分配）、
  按原图分组、`audit-data` 零错误。**待办**：抽样可视化。
- [ ] 逐主干通过冻结 smoke 的梯度/显存/吞吐门槛；测出单模型及全组预算（4090 正在跑 iSAID
  v3，检测 smoke 排在其后）。

### P1/P2：形成正式比较

- [ ] iSAID 首批 B2U 后评审检测 GPU 预算；锁定统一 Rotated Faster R-CNN、输入/适配、类别顺序、种子和验证搜索规则。
- [ ] 正式结果产生前修复 epoch/step 准入并回归 matrix/报告/Portal；37 类、五大类及尺寸召回同时验收。
- [ ] 所有拟纳入比较的模型跑相同正式协议；未实现/OOM/失败单列，不补零。
- [ ] 整图合并和主评价器对齐，通过每类 AP/GT 数与可视化复核。
- [ ] registry、worker、账本、报告和 Portal 的检测准入同步验收。
- [ ] 运行仓库要求的 lint、CPU 回归与 hygiene；GPU smoke/正式结果单独作为运行产物记录。

### 本版 review 需要给出的结论

1. 确认 torchvision + CPU 几何为 G0 首选，MMCV 限于参考复现，保留 G1 精度锚点。
2. 第一轮冻结、全量微调后置；MaRS/Galileo 先 iSAID B2U 再检测，R50 校准成本单列。
3. 官方 train 数据获取由谁落实；是否接受固定内部 train/dev 为默认比较，入口不可用时仅使用命名明确的子集 smoke。
4. P0 显存/吞吐测量完成后，是否接受 12 epoch × 三种子的实际算力预算；不预先承诺全组完成时间。
5. FAIR1M 相对既定 iSAID/T1S/T8 队列的优先级。本文不修改已有排期决定。

## 12. 参考资料与核验说明

- [FAIR1M 官方 benchmark](https://www.gaofen-challenge.com/benchmark)：任务、版本差异及官方数据入口；本次直接页面访问不稳定，未核实完整包直链。
- [FAIR1M 论文](https://arxiv.org/abs/2103.05569)：原始任务与数据集说明，不能单独证明任一镜像是完整 2.0。
- [TorchGeo FAIR1M](https://docs.torchgeo.org/en/stable/api/datasets/fair1m.html)：影像/标注格式和类别参考。
- [JDet FAIR 数据说明](https://github.com/Jittor/JDet/blob/master/docs/fair.md)：大图转换、切片和合并的实现参考；不直接采用其合并 train/val 的训练设置。
- [MMRotate 基础说明](https://mmrotate.readthedocs.io/en/latest/intro.html)：旋转框表示、方向、算子与评价。
- [MMRotate 安装说明](https://mmrotate.readthedocs.io/en/latest/install.html)：依赖边界参考；本机兼容版本必须另验收。
- [Hugging Face FAIR1M 镜像数据卡](https://huggingface.co/datasets/blanchon/FAIR1M)：部分数据身份说明，非完整 2.0 背书。
- [HF CLI 下载文档](https://huggingface.co/docs/huggingface_hub/guides/cli)：revision、本地目录和 dry-run 参数。
- [本仓库 iSAID 与 LIM 方案](iSAID与LIM评测方案_2026-10-29.md)：既定排期、LIM metadata 及输入适配背景。
- [自研骨干接入计划](../models/自研骨干接入现有评测计划.md)：冻结/微调分轨、可比性和结果解释边界。

本文件是一份可实施的设计与评审清单。下载命令示例不代表下载已经执行；候选训练配方、后端兼容性、正式算力和模型精度均须经过上述 gate 后才能作为已验证结论。
