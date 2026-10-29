# MaRS 与 Galileo 接入评测可行性

日期：2026-10-29  
版本：v2，第二轮审查修订（污染审计、iSAID 优先和预算边界）。  
状态：供上级 review；已核查官方代码、配置与权重清单，尚未下载权重、创建 worker 或运行 GPU 验证。  
目标：在 `rs-backbone-eval` 中增加两个主干候选，重点评估 **各主干 + 统一 Faster R-CNN 检测器**，并说明其对现有任务的适用范围。

## 1. 建议结论

| 候选 | 首选版本 | 判断 | 最先验证的事项 |
| --- | --- | --- | --- |
| MaRS | Base RGB encoder | 工程适配较直接，建议优先接入 | 权重完整加载、RGB 标准化、四级特征及 1024 输入 |
| Galileo | Base 为目标；Nano/Tiny 用于低成本验证 | 有条件可行，不能仅凭可加载权重就承诺 4090 正式检测 | RGB 缺失模态 mask、辐射归一化、稠密 token 恢复及注意力显存 |

两者分别作为独立 backbone，不互相拼接，也不与 LIM 或 ResNet-50 融合。FAIR1M 使用统一 **Rotated Faster R-CNN** 保留旋转框；普通水平框 Faster R-CNN 需另建 HBB 任务。共享结构包含 FPN、RPN、RoI head、框编码和后处理，不能每个主干选不同检测器。

MaRS 在高分辨率 RGB 任务上标为 **home（输入域主场）**，预训练与评测影像重叠标为 **unknown / 未排除**；Galileo 的 S2 RGB 入口在该场景标为 **away**。home/away 与是否见过评测图是两条独立轴；客场也不等于无污染。这里的 home/away 是输入域与结构适配判断，**不是精度排名或显存实测结论**。既有 LIM、AnySat、Clay、Prithvi、OlmoEarth Tiny/Base 比较范围保持；新增两个候选通过准入后再进入同一协议矩阵。

检测公共后端及复现锚点见 [FAIR1M 主方案](../design/FAIR1M-2.0旋转目标检测评测方案_2026-10-29.md)和[审查响应](../reviews/审查响应_2026-10-29_FAIR1M方案_第1轮.md)。本文件独立讨论两个新模型，不表示公共检测后端已实现。新增 MaRS、Galileo 是用户此前明确转达的上级要求，候选来源不是本次自行扩项；研究与分阶段接入不等于一次性启动全部模型、任务和三种子。**两者先走已具备 UPerNet/T9 链路的 iSAID B2U，再接检测**；iSAID 接入不依赖检测 G0/G1 完成。

## 2. 模型身份与来源

| 项目 | MaRS | Galileo |
| --- | --- | --- |
| 本文具体指代 | 武汉大学团队的多模态遥感 MaRS | NASA Harvest 团队的 Galileo |
| 官方代码 | [WanderRainy/MaRS](https://github.com/WanderRainy/MaRS) | [nasaharvest/galileo](https://github.com/nasaharvest/galileo) |
| 论文/项目 | [官方项目页](https://rsidea.whu.edu.cn/mars.htm)、[论文](https://ojs.aaai.org/index.php/AAAI/article/view/38153/42115) | [论文](https://arxiv.org/abs/2502.09356) |
| 官方权重 | [Zenodo 17800805](https://zenodo.org/records/17800805) | [Hugging Face nasaharvest/galileo](https://huggingface.co/nasaharvest/galileo) |
| 本次代码快照 | `7ce79aea78c228db9b292e0cb5751fd42cc69c4e` | `0f0b5b95ac81acef4b74cf4686dc877202f4541b` |

MaRS 存在同名项目，不能误下载其他领域的 MarS。下载和实现时固定完整 revision，保留上游代码许可及权重来源。若后续采用更新版本，重新核对接口和权重，不把本文快照当成永远有效的描述。

## 3. MaRS：优先接 RGB 编码器

### 3.1 结构与特征接口

官方发布基于 SwinV2 的 RGB 与 SAR 编码器，并提供单独的 RGB encoder-only 权重。FAIR1M/iSAID 只有 RGB 时，应使用 RGB 分支，不需要制造 SAR 数据，也不需要下载预训练全集。[官方使用示例](https://github.com/WanderRainy/MaRS/blob/7ce79aea78c228db9b292e0cb5751fd42cc69c4e/README.md)

官方示例的模型构造为：

```python
import timm

encoder = timm.create_model(
    "swinv2_base_window8_256",
    pretrained=False,
    features_only=True,
    in_chans=3,
    img_size=512,
    checkpoint_path="mars_base_rgb_encoder_only.pth",
)
```

这段仅为上游加载参考，不是已实现的项目 worker。Base 预期四级通道为 128/256/512/1024、步长为 4/8/16/32；实现时必须读取 `feature_info` 并实际打印尺寸确认。尤其核对 timm SwinV2 的输出 layout，若为 NHWC，显式转为检测契约的 NCHW。

正式 FAIR1M 建议使用 1024 tile，因此还需测试固定窗口、padding 和 `img_size` 配置从示例 512 改到 1024 的行为。不能为了模型容易运行，把该模型单独缩到 256/512 后与其他模型的 1024 结果直接合表。

### 3.2 归一化与权重转换

上游数据代码将 RGB 读为浮点像素值，使用 mean `[87.01, 91.52, 83.51]`、std `[62.66, 54.58, 53.11]`。浮点输入的 `ToTensor()` 不会像 uint8 输入一样自动除以 255；移植时要核对值域，避免额外除以 255 或直接套 ImageNet 均值方差。[训练数据处理代码](https://github.com/WanderRainy/MaRS/blob/7ce79aea78c228db9b292e0cb5751fd42cc69c4e/mars_dataset.py)

上述是代码快照中的训练处理依据；还需在选定 checkpoint 上验证官方加载路径与项目适配器的同输入特征一致性。上游转换脚本处理 `encoder_rgb.encoder.` 前缀及层名；已发布的 encoder-only 文件按官方说明已转换，不应未经检查再转换一次。[转换脚本](https://github.com/WanderRainy/MaRS/blob/7ce79aea78c228db9b292e0cb5751fd42cc69c4e/utils/convert_mars_checkpoints_to_swin.py)

### 3.3 环境和许可

官方示例采用 Python 3.11.13、torch 2.7.0、timm 1.0.15；这是可参考的环境，不等于这些版本均为不可改变的最低/最高限制。建议独立 `workers/mars` 环境，优先固定 timm 版本和权重加载语义，再验证公共检测算子，不改动其他 worker 或根环境。

代码 README 声明严格学术研究用途，而 Zenodo 记录的许可元数据为 CC-BY-4.0。两处表述不完全一致，应分别保存并确认适用范围，不能用权重页面的许可替代代码条款或自行推断可商用。[代码说明](https://github.com/WanderRainy/MaRS)、[发布记录元数据](https://zenodo.org/api/records/17800805)

### 3.4 接入前必查：预训练污染与 home/away

沿用 [iSAID / LIM 方案 §6](../design/iSAID与LIM评测方案_2026-10-29.md) 的口径：**输入域主场不等于已证实污染；没有找到评测集名称也不等于已排除污染。** 对 MaRS/iSAID、MaRS/FAIR1M 分别建记录，不能以一个总标签替代。

第二轮已核查的公开证据：

- 官方论文描述 MaRS-16M 由 Umbra/Capella SAR 及对应高分光学影像构成，约 0.35 m；光学分支继续预训练也指向该多模态数据。本文未找到直接使用 DOTA、FAIR1M 或 Million-AID 的明确声明，不能据此推断未用过或没有同源图。[论文数据与方法部分](https://ojs.aaai.org/index.php/AAAI/article/view/38153/42115)
- 官方仓库称完整语料尚未公开提供，现有公开材料不足以逐图核对重叠。[数据说明](https://github.com/WanderRainy/MaRS#2-data-preparation)
- 公开训练代码中 RGB/SAR 编码器及 teacher 使用 `pretrained=True` 构造 timm 模型。因此来源审计还必须追溯初始化及 teacher 权重，不能仅审 MaRS-16M。此代码路径不证明发布 checkpoint 的完整训练血缘，需对应的训练记录或作者说明。[模型构造代码](https://github.com/WanderRainy/MaRS/blob/7ce79aea78c228db9b292e0cb5751fd42cc69c4e/models/MaRS.py)

接入清单：

1. 记录具体 checkpoint 的所有阶段：初始化、teacher、持续预训练、是否用下游影像/标签。单列 DOTA、iSAID、FAIR1M、Million-AID 的使用/未使用/未知证据，区分“用过 Million-AID”与“已证实重叠 DOTA”。
2. 能取得清单时，按源影像 ID、文件 hash、地理足迹及采集时间交叉核对；重采样、裁片、转码图还需近重复检测和人工复核。hash 不同不能单独排除相同场景。
3. iSAID 复用 DOTA 影像，审计粒度必须到原图/场景，不能只比 iSAID 标注文件；FAIR1M 同样在切片之前检查。所有证据留来源、版本、日期和覆盖范围。
4. 尚无完整清单时，状态记 `home / overlap=unknown`，允许作披露局限的工程/能力验证；**不进入声称严格未见图/区域泛化的结论**。若确认评测原图参与预训练，另列“见过影像”的结果，监督标签参与则再单独披露，不能只标普通 home。
5. 若需严格无污染证据，使用有排除记录的 checkpoint 或新的独立场景评测；作者确认也是来源证据，不替代可用清单。是否联系作者由负责人执行，本次未对外发送消息。

当前结论是“污染未排除”，不是“已经发现污染”。相同审计规则也适用于 LIM、Galileo 和其余主干。

## 4. Galileo：能取空间特征，但 RGB 接入需要专门适配

### 4.1 版本和依赖选择

官方 Base 配置为 embedding 768、12 层、12 heads；Tiny 为 192、12 层、3 heads；Nano 为 128、4 层、8 heads。正式候选优先讨论 Base，Nano/Tiny 可用来验证链路；若最终比较 Tiny，必须独立 model ID，不能记成 Base 的结果。[官方配置目录](https://huggingface.co/nasaharvest/galileo/tree/f039dd5dde966a931baeda47eb680fa89b253e4e/models)

完整上游项目声明 Python >=3.10,<3.12，固定 torch 2.2.1、torchvision 0.17.1 等依赖；建议独立 worker。上游另有单文件实现，可避免把下载/训练工具的全部依赖带入检测环境，但迁移到较新 torch 仍需与参考输出对齐。[依赖声明](https://github.com/nasaharvest/galileo/blob/0f0b5b95ac81acef4b74cf4686dc877202f4541b/pyproject.toml)、[单文件实现](https://github.com/nasaharvest/galileo/blob/0f0b5b95ac81acef4b74cf4686dc877202f4541b/single_file_galileo.py)

代码及模型页标注 MIT；复制单文件实现时保留版权、许可证和上游 commit，不把第三方来源隐藏为项目原创。[代码许可](https://github.com/nasaharvest/galileo/blob/0f0b5b95ac81acef4b74cf4686dc877202f4541b/LICENSE)、[模型页](https://huggingface.co/nasaharvest/galileo)

### 4.2 RGB 输入与 mask 是首要正确性风险

Galileo 的 `S2_RGB` 组顺序为 B2/B3/B4，即蓝/绿/红；项目 RGB 输入必须映射 R→B4、G→B3、B→B2。单时相输入设 T=1，RGB 组可见；其余波段组和模态标为不可见，不能把补零的 NIR/SAR 当成真实观测。

特别注意：对 `construct_galileo_input` 传入完整 S2 张量时，辅助函数会将 S2 各组设为可见。简单地把不存在的波段补零再调用这个函数，会错误暴露缺失组。必须显式覆盖 mask，并测试改变被屏蔽位置的数值不会改变有效输出。[输入构造与波段分组](https://github.com/nasaharvest/galileo/blob/0f0b5b95ac81acef4b74cf4686dc877202f4541b/src/data/utils.py)

FAIR1M 的 8-bit RGB 不是 S2 物理反射率，不能直接套用 S2 统计量，也不能将像素乘一个常数就声称恢复了反射率。建议先评审 RGB 适配策略，例如仅用训练集计算的 RGB 标准化，并通过不重复归一化的输入路径送入模型；这属于明确标记的 RGB 域适配版本，需要用固定验证预算选择，不能称为官方原生 S2 输入。

GSD 和月份也应记录来源：输入接口默认分辨率为 10 m，不能用于高分影像却不作说明；有可靠 GSD 用真实值，没有则锁定声明值。未知月份使用固定占位值并披露，上游辅助函数默认值 5 不能被当作真实拍摄月份。

### 4.3 保留空间网格，显式构建特征金字塔

检测需要空间特征，不使用全局平均池化后的向量。保留 RGB 组、单时相的空间 token，按输出索引和 mask 恢复二维网格，再接已声明的特征适配器与统一 FPN。

Galileo 不是原生四阶段 Swin。可评审最后一层网格生成多尺度特征，或选定中间层；两者都要固定结构、训练状态和参数量。插值得到 /4 网格不能称为原生 /4 分辨率。若使用中间层 hook，必须验证 token 顺序、被移除位置和恢复逻辑，不能直接 reshape 失序的 token。[编码器实现](https://github.com/nasaharvest/galileo/blob/0f0b5b95ac81acef4b74cf4686dc877202f4541b/src/galileo.py)

### 4.4 4090 上最大的风险：高分辨率注意力

官方配置最大 patch size 为 8。只考虑 1024×1024、T=1、一个可见 RGB 组，patch=8 已有 16,384 个空间 token。单文件实现的注意力路径会将 mask 按 heads 和 query 长度 repeat，产生二次方大小的布尔张量。

对 Base 的 12 heads，`12 × 16384²` 个 bool 约为 **3 GiB**；patch=4 时 65,536 tokens，仅该 mask 就约 **48 GiB**。这是依据形状的静态估计，不是峰值显存测量，还没包括编码器其余张量、FPN 和检测头。启用 SDPA/Flash Attention 本身不保证消除显式 repeat 的开销。

P0 应先以 Nano/Tiny 小输入验证语义，再测 Base 的 512 和 1024。可以研究广播 mask 或无 mask 快路径，但必须证明与参考实现一致，尤其覆盖 padding/部分缺失情况。不得偷偷把 patch 改到未验证的 >8，或只给 Galileo 降低正式输入尺寸。若 1024 仍不可行，报告硬件/接入阻断，或另开全模型统一 512 的协议；不记零分。[单文件注意力代码](https://github.com/nasaharvest/galileo/blob/0f0b5b95ac81acef4b74cf4686dc877202f4541b/single_file_galileo.py)

## 5. 权重下载清单与复核

仅下载下游编码器及其配置，无需预训练数据和重建 decoder。下列大小来自官方发布 API 的文件元数据，**不是本机已完成下载的记录**。

| 文件 | 字节数 | 约 MiB | 用途 |
| --- | ---: | ---: | --- |
| MaRS Base RGB encoder-only | 347,703,439 | 331.6 | 建议首选 |
| MaRS Large RGB encoder-only | 634,126,336 | 604.8 | 可选扩展，另立模型配置 |
| Galileo Base encoder.pt | 346,177,246 | 330.1 | 正式目标版本 |
| Galileo Tiny encoder.pt | 22,906,462 | 21.8 | 低成本验证/独立规模对照 |
| Galileo Nano encoder.pt | 4,189,406 | 4.0 | 最小链路验证 |

MaRS Base + Galileo Base 的编码器合计约 662 MiB，不是主要存储压力；计算瓶颈主要来自输入分辨率、特征和检测训练。Galileo 每个版本另需数 KB 的 `config.json`。[MaRS 文件清单](https://zenodo.org/api/records/17800805)、[Galileo 固定版本](https://huggingface.co/nasaharvest/galileo/tree/f039dd5dde966a931baeda47eb680fa89b253e4e/models)

### 5.1 MaRS Base RGB

部署权重根目录为 `/geodata/rs-eval/weights`；正式接入应登记 `configs/downloads/catalog.yaml` 并遵循现有下载器 schema。以下是可复核的原始下载示例，本次未执行，不要覆盖已有文件：

```bash
MARS_WEIGHT_DIR=/geodata/rs-eval/weights/mars/base-rgb
mkdir -p "$MARS_WEIGHT_DIR"
curl --fail --location --retry 5 --continue-at - \
  --output "$MARS_WEIGHT_DIR/mars_base_rgb_encoder_only.pth.part" \
  https://zenodo.org/api/records/17800805/files/mars_base_rgb_encoder_only.pth/content
printf '%s  %s\n' '5ab1b978f783cbded01427c50b0c0dbb' \
  "$MARS_WEIGHT_DIR/mars_base_rgb_encoder_only.pth.part" | md5sum --check -
# 校验成功且目标文件不存在时，再将 .part 提升为正式文件；失败则停止。
sha256sum "$MARS_WEIGHT_DIR/mars_base_rgb_encoder_only.pth.part"
```

发布方提供的是 MD5；另外保存本地 SHA-256、来源 URL、记录号、许可与下载时间。MD5 用于核对发布元数据，不能替代可信来源验证。

### 5.2 Galileo Base

固定权重 revision：`f039dd5dde966a931baeda47eb680fa89b253e4e`。上游默认加载的是 `encoder.pt`，不是 `target_encoder.pt`；不能未经说明换成 EMA 编码器。[加载实现](https://github.com/nasaharvest/galileo/blob/0f0b5b95ac81acef4b74cf4686dc877202f4541b/single_file_galileo.py)

```bash
GALILEO_WEIGHT_DIR=/geodata/rs-eval/weights/galileo/base
GALILEO_REVISION=f039dd5dde966a931baeda47eb680fa89b253e4e
mkdir -p "$GALILEO_WEIGHT_DIR"
curl --fail --location --retry 5 --continue-at - \
  --output "$GALILEO_WEIGHT_DIR/encoder.pt.part" \
  "https://huggingface.co/nasaharvest/galileo/resolve/$GALILEO_REVISION/models/base/encoder.pt"
printf '%s  %s\n' \
  '935f4912f08483d052d69d8e96518f52ef6827fe718a435cb5536fd9bb7737a7' \
  "$GALILEO_WEIGHT_DIR/encoder.pt.part" | sha256sum --check -
curl --fail --location --retry 5 \
  --output "$GALILEO_WEIGHT_DIR/config.json.part" \
  "https://huggingface.co/nasaharvest/galileo/resolve/$GALILEO_REVISION/models/base/config.json"
# 两个文件验收通过且目标不存在时，再去除 .part；记录配置的本地 SHA-256。
```

上表另外两个 encoder 的发布 SHA-256 为：

- Tiny：`b0d7172a9bac1cdaf5f3294abf3797b3239a7fdde2747aa666049a1509785ebc`
- Nano：`eab6acacf527886a508a555bc24efe4b4a2cf91b162e84cd1c4a1807e942180c`

不要只修改下载 URL 中的版本名称而沿用 Base 的配置或 checksum。配置、代码和权重应作为一套来源记录。

## 6. 与现有任务的关系

| 任务输入 | MaRS | Galileo |
| --- | --- | --- |
| FAIR1M / iSAID 高清 RGB | RGB 分支、四级特征适配直接；检测/分割头仍需实现 | 可构造仅 RGB 的输入，但存在领域和空间尺度差异 |
| 原生多光谱 / S2 / 时序 | 不能把 RGB 编码器说成原生多光谱时序模型 | 值得另外评估，需逐任务核对真实波段和时间信息 |
| SAR 任务 | 有独立 SAR 权重，但单通道高分 SAR 不等于可直接接入双通道 VV/VH | 支持相应模态输入仍不等于现有 SAR 任务已接通 |

FAIR1M 回答的是高清 RGB 检测能力，不能代表 Galileo 的全部遥感能力，也不能因为 MaRS 在相近输入域上有优势就推断通用架构更强。新增模型只在双方均完成相同任务与协议时比较；不能把尚未实现的任务记零或直接改变原有 D1 的含义。

### 6.1 统一的隐含 GSD / 原生步长表

下表针对本项目拟采用的 RGB 入口；预训练影像 GSD、传入元数据和特征步长分别记录。既有模型路径参考 [iSAID 方案 §2.4](../design/iSAID与LIM评测方案_2026-10-29.md)，实际接入须用特征尺寸验证，不能把派生金字塔的 /4 当原生细节。

| 主干配置 | 输入尺度背景 / 元数据 | 原生 patch 或最细输出步长 | 高分 RGB 域标签 |
| --- | --- | --- | --- |
| LIM / MiT-Huge | Million-AID 高分 RGB；无统一逐图 GSD 声明 | 原生 /4、/8、/16、/32 | home；与评测图重叠未排除 |
| MaRS Base RGB | MaRS-16M 约 0.35 m；RGB 分支无显式 GSD 参数 | SwinV2 patch 4；四级 /4、/8、/16、/32，待实测确认 | home；重叠 unknown |
| Prithvi | HLS 预训练约 30 m；当前 RGB 入口不传 GSD | patch 16，派生多级特征 | away；污染另审 |
| Clay | 多传感器预训练；iSAID 路径声明 gsd=1.0 m，不是实测 GSD | patch 8，抽层/派生金字塔 | away；显式尺度不能替代真实分布匹配 |
| AnySat | 本项目拟用 S2 入口约 10 m 假设；不代表模型只支持 S2 | 既有方案按等效 /16；实际取决于入口及 patch_size_m，接入验证 | away（所选入口） |
| OlmoEarth Tiny / Base | 所选 S2 入口 10 m 组；不代表全部预训练模态均为 10 m | 既有方案选择 patch 16，需验证输出网格 | away（所选入口） |
| Galileo Base RGB | S2 RGB 路径以 10 m 为基准；`input_resolution_m` 可显式指定，实际/声明来源必记 | 本方案选 patch 8 → /8；不是固定仅能 patch 8 | away；额外 RGB 辐射与尺度错配 |
| R50 参考 | ImageNet 自然图像初始化，无物理 GSD | 常规 C2–C5 为 /4、/8、/16、/32 | 通用视觉校准，独立参考轨 |

Galileo 是多模态模型，不能简化成所有数据都以 10 m 预训练；这里的 10 m 指所选 S2 RGB 路径和位置编码基准。它也能显式传尺度，所以扩大候选范围后，“只有 Clay 能传 GSD”的旧表述不再适用。改变 GSD 声明不会创造缺失的空间信息；同样 /8 在不同输入 GSD 下对应不同地面尺度。Galileo 参考依据为[单文件实现](https://github.com/nasaharvest/galileo/blob/0f0b5b95ac81acef4b74cf4686dc877202f4541b/single_file_galileo.py)及[官方配置](https://huggingface.co/nasaharvest/galileo/tree/f039dd5dde966a931baeda47eb680fa89b253e4e/models/base)。

## 7. 实施与验收门槛：先 iSAID B2U，再检测

| 顺序 | 工作 | 通过证据 |
| --- | --- | --- |
| A0 | 两个候选来源及污染审计 | 本文 §3.4 清单、home/away、重叠状态和可解释范围；未知项显式披露 |
| A1-M | MaRS Base RGB 独立加载 | 权重覆盖率、NHWC→NCHW、四级尺寸、参考路径特征对齐 |
| A1-G | Galileo Nano/Tiny 验证输入，Base 验证目标版本 | RGB mask 不变性、归一化、GSD、token 恢复；Base 独立验收 |
| A2 | 接已有 T9 / UPerNet，做 B2U smoke | 冻结状态、head/必要适配器梯度、loss、1024 显存/吞吐、整图分割评测 |
| A3 | iSAID B2U 完整预算首轮 | 使用锁定 T9 协议和预声明 seed 17；预算统一准入，单种子仅作单种子报告 |
| D0 | 公共检测 G0/G1（可与 A 阶段独立推进） | torchvision/CPU 几何可用与数值一致、R50 参考推理及标准训练校准 |
| D1 | 新模型接统一检测器 | 复用已验收输入与特征接口；检测梯度、过拟合、显存/吞吐与整图合并验证 |
| D2 | 资源核算后展开正式比较 | 同数据/尺寸/head/预算；种子 17/42/73，失败及阻断单列 |

**MaRS A2 的位置处理验收**：记录 timm 版本、checkpoint 原始输入/窗口配置、实际构造的 `img_size` 和 window size；在 1024 输入上检查相对位置偏置/连续位置偏置、窗口坐标与 padding 的实际行为。对“保持窗口但显式设 img_size=1024”和上游允许的动态尺寸路径逐项核对，不预设 timm 必然插值或输入大小可自动变化。验证所有阶段形状、边缘窗口和同输入重复输出，保存权重加载及位置处理报告。run tags 至少记录 `backbone_input_px`、`window_size`、`positional_resize_policy`、`position_reference_config`、`position_impl_version`；LIM 同口径记录阶段 1 absolute pos_embed 双三次插值策略。字段名为待实施提案，本次没有声称已写入 run。

MaRS 自带四级特征，优先复用已有 UPerNet 契约。Galileo 的空间适配先在 T9 验证，不能将 Nano/Tiny 的成功当成 Base 的 1024 显存通过。iSAID 的工程结果用于判断接入质量和成本，低分本身不自动取消上级要求的检测候选，也不能据此反复选择最有利输入或种子。

MaRS 与 Galileo 分别建隔离 worker，根 CLI 不加载 torch。检测首选共享 torch head + 配套 torchvision.ops，旋转后处理/评测采用 CPU Shapely，MMCV 留在参考环境；不要求每个 worker 编译 MMCV。torchvision 的 wheel/torch/CUDA 配套与 RoIAlign 数值仍需验证。

冻结主干而训练入口适配器时，仍须保留对输入的梯度，不能一律 `no_grad()`。建议模型身份为 `mars_base_rgb`、`galileo_base_rgb`，尚未注册；Galileo patch、归一化和适配器策略必须记录，MaRS Large 独立 ID。

### 7.1 范围与单卡成本

正式候选来源已确认，但训练规模分阶段推进。原有 6 个模型配置加两候选为 8 个，检测三种子完整展开为 **24 个 run + R50 校准成本**；R50 若额外进入冻结主表，再增加 3 个 run。不能直接把独立锚点当成“第九个主干三种子”，也不能漏算此前 iSAID、smoke、参考复现和 CPU 后处理。

首轮先 MaRS iSAID 接入，同时做 Galileo 输入和显存 smoke；Galileo Base 在固定正式输入上通过后再安排其 iSAID 完整预算。完整三种子和检测队列以实测吞吐核算，不一次性承诺 27 组实验。所有长任务服从现有单卡 lane，不抢占活动训练。

### 7.2 在正式指标可见前锁定输入尺寸

本轮默认规则明确为：**1024 是正式主协议；OOM 的候选标资源阻断，512 仅作诊断。** 不因看到某主干精度低或高而切换整组输入。若要另立全组 512 附加协议，必须在该组任何正式指标可见之前预声明触发条件、模型范围、tile/重叠、预算、patch 与适配策略，且全模型重跑并分表。已经有结果的 iSAID 1024 轨道保留，不能回填成“事前决定”。

## 8. 已确认事项与待 review 决策

1. **已确认：两个新增候选来自上级明确要求。** 目前授权范围是调研与方案交付，未自动启动模型下载/长训练；不再询问是否由我们自行提议。
2. 是否接受先 MaRS Base RGB、再 Galileo Base 的 iSAID B2U 资源门槛；检测接入随后进行，G0/G1 不阻塞 iSAID？
3. 是否接受 MaRS 的 home / overlap unknown 暂定标签与接入前审计清单；资料不足时只作明确披露的能力评测，不作严格未见图泛化结论？
4. 是否接受默认 1024、OOM 标阻断，512 只作为预先声明的独立全组附加协议？
5. 是否按“8 个主干最多 24 个检测 run + R50 独立校准”分阶段核算资源，而非一次性安排九配置三种子？

本次交付为文档与来源核查，未下载模型权重或执行 GPU 验证。链接及 hash 来自发布元数据，不是本机权重文件已经校验的声明。正式实现和实验需逐项留存准入证据；第二轮核查详见[审查响应](../reviews/审查响应_2026-10-29_FAIR1M与新主干_第2轮.md)。
