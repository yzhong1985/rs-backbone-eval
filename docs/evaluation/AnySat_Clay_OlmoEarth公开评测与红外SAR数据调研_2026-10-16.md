# AnySat、Clay、OlmoEarth 公开评测与红外／SAR 数据调研

核对日期：2026-10-16。项目版本基于当前工作区；模型分别参考 AnySat CVPR 2025 / arXiv v3、Clay v1.5 官方文档和当前示例、OlmoEarth v1 与 v1.2 技术报告及官方评测仓库。以下没有下载完整影像数据，也没有执行新增实验。

后续实验目标已明确为“已有强 RGB 主干，新增主干负责非 RGB 波段解析与互补”。对应的对照组、输入契约和实施顺序见[非 RGB 波段解析主干补充实验设计](./非RGB波段解析主干补充实验设计_2026-10-16.md)。本文保留为数据与原始评测来源索引。

## 1. 当前 T1～T7 覆盖范围

项目 T1～T7 来自 Prithvi-EO-2.0 下游任务组，不等于 Prithvi 创建了这些数据，也不等于只有 Prithvi 评测过它们。T6 分成 a/b，所以实际是八份配置。波段以 `configs/tasks/` 为准。

| 项目任务 | 数据集 | 当前主配置输入 | 红外／SAR 覆盖 |
| --- | --- | --- | --- |
| T1 洪水 | Sen1Floods11 | S2 六波段 | NIR、SWIR；主配置不使用 S1 |
| T2 火烧迹地 | HLS Burn Scars | 单期 HLS 六波段 | NIR、SWIR；没有输入热红外 |
| T3 火烧强度 | Burn Intensity | 三期 HLS 六波段 | NIR、SWIR；没有输入热红外 |
| T4 滑坡 | Landslide4Sense | S2 六波段 | NIR、SWIR；没有输入地形数据 |
| T5 美国作物 | Multi-temporal Crop Classification | 三期 HLS 六波段 | NIR、SWIR；没有输入热红外 |
| T6a/b 土地覆盖／作物 | Sen4Map | 十二期 S2 十波段 | NIR、SWIR、红边 |
| T7 生物量 | BioMassters | 四期 S2 六波段 | NIR、SWIR；原数据另有 S1，但主配置不用 |

当前另有 `configs/tasks/T1_flood_sar.yaml`，任务 ID 为 **T1S**，输入 VV/VH。AnySat、Clay worker 已有相关入口；这不代表当前 OlmoEarth worker 或 Colab 包已接通同样功能。

区分三类信息：NIR/SWIR 是反射光谱；TIR 是热红外及其温度派生产品；SAR 是主动微波。不能因为数据名包含 Landsat、HLS 或 MODIS，就推断某次实验使用了热红外。HLS L30 产品可以含热红外，但本项目选取的六波段没有包含它。参见 [NASA HLS 算法说明](https://hls.gsfc.nasa.gov/algorithms/)和 [USGS Landsat 波段说明](https://www.usgs.gov/faqs/what-are-band-designations-landsat-satellites)。

## 2. AnySat 原始验证实验

以[论文正文第 4 节及附录](https://arxiv.org/html/2412.14123v3)为准。正文列出四个有标签的 GeoPlex 数据集和六个外部评测集；README 与部分图注的数量表述存在版本差异，不使用其“9 个数据集”概括最终清单。

| 数据集 | 验证任务 | 数据模态／波段 | 公开入口 |
| --- | --- | --- | --- |
| TreeSatAI-TS | 多标签树种分类 | 航空 RGB+NIR、S2、S1 时间序列 | [作者发布 HF](https://huggingface.co/datasets/IGNF/TreeSatAI-Time-Series) |
| PASTIS-HD | 作物分类、语义分割 | SPOT 高分辨率影像、S2、S1 时间序列 | [作者发布 HF](https://huggingface.co/datasets/IGNF/PASTIS-HD) |
| FLAIR | 土地覆盖分割 | 航空 RGB/NIR/高程、S2 时间序列 | [FLAIR-1-2](https://huggingface.co/datasets/IGNF/FLAIR-1-2) |
| PLANTED | 种植林树种分类 | S2、Landsat-7、MODIS、S1、ALOS-2 | [原作者数据说明](https://github.com/google-deepmind/forest_typology/tree/main/planted) |
| BraDD-S1TS | 森林砍伐变化分割 | S1 SAR 时间序列 | [Zenodo](https://zenodo.org/records/8060250) |
| SICKLE | 稻田／非稻田分割 | 原数据含 S1、S2、Landsat-8；具体运行按 modality 配置选取 | [原作者仓库／申请入口](https://github.com/Depanshu-Sani/SICKLE) |
| TimeSen2Crop | 作物类别分类 | 单像素 S2 时间序列 | [论文及作者数据入口见 AnySat 参考文献 75](https://arxiv.org/html/2412.14123v3#bib.bib75) |
| Sen1Floods11 | 洪水分割 | S1、S2，需区分单模态和融合设置 | [原作者仓库](https://github.com/cloudtostreet/Sen1Floods11) |
| So2Sat LCZ42 | 局地气候区分类 | 配准 S1/S2 | [原作者仓库](https://github.com/zhu-xlab/So2Sat-LCZ42) |
| HLS Burn Scars | 火烧迹地分割 | HLS 六波段 | [HF 数据](https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars) |

前四项也属于 AnySat 的 GeoPlex 预训练数据来源：使用其测试划分评测，与整个外部数据集未参与预训练，是两种不同的泛化设置。S2NAIP-URBAN 主要用于预训练，没有同一套官方有标签下游评测。官方模态及运行入口见 [AnySat README](https://github.com/gastruc/AnySat)。

PASTIS-HD 与 PASTIS-R、TreeSatAI-TS 与 TreeSatAI 原始单期数据不能互换名称或直接混用成绩。[OmniSat 作者提供了增强版本的公开入口](https://github.com/gastruc/OmniSat#datasets)。

## 3. Clay 官方验证例子

Clay v1.5 的公开文档以应用示例为主。没有从其发布页确认到一套与“Prithvi 七任务”同定义的固定评测清单；第三方论文评测 Clay 的任务，不应写成 Clay 团队原始实验。

| 官方示例 | 数据 | 实际输入／协议 | 公开来源 |
| --- | --- | --- | --- |
| 土地利用场景分类 | EuroSAT MS | S2 多光谱，冻结编码器加分类头 | [教程](https://clay-foundation.github.io/model/finetune/classify.html)、[数据作者](https://github.com/phelber/EuroSAT) |
| 高分辨率土地覆盖分割 | Chesapeake Bay CVPR | NAIP RGB+NIR，冻结编码器加分割头 | [教程和下载命令](https://clay-foundation.github.io/model/finetune/segment.html) |
| 森林生物量回归 | BioMassters | 当前示例取 S2 光学通道，并把时间维平均 | [教程](https://clay-foundation.github.io/model/finetune/regression.html) |

代码核对：BioMassters 预处理虽读取 S1/S2，但 [datamodule 的 `cube[2:, ...]`](https://github.com/Clay-foundation/model/blob/main/claymodel/finetune/regression/biomasters_datamodule.py) 去掉了预处理后前两个 S1 通道。因此这个官方示例不能当作“SAR+光学融合已验证”的证据。

Clay v1.5 的预训练来源包含 S1 SAR，说明模型有 SAR 训练背景；其标准 Landsat 配置则只有 RGB/NIR/SWIR 六波段，没有热红外。接口支持可变波段，不等于已经预训练或验证了热红外。[预训练数据说明](https://clay-foundation.github.io/model/release-notes/data_sampling.html)、[标准波段元数据](https://github.com/Clay-foundation/model/blob/main/configs/metadata.yaml)。

## 4. OlmoEarth 的验证实验

本项目配置为 OlmoEarth v1.2 Tiny。参考 [v1.2 技术报告表 1～3](https://allenai.org/papers/olmoearth-v1-2)、[研究基准复现说明](https://github.com/allenai/olmoearth_pretrain/blob/main/docs/Evaluation.md)，同时核对原始 v1 的任务映射。不同版本、kNN、线性探针和全量微调的结果要分开。

| 任务组 | 数据／任务 | 关注模态 |
| --- | --- | --- |
| 场景／多标签分类 | GEO-Bench 的 m-bigearthnet、m-so2sat、m-brick-kiln、m-forestnet、m-eurosat | 前四者中的 m-forestnet 为 Landsat，其余此处评测为 S2；不能把同名原数据有 S1 理解为这次用了 S1 |
| 时序分类 | BreizhCrops、CropHarvest-PRC、CropHarvest-Togo | S2；CropHarvest 还分别评测 S1、S2、S1+S2 |
| 像素分割 | m-cashewplant、m-SA-crop-type、MADOS | S2，多光谱光学 |
| SAR／光学分割 | PASTIS、Sen1Floods11 | PASTIS 分别评测 S1/S2；该论文 Sen1Floods11 使用 S1 |
| 合作伙伴任务 | AWF 土地覆盖、Nandi 作物类型 | 分别使用 Landsat、S1、S2 时间序列 |

原 v1 的表 2 包含 24 个“数据任务 × 模态”列，不是 24 个独立数据集。对应任务和已计算特征公开在 [OlmoEarth paper embeddings](https://huggingface.co/datasets/allenai/olmoearth-paper-embeddings)。可以仅下载目标模型／任务的特征，先复现冻结评测；特征数据不能替代波段消融所需的原始影像。

官方还单独发布了[合作伙伴任务数据及代码](https://github.com/allenai/rslearn_projects/blob/master/rslp/olmoearth_evals/README.md)：LFMC 活体燃料含水率、红树林分类、海上设施检测、Landsat/S1/S2 船舶检测、S2 船长／类型预测、光伏电站分割。这个当前公开列表不应与原论文 AWF/Nandi 的评测列混为一套数据。

研究基准的公开下载桶在[官方安装说明](https://github.com/allenai/olmoearth_pretrain/blob/main/docs/Pretraining.md#evaluation-datasets)中；匿名目录查询确认含 geobench、breizhcrops、floods、mados、pastis、pastis_r 等前缀。部分内部默认路径不可直接用，但并不意味着原始公开数据不存在。

## 5. 最适合补充的 SAR／红外数据

体积使用发布页或 2026-10-16 HTTP HEAD 信息，为压缩文件／下载对象体积，非解压后空间。HEAD 仅确认访问和长度，没有检验完整内容。

| 候选 | 模态及验证目标 | 数据获取与限制 | 当前建议 |
| --- | --- | --- | --- |
| Sen1Floods11 | S1 VV/VH 洪水分割；可做 S2-only / S1-only / S1+S2 | [公开桶与清单](https://github.com/cloudtostreet/Sen1Floods11) | 优先；复用 T1 标签和分区，已有 T1S 配置 |
| BraDD-S1TS | S1 SAR 时序砍伐检测 | [Zenodo，17.7 GB](https://zenodo.org/records/8060250) | 优先；独立 SAR 外部数据，AnySat 直接有论文实验 |
| PASTIS-R | S1/S2 作物分割，可比较模态互补与时序 | [Zenodo，53.7 GB](https://zenodo.org/records/5735646) | 适合三模型多模态比较；注明与 PASTIS-HD 的预训练来源关联 |
| BioMassters | S1/S2 生物量回归 | [项目已有 HF 入口](https://huggingface.co/datasets/ibm-nasa-geospatial/BioMassters)，体积较大 | 扩展 T7，而不是另造一个完全无关任务 |
| SICKLE | S1/S2/L8，含热红外派生 ST_B10；作物分割、季节、产量等标签 | [需填表申请，完整测试集未公开](https://github.com/Depanshu-Sani/SICKLE) | 热红外且与 AnySat 实验关联最直接，但不能承诺即下即跑或官方 test 复现 |
| SPARCS | Landsat 全波段，人工云／云影等标签；可增加／移除 B10/B11 | [USGS，页面标注约 1.4 GB，80 个 1000×1000 影像子集](https://www.usgs.gov/landsat-missions/spatial-procedures-automated-removal-cloud-and-shadow-sparcs-validation-data) | 公开小规模热红外补充；本次未确认它是上述模型原始评测项 |
| S1 vessel detection | SAR 船舶目标检测 | [Ai2 发布说明](https://github.com/allenai/rslearn_projects/blob/master/rslp/olmoearth_evals/README.md#sentinel-1-vessel-detection)，公开 tar HEAD 为 9,589,923,840 bytes | 约 9.59 GB，补充检测任务；不能沿用当前分割头 |
| LFMC | S1+S2 植被含水率回归 | [官方项目及数据](https://github.com/allenai/olmoearth_projects/blob/main/docs/lfmc.md)，公开 eval tar HEAD 为 26,479,308,800 bytes | 约 26.48 GB；适合 NIR/SWIR+SAR，不能称热红外实验 |

### 热红外证据与局限

1. [AnySat SICKLE loader](https://github.com/gastruc/AnySat/blob/main/src/data/Sickle.py) 明确读取 `ST_B10`，这是热红外派生的地表温度产品；它与原始 Level-1 热辐射 DN 不是同一个数值域。该 loader 还为缺失的 L8 通道使用填充值并生成 mask，移植时必须检查实际前向路径是否消费 mask，不能直接假定满足本项目输入契约。
2. [OlmoEarth 预训练数据](https://github.com/allenai/olmoearth_pretrain/blob/main/docs/Pretraining-Dataset.md) 明确使用 Landsat 8/9 Level-1 全 11 波段，包括 B10/B11。但“预训练含热红外”不等于“已证明某项下游成绩来自热红外”。
3. Clay 标准 Landsat 六波段、Prithvi 当前六波段入口没有相同热红外预训练证据。做共同 TIR 输入时要明确额外适配器／新波段处理，不能称四模型都原生支持。
4. USGS SPARCS 是可操作的新增 TIR 基准建议。应按原场景划分 train/validation/test，裁块后禁止同一原场景跨集合；QA/cloud mask 不作为输入，否则可能泄露云标签。历史 Pre-Collection 产品应使用其配套元数据校准，不能套用 Collection 2 的温度缩放常数。

### 推荐的最小增补实验

- 保留现有 T2/T3 作为 SWIR 重点应用；保持样本、训练预算相同，比较 RGB、RGB+NIR、RGB+NIR+SWIR。这里是待验证的实验设计，不是已观察到的波段增益。
- 先完成 T1S，沿用 T1 的水体标签／划分；随后补 BraDD-S1TS，形成单日期与时间序列两种 SAR 测试。
- 开放且小规模的热红外实验先做 SPARCS；SICKLE 获批后，再加入带农业标签的热红外／光学／SAR 时序实验。
- 若重点比较融合，选 PASTIS-R 或 BioMassters，分别报告 S1、S2、S1+S2，固定日期窗口与样本交集。不要把融合信息更多的结果与六波段光学主表混排。
- SAR 原生比较优先 AnySat、Clay、OlmoEarth。Prithvi 需要单列跨模态适配实验；当前项目中的可训练入口也仍需真实数据与梯度验收。
