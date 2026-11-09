# PASTIS-HD 作物分割：公开成绩、模型权重与复现指南

本文汇总前两轮调研，范围为**作物类型语义分割**。分数均为 mIoU（%），即对参与评估的类别取交并比的平均。成绩为论文或官方仓库报告值，权重状态来自公开发布页面，尚未实际运行推理核验。

## 1. 数据集版本

| 数据集 | 可用输入 |
|---|---|
| PASTIS | Sentinel-2 多光谱时间序列 |
| PASTIS-R | Sentinel-2 多光谱时间序列 + Sentinel-1 SAR 时间序列 |
| PASTIS-HD | PASTIS-R 的输入 + SPOT 6/7 高分辨率 RGB 影像 |

PASTIS-HD 延续原有农业地块标注，SPOT 影像对齐到 1 m 网格，作物分割仍沿用原 PASTIS 的 10 m 标签网格。以下成绩评估的是逐像素作物类别。若要评估能否把相邻的同类地块分别勾出来，应使用全景分割协议和 PQ 指标。

来源：[PASTIS 官方基准](https://github.com/VSainteuf/pastis-benchmark) · [PASTIS-HD 数据卡](https://huggingface.co/datasets/IGNF/PASTIS-HD)

## 2. 公开成绩与权重

| 模型 | 报告 mIoU | 评测口径 | 公开权重与复现条件 |
|---|---:|---|---|
| MAESTRO† | **69.0** | PASTIS-HD Fold 1，数据集内预训练后全量微调。† 表示预训练轮数翻倍 | 未查到对应的完整分割权重。按论文与官方代码重跑预训练和下游训练 |
| MAESTRO | **68.6** | PASTIS-HD Fold 1，数据集内预训练后全量微调 | 官方仓库有从头训练示例，未查到这组实验的完整分割权重 |
| 大型 MAE 变体 | **69.5** | MAESTRO 论文附录表 16，PASTIS-HD Fold 1，Large 模型使用基于 token 的光谱融合 | 未查到对应成品权重。按附录配置重跑 |
| MAESTRO（FLAIR-HUB 初始化） | **68.0** | FLAIR-HUB 预训练，PASTIS-HD Fold 1 微调 | 提供预训练权重、数据划分和迁移命令，需要训练下游任务 |
| MAESTRO（S2-NAIP-urban 初始化） | **67.4** | S2-NAIP-urban 预训练，PASTIS-HD Fold 1 微调 | 提供预训练权重，需要训练下游任务 |
| AnySat | **66.5** | PASTIS-HD，MAESTRO 主表引用的结果 | 提供预训练权重和 PASTIS-HD 语义分割微调配置。未查到该成绩对应的完整分割权重 |
| U-TAE-MM（后期融合） | **66.3** | PASTIS-R 官方五折评测，使用 S1 + S2，约 170 万参数 | 未确认有对应融合版本的成品权重。可依据论文和 U-TAE 实现复现训练 |
| TSViT | **65.9**（旧版 65.4） | PASTIS24 五折，仅 S2。65.9 对应修复 patch 空间重建后重新训练的版本 | 官方提供五折任务权重，可按相应数据切块、划分和代码直接评估 |
| U-TAE | **63.1** | 原始 PASTIS 官方五折，仅 S2，约 110 万参数 | 官方提供语义分割权重和测试脚本，可直接评估 |

**PASTIS-HD 的公开参考成绩**可引用 MAESTRO† 主表的 **69.0**。附录的 **69.5** 对应另一套大型变体配置，汇报时应分别列出。表中的 PASTIS-HD Fold 1、PASTIS-R 五折和 PASTIS24 五折使用不同输入或评测设置，不宜直接合成一个排名。

来源：[MAESTRO 论文主表与附录](https://arxiv.org/html/2508.10894v2) · [PASTIS 官方榜单](https://github.com/VSainteuf/pastis-benchmark#leaderboard) · [TSViT 官方更新说明](https://github.com/michaeltrs/DeepSatModels/blob/main/README_TSVIT.md)

## 3. 复现路径

### 用成品权重跑通推理

- **TSViT**：下载五折任务权重和 PASTIS24 数据，用仓库中对应折的配置评估。使用新版成品权重时，参考值为 **65.9**。  
  [五折权重](https://drive.google.com/file/d/1GQSNt4DzXt9tBwANnURYQNsVtQtRLRWW/view) · [数据与配置说明](https://github.com/michaeltrs/DeepSatModels/blob/main/README_TSVIT.md)
- **U-TAE**：下载语义分割任务权重，准备原始 PASTIS，运行官方 `test_semantic.py`。该模型可作为 S2 时序专用模型基线。  
  [语义分割权重](https://zenodo.org/records/5172293) · [推理说明](https://github.com/VSainteuf/utae-paps#inference-with-pre-trained-models)

### 用预训练权重复现 PASTIS-HD 下游训练

- **AnySat**：使用官方预训练权重和 `Pastis_AnySat_FT_SemSeg.yaml`。该配置使用 SPOT、S2、S1，并从 `AnySat_full.pth` 初始化。按官方划分训练和评估后，将实测成绩与论文的 **66.5** 对照。  
  [权重目录](https://huggingface.co/g-astruc/AnySat/tree/main/models) · [官方分割配置](https://github.com/gastruc/AnySat/blob/main/configs/exp/Pastis_AnySat_FT_SemSeg.yaml)
- **MAESTRO**：最直接的公开权重复现路线是从 FLAIR-HUB 预训练模型出发，按模型卡的 PASTIS-HD 命令训练任务头并微调全模型，参考 **68.0**。若目标是 **69.0**，应按论文的数据集内预训练设置和对应下游配置重跑。  
  [FLAIR-HUB 权重、数据划分与命令](https://huggingface.co/IGNF/MAESTRO_FLAIR-HUB_base) · [S2-NAIP-urban 权重](https://huggingface.co/IGNF/MAESTRO_S2-NAIP-urban_base) · [训练代码](https://github.com/IGNF/MAESTRO)
- **U-TAE-MM**：它面向作物时序制图，分别处理光学和雷达序列，再融合特征做分割。若需要对照 **66.3**，使用 PASTIS-R 和论文的后期融合方案重跑。  
  [光学与 SAR 融合论文](https://arxiv.org/html/2112.07558v1) · [U-TAE 实现](https://github.com/VSainteuf/utae-paps)

## 4. 用于项目对比

若先验证评测与可视化流程，用 **TSViT 或普通 U-TAE 的成品权重**跑通一个官方测试折。若要验证 PASTIS-HD 全输入的效果，再训练 **AnySat 或 MAESTRO**，并用 **U-TAE-MM**作为只看 S1 + S2 的专用小模型对照。

每次实验记录**数据版本、输入模态与日期、测试折、标签网格与忽略类别、权重来源、训练方式**。尤其要区分“冻结主干＋训练分割头”和“全量微调”。若 LIM 只使用单日期 SPOT RGB，可单独做一组同输入对比，同时展示各模型使用其可用模态时的结果。
