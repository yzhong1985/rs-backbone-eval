# 文档导航

文档按用途归档，子目录采用英文名称。文档中的状态和结论以各自记录的日期与代码版本为准。

| 目录 | 内容 |
| --- | --- |
| [architecture/](architecture/) | 架构与工程设计 |
| [evaluation/](evaluation/) | 评测设计与协议 |
| [models/](models/) | 模型设计与接入 |
| [data/](data/) | 数据筛选与核查 |
| [reports/](reports/) | 实验结果与阶段报告 |
| [reviews/](reviews/) | 代码复查与运行审计 |

建议先阅读[项目架构图](architecture/项目架构图.md)和[七任务评测设计](evaluation/七任务冻结主干评测设计_AnySat_Clay.md)；自研模型从[光学与 SAR 骨干设计草案](models/自研轻量光学SAR骨干设计草案.md)开始，数据准备参考[NAS 核查与拷贝建议](data/NAS核查与2TB拷贝建议_2026-10-15.md)。

## architecture — 架构与工程设计

- [项目架构图](architecture/项目架构图.md)
- [代码基础设施与Web_Portal设计](architecture/代码基础设施与Web_Portal设计.md)
- [数据与模型下载管理设计](architecture/数据与模型下载管理设计.md)
- [Colab算力接入与队列运行方案](architecture/Colab算力接入与队列运行方案.md)
- [断点续训与Colab运行](architecture/断点续训与Colab运行.md)

## evaluation — 评测设计与协议

- [七任务冻结主干评测设计_AnySat_Clay](evaluation/七任务冻结主干评测设计_AnySat_Clay.md)
- [骨干能力评分卡与扰动协议](evaluation/骨干能力评分卡与扰动协议.md)
- [遥感Backbone评价体系改进建议](evaluation/遥感Backbone评价体系改进建议.md)
- [冻结主干任务头训练实验设计_AnySat_Clay](evaluation/冻结主干任务头训练实验设计_AnySat_Clay.md)
- [多光谱与多传感器模型验证实验计划_AnySat_Clay](evaluation/多光谱与多传感器模型验证实验计划_AnySat_Clay.md)

## models — 模型设计与接入

- [自研轻量光学SAR骨干设计草案](models/自研轻量光学SAR骨干设计草案.md)
- [自研多光谱模型实验方案](models/自研多光谱模型实验方案.md)
- [自研骨干接入现有评测计划](models/自研骨干接入现有评测计划.md)
- [模型扩展与低成本评测计划](models/模型扩展与低成本评测计划.md)
- [外部模型可比性评估](models/外部模型可比性评估.md)
- [OlmoEarth接入进展](models/OlmoEarth接入进展.md)

## data — 数据筛选与核查

- [NAS数据用于自研骨干的筛选建议](data/NAS数据用于自研骨干的筛选建议.md)
- [NAS现场核查表](data/NAS现场核查表.md)
- [NAS核查与2TB拷贝建议_2026-10-15](data/NAS核查与2TB拷贝建议_2026-10-15.md)
- [遥感模态与Prithvi输入范围说明](data/遥感模态与Prithvi输入范围说明.md)

## reports — 实验结果与阶段报告

- [实验报告_当前结果](reports/实验报告_当前结果.md)
- [实验报告_给上级](reports/实验报告_给上级.md)

## reviews — 代码复查与运行审计

- [项目审查_2026-10-14](reviews/项目审查_2026-10-14.md)
- [评测框架复审_3b4460a](reviews/评测框架复审_3b4460a.md)
- [历史运行审计](reviews/历史运行审计.md)
- [最新修复复查_2026-10-15](reviews/最新修复复查_2026-10-15.md)
- [最新推送复查_2026-10-15](reviews/最新推送复查_2026-10-15.md)
- [Colab运行与断点续训审查_2026-10-15](reviews/Colab运行与断点续训审查_2026-10-15.md)
