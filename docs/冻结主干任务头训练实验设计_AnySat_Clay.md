# AnySat 与 Clay 冻结主干任务头训练实验设计

版本：v1.0 ｜ 日期：2026年10月12日 ｜ 状态：待实现

> 本文是水体与火烧迹地两个任务的早期冻结主干设计，已被[《AnySat 与 Clay 七任务下游评测设计（Prithvi-EO-2.0 对照）》](./七任务冻结主干评测设计_AnySat_Clay.md)取代。保留本文仅用于追溯，不再作为完整实施范围。

## 一、定位与结论口径

本阶段允许使用任务标签训练新任务头，但预训练主干保持冻结。目标是回答两个问题：

1. AnySat、Clay 的现成多光谱表征是否容易被简单模型解码为水体或火烧迹地；
2. 在相同的数据、训练预算和评价协议下，冻结主干加轻量分割头能达到什么任务效果和运行成本。

本阶段不重新预训练主干，不训练新的 AnySat 传感器投影入口，不更新主干参数。若后续允许微调主干，应建立单独实验组，不能与冻结主干结果混合排名。

结果必须使用以下名称：

- **线性探针结果**：冻结主干，只训练逐像素线性层；
- **轻量头结果**：冻结主干，只训练统一的轻量分割头；
- **现成任务模型结果**：加载发布方已经训练完整的模型；
- **主干微调结果**：至少一个主干参数发生更新，本阶段不包含。

线性探针衡量表征的线性可分性，轻量头衡量有限解码能力下的任务迁移效果。二者分别报告，不合成一个总分。

## 二、任务和首轮实验矩阵

### 2.1 任务定义

| 任务 | 输入数据 | 目标 | 标签 | 主指标 |
| --- | --- | --- | --- | --- |
| S2 水体分割 | Sen1Floods11 手工标注区域的 Sentinel-2 | 水／非水 | `-1`忽略，`0`非水，`1`水 | 水体 IoU |
| HLS 火烧迹地分割 | HLS Burn Scar | 烧毁／未烧毁 | `-1`忽略，`0`未烧毁，`1`烧毁 | 火烧迹地 IoU |

Sen1Floods11 的标签表示水体，不足以单独区分永久水体与新增洪水，因此实验名称统一写成“S2 水体分割”。

### 2.2 模型矩阵

| 数据／任务 | AnySat | Clay v1.5 | LIM | Prithvi参考 |
| --- | --- | --- | --- | --- |
| Sen1Floods11 S2水体 | 主实验 | 主实验 | 有可用RGB接口时加入 | 有匹配现成模型时只作外部参考 |
| HLS Burn Scar | 缺少匹配HLS入口时不运行 | 主实验 | 有可用RGB接口时加入 | 现成任务模型只作外部参考 |

每个可训练的“主干 × 任务”组合运行两个头：

1. `LinearProbeHead`；
2. `LiteSegHead`。

正式结果使用3个固定随机种子：`17, 42, 73`。开发阶段先以种子`42`运行3–5景和一个短训练，确认数据、梯度、显存及指标无误后再展开正式实验。

## 三、数据划分与防泄漏

### 3.1 Sen1Floods11

只使用446景手工标注数据作为正式监督数据。输入为13波段 Sentinel-2 L1C TOA 反射率，原始数值按10000缩放；模型使用其中约定的10个波段：

`B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12`

处理规则：

- 以官方 `flood_train_data.csv`、`flood_valid_data.csv`、`flood_test_data.csv` 为首要划分；
- 通过唯一 chip ID 将划分文件映射到 `S2Hand` 和 `LabelHand`，生成后检查三个集合无交集；
- 读取事件元数据中的 `s2_date` 生成 AnySat 日期输入和 Clay 时间元数据；
- 检查同一洪水事件是否跨集合。如果官方划分存在事件交叉，保留官方划分用于复现，并增加一个按事件分组的稳健性划分；
- 不使用弱标签数据参与首轮正式训练；以后若使用，必须单列为“弱监督扩展实验”。

所有归一化统计、类别比例、阈值和超参数只能由训练集合或发布方预训练配置得到。

### 3.2 HLS Burn Scar

数据包含6波段反射率：Blue、Green、Red、NIR、SWIR1、SWIR2。正式使用压缩包中的实际目录建立 manifest，不直接采用数据加载脚本生成的 `test` split，因为该脚本把 validation 目录再次暴露为 test。

划分规则：

- 官方 `training/` 中按固定种子划出10%作为项目内部 validation；
- 官方 `validation/` 在配置和报告中命名为 `official_validation`，在所有超参数锁定后只运行一次；
- 数据没有独立官方测试集，因此不得把该结果描述为独立测试成绩；
- 检查文件ID和地理范围是否重复，保留检查报告；
- HLS归一化统计只能从项目训练子集计算。

官方数据卡称原始数据有804景，而 Hugging Face 数据加载视图可能因重复暴露 validation 显示更多行。下载后以唯一影像路径和校验和去重，并在 manifest 中记录最终景数。

### 3.3 统一 manifest

每一景至少记录：

```text
sample_id,dataset,split,event_id,image_path,label_path,sensor,
bands,acquired_at,crs,transform,gsd_m,nodata_value,checksum
```

训练启动前自动执行以下硬检查：

- `sample_id`、影像路径和校验和不跨集合重复；
- 影像、标签、有效像素掩码尺寸及地理变换一致；
- 波段数量、顺序、数值范围和数据类型符合模型配置；
- 标签值只包含 `-1, 0, 1`；
- train/validation/test 或 official_validation 的用途没有混用。

## 四、主干输入和特征协议

### 4.1 原始数据边界

公共数据层输出未经模型归一化的 `GeoSample`。缩放、波段选择、归一化、日期及位置编码全部由模型适配器处理，避免把一个模型的统计量错误用于另一个模型。

### 4.2 AnySat

- S2张量形状为 `B × T × 10 × H × W`，单日期保留 `T=1`；
- 日期张量形状为 `B × T`，数值为当年的 day-of-year；
- 使用官方S2波段顺序及预训练统计量；
- 提取 `output="dense"`、`output_modality="s2"` 的空间特征；
- 先在候选裁块大小和 patch size 上做显存与输出步长检查，正式配置保证每景裁块规则固定；
- 没有官方支持的缺失通道策略时，不运行 AnySat RGB 或 RGB+NIR 子集实验；
- 没有匹配HLS投影入口权重时，不把HLS伪装成S2或Landsat输入。

### 4.3 Clay v1.5

- 输入影像形状为 `B × C × H × W`；
- 同步提供波长以及 `[week, hour, lat, lon]`；
- S2使用官方 `sentinel-2-l2a` 波段顺序、波长和预训练统计量；
- HLS建立独立的 `hls-burn-scars` 元数据配置，记录六波段波长、30米GSD和训练集统计量；
- 优先提取最终空间 token，而不是池化后的单一1024维场景 embedding；
- 若需要修改官方模块才能暴露空间 token，适配器必须记录上游commit和修改点，并用池化结果一致性做回归检查。

### 4.4 统一特征对象

模型适配器统一返回：

```python
SpatialFeatures(
    values,          # [B, C, h, w]
    stride_y,        # 相对原输入像素的纵向步长
    stride_x,        # 相对原输入像素的横向步长
    valid_mask,      # 特征网格上的有效区域
    encoder_id,
    checkpoint_sha256,
    preprocessing_id,
)
```

不得仅根据张量形状猜测空间步长。适配器需用一个已知位置的合成或真实小样本验证特征方向、裁块和标签对齐。

## 五、任务头设计

### 5.1 线性探针

```text
[B,C,h,w]
  → 1×1 Conv(C,1)
  → bilinear resize到标签网格
  → 单通道logit
```

除偏置和1×1卷积外没有可训练参数。它作为主干表征的基础比较，不加入隐藏层、跳接或多尺度特征。

### 5.2 统一轻量分割头

```text
[B,C,h,w]
  → 1×1 Conv(C,256) + GroupNorm + GELU
  → 3×3 Conv(256,256) + GroupNorm + GELU
  → 3×3 Conv(256,256) + GroupNorm + GELU + residual
  → Dropout2d(0.1)
  → 1×1 Conv(256,1)
  → bilinear resize到标签网格
```

两个主干使用相同隐藏宽度、层数、激活、归一化和插值设置。输入投影层会因主干通道数不同而产生少量参数量差异，因此必须同时报告任务头参数量和MACs。

首轮公平比较只使用主干最终空间特征。Clay官方示例所建议的多层特征解码器可作为后续“原生优化头”实验，但不得与统一轻量头放在同一排行榜。

### 5.3 冻结保证

训练代码必须同时满足：

- 主干全部参数 `requires_grad=False`；
- 主干始终处于 `eval()`，任务头单独处于 `train()`；
- optimizer 参数集合只来自任务头；
- 在线训练时主干前向使用 `torch.no_grad()`，输出立即 `detach()`；预计算缓存时可使用 `torch.inference_mode()`，缓存重新加载后再送入任务头；
- 训练前后计算主干 `state_dict` 哈希，要求完全一致；
- 单元测试执行一次反向传播，确认所有主干梯度为 `None`。

## 六、训练协议

### 6.1 固定设置

| 项目 | 配置 |
| --- | --- |
| 优化器 | AdamW，仅任务头参数 |
| 初始学习率 | `1e-3` |
| weight decay | `1e-4` |
| 最大epoch | 50 |
| 学习率计划 | 5% warmup + cosine decay |
| 梯度裁剪 | global norm 1.0 |
| 混合精度 | CUDA上使用 BF16；不支持时FP16 |
| 模型选择 | validation 前景IoU最高的checkpoint |
| early stopping | patience 10，最少训练10个epoch |
| 随机种子 | 17、42、73 |

batch size 由显存冒烟测试确定；同一任务内尽量一致。若不同主干必须使用不同 batch size，则采用梯度累积保持相同 effective batch size，并在报告中注明。

### 6.2 损失函数

首轮统一使用：

```text
loss = 0.5 × BCEWithLogits + 0.5 × SoftDiceLoss
```

两项损失都忽略 `label=-1`。Dice只针对前景类计算，分母加固定平滑项。首轮不为某个模型单独调类别权重或损失比例。

### 6.3 数据增强

只采用不改变光谱语义且能对特征与标签同步执行的增强：

- 水平翻转；
- 垂直翻转；
- 90度整数倍旋转；
- 固定规则的随机裁块。

不使用RGB颜色抖动、逐波段随机缩放或MixUp。若使用预计算特征，增强在空间特征和标签上同步执行。

### 6.4 特征缓存

冻结主干时默认先缓存空间特征，再重复训练不同种子的任务头。缓存键包含：

```text
dataset_manifest_hash
encoder_code_commit
checkpoint_sha256
preprocessing_config_hash
band_subset
crop_and_patch_config
```

任一键变化都必须重新提取特征。缓存采用Zarr或等价的分块数组格式，保存 `float16`/`bfloat16` 特征，并随机抽样与在线前向结果对比误差。正式指标计算仍使用 `float32` logits和累计量。

## 七、评价协议

### 7.1 主指标

主指标通过整个集合的有效像素混淆矩阵汇总后计算：

- 前景 IoU；
- 前景 F1；
- Precision；
- Recall；
- 两类 mIoU；
- 混淆矩阵。

默认分类阈值固定为0.5。可以额外报告由 validation 选择的阈值，但必须标为次要结果，且该阈值只能应用到尚未查看的最终集合。

### 7.2 辅助结果

- 每景指标的均值、中位数和分布；
- 对每景做bootstrap得到的95%置信区间；
- 含前景景与不含前景景的分组结果；
- 成功、漏检、误检和边界失败案例；
- 每景端到端推理时间、仅任务头时间、峰值显存；
- 主干参数量、任务头参数量和输出特征步长。

3个种子的结果报告均值和标准差，同时保留每个种子的原始结果。不得只选择成绩最好的种子。

### 7.3 排名规则

比较表按“任务 × 输入波段 × 任务头类型”分组。只有使用相同数据划分和同一类任务头的结果可以直接横向排名。

以下结果单列：

- 现成Prithvi任务模型；
- LIM已有任务模型；
- 原生优化或多尺度任务头；
- 主干微调；
- 非参数近邻方法；
- 不同输入波段组合。

## 八、谱段实验

完整波段是主实验。谱段实验分成两种，避免混淆：

1. **移除稳健性**：用完整波段训练好的任务头，推理时移除部分波段；只有模型接口存在经过验证的缺失波段处理时才运行。
2. **子集可学习性**：冻结同一主干，为 RGB+NIR 或 RGB 输入重新训练相同任务头；结果表示该输入子集下的最佳头部解码能力。

Clay可在同步修改波长和归一化配置后进行子集可学习性实验。AnySat固定S2入口若不支持变长通道，则记为不支持，不以补零替代正式实验。

两类谱段实验都不能把性能差全部解释为某个波段的物理贡献，因为模型预训练分布和输入接口也会影响结果。

## 九、代码基础设施

在纯推理框架基础上新增：

```text
src/rs_backbone_eval/
├── features/
│   ├── schema.py
│   └── cache.py
├── heads/
│   ├── base.py
│   ├── linear_probe.py
│   └── lite_seg.py
├── training/
│   ├── trainer.py
│   ├── losses.py
│   ├── checkpoint.py
│   └── freeze_guard.py
├── evaluation/
│   ├── segmentation.py
│   ├── bootstrap.py
│   └── threshold.py
└── reporting/
    └── comparison.py
```

建议提供以下命令：

```bash
rs-eval prepare-data --config configs/datasets/sen1floods11.yaml
rs-eval cache-features --config configs/experiments/s2_anysat.yaml
rs-eval train-head --config configs/experiments/s2_anysat_linear.yaml --seed 42
rs-eval evaluate --run outputs/<run_id> --split test
rs-eval compare --task s2-water --head linear
```

每个训练目录保存：

```text
outputs/<run_id>/
├── resolved_config.yaml
├── environment.json
├── manifest.parquet
├── encoder.json
├── head_checkpoint.safetensors
├── training_history.parquet
├── metrics.json
├── metrics_per_scene.parquet
├── predictions/
├── cases/
└── report.md
```

`encoder.json` 至少记录上游仓库、commit、checkpoint URL、SHA256、预处理配置哈希和训练前后主干哈希。

## 十、实施顺序与验收

### 阶段A：数据和冻结机制

- 建立两个数据集的唯一 manifest；
- 完成划分、防重复和地理对齐检查；
- 接入主干并验证空间特征；
- 实现冻结检查和特征缓存。

验收：3–5景能够完成“读取 → 特征 → 任务头 → 损失 → 反向传播”，且主干参数和梯度保持不变。

### 阶段B：单种子基线

- 每个主实验组合运行 LinearProbeHead；
- 再运行 LiteSegHead；
- 检查学习曲线、类别塌缩、阈值和预测空间位置；
- 固定正式超参数，不按最终集合结果反复调整。

验收：训练损失可下降，validation预测不全为单一类别，输出可正确恢复到标签网格。

### 阶段C：正式重复实验

- 使用3个固定种子运行锁定配置；
- 对最终集合运行一次评价；
- 汇总均值、标准差、置信区间、成本和失败案例。

验收：每个结果都能追溯到数据manifest、主干权重、预处理配置、任务头checkpoint和完整运行配置。

### 阶段D：可选扩展

- Clay谱段子集可学习性；
- LIM统一任务头；
- AnySat/Clay原生优化头；
- 主干部分微调。

扩展实验不阻塞阶段A–C的交付。

## 十一、首轮交付表

| 交付物 | 内容 |
| --- | --- |
| 数据审计 | 样本数、划分、重复、标签比例、无效像素、日期和地理元数据 |
| 模型审计 | 代码版本、权重哈希、输入约定、输出步长、是否严格冻结 |
| 训练记录 | 两种任务头、3个种子、学习曲线、最佳epoch、任务头参数量 |
| 定量结果 | IoU、F1、Precision、Recall、mIoU、混淆矩阵和置信区间 |
| 定性结果 | 每个任务的成功、漏检、误检和边界失败案例 |
| 成本结果 | 特征提取时间、头训练时间、端到端推理时间、峰值显存和缓存空间 |
| 结论 | 分开陈述表征可分性、轻量迁移能力、输入适配限制和下一步建议 |

## 参考资料

- [AnySat官方代码与特征接口](https://github.com/gastruc/AnySat)
- [Clay v1.5 Basic Use](https://clay-foundation.github.io/model/getting-started/basic_use.html)
- [Clay分割头训练示例](https://clay-foundation.github.io/model/finetune/segment.html)
- [Sen1Floods11官方仓库](https://github.com/cloudtostreet/Sen1Floods11)
- [HLS Burn Scar数据卡](https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars)
