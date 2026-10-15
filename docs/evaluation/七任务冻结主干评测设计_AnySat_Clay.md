# AnySat 与 Clay 七任务下游评测设计（Prithvi-EO-2.0 对照）

版本：v3.0 ｜ 日期：2026年10月12日 ｜ 状态：目标已校正，待实现

## 一、范围与比较口径

任务范围采用 Prithvi-EO-2.0 官方仓库列出的7个下游应用任务。欧洲 Sen4Map 同时提供土地覆盖和作物分类两份配置，因此工程上是**7个应用任务、8份任务配置**。

这7项更准确的名称是“Prithvi-EO-2.0 下游任务组”，不是论文中另一个独立实验——GEO-Bench 12数据集基准。各数据集也不是都由Prithvi团队创建；它们是Prithvi官方选择、发布配置并在论文中报告结果的下游数据集。

**主实验目标**是：在同一批下游数据、官方划分和指标上训练 AnySat、Clay，并与复现的 Prithvi-EO-2.0 结果比较。所有任务都需要监督训练任务头；为了和Prithvi论文的全量微调结果形成有效对照，主实验还需要允许微调主干。冻结主干训练任务头保留为第二条实验轨道，用来比较预训练表征本身。

Prithvi 的 TerraTorch 配置用于锁定数据结构、标签、波段、时相、划分、指标和参考训练做法，不代表 AnySat、Clay 自动兼容相同输入。每个主干必须先经过传感器、波段、时间维度和空间输出兼容性检查。

本项目主要回答：

1. AnySat、Clay 在7项Prithvi下游任务上的可达到性能；
2. 在相同数据划分、输入信息、任务头和训练预算下，三种主干的冻结迁移能力；
3. 模型原生输入能力、全量微调和传感器适配分别带来多少收益；
4. 复现结果与Prithvi论文值是否同量级，差异能否由协议、输入或优化解释。

论文中的Prithvi数值先作为sanity check。仓库文件名虽是官方sample config，其中若干训练轮数、损失、裁剪和decoder与论文实验描述不同，不能直接把运行YAML所得数值称为论文复现。正式结论优先使用本项目在相同代码、数据版本和评价器下重跑的Prithvi结果。不同输入波段、不同任务头类型、冻结主干与全量微调不得放进同一个排行榜。

## 二、七任务清单

| ID | 应用任务 | 数据集 | 标准输入 | 输出类型 | 类别／通道 | 主指标 |
| --- | --- | --- | --- | --- | --- | --- |
| T1 | Flood Detection | Sen1Floods11 | 单日期S2六波段 | 像素二分类 | 水／非水 | 水体IoU、mIoU |
| T2 | Wildfire Scar Detection | HLS Burn Scar | 单日期HLS六波段 | 像素二分类 | 烧毁／未烧毁 | 火烧迹地IoU |
| T3 | Burn Scar Intensity | Burn Intensity | pre/during/post三期HLS六波段 | 像素多分类 | 5个强度等级 | mIoU |
| T4 | Landslide Detection | Landslide4Sense | 单日期S2六波段 | 像素二分类 | 滑坡／非滑坡 | mIoU、F1 |
| T5 | Multi-temporal Crop Segmentation (US) | HLS Multi-temporal Crop Classification | 三期HLS六波段 | 像素多分类 | 13类，0为无效标签 | mIoU |
| T6a | Land Cover Classification (Europe) | Sen4Map | 12期S2十波段 | 场景多分类 | 10类 | Mean/weighted F1，按官方实现锁定 |
| T6b | Crop Classification (Europe) | Sen4Map | 12期S2十波段 | 场景多分类 | 8类 | Mean/weighted F1，按官方实现锁定 |
| T7 | Above Ground Biomass Estimation | BioMassters | 四期S2六波段 | 像素回归 | 1个AGB值 | RMSE、R² |

T6a与T6b共享影像数据，但标签映射、样本筛选和输出空间不同，必须生成两个独立 `TaskSpec`、checkpoint 和结果表。

### 2.1 多光谱结论

官方8份配置**全部使用RGB以外的光学波段**：

- T1、T2、T3、T4、T5、T7使用 `Blue, Green, Red, NIR, SWIR1, SWIR2` 六波段；
- T6a、T6b使用S2十波段，在上述波段之外还包含三个红边波段和另一条NIR波段；
- Landslide4Sense原数据另有DEM和坡度，BioMassters原数据另有Sentinel-1，但Prithvi参考配置没有使用这些额外模态，因此主对照实验也不使用。

这说明它们适合检验AnySat、Clay相对RGB模型的多光谱价值。它不表示任务理论上不能只靠RGB完成；准确表述是“Prithvi官方协议使用并依赖多光谱输入，RGB-only结果属于消融实验”。每项任务都应补一个RGB消融，量化非RGB波段的实际增益。

## 三、模型兼容性门禁

### 3.1 预期兼容矩阵

| 配置 | AnySat | Clay v1.5 | 说明 |
| --- | --- | --- | --- |
| T1 洪水 | 可运行；原生S2入口需十波段 | 可运行；需六波段S2元数据 | 官方对照是S2六波段；AnySat需输入适配或另列十波段原生轨道 |
| T2 火烧迹地 | 条件支持 | 条件支持 | HLS六波段；AnySat先验证L7入口，Clay建立HLS元数据 |
| T3 火烧强度 | 条件支持 | 条件支持 | 同上，并处理三时相融合 |
| T4 滑坡 | 可运行；原生S2入口需十波段 | 可运行 | 原数据包含AnySat所需S2十波段；官方六波段对照仍需适配 |
| T5 美国作物分割 | 条件支持 | 条件支持 | HLS六波段、三时相 |
| T6a 欧洲土地覆盖 | 原生支持 | 可运行 | AnySat原生S2时间序列；Clay逐日期编码后融合 |
| T6b 欧洲作物 | 原生支持 | 可运行 | 与T6a共享输入，标签映射不同 |
| T7 地上生物量 | 可运行；原生S2入口需十波段 | 可运行 | 官方四期S2六波段；需锁定缺测和选帧规则 |

“条件支持”表示公开权重没有完全匹配的HLS入口，需要先做小样本前向验证，并在以下两种办法中锁定一种：使用语义最接近的预训练入口，或训练轻量光谱输入适配器。输入适配器的参数量和训练状态必须单列，不能把它计作“仅任务头”。

### 3.2 四条结果轨道

为避免把不同能力混在一起，结果分四条轨道：

1. **B0：Prithvi参考复现轨道**：先建立论文协议与仓库sample YAML的差异表；信息充分时按论文协议复现，信息不足时按固定commit的YAML复现并明确标为“repo-config reproduction”。它既校验数据管线，也提供本机基线。
2. **B1：全量下游适配轨道（主结果）**：三种模型都训练任务头并微调主干；分割/回归尽量使用同等容量的decoder，分类使用同等容量的MLP头；每个模型获得相同的超参数搜索预算。它回答实际下游效果，并与论文全量微调结果对照。
3. **B2：冻结表征轨道（辅助结果）**：三种主干都冻结，使用同一线性头和同一轻量任务头。单日期任务提取空间特征，多日期任务按各模型公开接口编码后由统一头融合。它回答预训练表征质量，不能直接与论文全量微调数值比较。
4. **B3：模型原生输入轨道（能力上限补充）**：AnySat在可用时接收完整S2十波段和原生时间序列，Clay使用其官方传感器配置；Prithvi也在相同新增波段上重跑。它衡量模型原生能力，不与六波段论文值混表。

同一张严格比较表中的模型必须使用相同样本、波段、日期选择、地面范围、标签、训练状态和指标实现。无法满足时拆表并标为“协议差异对照”。

## 四、数据协议

### 4.1 统一样本与时间序列

```python
GeoSequenceSample(
    sample_id,
    group_id,
    images,          # [T,C,H,W]
    bands,
    timestamps,      # [T]，真实日期或明确的pre/during/post顺序
    temporal_mask,   # [T]，缺测日期为False
    label,
    label_mask,
    crs,
    transform,
    gsd_m,
    split,
)
```

manifest至少记录：

```text
task_id,sample_id,group_id,split,image_paths,label_path,sensor,
bands,timestamps,temporal_mask,crs,transform,gsd_m,checksum
```

所有日期必须作为一个 `sample_id` 的整体划分，禁止同一地点或事件的不同日期跨 train、validation 和 test。

### 4.2 各任务划分规则

| 配置 | 划分规则与审计重点 |
| --- | --- |
| T1 | 使用Sen1Floods11官方train/validation/test；按chip ID映射S2与标签，并检查洪水事件是否跨集合 |
| T2 | 官方training内固定划出项目validation；官方validation锁定为最终 `official_validation`，不称为独立test |
| T3 | 以火灾case为group，将pre/during/post绑定；优先复现论文80/20和发布数据的split字段，若两者不一致则分别固化为paper/repo协议，不能临时另造划分冒充官方结果 |
| T4 | 使用Landslide4Sense官方train/validation/test；首轮只取TerraTorch参考配置的六个光学波段 |
| T5 | 官方train内划项目validation，官方validation只用于最终评价；三个时相绑定且0标签映射为ignore |
| T6a/T6b | 使用Sen4Map提供的train/validation/test HDF5及key文件；两种标签映射分别审计类别数和样本数 |
| T7 | 使用BioMassters既有train/validation/test；四帧由同一固定算法选择，缺测帧用mask处理，不以零值冒充真实观测 |

归一化统计、类别权重、回归目标均值方差和分类阈值只能从训练集合得到。发布方提供且与传感器完全匹配的预训练统计量优先于重新计算。

### 4.3 输入波段

主轨道首先复现Prithvi官方输入：

- T1：S2 `Blue,Green,Red,NIR_NARROW,SWIR1,SWIR2`；
- T2：HLS `Blue,Green,Red,NIR_BROAD,SWIR1,SWIR2`；
- T3：三期HLS `Blue,Green,Red,NIR,SWIR1,SWIR2`；
- T4：S2 `Blue,Green,Red,NIR_BROAD,SWIR1,SWIR2`，不输入坡度和DEM；
- T5：三期HLS `Blue,Green,Red,NIR_NARROW,SWIR1,SWIR2`；
- T6a/T6b：12期S2十波段 `B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12`；
- T7：四期S2 `Blue,Green,Red,NIR_NARROW,SWIR1,SWIR2`，不输入Sentinel-1。

AnySat的公开 `s2` 入口固定为S2十波段，不能靠补零伪装六波段。B1主轨道若要保持官方六波段信息量，必须使用经验证的六波段入口或训练一个显式记录的光谱适配器；B3再使用完整十波段。Clay按波长编码输入，但HLS仍需明确产品来源、中心波长、反射率尺度和归一化统计，不能只改传感器名称。

公共数据层保留原始数值，模型适配器分别执行官方缩放和归一化。

## 五、主干与特征协议

### 5.1 AnySat

- S2输入形状为 `B,T,10,H,W`，日期为 `B,T` day-of-year；
- 单日期任务仍保留 `T=1`；
- 空间任务提取 `dense` 特征；场景分类同时保留 `tile` 和由dense特征池化的表示，正式配置预先锁定其一；
- B2冻结轨道把每个日期作为 `T=1` 分别编码；B1/B3在接口允许时一次输入完整序列；
- Landslide4Sense原始12条S2光学波段包含AnySat `s2` 入口需要的十波段，B3可以直接使用；
- 对HLS六波段先验证AnySat公开 `l7` 入口的波段语义和数值域。若样本来自HLS S30或统计分布不匹配，必须训练独立HLS输入适配器并单列结果；
- 不使用补零把六波段数据伪装成S2十波段。

### 5.2 Clay v1.5

- 每个日期输入 `B,C,H,W`，并同步提供波长及时间、位置元数据；
- 多日期样本逐日期运行同一个冻结编码器，返回 `B,T,C,h,w`；
- 空间任务必须取得空间token或feature map，池化后的1024维场景embedding只用于场景分类；
- S2和Landsat使用Clay官方元数据；HLS和Landslide六波段子集建立可审计的派生元数据，保存波长、均值、标准差、GSD和数值缩放；
- 若需修改上游模块以暴露空间token，固定上游commit，并用官方池化输出做一致性回归检查。

### 5.3 统一特征对象

```python
BackboneFeatures(
    spatial,          # [B,T,C,h,w]，空间任务必需
    pooled,           # [B,T,C]，分类任务可用
    temporal_mask,    # [B,T]
    timestamps,
    stride_y,
    stride_x,
    encoder_id,
    checkpoint_sha256,
    preprocessing_id,
)
```

特征缓存键必须包含数据manifest、主干代码commit、权重哈希、预处理配置、波段、选帧算法、裁块和输出类型。

## 六、任务头与微调设计

7项任务对AnySat和Clay都没有可直接复用的统一标签头，因此**8份配置全部需要监督训练任务头**。区别在于主干是否冻结，以及是否还需要训练输入适配器。

### 6.1 线性探针组

| 输出类型 | 线性探针 |
| --- | --- |
| 单日期像素分类 | `1×1 Conv(C,K)` 后双线性恢复到标签网格 |
| 多日期像素分类 | 对有效日期特征做masked mean，再接 `1×1 Conv(C,K)` |
| 多日期场景分类 | 空间与有效日期masked mean，再接 `Linear(C,K)` |
| 多日期像素回归 | 有效日期masked mean，再接 `1×1 Conv(C,1)` |

线性探针不使用隐藏层。时间平均是无参数操作，因此结果表示最基本的可解码性，不表示模型充分利用了物候顺序。

### 6.2 统一轻量空间头

用于T1、T2和T4：

```text
[B,C,h,w]
  → 1×1 Conv(C,256) + GroupNorm + GELU
  → 2个残差3×3 Conv块
  → Dropout2d(0.1)
  → 1×1 Conv(256,K)
  → resize到标签网格
```

### 6.3 统一轻量时间头

用于T3、T5、T6a、T6b和T7：

```text
[B,T,C,h,w]
  → 各日期共享1×1投影到256维
  → 加真实日期编码和temporal mask
  → 2层、4头的轻量Temporal Transformer
  → 空间任务：2个残差卷积块 + 像素输出层
  → 分类任务：空间池化 + Linear(256,K)
  → 回归任务：残差卷积块 + 1×1 Conv(256,1)
```

Temporal Transformer在同一空间位置上沿时间轴运行。所有模型使用相同宽度、层数和时间编码。任务头参数量、MACs和输入特征步长随结果一起报告。

### 6.4 任务到输出头的映射

| 配置 | `K` | 头类型 |
| --- | ---: | --- |
| T1/T2/T4 | 1 | BinarySpatialHead |
| T3 | 5 | TemporalSegmentationHead |
| T5 | 13 | TemporalSegmentationHead |
| T6a | 10 | TemporalClassificationHead |
| T6b | 8 | TemporalClassificationHead |
| T7 | 1 | TemporalRegressionHead |

### 6.5 主结果decoder

B1全量适配轨道不能只用线性探针去对照Prithvi论文结果：Prithvi官方分割和回归配置使用UPerNet类decoder并更新主干，Sen4Map使用分类头并更新主干。因此：

- T1、T2、T3、T4、T5、T7：为三种主干接入同等输出尺度和相近参数量的多尺度decoder；若某主干无法稳定提供多层特征，则使用单尺度decoder并拆表报告；
- T6a、T6b：使用相同宽度和层数的分类MLP；
- 任务头参数量、可训练总参数量、训练FLOPs和峰值显存随成绩报告；
- B2继续使用本节的线性探针和统一轻量头，不与B1混排。

## 七、冻结与训练协议

### 7.1 冻结保证

以下保证只适用于B2冻结表征轨道：

- 主干参数全部 `requires_grad=False`；
- 主干始终为 `eval()`，任务头单独为 `train()`；
- optimizer只接收任务头参数；
- 在线主干前向使用 `torch.no_grad()` 并立即 `detach()`；缓存提取可使用 `torch.inference_mode()`；
- 训练前后主干 `state_dict` 哈希必须相同；
- 自动测试确认反向传播后所有主干梯度为 `None`。

### 7.2 sample YAML损失锚点

| 配置 | 损失 |
| --- | --- |
| T1/T2 | sample YAML的CrossEntropy，忽略 `-1` |
| T3 | sample YAML类别权重的CrossEntropy，忽略 `-1`；严重度等级MAE只作辅助监控 |
| T4 | sample YAML的Focal loss；论文协议另有Lovasz和2:8加权CE |
| T5 | sample YAML的13类权重CrossEntropy，执行 `reduce_zero_label` 并忽略 `-1` |
| T6a/T6b | CrossEntropy |
| T7 | 由锁定的TerraTorch版本解析 `PixelwiseRegressionTask` 默认损失并写入resolved config；辅助报告MSE/MAE |

同一B1任务内三种模型使用完全相同的损失定义、标签映射和类别权重。若另做Dice组合损失实验，单独作为消融，不替换锁定协议结果。

### 7.3 仓库sample YAML训练预算锚点

| 配置 | epochs | AdamW初始学习率 | weight decay | 调度器 |
| --- | ---: | ---: | ---: | --- |
| T1 | 50 | `5e-5` | `0.05` | cosine |
| T2 | 50 | `5e-5` | `0.05` | cosine |
| T3 | 50 | `1e-5` | `0.10` | StepLR(3, 0.9) |
| T4 | 50 | `5e-5` | `0.10` | cosine |
| T5 | 125 | `3e-4` | `0.35` | StepLR(10, 0.9) |
| T6a/T6b | 20 | `6e-5` | `0.05` | ReduceLROnPlateau |
| T7 | 125 | `1e-4` | `0.05` | StepLR(10, 0.9) |

上表只描述当前仓库sample YAML，不能替代论文协议。已确认的差异至少包括：

| 配置 | 论文实验描述 | sample YAML |
| --- | --- | --- |
| T1/T2 | 512裁到448，再随机裁224；10次超参试验、最佳配置10个种子、50 epochs | 直接Resize 224；单个seed字段、50 epochs |
| T3 | 80/20划分、150 epochs、加权CE、卷积上采样decoder | 50 epochs、加权CE、UPerNet |
| T4 | 100 epochs、Adam `1e-5`、Lovasz或2:8加权CE、自定义反卷积decoder | 50 epochs、AdamW `5e-5`、focal、UPerNet |
| T5 | 80 epochs、加权CE、简单卷积上采样decoder | 125 epochs、加权CE、UPerNet |
| T6a/T6b | 论文按多个数据比例和多个种子报告 | YAML给出全量数据训练入口，20 epochs |
| T7 | 论文中的Prithvi-UNet数据过滤和训练过程需从正文补齐 | YAML给出四帧S2、UPerNet、125 epochs |

B0分别保存 `paper_protocol` 和 `repo_config_protocol`，不得混用名称。B1对AnySat、Clay和本项目重跑的Prithvi使用相同epoch/step预算、增强和验证选择规则，但学习率不能机械照搬某个主干：每个模型在同一对数网格上获得相同次数的验证集搜索，主干和新任务头分别设学习率。B2只优化任务头及显式声明的输入适配器。

开发阶段用一个固定种子做smoke test；配置锁定后运行3个种子 `17,42,73`。不得查看最终测试集后为单个模型单独调参。混合精度优先BF16，记录梯度累积、有效batch size、训练step和提前停止点。

### 7.4 数据增强

B0严格跟随所选paper/repo协议。B1三种模型统一使用水平翻转、垂直翻转和任务协议规定的resize/crop；只有协议明确采用时才加入90度旋转。所有日期和标签同步变换。场景分类不做会改变标签语义的空间裁剪。首轮不使用逐波段颜色抖动、MixUp或时间顺序打乱。

## 八、评价协议

| 输出类型 | 主指标 | 辅助指标 |
| --- | --- | --- |
| 二分类分割 | 前景IoU | F1、Precision、Recall、mIoU、混淆矩阵 |
| 多分类分割 | macro mIoU | macro F1、每类IoU、频率加权IoU |
| 火烧强度分割 | macro mIoU | 每类IoU、预测等级MAE、严重类别召回率 |
| 场景多分类 | Macro-F1 | Accuracy、Balanced Accuracy、每类Recall、混淆矩阵 |
| 像素回归 | RMSE | MAE、R²、Bias、按AGB区间误差 |

正式主指标必须先通过Prithvi发布代码或锁定版TerraTorch确认平均方式。特别是Sen4Map论文正文使用weighted-averaged F1，不能未经核对改成macro-F1。结果文件同时保存逐类混淆矩阵，使macro、weighted和micro口径都能复算。

指标只在有效标签上计算。像素任务先跨整个集合累计混淆矩阵或误差和，再计算全局指标；同时报告每景分布和按样本bootstrap的95%置信区间。

三个种子报告均值、标准差和每个种子的原始结果。论文若使用10个种子，则B0按论文设置或明确标注未完全复现。推理成本同时报告端到端时间、仅任务头时间、峰值显存和异常率。

### 8.1 论文结果sanity check

以下数值只用于检查数量级，不能替代本项目同协议重跑：洪水最佳报告水体IoU为83.1；火烧迹地最佳报告火烧类IoU为83.2；火烧强度300M报告mIoU为29.4；美国多时相作物600M报告mIoU为50.7；Sen4Map全量数据的土地覆盖和作物分类600M F1分别为76.1和84.6；BioMassters中Prithvi-UNet 300M测试RMSE为54.51。Landslide论文表格与相邻文字对最佳模型标识存在不一致，复现注册表需保留原表行并记录该问题。

每项任务另做一次固定种子的RGB消融。RGB消融只回答“非RGB波段贡献多少”，不进入官方六/十波段排行榜。

## 九、代码基础设施

可执行仓库结构、外置存储、worker隔离、运行契约和Portal详见[《代码基础设施与 Web Portal 设计》](../architecture/代码基础设施与Web_Portal设计.md)。

### 9.1 配置和注册表

```text
configs/
├── protocols/
│   ├── paper_reference/
│   ├── prithvi_repo_config/
│   ├── full_finetune/
│   └── frozen_probe/
├── tasks/
│   ├── flood.yaml
│   ├── fire_scars.yaml
│   ├── burn_intensity.yaml
│   ├── landslide.yaml
│   ├── us_multicrop.yaml
│   ├── sen4map_landcover.yaml
│   ├── sen4map_crops.yaml
│   └── biomass.yaml
├── models/
│   ├── anysat.yaml
│   ├── clay_v1_5.yaml
│   └── prithvi_reference.yaml
└── heads/
    ├── linear.yaml
    ├── lite_spatial.yaml
    └── lite_temporal.yaml
```

`TaskSpec`至少包含：

```text
task_id, task_family, dataset_adapter, sensor, bands, time_steps,
class_names, ignore_index, split_policy, loss, metrics, primary_metric
```

### 9.2 源码模块

```text
src/rs_backbone_eval/
├── data/
│   ├── manifest.py
│   ├── sequence.py
│   └── datasets/{sen1floods11,hls_burn_scars,burn_intensity,
│                 landslide4sense,multicrop,sen4map,biomassters}.py
├── models/{base,anysat,clay,prithvi}.py
├── adapters/{spectral,temporal,multiscale}.py
├── features/{schema,cache}.py
├── heads/{linear,spatial,temporal,classification,regression}.py
├── training/{trainer,losses,freeze_guard,checkpoint}.py
├── evaluation/{segmentation,classification,regression,bootstrap}.py
└── reporting/{task_report,comparison}.py
```

TerraTorch数据模块和配置可以作为数据语义参考或Prithvi runner使用，但公共评测协议由本项目控制，避免AnySat、Clay因不同上游训练器获得不同评价规则。另建 `references/prithvi_protocol_registry.yaml`，逐任务记录论文版本、论文表号、官方仓库commit、YAML哈希、已知协议差异和目标数值。

### 9.3 命令行

```bash
rs-eval audit-data --task T5
rs-eval probe-model --model clay --task T5
rs-eval cache-features --model clay --task T5 --protocol frozen-probe
rs-eval reproduce --model prithvi_eo_v2_300 --task T5 --protocol repo-config
rs-eval finetune --model clay --task T5 --protocol full-finetune --seed 42
rs-eval train-head --model clay --task T5 --protocol frozen-probe --head linear --seed 42
rs-eval evaluate --run outputs/<run_id> --split official_validation
rs-eval compare --task T5 --protocol full-finetune
```

每次运行保存protocol ID、resolved config、manifest哈希、环境、代码commit、主干权重哈希、输入波段、可训练模块与参数量、checkpoint、训练曲线、逐样本指标、预测、案例图和报告。

## 十、执行顺序

### P0：兼容性和数据审计

逐项下载元数据或小样本，核对波段、时间、标签、划分、缺测和空间对齐。完成论文与sample YAML协议注册表。每个“模型 × 任务 × 输入协议”先生成 `supported/conditional/unsupported` 报告和一次真实前向结果。

### P1：Prithvi基线和框架原型

先完成T1洪水：跑通Prithvi repo-config基线、AnySat和Clay，再完成B1全量适配与B2冻结探针。随后用T2验证HLS输入适配。两项共同覆盖S2与HLS、二分类空间decoder和完整评价链路。

### P2：其余空间分割

加入T4滑坡，并完成统一空间头、类别不平衡处理和小目标案例分析。

### P3：多时相分割

加入T3火烧强度和T5美国作物，完成时间序列manifest、temporal mask和统一轻量时间头。

### P4：分类与回归

加入T6a/T6b Sen4Map分类和T7 BioMassters像素回归，补齐分类、回归指标及报告模板。

### P5：正式重复实验

锁定配置后运行3个种子；对论文明确使用10个种子的B0实验，资源允许时按10个种子复现。生成分任务结果表、协议差异表、兼容性矩阵、成本表、RGB消融和选型建议。

不再沿用旧版“72次任务头训练”的估算，因为主实验已经改为全量下游适配，且8份配置的训练预算差异很大。P0/P1测得单step显存与耗时后再生成资源表。最低正式矩阵为 `3个模型 × 8份配置 × 3个种子 = 72次B1训练`；B0协议复现、B2冻结探针、B3原生输入和RGB消融分别计费，不能藏在这72次内。

## 十一、验收标准

1. 7个应用任务和8份配置都有B1目标结果；若某组合不能运行，必须给出真实前向失败证据和适配尝试记录。
2. 每个已支持任务都有固定manifest、标签映射、输入波段、时间选择和评价指标。
3. B2主干训练前后哈希一致；B1保存完整checkpoint、可训练参数清单和训练日志。
4. 同表比较的结果满足相同输入、划分、头类型、训练状态和搜索预算。
5. 每个正式配置保留3个种子的完整记录，不只报告最好结果。
6. 交付分割、分类和回归三类报告，以及论文/YAML协议差异、兼容性、精度和成本的综合结论。

## 参考资料

- [Prithvi-EO-2.0七任务与TerraTorch配置清单](https://github.com/NASA-IMPACT/Prithvi-EO-2.0)
- [AnySat官方代码与输入接口](https://github.com/gastruc/AnySat)
- [Clay v1.5 Basic Use](https://clay-foundation.github.io/model/getting-started/basic_use.html)
- [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11)
- [HLS Burn Scar](https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars)
- [Burn Intensity](https://huggingface.co/datasets/ibm-nasa-geospatial/burn_intensity)
- [Landslide4Sense](https://huggingface.co/datasets/ibm-nasa-geospatial/Landslide4sense)
- [HLS Multi-temporal Crop Classification](https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification)
- [Sen4Map](https://datapub.fz-juelich.de/sen4map/)
- [BioMassters](https://huggingface.co/datasets/ibm-nasa-geospatial/BioMassters)
