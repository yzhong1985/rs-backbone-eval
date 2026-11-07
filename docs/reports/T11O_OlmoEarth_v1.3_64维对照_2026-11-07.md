# T11O：OlmoEarth v1.3 64 维年度 embedding 与 AEF 对照（2026-11-07）

- 任务：T11O（PASTIS-HD，与 T11 同一批 patch、标签、Fold 1–3/4/5）
- 输入：OlmoEarth v1.3 2019 年度 embedding，64 维（128 维的 Matryoshka 前缀），10 m
- 头与预算：AEF worker 的 `MLPHead` / `UPerNetHead`，固定第 8 轮，不做 validation 选模
- seed：17 / 42 / 73；设备：CPU
- manifest：`d8f46dcf2f331156e18e01638b8aa56b98a1b15c387055849154a8eeb095a807`
  （2,433；train 1,455 / validation 482 / test 496，sample_id 与 T11 完全一致）
- 代码：`2786489`；`scripts/audit_runs.py` 判定 6/6 满足正式预算

## 1. 结论

**在完全相同的数据、划分、头和 8-epoch 预算下，OlmoEarth v1.3 64 维在 PASTIS-HD
上比 AEF 2019 64 维高约 +0.09 mIoU，三个种子方向一致：**

| 输入（64 维） | 头 | test mIoU（mean ± sd, n=3） |
|---|---|---:|
| AEF 2019 | MLP | 0.4849 ± 0.0081 |
| AEF 2019 | UPerNet | 0.4879 ± 0.0099 |
| **OlmoEarth v1.3 2019** | **MLP** | **0.5759 ± 0.0120** |
| **OlmoEarth v1.3 2019** | **UPerNet** | **0.5756 ± 0.0110** |

这是**年度 embedding 产品**的对比，不是骨干排名；不能外推成“OlmoEarth 模型更强”。

## 2. 逐 seed

| 输入 | 头 | s17 | s42 | s73 | mean ± sd |
|---|---|---:|---:|---:|---:|
| AEF | MLP | 0.4760 | 0.4921 | 0.4866 | 0.4849 ± 0.0081 |
| AEF | UPerNet | 0.4777 | 0.4886 | 0.4974 | 0.4879 ± 0.0099 |
| OlmoEarth v1.3 | MLP | 0.5622 | 0.5819 | 0.5838 | 0.5759 ± 0.0120 |
| OlmoEarth v1.3 | UPerNet | 0.5711 | 0.5676 | 0.5882 | 0.5756 ± 0.0110 |

配对差值（同一 seed，OlmoEarth − AEF）：

- MLP：+0.0861 / +0.0898 / +0.0972，均值 **+0.0910**
- UPerNet：+0.0935 / +0.0790 / +0.0908，均值 **+0.0878**

三个种子全部为正，且两种头给出的差值一致（约 +0.09），不是单一种子或单头驱动。

## 3. 训练与准入

| 项 | 值 |
|---|---|
| 编码 | `olmoearth_int8`，`sign(v)*(|v|/127.5)^2`，-128=NODATA |
| 无效像元口径 | `zero-fill-full-label`（与 AEF 一致：特征置零，标签全范围参与 loss/mIoU） |
| preprocessing | `olmoearth_signed_square_imagenetfree` |
| 头 / lr | MLP 3e-3；UPerNet 1e-3；AdamW，weight decay 1e-4 |
| 统计量 | 训练集 tile-balanced 80 patch，逐 run 计算 |
| 评测样本 | test 496/496，validation 482/482（硬锚点） |
| 耗时 | MLP 2.2–2.3 min；UPerNet 27.2–27.6 min |
| run id | `olmoearth_v13_embedding-T11O-olmoearth_v13_probe_{mlp,upernet}-train-{17,42,73}-v1` |

## 4. 解读边界

1. **产品对比**：AEF 与 OlmoEarth v1.3 都是年度合成 embedding；OlmoEarth 的预训练
   输入含 Sentinel-1/2 与 Landsat，AEF 主要是 S2。差值不能归因到单一传感器或模型。
2. **64 维是 Matryoshka 前缀**：OlmoEarth 128 维包含 64 维之外的信息，128 维结果
   单独成 cell，不与本表混排。
3. **对齐残余**：OlmoEarth 用 10 m 网格最近像元，24-patch QC 显示约 ±1 像元不确定性
   （8/24 为 (0,+1)），未做平移修正；+0.09 远大于该量级，但小于 1–2 个 IoU 点的
   差异不应解读。
4. **不并入 D1**：T11O 不在 Prithvi-suite D1，也不与 T11 的 BBF/BBT 全量微调混排。
5. **LPIS 标签年**沿用 T11 的 2019 技术假设，未获数据发布说明确认。
6. 弱类（Grapevine/Beet/Orchard）仍接近 0，本表只报总体 mIoU；类别级差异需另出表。

## 5. 下一步

- 128 维：镜像已包含 128 维（同一 shard 对象），无需再下载；按方案建 T11O128
  task/model/protocol 后提取并跑同款 6 个 run。
- 类别级 IoU 与混淆矩阵：当前 worker 未持久化 per-group confusion，若要做 bootstrap
  置信区间需要补一次带混淆矩阵的重跑（AEF 若要同口径也需重跑）。
