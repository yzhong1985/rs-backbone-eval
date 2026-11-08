# T11O128：OlmoEarth v1.3 128 维年度 embedding 与 64 维对照（2026-11-08 定稿）

- 任务：T11O128（PASTIS-HD，与 T11/T11O 同一批 patch、标签、Fold 1–3/4/5，sample_id 一一对应）
- 输入：OlmoEarth v1.3 2019 年度 embedding 的**完整 128 维**（64 维是其 Matryoshka 前缀），10 m
- 头与预算：AEF worker 的 `MLPHead` / `UPerNetHead`，固定第 8 轮，不做 validation 选模
- seed：17 / 42 / 73；设备：CPU（32 cores）
- manifest：`5c0fbe78803b3a071d13b8d79cbec82a0f376d783a96133a4e29ff7037135a17`
  （2,433；train 1,455 / validation 482 / test 496）
- 代码：`37883a4`（干净树）；`scripts/audit_runs.py` 判定 T11O128 6/6 满足正式预算

## 1. 结论

**在完全相同的数据、划分、头、8-epoch 预算和种子上，完整 128 维在两个头上都优于
64 维，6 个配对差值全为正：**

| 输入 | 头 | test mIoU（mean ± sd, n=3） | 相对 64 维（配对） |
|---|---|---:|---:|
| 64 维 | MLP | 0.5788 ± 0.0107 | — |
| **128 维** | **MLP** | **0.6137 ± 0.0132** | **+0.0349 ± 0.0027** |
| 64 维 | UPerNet | 0.5799 ± 0.0105 | — |
| **128 维** | **UPerNet** | **0.6027 ± 0.0143** | **+0.0228 ± 0.0057** |

配对差值（同 seed，128 − 64）：

- MLP：+0.0321 / +0.0375 / +0.0353，均值 **+0.0349**（sd 0.0027）
- UPerNet：+0.0170 / +0.0229 / +0.0284，均值 **+0.0228**（sd 0.0057）

即：**64 维确实截断了有用信息**。这是输入预算的收益，不是模型或产品差异——两行是
同一个产品的两个条件，不能说“OlmoEarth 架构更强”。

## 2. 逐 seed

| 输入 | 头 | s17 | s42 | s73 | mean ± sd |
|---|---|---:|---:|---:|---:|
| 64 维 | MLP | 0.5664 | 0.5846 | 0.5854 | 0.5788 ± 0.0107 |
| 128 维 | MLP | 0.5985 | 0.6220 | 0.6207 | 0.6137 ± 0.0132 |
| 64 维 | UPerNet | 0.5794 | 0.5697 | 0.5907 | 0.5799 ± 0.0105 |
| 128 维 | UPerNet | 0.5964 | 0.5926 | 0.6191 | 0.6027 ± 0.0143 |

64 维数字来自 `-v3`（`364b948`，manifest `ef75efa9…`）；128 维来自 `-v1`
（`37883a4`，manifest `5c0fbe78…`）。两批 run 的 patch/fold/标签完全同一份。

## 3. 训练与准入

| 项 | 值 |
|---|---|
| 编码 | `olmoearth_int8`，`sign(v)*(|v|/127.5)^2`，-128=NODATA |
| 无效像元口径 | `zero-fill-full-label`（与 AEF、T11O 一致） |
| preprocessing | `olmoearth_signed_square_imagenetfree` |
| 头 / lr | MLP 3e-3；UPerNet 1e-3；AdamW，weight decay 1e-4 |
| 输入通道 | 128（`OLE_00`–`OLE_127`，头按 `len(task_spec.bands)` 自动适配） |
| 统计量 | 训练集 tile-balanced 80 patch，逐 run 计算 |
| 评测样本 | test 496/496，validation 482/482（硬锚点） |
| 耗时 | MLP 3.0–3.1 min；UPerNet 29.2–29.4 min |
| worker 校验 | `payload.ok` 2433/2433（启动时逐样本核对 sha256） |
| run id | `olmoearth_v13_embedding-T11O128-olmoearth_v13_probe_{mlp,upernet}_128-train-{17,42,73}-v1` |

## 4. 与预算扩展的关系（3 seed）

| 产品 | 维度 | 头 | 8 ep | 20 ep | 20ep − 8ep（同 seed） |
|---|---|---:|---:|---:|---:|
| AEF | 64 | MLP | 0.4849 ± 0.0081 | 0.5037 ± 0.0028 | +0.0188 ± 0.0079 |
| AEF | 64 | UPerNet | 0.4879 ± 0.0099 | 0.5112 ± 0.0131 | +0.0233 ± 0.0218 |
| OlmoEarth | 64 | MLP | 0.5788 ± 0.0107 | 0.5956 ± 0.0041 | +0.0168 ± 0.0131 |
| OlmoEarth | 64 | UPerNet | 0.5799 ± 0.0105 | 0.6013 ± 0.0033 | +0.0213 ± 0.0133 |
| OlmoEarth | 128 | MLP | 0.6137 ± 0.0132 | **0.6314 ± 0.0022** | +0.0177 ± 0.0135 |
| OlmoEarth | 128 | UPerNet | 0.6027 ± 0.0143 | **0.6183 ± 0.0211** | +0.0156 ± 0.0322 |

- 12 条 20ep 诊断（6 cell × 3 seed）全部 `panel=diagnostic`，不进矩阵/Portal。
- **预算没有放大 128 维的优势**：128 − 64 在 8ep 是 +0.0349 / +0.0228
  （MLP / UPerNet），20ep 是 +0.0358 / +0.0171。UPerNet 的 20ep s73 甚至是 −0.019。
  所以 8ep 的配对差已经能代表这个量级，不能说“8ep 低估 128 维”或“是下界”。
- 产品差距同样稳定：AEF 仍低 +0.0919（MLP）/ +0.0901（UPerNet），8ep 是
  +0.0939 / +0.0921；已测预算下产品差距不是预算造成的。

## 5. 解读边界

1. **输入预算差异，不是模型差异**：64/128 是同一产品、同一任务的两行条件；
   报告和兼容性表按“声明波段数”并列，不合并成一个 cell。
2. **AEF 对照口径不变**：AEF 年度 embedding 是 64 维，没有 128 维版本；产品对照
   仍用 64 vs 64（+0.0939 / +0.0921），不能拿 128 维与 AEF 比。
3. **不并入 D1**：T11O128 不在 Prithvi-suite D1，也不与 T11 的 BBF/BBT 混排。
4. **单数据集、单年份**：PASTIS-HD 法国农田 2019，不能外推到其他区域/传感器。
5. **LPIS 标签年**沿用 T11 的 2019 技术假设。
6. 弱类（Grapevine/Beet/Orchard）仍接近 0；类别级差异需另出表与混淆矩阵。

## 6. 复现

```bash
export RS_EVAL_MANIFEST=/geodata/rs-eval/data/manifests/T11O128/5c0fbe78….jsonl
RS_EVAL_T11O128_SUFFIX=v1 bash scripts/rerun_baselines.sh --detached T11O128
```

lane 串行跑 6 条，实测约 1.6 h（00:32 → 02:10）。
