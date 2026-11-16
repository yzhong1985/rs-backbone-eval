# OlmoEarth 在 PASTIS 上的探针：head 与超参数（2026-11-16）

给同事查用。覆盖两组实验：

- **T11O / T11O128**：把 OlmoEarth v1.3 年度 embedding（64 / 128 维）当**输入**，冻结，训一个
  轻量 head（教师侧的探针，也是蒸馏的教师基线）。
- **T11**：OlmoEarth v1.2 tiny **端到端**冻结主干，用 PASTIS 原始 S2 影像，训 UPerNet。

数据/口径：PASTIS-HD，1455 / 482 / 496（train/val/test），20 类，`ignore_index=19`；
test mIoU 为三种子（17/42/73）均值。所有数字来自 `/geodata/rs-eval/runs`。

---

## 1. 共同训练配方

| 项 | 值 |
|---|---|
| 优化器 | AdamW，`weight_decay=1e-4` |
| 学习率 | **恒定**，无 warmup / 无 cosine（本组实验全部没有调度器） |
| 损失 | 交叉熵，`ignore_index=19`；无类权重、无辅助损失 |
| 轮数 | 默认 **8 epoch**；`*_20ep` 是单独注册的**诊断**预算轴 |
| 选模 | **不按 validation 选 epoch**，直接评最后一轮权重 |
| 数据增强 | 无 |
| 归一化 | 特征按**训练折**逐通道均值/标准差标准化后再送 head（val/test 用同一套统计量） |

## 2. T11O / T11O128：OlmoEarth v1.3 embedding 探针

### 2.1 输入

| 项 | 值 |
|---|---|
| 特征 | OlmoEarth v1.3 `annual` 2019，**64 维**（T11O）/ **128 维**（T11O128） |
| 编码 | `olmoearth_int8`（int8 解码后转 float32），preprocessing `signed_square_imagenetfree` |
| 时间步 | 1（年度 embedding，已按年聚合） |
| 无效像元 | 0 填充特征，但**标签与 mIoU 保留全部像元**（策略 A） |

### 2.2 head

**MLP**（`head_id=mlp`，lr **3e-3**）

```
Conv2d(in_ch, 128, 1)  →  GroupNorm(8, 128)  →  GELU  →  Conv2d(128, 20, 1)
```

- 参数量：64 维输入 **11,156**；128 维输入 **19,348**

**UPerNet**（`head_id=upernet`，lr **1e-3**）

```
stem:    Conv3x3(in_ch→128) + GroupNorm(8) + GELU
down2:   Conv3x3(128→256, stride 2)
down3:   Conv3x3(256→256, stride 2)
down4:   Conv3x3(256→256, stride 2)
PPM:     AdaptiveAvgPool(1/2/3/6) → Conv1x1(256→128) + GN + GELU，四支拼接后
         Conv1x1(512→256) + GN + GELU
FPN:     4 级 lateral Conv1x1 → 128，4 个 smooth Conv3x3(128→128)
head:    Conv3x3(512→256) + GN + GELU → Conv1x1(256→20)
```

- 参数量：64 维输入 **3,770,644**；128 维输入 **3,844,372**
- **这个 head 没有 dropout**（注意：`src/rs_backbone_eval/heads.py` 里那个共享版
  `UPerNetHead` 默认 `dropout=0.1`，T11 的 OlmoEarth worker 用的是共享版；
  T11O 用的是 `workers/aef/src/aef_worker/heads.py` 里的本地版，无 dropout）

### 2.3 结果（test mIoU，3 种子）

| 协议 | head | 轮数 | 参数 | test mIoU |
|---|---|---:|---:|---:|
| `olmoearth_v13_probe_mlp`（T11O） | MLP | 8 | 11,156 | 0.5788 |
| `olmoearth_v13_probe_upernet`（T11O） | UPerNet | 8 | 3,770,644 | 0.5799 |
| `olmoearth_v13_probe_mlp_20ep`（T11O，诊断） | MLP | 20 | 11,156 | 0.5956 |
| `olmoearth_v13_probe_upernet_20ep`（T11O，诊断） | UPerNet | 20 | 3,770,644 | 0.6013 |
| `olmoearth_v13_probe_mlp_128`（T11O128） | MLP | 8 | 19,348 | 0.6137 |
| `olmoearth_v13_probe_upernet_128`（T11O128） | UPerNet | 8 | 3,844,372 | 0.6027 |
| `olmoearth_v13_probe_mlp_128_20ep`（T11O128，诊断） | MLP | 20 | 19,348 | 0.6314 |
| `olmoearth_v13_probe_upernet_128_20ep`（T11O128，诊断） | UPerNet | 20 | 3,844,372 | 0.6183 |

## 3. T11：OlmoEarth v1.2 tiny 端到端冻结探针

| 项 | 值 |
|---|---|
| 主干 | OlmoEarth v1.2 tiny，**冻结**，`embed_dim=192` |
| 输入 | PASTIS S2 L2A 10 波段 → 12 波段适配（set0/set1 online，**set2 [B01,B09] MISSING**），编码器输入 **224 px @ 5.714 m** |
| 归一化 | 训练折 tile 统计量（`norm_source=train_fold_tile_stats`） |
| 优化器 | AdamW，lr **1e-3**，`weight_decay=1e-4`，恒定 |
| batch | **4 × 2 梯度累积 = 有效 8** |
| 轮数 | 8 |
| 头 | `UPerNetHead([192], 20, fpn_dim=256)`（共享版，`dropout=0.1`）——**2,481,940** 参数 |
| 时序变体 | `20bin`：先 `LTAE2d(192→256, n_head=16, d_k=8, mlp=[256,256], dropout=0.2, T=1000 天)`，再过 `UPerNetHead([256], 20, fpn=256)`；可训练 **2,647,956** / 总 15,112,708（含冻结主干） |

结果（test mIoU，3 种子）：`olmoearth_s2_single` **0.1975**、`olmoearth_s2_20bin` **0.3305**。

## 4. 这些超参数记录在哪

- `run.json` 里能直接查到：`training.head_id`、`trainable_parameters`、`total_parameters`、
  `epochs_completed`、`optimizer_steps`、`wall_time_seconds`；输入维度和编码在 `tags`
  （`embedding_dims`、`embedding_encoding`）。
- **学习率、weight decay、batch、优化器类型没有写进 `run.json`**，只写死在 worker 代码里：
  - T11O：`workers/aef/src/aef_worker/runner.py`（`lr = 3e-3 if head_name == "mlp" else 1e-3`、
    `AdamW(... weight_decay=1e-4)`、`BATCH = 8`、`DEFAULT_EPOCHS = 8`），head 在
    `workers/aef/src/aef_worker/heads.py`；
  - T11：`workers/olmoearth/src/olmoearth_worker/t11.py`（`EPOCHS/BATCH/ACCUM/LTAE` 常量、
    `AdamW lr=1e-3`），共享 head 在 `src/rs_backbone_eval/heads.py`。
- 这算一个记录缺口：如果同事需要按 run 追溯，建议下一步把 `lr / weight_decay / batch /
  optimizer / scheduler` 写进 `run.json.training`（改动很小，但要重跑才对齐，旧 run 只能靠代码版本推断）。

## 5. 复现

```bash
export RS_EVAL_MANIFEST=/geodata/rs-eval/data/manifests/T11O/<sha>.jsonl
uv run rs-eval dispatch --task T11O --model olmoearth_v13_embedding \
  --protocol olmoearth_v13_probe_upernet --action train --seed 17 \
  --run-id <id>
# 20 epoch 诊断预算用 *_20ep 协议；不要用环境变量改 epochs。
```
