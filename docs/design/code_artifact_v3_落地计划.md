# 无教师部署（Image-Only）落地计划 v3（2026-11-16）

- 上游方案：`docs/design/code_artifact_v2.md`（六臂设计）与
  `docs/design/code_artifact_v2_修订意见_2026-11-16.md`（验收门槛）。
- 本文只回答一件事：**这套方案在本仓库怎么落**，不改研究路线、不改判据。
- 一句话结论：**不采用 v2 §5 的两个脚本**（修订意见已判定为原型），把六个臂实现进
  `workers/lim` + 新 task，复用现有账本、缓存、队列与审计；先过验收，再开正式实验。

## 1. 范围与不做的事

做：

- 新 task `T13`（WHU-OPT-SAR，8 类，4 波段）；
- LIM worker 内的六臂（B0 / R / R_null / Bcap / D / P，A 留到阶段四）；
- 修订意见 §6 的全部验收项，作为 `tests/` 里的可执行测试；
- AEF 教师的获取、对齐与缓存。

不做（本轮）：

- 不复制、不修补 v2 §5 的 `six_arm_models.py` / `train_eval_six_arms.py`；
- 不与同事的 `0.4909 / 0.5112` 做绝对对齐（头结构不同，见 §5.2）；
- 不并行展开多主干（mars/prithvi）、多教师、多注入位置、多损失。

## 2. 仓库现状映射

| 需要 | 现状 | 结论 |
|---|---|---|
| LIM（MiT-Huge）+ 分割头 | `workers/lim` + 共享 `rs_backbone_eval/heads.py:UPerNetHead`（T9/T10 在用） | 复用 |
| AEF / OlmoEarth int8 解码与 NoData | `workers/aef/.../runner.py`：`aef_signed_square_imagenetfree`、`olmoearth_int8`，`-128` 作 NoData | **复用，不重写**；只补验收测试 |
| 冻结特征缓存 + 指纹 | T11 的 `feature_fingerprint` 缓存 | 复用（注意修订意见 §2：禁用几何增广） |
| 账本 / 协议 / 队列 / 审计 | `run.json`、`configs/protocols`、`rs-eval queue`、`audit_runs.py` | 直接获得 |
| WHU-OPT-SAR 数据 | 7.3 GB、100 幅、8 类；`scripts/fetch_whu_opt_sar.sh`、`scripts/whu_tile_bbox.py` | 已落地 |
| WHU 的 AEF 教师 | 仅 1 景（EPSG:32649，82×82 km，2.2 GB）；全域需 30–40 景 | **最大依赖**，见 §6 |
| LIM on T8（FBP） | `workers/lim` 的 `TASK_IDS={T9,T10,T11}` | 阶段五再补 |
| 同事的 TinyUPerNet | 无 | 自建 B0/R 作为唯一可比基线 |

## 3. 目标架构

```
configs/tasks/T13_whu_opt_sar.yaml        新任务:8 类、4 波段、图幅级 split
configs/protocols/whu_{b0,r,r_null,bcap,d,p}.yaml
workers/lim/src/lim_worker/whu.py         数据:tiling / 图幅 split / 同步增广
workers/lim/src/lim_worker/arms.py        六臂模块(注入器、预测头、校准、损失)
workers/lim/src/lim_worker/runner.py      新增 T13 分支(不改 T9/T10/T11 路径)
tests/test_whu_arms.py                    修订意见 §6 的验收测试
tests/test_whu_task.py                    task/协议注册、split、类别映射
```

### 3.1 task 定义

| 项 | 值 | 说明 |
|---|---|---|
| `task_id` | `T13` | T12 已被 FLAIR-HUB 设计占用 |
| 类别 | 8 类（标签 `0,10,…,70` → `0..7`） | 映射写进 task yaml 与 manifest 审计 |
| 训练目标 | 8 类 CE（背景是真实类） | **不设 `ignore_index`**；评测另算 7 类封闭 mIoU |
| 指标 | `mean_iou`（8 类）+ `mean_iou_closed7` | 两个都记，避免 7/8 类口径混淆 |
| 划分 | 官方 80/20 大图；验证集从 train 图幅中留 8 幅 | 图幅级隔离，不切 patch 随机分 |
| 切片 | 512×512，stride 384 | 记录每块的图幅号与 bbox |
| 输入 | GF-1 光学 4 波段（BLUE/GREEN/RED/NIR） | SAR 分支不在本轮 |
| 标定 | 逐通道 z-score，**只用 train 有效像素拟合，存 buffer** | 验收:同 tile 单独/拼 batch 结果一致 |

### 3.2 六臂（`arms.py`）

| 臂 | 训练输入 | 推理输入 | 结构/损失 | 对照 |
|---|---|---|---|---|
| B0 | 影像 | 影像 | LIM → UPerNetHead，纯 CE | — |
| R | 影像 + 真 AEF | 影像 + 真 AEF | `F' = F + Injector(AEF)`，末层 zero-init；纯 CE | B0 / R_null |
| R_null | 影像 + 高斯噪声 | 同左 | 结构同 R，输入换成同均值方差噪声（固定种子） | R |
| Bcap | 影像 | 影像 | `F' = F + Injector(Predictor(F))`，**纯 CE** | B0 |
| D | 影像 + 冻结 R 教师 | **影像** | `CE + βτ²KL`（先 γ=0），部署删 Q | B0 |
| P | 影像 + 真 AEF 监督 | **影像** | 同 Bcap 逐位一致 + `λ·L_align(ÂEF, AEF)` | Bcap / B0 |
| A（阶段四） | 影像 + 真 AEF 监督 | 影像 | 共享 Adapter 上的辅助头 | 同 adapter 的 B_A |

实现约束（直接写进代码注释与测试）：

1. 注入器零初始化：`fc2` 全零、`fc1` 随机；第 0 步输出恒等于基线。
2. 预测器（P/Bcap）只经 CE 也要拿得到梯度 —— 用「先 `optimizer.step()` 再单 CE 反传」验证，
   参考修订意见 §3。
3. 纯影像模型（B0/Bcap/D/P 推理路径）收到 AEF 输入**必须抛异常**。
4. 损失一律带 `valid_mask`；分母加 eps；全无效教师、全忽略标签不得产生 NaN。
5. 教师推理 `eval()` + 冻结 + `no_grad` + `sg`。

## 4. 阶段计划

### 阶段 0：工程验收（**现在开工，不依赖 AEF 下载**）

交付：

- `T13` task + tiling + manifest + 图幅 split；
- `arms.py` 六臂骨架 + 协议六条；
- `tests/test_whu_arms.py`（见 §4.1）；
- 真实小批次 smoke（2 幅图 × 1 epoch，`RS_EVAL_TRAIN_LIMIT` 小值），产出 `tags.smoke=true` 的记录。

验收（= 修订意见 §6 逐条落地）：见 §4.1。**未全绿不得进入阶段 1。**

估算：3–5 天。GPU 需求小（仅 smoke 与单元测试）。

### 阶段 1：参照基线

- B0 三种子 × 30 epoch（±z-score 两档，只有一档进主表）；
- R 三种子（需 AEF 覆盖区）；
- 口径核对：混淆矩阵全量累计、7 类封闭 mIoU、图幅级划分；
- 冻结 R 检查点（D 的教师）。

验收：B0 与 R 在同一管线可复现；R 优于 B0 且曲线收敛；`run.json` 记录 host/commit。

估算：3–4 天（含缓存提取）。

### 阶段 2：D（只 CE + 输出 KL，γ=0）

三种子，对照 B0。判据见 §5.1。估算 1–2 天。

### 阶段 3：P vs Bcap

三种子，先 AEF、先 cosine；InfoNCE 只作为后续变体。估算 1–2 天。

### 阶段 4：A（共享 Adapter 或解冻主干）

必须先证明辅助损失更新了**部署时保留**的参数；配同预算的纯 CE 对照（B_A / 微调基线）。
估算 2–3 天。

### 阶段 5：迁移

FBP/T8：补 LIM-on-T8，重跑 B0 与胜出臂；24 类封闭 mIoU、`unlabeled=0` 处理按修订意见 §6。
估算 3–5 天。

### 4.1 阶段 0 验收清单（对应 `tests/test_whu_arms.py`）

- [ ] 解码：`+值 / −值 / 0 / −128 / NaN / Inf / 已解码输入` 六种情况；
  AEF 与 OlmoEarth 两套约定分开测。
- [ ] z-score：统计量只来自 train；评测模式同 tile 单独推理 vs 拼 batch 一致。
- [ ] 空间对齐：小栅格人工核对位置、聚合值、有效覆盖率；边缘 NoData。
- [ ] 增广一致性：影像 / 标签 / 教师 / 掩膜同步；缓存模式禁止几何增广。
- [ ] 微调模式传 `cached_feats` 直接报错。
- [ ] 梯度路由：Bcap 的 Predictor 收到 CE 梯度；P 的 Predictor 收到对齐梯度；
  A 的共享 Adapter 收到两个损失；教师参数不变。
- [ ] 优化器：一次 `step()` 后预期参数确实变化。
- [ ] 非法输入：纯影像模型传 AEF → 抛异常。
- [ ] 边界：全无效教师、全忽略标签、单像素有效 → 无 NaN。
- [ ] 端到端：原始影像 → 预测，再跑一次部署路径（不带教师/预测头）。
- [ ] 真实数据：至少一个真实小批次前向+反传。

## 5. 预声明

### 5.1 判据（沿用 v2 §4.3，跑之前写死）

- **D vs B0**、**P vs Bcap**：3 种子平均 ≥ **+1.0 pt**，且最差种子 ≥ 对照 **−0.3 pt**；
- `P − Bcap < 0.6 pt` ⇒ 收益来自旁路容量而非教师，采纳更简单的 Bcap；
- `R` 无稳定收益 ⇒ 先不判死 D/P，按 v2 §4.3 进入微调档再判一次。

### 5.2 可比性边界

- 我们的 `UPerNetHead` 与同事 `TinyUPerNet(ch=128)` 不是同一实现，**0.4909 / 0.5112 只作参考**；
- 所有结论只在同管线内相对成立；B0/R 由我们重跑；
- 种子用仓库约定 `17/42/73`（同事的 `20261103` 不混用），并报告成对差值。

### 5.3 记录要求

每条 run 的 `run.json` 必须能回答：训练输入、部署输入、教师产品版本与年份、图幅 split、
z-score 拟合集、随机种子与初始化方式、commit。缺一项视为实验记录不合格。

## 6. 数据依赖：AEF 覆盖（当前唯一阻塞项）

现状：本机 AEF 只有法国块（EPSG:32630）；湖北 1 景（EPSG:32649，82×82 km，2.2 GB）。
全域按 10 m、82 km 一景估算需要 30–40 景、60–90 GB。

速度实测：本机 → GCS AEF 约 **0.25–0.44 MB/s**（60 GB ≈ 2 天以上）。

三条路线，按优先级：

1. **Colab 中转**：Colab 从 GCS 拉瓦片（机房带宽），打成 tar 写进 Drive；本机从 Drive 拉。
   **先测本机 ← Drive 的速度**（取一个 100 MB 文件），高于 5 MB/s 就值得。
2. **子区域**：只做现有 1 景覆盖到的图幅（约 1–2 幅大图），先把阶段 1–3 的管线走通；
   结论标注为"子区域初筛"，不升级为全域结论。
3. **无 AEF 先行**：B0 / Bcap / D_img（纯影像教师）不依赖 AEF，可先跑，只有 R / P / A 需要。

建议：**1 和 3 并行** —— 后台拉 AEF，同时把不依赖 AEF 的臂跑起来。

## 7. 风险

| 风险 | 影响 | 对策 |
|---|---|---|
| AEF 下载慢 | 阶段 1 推迟 | 子区域先行 + Colab 中转 |
| 头结构不一致 | 无法对标旧分数 | 只做臂内相对比较，B0/R 重跑 |
| 7/8 类与 ignore 语义混淆 | 指标不可比 | 双指标（8 类 + closed7），训练不 ignore |
| 图幅相邻仍有自相关 | 高估泛化 | 报告写"图幅级隔离"；测试集不参与调参 |
| 低维瓶颈 | P/D 无收益 | 按 §5.1 判，不事后改判据 |

## 8. 排期与下一动作

| 周 | 内容 |
|---|---|
| W1 | 阶段 0（task + arms + 验收测试 + smoke）；并行拉 AEF |
| W2 | 阶段 1（B0/R 三种子）+ 阶段 2（D） |
| W3 | 阶段 3（P/Bcap）+ 阶段 4 决策 |
| W4+ | 阶段 4/5 视结果 |

下一步（本文件落地后的第一个 commit）：

1. `configs/tasks/T13_whu_opt_sar.yaml` + `whu.py` 数据管线 + `tests/test_whu_task.py`；
2. `arms.py` 的 `FeatureNorm` / `AEFInjector` / losses + 对应测试；
3. AEF 拉取脚本（Colab → Drive → 本机）与测速记录。
