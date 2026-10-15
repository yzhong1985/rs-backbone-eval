# Colab 算力接入与队列运行方案

日期：2026-10-15。初版代码版本：`5753749`。断点续训需要使用包含 `TrainingCheckpoint` 和 `dispatch --resume-from` 的新版代码，旧网盘包需要重新打包。
关联：[外部模型可比性评估](../models/外部模型可比性评估.md)、[历史运行审计](../reviews/历史运行审计.md)。

## 1. 目标与分工

Colab A100（40GB）承担本地卡装不下或太慢的 AnySat 单元；本地卡继续的一般分工：

| 资源 | 承担 | 预算 |
|---|---|---|
| **Colab A100** | ① AnySat T5 冻结种子（batch 2 原生，无累积：`…-T5-frozen_probe-<seed>-colab`）；② **B1A**（AnySat T1 微调 ×3，batch 2：`…-T1-full_finetune-<seed>-colab`）；③ T3/T4 的 AnySat B1（单种子 42） | 每条估算见 §4 |
| 4090 | T5A 仅 seed 17（进行中）→ 视选定方案接 T2 冻结 ×9（需要新 case） | T5 后 ~1 天 |
| 3080 | 同步环境 → O（OlmoEarth ×3）→ B1（仅 Prithvi/Clay）→ B0（11 条） | ~5–8h |

**约束**：同一实验在不同设备上保持项目约定的有效 batch 2；本地可走 1×2 累积，A100 是否能容纳原生 batch 2 仍需逐任务实测。这里的 batch 2 是项目自定义预算；配置链接所指的 Prithvi 官方 T1 batch 为 16，T5 为 8，不能称为官方 batch。Colab 使用固定提交的代码；续训要求代码版本和训练配置一致。

**公平性注记**：保持协议相同仍不代表不同 GPU 上数值逐位一致；混合精度和计算内核可能产生差异。`nvidia-smi -L` 的卡型号记录在交付汇报里。断点恢复的配置检查包含 micro-batch，恢复同一实验时继续使用原来的 2×1 或 1×2 设置。

## 2. 需要准备的东西

| 项 | 内容 |
|---|---|
| 算力包 | 按分配到的 GPU 实时单位消耗估算。原先 5.4–10.5 单位/小时只是估值；若运行 40–55h，对应约 216–578 单位，400 单位不保证够用 |
| 网盘包 | `~/rs-colab-pack.tgz`（**449MB**，含 repo `5753749`、manifests、AnySat 权重、`/content` 路径的 `.env`、空 runs/cache 骨架） |
| Runtime | Colab：申请 **A100**，实际分配受可用性影响；CPU RAM、VRAM、磁盘需通过每个任务的 smoke 实测；磁盘除数据外还要容纳本地检查点和临时副本，Drive 也需留足空间 |
| 网络 | Colab 直连 HF/GCS（机房带宽），**不需要**把 24GB 数据从家里传 |

**bootstrap 数据源**（`scripts/colab_bootstrap.sh` 自动完成，幂等）：

| 数据集 | 大小（压缩） | 来源 |
|---|---|---|
| T5 multi-temporal-crop | ~11.8GB（2 tgz） | HF `ibm-nasa-geospatial/multi-temporal-crop-classification` |
| T3 burn intensity | ~894MB（3 zip） | 同 HF `burn_intensity` |
| T1 Sen1Floods11 | ~1.8GB（~670 行 ×2 文件） | GCS 公开桶 `gs://sen1floods11`（按 manifest 相对路径逐文件映射，无鉴权） |

manifest 是**在本地生成的同一批文件**（数据包里含 `rs-eval/data/manifests`），Colab 数据布局与本地一致 → 该清单一个字节都不用改。

T4 尚未接入此 bootstrap 脚本。开始 T4 前需按 `configs/downloads/catalog.yaml` 中 `data_landslide4sense` 的固定 revision 准备 `raw/landslide4sense/main/images`、`annotations` 和同一份 T4 manifest。AnySat 的固定版本源码目录也必须随包携带或单独克隆，并设置 `ANYSAT_SRC`；仅有模型权重不足以运行。

## 3. Notebook 格子（按顺序执行）

格子 1：挂载 + 解包 + 硬件确认。

```python
from google.colab import drive
drive.mount('/content/drive')
!tar xzf /content/drive/MyDrive/rs-colab-pack.tgz -C /content/
!nvidia-smi -L && df -h /content | tail -1   # 卡型号发给主控侧记录
```

格子 2：环境（root + anysat 两套 venv，Python 3.12）。

```bash
%%bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
uv python install 3.12 > /dev/null 2>&1
cd /content/rs-backbone-eval
git log --oneline -1     # 固定使用包含断点续训功能的提交
uv sync --python 3.12 --no-progress && uv sync --project workers/anysat --python 3.12 --no-progress
echo READY
```

格子 3：数据 bootstrap（先只 T3 校准流程；正式种子前再拉 T5/T1）。

```bash
%%bash
export PATH="$HOME/.local/bin:$PATH"
cd /content/rs-backbone-eval
bash scripts/colab_bootstrap.sh T3          # ~894MB，1 分钟量级
# 通过后再执行：bash scripts/colab_bootstrap.sh "T5 T1"（~15GB，几分钟）
```

预期末行 `BOOTSTRAP-OK`，且各 task 打印的 tiff 条数 > 0。

格子 4：smoke 校准（正式 run 之前必须先跑一条）。

```bash
%%bash
export PATH="$HOME/.local/bin:$PATH"
cd /content/rs-backbone-eval
set -a && . ./.env && set +a
export RS_EVAL_RUNS_ROOT=/content/rs-eval/smoke-runs
export RS_EVAL_CHECKPOINT_ROOT=/content/drive/MyDrive/rs-eval-smoke-checkpoints
ANYSAT_WORKER_EPOCHS=1 ANYSAT_WORKER_LIMIT_BATCHES=3 \
uv run rs-eval dispatch --action train --task T3 --model anysat_base \
    --protocol full_finetune --seed 42 --run-id anysat_base-T3-full_finetune-42-colab-smoke
```

输出 `wall` 与显存/每-step 耗时发回主控侧校准。`[SMOKE]` 标记的记录不进比较表。

## 4. 正式 run 序列（smoke 校准通过后）

| 序 | 命令（都先 `set -a && . ./.env && set +a`） | 运行 id 模式 | 估时（A100） |
|---|---|---|---|
| 1 | T3 B1：`uv run rs-eval dispatch --action train --task T3 --model anysat_base --protocol full_finetune --seed 42 --run-id anysat_base-T3-full_finetune-42-colab` | `…-42-colab` | ~2h |
| 2 | T4 B1：同上换 `--task T4 --seed 42` | `…-42-colab` | ~2–3h（smoke 后定） |
| 3 | T1 B1A ×3：`--task T1 --protocol full_finetune --seed {17,42,73} --run-id anysat_base-T1-full_finetune-<seed>-colab` | `…-colab` | ~1–1.5h/seed |
| 4 | T5 冻结种子 ×2–3：`--task T5 --protocol frozen_probe --seed {42,73(,17)} --run-id anysat_base-T5-frozen_probe-<seed>-colab` | `…-colab` | ~13–16h/seed |

以 3 个 T5 种子计总时长 ≈ **40–55h**（工程师按 credit 余量裁剪最后一步即可；官方 125 epoch 无早停——官方 patience 与 val 周期的组合在 125 epoch 内数学上无法触发，见 `configs/tasks/T5_us_multicrop.yaml` 的 `official_config_url` 指向的官方配方）。

## 5. 会话维护与风险

- AnySat T1/T3/T4/T5 已支持 epoch 边界断点续训：每轮训练及该轮验证结束后保存模型、优化器、最佳验证权重、RNG、曲线和累计步数。默认每轮保存一次，中途断开会重做尚未保存的 epoch。详见[断点续训与Colab运行](断点续训与Colab运行.md)。
- 训练前在每个训练用 `%%bash` 格子内设置 `export RS_EVAL_CHECKPOINT_ROOT=/content/drive/MyDrive/rs-eval-checkpoints`，否则检查点只存在临时 VM。
- [Colab 官方 FAQ](https://research.google.com/colaboratory/faq.html) 说明普通运行时通常最多 12 小时，Pro+ 在额度充足时支持最长 24 小时连续执行；保持浏览器窗口不能取消 VM 最长寿命。T5 的 13–16h/seed 必须按可恢复运行安排，A100 也不保证可分配。
- 每 10–15 分钟在主控侧保存一次输出（smoke 后正式 run 每条完成时打包）；
- 收尾打包（每条 run 完成后运行）：

```python
run_id = "anysat_base-T3-full_finetune-42-colab"
!cd /content/rs-eval/runs && zip -r /content/drive/MyDrive/{run_id}.zip {run_id}
```

主控侧把 zip 解包到 `/geodata/rs-eval/runs/` 后跑 `python3 scripts/audit_runs.py` 统一收账。
- credit 余额在 Colab 页面实时可见；到 ~10 单位余量时停止派新 run（避免种子跑到一半额度耗尽）。
- `[SMOKE]`、`…-colab-smoke` 的记录应放到独立的 runs 目录。当前 matrix 不会自动排除 smoke，`scripts/audit_runs.py` 也只检查代码修复的祖先关系，不是实验预算验收工具。
