# Colab 算力接入与队列运行方案

日期：2026-10-15。代码版本：`5753749`（含 `scripts/colab_bootstrap.sh`，`ee88473` 的 B1A/T5B case）。
关联：[外部模型可比性评估](../models/外部模型可比性评估.md)、[历史运行审计](../reviews/历史运行审计.md)。

## 1. 目标与分工

Colab A100（40GB）承担本地卡装不下或太慢的 AnySat 单元；本地卡继续的一般分工：

| 资源 | 承担 | 预算 |
|---|---|---|
| **Colab A100** | ① AnySat T5 冻结种子（batch 2 原生，无累积：`…-T5-frozen_probe-<seed>-colab`）；② **B1A**（AnySat T1 微调 ×3，batch 2：`…-T1-full_finetune-<seed>-colab`）；③ T3/T4 的 AnySat B1（单种子 42） | 每条估算见 §4 |
| 4090 | T5A 仅 seed 17（进行中）→ 视选定方案接 T2 冻结 ×9（需要新 case） | T5 后 ~1 天 |
| 3080 | 同步环境 → O（OlmoEarth ×3）→ B1（仅 Prithvi/Clay）→ B0（11 条） | ~5–8h |

**约束**：有效 batch 必须与官方一致（AnySat T5/T1 均 batch 2）；A100 的 40GB 是唯一允许"原生 batch 2"的环境，4090/3080 走 1×2 累积。**不改动 worker 代码**；Colab 只跑既有 committed 代码（运行前核对 `git log --oneline -1` ≥ `5753749`）。

**公平性注记**：不同 GPU 只影响时长，不影响协议；run 记录依赖协议/种子/代码版本，`nvidia-smi -L` 的卡型号记录在交付汇报里（必要时补写入 run notes）。

## 2. 需要准备的东西

| 项 | 内容 |
|---|---|
| 算力包 | Colab Pro credit ≥ 400 单位（A100 5.4–10.5 单位/小时 → ~40–70h） |
| 网盘包 | `~/rs-colab-pack.tgz`（**449MB**，含 repo `5753749`、manifests、AnySat 权重、`/content` 路径的 `.env`、空 runs/cache 骨架） |
| Runtime | Colab：GPU → **A100**（40GB 版本）；RAM 默认 12GB 够用（瓶颈在 VRAM）；磁盘 `/content` ≥ ~40GB（T5 解压+归档瞬态） |
| 网络 | Colab 直连 HF/GCS（机房带宽），**不需要**把 24GB 数据从家里传 |

**bootstrap 数据源**（`scripts/colab_bootstrap.sh` 自动完成，幂等）：

| 数据集 | 大小（压缩） | 来源 |
|---|---|---|
| T5 multi-temporal-crop | ~11.8GB（2 tgz） | HF `ibm-nasa-geospatial/multi-temporal-crop-classification` |
| T3 burn intensity | ~894MB（3 zip） | 同 HF `burn_intensity` |
| T1 Sen1Floods11 | ~1.8GB（~670 行 ×2 文件） | GCS 公开桶 `gs://sen1floods11`（按 manifest 相对路径逐文件映射，无鉴权） |

manifest 是**在本地生成的同一批文件**（数据包里含 `rs-eval/data/manifests`），Colab 数据布局与本地一致 → 该清单一个字节都不用改。

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
export PATH="$HOME/.local/bin:$PATH"
curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
uv python install 3.12 > /dev/null 2>&1
cd /content/rs-backbone-eval
git log --oneline -1     # ≥ 5753749
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

- **session 中断 = 该条 run 丢失**（我们无断点续训；Colab 空闲断开、机器重启都会杀会话）→ 运行期间保持窗口/启用 Pro 的后台执行；
- 每 10–15 分钟在主控侧保存一次输出（smoke 后正式 run 每条完成时打包）；
- 收尾打包（每条 run 完成后运行）：

```python
run_id = "anysat_base-T3-full_finetune-42-colab"
!cd /content/rs-eval/runs && zip -r /content/drive/MyDrive/{run_id}.zip {run_id}
```

主控侧把 zip 解包到 `/geodata/rs-eval/runs/` 后跑 `python3 scripts/audit_runs.py` 统一收账。
- credit 余额在 Colab 页面实时可见；到 ~10 单位余量时停止派新 run（避免种子跑到一半额度耗尽）。
- `[SMOKE]`、`…-colab-smoke` 的记录**不进**任何比较表；正式记录以 `runs/audit_runs.py` 的逐条判定为准。
