# Colab High-RAM Runbook：AnySat T1HOLDA / T3 B1

适用：Colab **High-RAM + A100 40GB**；代码固定 `b7c1714`；数据只跑 T1、T3。
目标不是让最终报告提前（最终报告仍卡在 3080 的 T4 B2 seed 73），而是：

1. 把 4090 的 **T1HOLDA** 长尾搬走；
2. 补上 **T3 AnySat B1** 这个缺口。

> 不要用 10-15 的旧网盘包（`5753749`）。它没有 T1S/T8/D2 和后续累积修复。

## 1. 先确认硬件

```bash
nvidia-smi -L
free -g
df -h /content | tail -1
```

必须看到 **A100 40GB 或 80GB** 才继续 T1HOLDA B1 和 T3 B1。
如果是 L4 24GB / T4 16GB：T1HOLDA B2 可以试，T1 B1 和 T3 B1 不要跑，AnySat B1 会 OOM。

### 1.1 80GB A100 能快多少

A100 40GB 与 80GB 的 SM/算力基本同档，80GB 主要多显存和带宽；**不要期待 2×**。
真正的收益来自：B1/T3 B1 可以不用 1×2 累积，改成 **native batch 2（micro 2 × accum 1）**，
少一次累积窗口开销、也少 OOM 重试。经验上大约快 **10–30%**，不是翻倍。

- **T3 B1**：40GB 用 1×2，3 seeds 约 6–9h；80GB native 2 约 5–7h。
- **T1HOLDA B1**：40GB 1×2 约 5–6h；80GB native 2 约 4.5–5.5h。
- **T1HOLDA B2**：40GB 已够，80GB 收益很小；保持 1×2，和 4090 的设置一致。
- 有效 batch 必须仍是 **2**；不要因为 80GB 就开 4。
- 同一个 cell 的 3 个 seed 必须全部在同一张 80GB 上；不要混 1×2 和 2×1。
- 如果最终报告要严格对齐 4090 的 T1HOLDA B1（1×2），就继续用 1×2；接受微批次差异再用 2×1。
- Colab 的耗时/显存不要直接和 4090 做 D3 成本比较，需要单列 host/GPU。
- 成本：80GB 每小时贵 50% 时，需要 **1.5× 加速**才打平；10–30% 加速对应单任务贵约
  **15–36%**。纯按 credit 算不划算；只有 40GB 排队/额度导致额外会话成本，或 wall-clock
  比 credit 更重要时才值得。

## 2. 包内容

本地目录：`/home/yzhong/rs-eval/colab-2026-10-26/`

上传到 Drive 的 `MyDrive/rs-eval-colab/`：

| 文件 | 用途 |
|---|---|
| `rs-eval-b7c1714.bundle` | 当前代码（git bundle） |
| `manifests_T1_T3.tar.gz` | T1/T3 的 manifest |
| `anysat-src-5f6f475.tar.gz` | AnySat 上游源码（commit `5f6f475`） |
| `AnySat.pth` | AnySat base 权重，sha256 `ce124652…c93d4f` |
| `RUNBOOK.md` | 本文件 |

权重也可以直接上传到 `MyDrive/rs-eval-colab/AnySat.pth`，不必放仓库。

## 3. Colab 初始化（每个新 session 都做）

### 3.1 挂载 + 解包

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%%bash
set -euo pipefail
DRIVE=/content/drive/MyDrive/rs-eval-colab
mkdir -p /content/rs-eval /content/tools
rm -rf /content/rs-backbone-eval
git clone "$DRIVE/rs-eval-b7c1714.bundle" /content/rs-backbone-eval
cd /content/rs-backbone-eval
git checkout b7c1714
git status --porcelain
git rev-parse HEAD
tar xzf "$DRIVE/manifests_T1_T3.tar.gz" -C /content/rs-eval/data
tar xzf "$DRIVE/anysat-src-5f6f475.tar.gz" -C /content/tools
mkdir -p /content/rs-eval/weights/anysat/base
cp "$DRIVE/AnySat.pth" /content/rs-eval/weights/anysat/base/AnySat.pth
sha256sum /content/rs-eval/weights/anysat/base/AnySat.pth
```

最后一行应为：

```text
ce1246527a28bade624048c448a1fa473ca85791b998f562ce2bcc3dcac93d4f
```

### 3.2 写 Colab `.env`

```bash
%%bash
set -euo pipefail
cat > /content/rs-backbone-eval/.env <<'EOF'
RS_EVAL_DATA_ROOT=/content/rs-eval/data
RS_EVAL_WEIGHTS_ROOT=/content/rs-eval/weights
RS_EVAL_RUNS_ROOT=/content/rs-eval/runs
RS_EVAL_CACHE_ROOT=/content/rs-eval/cache
RS_EVAL_CHECKPOINT_ROOT=/content/drive/MyDrive/rs-eval-colab/checkpoints
RS_EVAL_CHECKPOINT_EVERY=1
ANYSAT_CHECKPOINT=/content/rs-eval/weights/anysat/base/AnySat.pth
ANYSAT_SRC=/content/tools/anysat-src
RS_EVAL_DOWNLOADS_ENABLED=false
EOF
mkdir -p /content/rs-eval/runs /content/rs-eval/cache \
         /content/drive/MyDrive/rs-eval-colab/checkpoints
```

### 3.3 安装环境

```bash
%%bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
curl -LsSf https://astral.sh/uv/install.sh | sh
cd /content/rs-backbone-eval
uv sync --python 3.12
uv sync --project workers/anysat --python 3.12
workers/anysat/.venv/bin/python - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0).total_memory / 1e9)
PY
```

若 `torch.cuda.is_available()` 为 False，先停下；不要继续跑正式 run。

### 3.4 下载 T1/T3 数据

```bash
%%bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /content/rs-backbone-eval
bash scripts/colab_bootstrap.sh "T3 T1"
```

预期最后一行 `BOOTSTRAP-OK`。T3 约 894MB，T1 约 1.8GB。

### 3.5 数据审计

```bash
%%bash
set -euo pipefail
cd /content/rs-backbone-eval
uv run rs-eval audit-data \
  /content/rs-eval/data/manifests/T3/e4972d4a1cacc88d26b038bdd1771732200b6bf80d32b423a8f858250284a638.jsonl \
  --task T3 || true
uv run rs-eval audit-data \
  /content/rs-eval/data/manifests/T1/6a02296af3c2d669c7b92a6700729b98e89e51cb75d0cde801c1f97d56a3d0e8.jsonl \
  --task T1 || true
```

T1/T3 官方 split 的 `leakage` 是已知现象，可以继续；如果有 **schema / band / sensor / time-step / unknown split / missing split** 错误，停止。
`audit-data` 因 leakage 非零退出是正常的，所以上面加了 `|| true`。

## 4. 冒烟（必须做，且不进正式账本）

冒烟使用独立 runs root，避免污染 `/content/rs-eval/runs`。

```bash
%%bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /content/rs-backbone-eval
set -a; . ./.env; set +a

export RS_EVAL_RUNS_ROOT=/content/rs-eval/smoke-runs
export ANYSAT_WORKER_EPOCHS=1
export ANYSAT_WORKER_LIMIT_BATCHES=3
export ANYSAT_WORKER_EVAL_LIMIT=1
export ANYSAT_WORKER_BATCH_SIZE=1
export ANYSAT_WORKER_GRAD_ACCUM=2
export ANYSAT_WORKER_LR=1e-4
export ANYSAT_WORKER_TRAIN_CROP=224

# T3 B1
uv run rs-eval dispatch --action train --task T3 --model anysat_base \
  --protocol full_finetune --seed 42 \
  --run-id anysat_base-T3-full_finetune-42-colab-smoke

# T1 B2（不要带 LR/crop，B2 用 worker 默认）
unset ANYSAT_WORKER_LR ANYSAT_WORKER_TRAIN_CROP
RS_EVAL_EXTRA_EVAL_SPLITS=bolivia_holdout RS_EVAL_PERTURB="default;metadata" \
uv run rs-eval dispatch --action train --task T1 --model anysat_base \
  --protocol frozen_adapter --seed 42 \
  --run-id anysat_base-T1-frozen_adapter-42-colab-smoke

# T1 B1
export ANYSAT_WORKER_LR=1e-4
export ANYSAT_WORKER_TRAIN_CROP=224
RS_EVAL_EXTRA_EVAL_SPLITS=bolivia_holdout \
uv run rs-eval dispatch --action train --task T1 --model anysat_base \
  --protocol full_finetune --seed 42 \
  --run-id anysat_base-T1-full_finetune-42-colab-smoke
```

检查三个 smoke 的 `run.json`：`status=completed`，并记录 `wall_time_seconds`、`peak_gpu_memory_mb`、`train.metrics`。
如果 B1 smoke OOM，不要继续正式 B1；T3 B1 也同理。B2 可以单独继续。
**80GB 想用 native 2**：把 smoke 里的 `ANYSAT_WORKER_BATCH_SIZE=2` / `ANYSAT_WORKER_GRAD_ACCUM=1`，
确认 `run.json` 的 `tags.micro_batch=2`、`tags.effective_batch=2`，再在正式 session 用同一设置。

## 5. 正式 Session A：T1HOLDA B2（3 seeds）

**先跑这个**，因为 4090 chain 最后会跑到 T1HOLDA；提前同步能让 4090 跳过。

```bash
%%bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /content/rs-backbone-eval
set -a; . ./.env; set +a

export RS_EVAL_RUNS_ROOT=/content/rs-eval/runs
export RS_EVAL_EXTRA_EVAL_SPLITS=bolivia_holdout
export RS_EVAL_MANIFEST=/content/rs-eval/data/manifests/T1/6a02296af3c2d669c7b92a6700729b98e89e51cb75d0cde801c1f97d56a3d0e8.jsonl
export RS_EVAL_PERTURB="default;metadata"
unset ANYSAT_WORKER_LR ANYSAT_WORKER_TRAIN_CROP
unset ANYSAT_WORKER_EPOCHS ANYSAT_WORKER_LIMIT_BATCHES ANYSAT_WORKER_EVAL_LIMIT
export ANYSAT_WORKER_BATCH_SIZE=1
export ANYSAT_WORKER_GRAD_ACCUM=2

for seed in 17 42 73; do
  rid="anysat_base-T1-frozen_adapter-${seed}-hold-v1"
  log="/content/rs-eval/runs/${rid}.log"
  uv run rs-eval dispatch --action train --task T1 --model anysat_base \
    --protocol frozen_adapter --seed "$seed" --run-id "$rid" 2>&1 | tee "$log"
  cp "$log" "/content/rs-eval/runs/${rid}/train.log"
done
```

每条约 1–1.5h，总计约 4h。

## 6. 正式 Session B：T1HOLDA B1（3 seeds）

B1 需要 A100 40GB 或 80GB。  
**80GB 可选**：把下面的 `ANYSAT_WORKER_BATCH_SIZE=1` / `ANYSAT_WORKER_GRAD_ACCUM=2` 改成
`2` / `1`（native batch 2），但 3 个 seed 必须全部用同一设置；否则保持 1×2。

```bash
%%bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /content/rs-backbone-eval
set -a; . ./.env; set +a

export RS_EVAL_RUNS_ROOT=/content/rs-eval/runs
export RS_EVAL_EXTRA_EVAL_SPLITS=bolivia_holdout
export RS_EVAL_MANIFEST=/content/rs-eval/data/manifests/T1/6a02296af3c2d669c7b92a6700729b98e89e51cb75d0cde801c1f97d56a3d0e8.jsonl
unset RS_EVAL_PERTURB
export ANYSAT_WORKER_BATCH_SIZE=1
export ANYSAT_WORKER_GRAD_ACCUM=2
export ANYSAT_WORKER_LR=1e-4
export ANYSAT_WORKER_TRAIN_CROP=224

for seed in 17 42 73; do
  rid="anysat_base-T1-full_finetune-${seed}-hold-v1"
  log="/content/rs-eval/runs/${rid}.log"
  uv run rs-eval dispatch --action train --task T1 --model anysat_base \
    --protocol full_finetune --seed "$seed" --run-id "$rid" 2>&1 | tee "$log"
  cp "$log" "/content/rs-eval/runs/${rid}/train.log"
done
```

每条约 1.5–2h，总计约 5–6h。

## 7. 正式 Session C：T3 B1（3 seeds）

**80GB 可选**：同 Session B，把 batch/accum 改成 `2`/`1`；3 个 seed 必须一致。

```bash
%%bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /content/rs-backbone-eval
set -a; . ./.env; set +a

export RS_EVAL_RUNS_ROOT=/content/rs-eval/runs
unset RS_EVAL_EXTRA_EVAL_SPLITS RS_EVAL_PERTURB
export RS_EVAL_MANIFEST=/content/rs-eval/data/manifests/T3/e4972d4a1cacc88d26b038bdd1771732200b6bf80d32b423a8f858250284a638.jsonl
export ANYSAT_WORKER_BATCH_SIZE=1
export ANYSAT_WORKER_GRAD_ACCUM=2
export ANYSAT_WORKER_LR=1e-4
export ANYSAT_WORKER_TRAIN_CROP=224

for seed in 17 42 73; do
  rid="anysat_base-T3-full_finetune-${seed}-colab"
  log="/content/rs-eval/runs/${rid}.log"
  uv run rs-eval dispatch --action train --task T3 --model anysat_base \
    --protocol full_finetune --seed "$seed" --run-id "$rid" 2>&1 | tee "$log"
  cp "$log" "/content/rs-eval/runs/${rid}/train.log"
done
```

每条约 2–3h，总计约 6–9h。

## 8. 断点续训

AnySat 的 T1/T3 支持 epoch 边界 checkpoint。每个 run 的 `latest.pt` 会写到：

```text
/content/drive/MyDrive/rs-eval-colab/checkpoints/<run_id>/latest.pt
```

如果断线，重新挂载 Drive、准备同样的代码/数据/权重、保持同样的 batch/accum/lr/crop，然后：

```bash
uv run rs-eval dispatch --action train --task T1 --model anysat_base \
  --protocol full_finetune --seed 42 \
  --run-id anysat_base-T1-full_finetune-42-hold-v1-r2 \
  --resume-from /content/drive/MyDrive/rs-eval-colab/checkpoints/anysat_base-T1-full_finetune-42-hold-v1/latest.pt
```

注意：换 run ID 后，4090 的 T1HOLDA 不会把 `hold-v1` 视为已完成，可能重复跑同一 seed。**尽量在 4090 到 T1HOLDA 之前完成原 ID**；如果只能 resume 成 `-r2`，要检查 `chain.out`，必要时在 4090 到 T1HOLDA 前停掉/协调该 lane。

## 9. 回传结果

在 Colab 打包 completed 的 run：

```bash
%%bash
set -euo pipefail
OUT=/content/drive/MyDrive/rs-eval-colab/out
mkdir -p "$OUT"
cd /content/rs-eval/runs
for d in */; do
  rid=${d%/}
  [ -f "$rid/run.json" ] || continue
  grep -q '"status": "completed"' "$rid/run.json" || continue
  tar czf "$OUT/${rid}.tgz" "$rid"
  echo "$rid"
done
```

本地解包（不要覆盖已有正式记录）：

```bash
cd /geodata/rs-eval/runs
for f in /path/to/out/*.tgz; do
  tar tzf "$f" >/dev/null && tar xzf "$f" -C /geodata/rs-eval/runs
done
```

如果目标 run 目录已存在，先停手检查，不要 `--overwrite`。
T1HOLDA 的 `hold-v1` 必须在 4090 跑到 T1HOLDA 之前回传；否则 4090 会先创建同名目录，Colab 结果只能改 ID。

回传后：

```bash
cd /home/yzhong/projects/rs-backbone-eval
uv run python scripts/audit_runs.py /geodata/rs-eval/runs .
```

报告由 post-chain watcher 在 chain + 3080 sync 都完成后自动生成，不需要手工覆盖。

## 10. 与 4090 的衔接

- T1HOLDA 在 `chain2` 最后一段；4090 跑到它时，如果看到 `anysat_base-T1-{frozen_adapter,full_finetune}-<seed>-hold-v1` 已 `completed`，会 **SKIP**。
- `run()` 把 SKIP 计入 `failed`，所以 4090 日志会显示失败计数；这是预期的，不是训练失败。
- 如果 4090 已经先创建了 `hold-v1` 目录，不要覆盖；让 4090 跑完，Colab 结果改用 `-hold-v1-colab` 之类的新 ID，并重新评估是否值得。
- T3 `-colab` 不在 4090 chain 里，不存在冲突。

## 11. 时间预算

| Session | 内容 | 预计 |
|---|---|---|
| A | T1HOLDA B2 ×3 | ~4h |
| B | T1HOLDA B1 ×3 | ~5–6h |
| C | T3 B1 ×3 | ~6–9h |

High-RAM 12h 会话建议一次只做一个 session；24h 会话可以 A+B，但留足余量，C 单独做。
最终报告仍要等 3080 T4 B2 seed 73（约 10/27 夜间），这些 Colab 任务不会把最终报告提前。

## 12. 不要做的事

- 不要把 seed 拆到 4090 和 Colab。
- 不要用 `-colab` ID 跑 T1HOLDA；那样 4090 不会跳过，会重复。
- 不要在正式 `RS_EVAL_RUNS_ROOT` 里跑 smoke。
- 不要改 B2 的 lr/crop；B2 使用 worker 默认（lr 1e-3，crop 256）。
- 不要改 B1 的 1×2、224 crop、lr 1e-4；否则与 4090 的 T1HOLDA 不是同一条件。
- 不要在 L4 24GB 上跑 T1 B1/T3 B1。
- 不要把 smoke / failed / 未完成 run 回传到正式账本。
- 不要把 Colab 的耗时直接和 4090 做 D3 成本比较；D3 需要单列 host/GPU。
