# AGENTS.md

本仓库是协议驱动的多光谱骨干评测框架。先跑 `uv run rs-eval doctor`，不要假设 README 里的
`/mnt/rs-eval` 路径；本部署的 `.env` 指向 `/geodata/rs-eval`。

## 分层与边界

- 控制层 `cli.py`/`dispatch.py`/`registry.py`/`matrix.py`/`report.py` 不加载模型；
  `heads.py`/`losses.py`/`checkpoint.py`/`temporal_io.py` 是给 worker 共享的
  torch/rasterio 模块，不要让 CLI 路径依赖它们。模型代码在 `workers/<name>/` 的独立
  `uv` 环境里，通过 `configs/models/*.yaml` 的 `worker_command` 启动；不要把 worker
  依赖装进根环境。
- `configs/` 是任务、模型、协议的唯一注册表；`registry.py` 用 `extra="forbid"` 校验。
  `docs/` 和 worker README 有历史快照，可能与代码不一致；以 `configs/`、`run.json` 和
  `rs-eval list-tasks` 为准。
- 数据、权重、缓存、run 记录都在仓库外；`.env` 不入库。`run.json` 是唯一权威结果，
  `docs/reports/` 只是快照。README 的“7 任务/8 配置”只指 Prithvi 官方套件；当前
  `configs/` 另有 T1S/T8，共 10 个 task ID。

## 常用命令

```bash
uv sync --extra portal --extra downloads --extra dev
uv run rs-eval doctor
uv run pytest
uv run pytest tests/test_dispatch.py -q          # 单文件
uv run pytest -k rerun_baselines -q              # 单主题
uv run ruff check . && uv run ruff format --check .
uv run python scripts/check_repo_hygiene.py
uv run rs-eval list-tasks
```

`make test|lint|hygiene` 是等价别名；`make lint` 只检查，不自动格式化。CI 顺序是
ruff check → ruff format --check → pytest → hygiene。测试必须 CPU-only、无外部数据，
使用 `tmp_path`/stub，不要在测试里加载真实 GPU 或数据集。

`scripts/rerun_baselines.sh` 没有可执行位，必须用 `bash scripts/...`；lane 列表：
`grep '^    [A-Z0-9]*)' scripts/rerun_baselines.sh`。

## Run ledger 与结果准入

- 只用 `uv run rs-eval dispatch ...` 创建 run；worker 负责 `run.json` 生命周期。
  不要手写/修改 `run.json`，不要删除 run 目录。无效/误配置的记录移到
  `runs_root/../runs_quarantine/`（报告会单独计数），保留审计轨迹。
- run 目录已存在时 dispatch 会拒绝；重试用新的 `-vN` run_id。旧记录用
  `tags.superseded` 标记并保留，不要覆盖或删除。
- 正式 run 前保持 worktree 干净；`run.json` 记录 `git_commit`，脏树会带 `-dirty`。
  `scripts/audit_runs.py` 用 git 祖先关系判断，不看提交时间。
- 只有 `completed` + 官方 epoch + 未被 superseded/审计排除的运行进入 headline、
  scorecard、Pareto 和 Portal；smoke、短跑、`REDUCED-BUDGET` 永不作为正式结果
  （见 `matrix.formal_candidates`）。
- `init`（pretrained/random）与 `label_fraction` 是独立 panel，不要混进主分数；同一
  seed 重跑不是新观测，`canonical_runs` 每个 seed 只保留一个。
- 改动会改变存储指标值的代码时，必须把修复 commit 加进
  `src/rs_backbone_eval/audit.py:required_fixes`，然后重跑：

  ```bash
  uv run python scripts/audit_runs.py /geodata/rs-eval/runs .
  ```

  否则旧 run 会被静默当成可比结果。纯报告/聚合改动不要加 fix，只需重生成报告。
- 报告与 Portal 共用准入规则（`excluded_run_ids` + `build_matrix`）；改一处必须同步
  验证另一处，否则会出现 500 或两边 cell 不一致。

## 轨道与种子

- B0 = Prithvi 参考复现（`prithvi_repo_config`/`prithvi_paper_protocol`）；
  B1 = 全量微调（主结果）；B2 = 冻结头（`frozen_probe`/`frozen_adapter`）；
  B2S = 两阶段（仅 AnySat T5，`frozen_adapter_two_stage`）；B3 = 原生输入探针。
  不要把 B2S 并进 B2，不要把 B3 探针当训练结果。
- 固定种子 17/42/73；T2 B2 的 AnySat cell 另有已声明的 7/123/2024。先声明种子再看
  结果，报告所有种子。
- D1 只在该 track 所有模型都跑过的任务上取平均（`scorecard` 的 common set）；新增
  模型/任务会改变分母，D1 只在同一份报告内可比。
- T5 官方预算是 125 epoch（旧文档里的 20 是历史/非正式）；T1–T4 是 50。

## 计算与队列

- 保持有效 batch = 2（官方口径）。显存不够时用 `*_WORKER_BATCH_SIZE=1` +
  `*_WORKER_GRAD_ACCUM=2`；不要静默降低有效 batch。AnySat B1 的 T1/T2 用 224 训练
  裁剪（256 在 24GB 上 OOM）。
- 长任务用 `bash scripts/rerun_baselines.sh --detached <LANE>`：它会等 GPU 空闲、串行
  执行、记录 PID/日志、跳过已完成 run、失败后继续并在末尾非零退出。不要手写长循环。
- 同一 cell 的种子留在同一 host/GPU class；不要把种子拆到不同 GPU，也不要在同一 cell
  混 1×2 和 2×1。
- 日志默认在 `/tmp/opencode`，重启会丢；用 `scripts/collect_run_logs.sh` 把
  `train.log` 镜像进 run 目录。
- 断点续训只支持 AnySat T1/T3/T4/T5：`RS_EVAL_CHECKPOINT_ROOT`、
  `RS_EVAL_CHECKPOINT_EVERY`、`--resume-from`。不要把 smoke checkpoint 续成正式预算。
- 本部署：4090 24GB 本机；3080 20GB（`ssh yzhong@192.168.1.104`）不能访问 GitHub，
  代码用 git bundle 同步。T5 两阶段 dense cache 放 NVMe（`RS_EVAL_CACHE_ROOT`），
  成功后自动清理。
- Portal：`scripts/portal_server.sh start|stop|restart|status`；远程用
  `scripts/portal_client.sh` 建单会话 SSH 隧道。Portal 只读 `run.json`，不改结果。

## 数据与 manifest

- `uv run rs-eval build-manifest --task T1` 写入
  `data_root/manifests/<task>/<sha256>.jsonl`；`audit-data` 检查 schema、波段、传感器、
  时相、split 和组泄漏。
- 正式 run 建议 `export RS_EVAL_MANIFEST=...` 固定 manifest；否则 dispatch 取该任务
  目录下 mtime 最新的 `.jsonl`，可能不是你想要的。
- 官方 split 的组泄漏是已知事实，不要静默“修” split；在 run/report 里如实记录。
  需要 holdout 时用 `holdout-manifest` / `subset-manifest`。
- 数据/权重下载走 `configs/downloads/catalog.yaml` 和 Portal/`rs-eval list-downloads`；
  不要把 payload 放进仓库。

## 评测解释（避免过度结论）

- T1–T5 属于 Prithvi-EO-2.0 官方下游任务组；T2/T3/T5 是 HLS（Prithvi 预训练传感器），
  T1/T4 是 S2。B0 是 Prithvi 参考，不是中立第三方基线。
- T1/T2/T5 的官方 split 存在 train/test group 重叠（T1 10/10、T2 119/186、
  T5 240/253）；这些分数是同分布留出，不是跨区域/跨事件泛化。T1 Bolivia holdout
  是当前唯一独立证据。
- T1S（SAR）与 T8（GF-2）是域外/泛化轨，单列，不并入 Prithvi-suite D1；报告里标清
  sensor、分辨率、provenance 和 home/away。T1S 只有 AnySat/Clay 实现，Prithvi 是纯光学。
- 不要用 D1 总分直接宣布“最佳架构”；按传感器/任务来源分层，B0/B1/B2/B2S 分轨比较。

## 修改配置、报告与文档

- 新增 task/model/protocol：加 `configs/{tasks,models,protocols}/*.yaml`，更新 worker
  `runner.py` 的 task 允许列表，并更新 `tests/test_registry.py`（它断言精确集合）。
- OlmoEarth Tiny 和 Base 是两个 model ID，checkpoint 分别是 `OLMOEARTH_CHECKPOINT` /
  `OLMOEARTH_BASE_CHECKPOINT`，不要合并。
- 聚合/评分规则有很多审查回归：改 `matrix.py` / `report.py` 前先读
  `tests/test_review_fixes.py`、`tests/test_scorecard.py` 和 `docs/reviews/`。
- worker README 有旧状态（例如 AnySat/Clay 仍写“待实现”）；以 `runner.py` 和
  `configs/models` 为准。
- `web/` 是无构建静态前端；改 Portal API 时同时跑 `tests/test_portal.py`。
- 报告生成（只读账本）：

  ```bash
  uv run python - <<'PY'
  from datetime import date
  from pathlib import Path
  from rs_backbone_eval.paths import RuntimePaths
  from rs_backbone_eval.report import build_markdown

  p = RuntimePaths.from_environment()
  text = build_markdown(p.runs_root, p.data_root, p.config_root, Path("."))
  out = p.runs_root.parent / "reports" / f"实验报告_{date.today()}.md"
  out.parent.mkdir(exist_ok=True)
  out.write_text(text, encoding="utf-8")
  print(out)
  PY
  ```

  报告快照写到账本同级的 `reports/`，不入 git；`docs/reports/` 只放人工整理或存档的快照，
  当前结论仍以账本为准。
- 回答和评审文档用中文；评审响应放 `docs/reviews/审查响应_<日期>[_第N轮].md`，写清问题、
  修复 commit、pytest/audit/报告验证和快照路径。
- 提交到 `main`；CI 通过后再推送。3080 不能拉 GitHub，需要时用 bundle。
