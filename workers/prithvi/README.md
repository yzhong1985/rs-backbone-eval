# Prithvi worker

Prithvi-EO-2.0 评测的隔离执行环境，锁定 TerraTorch 1.2.x。不与其他模型共享 Python
环境，也不复制权重：通过 `PRITHVI_CHECKPOINT` 指向外置权重文件。

## 环境

```bash
uv sync                       # 在 workers/prithvi 内；需要网络时带 http_proxy/https_proxy
uv run python -c "import torch; print(torch.cuda.is_available())"
```

依赖通过 `tool.uv.sources` 以可编辑方式引入根项目 `rs-backbone-eval`，worker
直接复用根项目的契约（`WorkerRequest`/`RunRecord`）、manifest 审计和指标实现，
保证三个 worker 的数据与评价规则一致。

## 运行

正常入口是控制层派发，不在 worker 目录手动执行：

```bash
rs-eval dispatch --action probe --task T1 \
  --model prithvi_eo_v2_300_tl --protocol prithvi_repo_config --seed 42
```

控制层会：

1. 校验 task × model × protocol × seed 组合与 manifest 存在性；
2. 在外置 `runs/` 下创建 `prithvi_eo_v2_300_tl-T1-prithvi_repo_config-probe-42/`，
   写入 `request.json`（WorkerRequest 契约）；
3. 通过 `configs/models/prithvi.yaml` 的 `worker_command` 启动本 worker；
4. 逐行转发 stdout 上的 `WorkerEvent` JSONL 进度。

worker 负责 `run.json` 的全部生命周期：创建为 `running`，每步原子更新
（先写 `run.json.tmp` 再替换），结束置 `completed` 或 `failed`。Portal 直接
读取同一文件展示结果。

## 动作实现状态

- `probe`：已实现——manifest 契约审计（含组泄漏报告）、checkpoint 存在性与
  SHA-256、torch/CUDA 可用性、TerraTorch 可导入性；
- `train` / `evaluate` / `extract_features` / `export_previews`：待实现
  （T1 repo-config 基线训练是下一步）。
