# Run 配置

每个 `*.yaml` 描述一条要入队的 run，由 `scripts/runs.sh enqueue <file>` 读取。
字段：

- `task`、`model`、`protocol`、`seed`：必填，对应 `rs-eval queue add` 的参数。
- `action`：可选，默认 `train`。
- `env`：可选，键值对，传给 worker 的环境变量（如 `RS_EVAL_MANIFEST`）。
- `note`：可选，写入任务文件。

配置只描述要跑什么。结果仍以 `run.json` 为准，不要在这里写结果。
数据路径不写在这里：`data_root` 由部署环境决定，`/geodata/rs-eval/data` 下的软链接指向家盘数据。
