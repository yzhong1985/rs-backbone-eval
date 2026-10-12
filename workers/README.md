# Model workers

AnySat、Clay和Prithvi不要强行安装到同一个Python环境。控制层把任务、模型、协议和外置路径解析成worker request，各worker只负责：

1. 加载对应上游代码和模型权重；
2. 执行 `probe`、`extract_features`、`train` 或 `evaluate`；
3. 持续写入统一的 `<runs_root>/<run_id>/run.json`；
4. 将checkpoint、预测、日志和案例图写到同一外置运行目录；
5. 退出前用根项目的 `rs-eval validate-run` 校验记录。

建议分别使用：

```text
workers/
├── anysat/   # Python/CUDA版本跟随AnySat官方commit
├── clay/     # Python/CUDA版本跟随Clay v1.5官方commit
└── prithvi/  # Python/CUDA版本跟随TerraTorch/Prithvi锁定commit
```

每个worker最终应包含独立的 `pyproject.toml` 或容器镜像、锁文件、上游commit、adapter实现和smoke test。worker环境只安装当前模型，公共数据manifest和run契约由根项目维护。

模型权重不得复制到worker目录。通过 `ANYSAT_CHECKPOINT`、`CLAY_CHECKPOINT`、`PRITHVI_CHECKPOINT` 指向外置文件。
