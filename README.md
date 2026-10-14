# RS Backbone Eval

AnySat、Clay 与 Prithvi-EO-2.0 在7个官方下游应用任务（8份任务配置）上的可复现评测框架。

仓库只保存代码、配置、协议和文档。数据集、模型权重、特征缓存、checkpoint、预测影像和运行结果必须放在仓库外，并通过环境变量传入。

## 组成

- `src/rs_backbone_eval/`：协议、任务注册表、运行记录和命令行；
- `workers/`：AnySat、Clay、Prithvi隔离运行环境的接入点；
- `configs/`：任务、模型和实验协议；
- `configs/downloads/catalog.yaml`：官方数据与权重来源、大小、校验值和外置目标；
- `web/`：无构建步骤的结果与下载管理Portal前端；
- `docs/`：实验设计与基础设施说明；
- `tests/`：契约、注册表和Portal API测试。

## 快速开始

```bash
cp .env.example .env
uv sync --extra portal --extra downloads --extra dev
uv run rs-eval doctor
uv run rs-eval list-tasks
uv run rs-eval list-downloads
uv run rs-eval portal --runs-root /path/to/external/runs
```

浏览器打开 `http://127.0.0.1:8080`。Portal只读取每个运行目录中的 `run.json`，不会加载模型或修改实验结果；下载控制只允许执行目录配置中声明的官方资源，并写入外置 `data/weights` 根目录。

### Portal 启停与远程访问

```bash
scripts/portal_server.sh start|stop|restart|status|url   # 服务端：在本机管理 Portal
scripts/portal_client.sh yzhong@<本机IP> 8080            # 客户端：在任意SSH设备建立隧道
```

`portal_server.sh` 会读取 `.env` 中的 `RS_EVAL_PROXY`（下载子进程走代理）、`RS_EVAL_PORTAL_HOST/PORT`，启动后打印各远程设备的接入方式。默认只监听 `127.0.0.1`；从别的机器访问时，在该机器上运行 `portal_client.sh`（客户端脚本）建立 SSH 隧道后打开 `http://127.0.0.1:8080`。

隧道是**单会话**的：任意时刻只允许一台设备持有 Portal 会话，新设备接入会自动撤销前一台（其 SSH 会话被终止、隧道断开）。在评测机上用 `scripts/portal_server_session.sh status` 可查看当前持有者。若确需局域网直连，用 `RS_EVAL_PORTAL_HOST=0.0.0.0 scripts/portal_server.sh restart`（注意下载控制API会暴露给局域网，此时单会话限制不适用）。

数据清单用 `rs-eval audit-data <manifest.jsonl> --task T1` 校验：schema、波段、传感器、时相数、split命名、组泄漏和必备split。指标层（`evaluation/`）提供分割/分类/回归的全局累积指标和bootstrap置信区间，供全部worker共用。

普通HTTPS资源优先使用 `aria2c`，缺失时回退到Python续传下载器。Hugging Face仓库快照使用 `hf` CLI；Sen1Floods11需要额外安装 `gsutil` 或 `gcloud`。下载可在Portal暂停并续传，完成后会校验已提供的SHA-256。

## 外置目录

```text
/mnt/rs-eval/
├── data/          # 原始和处理后的数据
├── weights/       # AnySat、Clay、Prithvi权重
├── cache/         # 冻结特征、索引和临时文件
└── runs/          # <run_id>/run.json、曲线、预测和案例图
```

运行记录格式见 [`docs/代码基础设施与Web_Portal设计.md`](docs/代码基础设施与Web_Portal设计.md)，下载设计见 [`docs/数据与模型下载管理设计.md`](docs/数据与模型下载管理设计.md)。正式实验设计见 [`docs/七任务冻结主干评测设计_AnySat_Clay.md`](docs/七任务冻结主干评测设计_AnySat_Clay.md)。

新增模型的选择与分阶段预算见[模型扩展与低成本评测计划](docs/模型扩展与低成本评测计划.md)：默认只新增 OlmoEarth Tiny、评测两个任务，DOFA 与 TerraMind 按需二选一扩展。新增模型的可比性评估见[外部模型可比性评估](docs/外部模型可比性评估.md)：按容量/多模态/多分辨率/时序衡量外部参照能回答哪类问题；OlmoEarth Tiny 优先，架构结论由自研模型的受控消融支撑。历史运行与当前协议的一致性见[历史运行审计](docs/历史运行审计.md)（脚本 `scripts/audit_runs.py`）。
