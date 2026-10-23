# 分布式实验 Hub 设计

版本：v0.1（设计稿） ｜ 日期：2026年10月23日 ｜ 状态：仅设计，未实现；在 `feat/distributed-hub` 分支迭代
关联：[代码基础设施与Web_Portal设计](代码基础设施与Web_Portal设计.md)、[Colab算力接入与队列运行方案](Colab算力接入与队列运行方案.md)、[断点续训与Colab运行](断点续训与Colab运行.md)、[数据与模型下载管理设计](数据与模型下载管理设计.md)

## 一、目标

1. **配置驱动**：新增实验、新增主干只改配置（加上新主干自己的 worker 目录），不改控制层代码，也不改 bash 脚本。
2. **可部署**：任意实验（例如 T5B2）都能导出成独立部署包，目标环境有两种：普通 GPU 机器、Google Colab。
3. **中央 Hub**：节点把进度和结果汇报给 Hub；Hub 负责调度、展示进度、汇总入库、导出报告。
4. **容器化**：Hub 和本地 GPU 节点可以用容器部署；Colab 不支持 Docker，使用同源构建的部署包。

非目标：不改动 worker 的训练语义，不改动 `RunRecord`（`run.json`）的结构，不改动 matrix 和 report 的准入规则。现有审计和报告链路原样复用。

## 二、现状与问题

| 现状 | 问题 |
|---|---|
| `scripts/rerun_baselines.sh` 用 `case` 分支（T1、T5A、T5B、T5C、B0、B1A……）定义实验，batch、累积步数、种子、后缀写在环境变量里 | 新增实验要改脚本；实验参数没有版本化，出了问题也无法从配置追溯 |
| `dispatch.py` 写死了 `model_id == "anysat_base" and task_id in {...}` 这类能力判断 | 新增主干要改控制层 |
| Colab 流程：手工打 tgz → 在 notebook 格子里手敲命令 → zip 到 Drive → 主控侧解压，再跑 `audit_runs.py` | 步骤多、容易出错；断线后接续要手工换新 run id |
| `_code_version()` 依赖 `git rev-parse` | 用 `git archive` 打出的部署包没有 `.git`，`git_commit` 为空，审计会把这类运行当作不可验证排除 |
| `colab_bootstrap.sh` 下载后不校验 sha256，也不包含 T4 和固定版本的源码 | 节点环境的数据完整性没有保证 |
| Portal 只读本机 `runs/`，没有调度和节点的概念 | 看不到多台机器上的进度 |

## 三、总体架构

```
                    ┌──────────────────── Hub（4090 主机，CPU 侧）────────────────────┐
                    │ 注册表(configs/)  调度器  SQLite(jobs/nodes/events)  入库校验     │
                    │ 结果仓库 /geodata/rs-eval/runs   报告导出   Web(进度/节点/派发)   │
                    └────▲──────────────────────▲───────────────────────▲───────────────┘
                 localhost│              HTTPS（cloudflared，只发布 /api/agent/*）
                          │                     │                       │
                 ┌────────┴───────┐   ┌─────────┴────────┐    ┌─────────┴─────────┐
                 │ node-4090 agent│   │ 其他 GPU 节点 agent│    │ Colab agent        │
                 │ （同机，容器）  │   │ （镜像或部署包）    │    │ （部署包 + ipynb）  │
                 └────────┬───────┘   └─────────┬────────┘    └─────────┬─────────┘
                          └──── 都调用现有的 rs-eval dispatch → worker → run.json ────┘
```

分层原则：

- **worker 层不动**：仍然通过 `WorkerRequest` 输入、`WorkerEvent` JSONL 输出、原子写 `run.json`。
- **agent 是 dispatch 外面的一层**：负责领 job、调用 dispatch、转发 event、上报结果。
- **Hub 是 Portal 的升级版**：结果仍以 `runs_root/<run_id>/run.json` 为唯一事实来源，SQLite 只记录调度和进度这类运行时状态。

## 四、配置层

### 4.1 实验：`configs/experiments/*.yaml`

每个实验对应现在 `rerun_baselines.sh` 里的一个 case 分支，展开后是 模型 × 协议 × 种子 的一组 cell。

```yaml
experiment_id: T5B2
title: T5 冻结表征（三主干 × 3 种子）
task: T5
action: train
cells:
  - {model: prithvi_eo_v2_300_tl, protocol: frozen_probe, seeds: [17, 42, 73]}
  - {model: clay_v1_5,            protocol: frozen_probe, seeds: [17, 42, 73]}
  - model: anysat_base
    protocol: frozen_probe
    seeds: [17, 42, 73]
    env:                              # 训练相关的环境变量冻结在配置里
      ANYSAT_WORKER_BATCH_SIZE: 1
      ANYSAT_WORKER_GRAD_ACCUM: 2
    variants:                         # 可选：同一 cell 的超参变体（取代 B0 那种手写循环）
      - {suffix: lr1e4, env: {PRITHVI_WORKER_LR: 1e-4}}
run_suffix: v3                        # run_id = {model}-{task}-{protocol}-{seed}[-{variant}]-{suffix}
priority: 50
requires:
  resources: [data_multicrop, weights_anysat_base, weights_clay_v1_5, weights_prithvi_300_tl]  # catalog 资源 id
  sources: [anysat_src, clay_src]     # 固定 commit 的上游源码
  min_vram_gb: 22
  disk_gb: 40
  cache_gb: 0                         # T5 两阶段这类实验填写约 430
placement: [local_gpu, colab]         # 允许运行的节点类型
```

规则：

- 实验 yaml 用 pydantic 校验（`extra="forbid"`）。展开后的每条 run 都要通过现有 `plan_dispatch` 的组合校验，校验失败的实验不能导出，也不能入队。
- **实验内容哈希** = 规范化后的 yaml + 所引用的 task/model/protocol yaml + manifest 哈希。这个哈希写进每条 run 的 `tags.experiment_sha`，用于保证同一实验在不同节点上可比。
- `env` 只允许出现在模型白名单（见 4.2 的 `env_allowlist`）里的变量，避免借环境变量偷偷改变协议语义。
- 迁移：`rerun_baselines.sh` 的各个分支逐个改写成 yaml，脚本暂时保留为 `rs-eval experiment run <id> --local` 的薄壳，全部迁移完再删除。

### 4.2 主干：扩展 `ModelSpec`

把写死在 dispatch 里的能力判断移到模型 yaml：

```yaml
model_id: anysat_base
display_name: AnySat Base
worker_id: anysat
checkpoint_env: ANYSAT_CHECKPOINT
upstream_url: https://github.com/gastruc/AnySat
worker_command: [uv, run, --project, workers/anysat, python, workers/anysat/src/anysat_worker/runner.py]
capabilities:
  tasks: [T1, T2, T3, T4, T5, T8]
  protocols: [full_finetune, frozen_probe, frozen_adapter, frozen_adapter_two_stage]
  resume: [T1, T3, T4, T5]            # 取代 dispatch.py 中的硬编码
resources:
  weights: [weights_anysat_base]      # catalog 资源 id
  sources: [anysat_src]
  min_vram_gb: 22
env_allowlist: [ANYSAT_WORKER_BATCH_SIZE, ANYSAT_WORKER_GRAD_ACCUM, ANYSAT_WORKER_EPOCHS, ANYSAT_STAGE_B_START]
```

- 新字段全部可选，保证现有 yaml 和测试兼容；`dispatch.plan_dispatch` 改成读 `capabilities`。
- 协议侧原有的 `workers` / `tasks` 白名单继续保留，和模型能力取交集，两边都允许才放行。
- 提供 `rs-eval scaffold-model <id>`，生成 worker 骨架：`pyproject.toml`、读取 request、使用 `worker_runtime` 写 `run.json` 和发 event 的模板、契约测试。

**新增一个主干的完整步骤**：`scaffold-model` → 实现 `runner.py` → 写模型 yaml → 在 `catalog.yaml` 登记权重 → 跑契约测试 → 在实验 yaml 里引用。

### 4.3 节点：`configs/nodes/*.yaml`（也可以在 Hub 上注册）

```yaml
node_id: node-4090
kind: local_gpu                       # local_gpu | colab
gpu: {name: RTX 4090, vram_gb: 24}
paths:
  data_root: /geodata/rs-eval/data
  weights_root: /geodata/rs-eval/weights
  runs_root: /geodata/rs-eval/runs
  cache_root: /home/yzhong/rs-eval-cache   # 放在 / 所在 NVMe（剩余 1.4T）；/geodata 只剩 335G，放不下 T5 两阶段缓存
labels: [nvme-large]
```

Colab 节点不需要预先登记：agent 第一次连上时，用 `nvidia-smi` 和 `df` 的结果自动注册成临时节点，名字形如 `colab-<短 id>`。

### 4.4 源码依赖：`catalog.yaml` 增加 `kind: git_source`

AnySat 和 Clay 需要固定 commit 的上游源码（`ANYSAT_SRC`、`CLAY_SRC`）。它们作为新的资源类型登记在 catalog 里，字段为 repo、commit、目标目录，让下载、部署包导出、节点 fetch 走同一套逻辑。

## 五、部署包

### 5.1 导出命令

```bash
rs-eval bundle export T5B2 --target local  [--with-weights] [--with-data] -o dist/
rs-eval bundle export T5B2 --target colab  [--with-weights]               -o dist/
```

### 5.2 包内容

| 文件 | 说明 |
|---|---|
| `code/` | `git archive HEAD` 打出的代码。工作区不干净时拒绝导出（可用 `--allow-dirty` 强制，但版本会标记为 `-dirty`，这类结果不进正式表） |
| `bundle.json` | `bundle_id`、commit、实验哈希、manifest 哈希、目标环境、生成时间、包内所有文件的 sha256 |
| `plan.json` | 展开后的 run 列表：run_id、模型、协议、种子、env、Hub 侧 job id。节点上不再做任何决策 |
| `data/manifests/<task>/` | 和主控侧完全相同的 manifest 文件 |
| `weights/`（可选） | 带 `--with-weights` 时包含，并附带 sha256 |
| `.env.template` | local：填写外置目录；colab：预先填好 `/content/rs-eval/...` |
| `hub.json` | Hub 的 URL、**只对这个包有效的 token**（只能上报本包里的 job，可在 Hub 上吊销）、心跳间隔 |
| `run.sh` | local 的入口：装 uv → `uv sync` 根环境和所需 worker → `rs-eval fetch` → `rs-eval agent` |
| `T5B2.ipynb` | colab 的入口：生成好的 3 个格子（解包和环境准备 → fetch → 后台启动 agent 并查看日志） |

### 5.3 代码版本注入

`worker_runtime._code_version()` 改为：先读 `RS_EVAL_CODE_VERSION`，部署包的 `run.sh` 和容器镜像会从 `bundle.json` 或镜像 label 注入它；读不到时再回退到 git。Hub 入库时核对 `run.json.git_commit` 与 `bundle.json.commit` 是否一致，这样现有的 `audit.excluded_run_ids` 不需要修改。

### 5.4 节点侧下载：`rs-eval fetch`

不带权重或数据时，在节点上执行：

```bash
rs-eval fetch --plan plan.json        # 或 --experiment T5B2
```

- 从实验的 `requires` 和模型的 `resources` 推出需要的 catalog 资源 id，**复用 `downloads.py` 引擎**（aria2c / hf / gsutil，失败时用 Python 续传）。
- 下载后在节点上校验 sha256，已存在且校验通过的资源跳过；可以重复执行。
- 先校验 manifest 里引用的每个文件都存在（逻辑来自 `colab_bootstrap.sh` 的 `verify_task`）。
- 下载进度通过 agent 通道上报 Hub，Web 上可以看到节点处于“准备数据”阶段。
- 完成后 `colab_bootstrap.sh` 不再需要，T4 和上游源码这两处缺口也一起补齐。

## 六、节点 agent

```bash
rs-eval agent --bundle ./T5B2-colab        # 部署包模式：只执行包内的 job
rs-eval agent --node configs/nodes/node-4090.yaml   # 常驻模式：从 Hub 领取任意符合条件的 job
```

### 6.1 职责

1. **注册和心跳**（默认 30 秒）：上报 GPU 型号和显存占用、磁盘、当前 job、agent 版本。
2. **领取 job**：调用 `claim`，拿到 job 和租约（默认 10 分钟，每次心跳续期）。
3. **执行**：设置 env，调用现有的 `dispatch()`。保留 `rerun_baselines.sh` 里的两项保护：GPU 空闲等待、已完成的 run 不覆盖。
4. **上报**：把 `WorkerEvent` 批量转发给 Hub；`run.json` 的 status、curves、metrics 变化时同步上报一份快照。
5. **检查点**：检测到新的 `latest.pt` 时，分块上传到 Hub（Colab 上同时可选复制一份到 Drive）。
6. **收尾**：把运行目录打成 tar 分块上传，Hub 校验并入库后，job 进入 `ingested`。

### 6.2 进程保活

- 本地节点：容器使用 `restart: unless-stopped`；在宿主机上运行时，用 systemd user service，取代 `setsid --detached` 那套 PID 文件机制。
- Colab：notebook 格子用 `setsid nohup rs-eval agent ... &` 把 agent 放到后台，格子只负责启动和 `tail` 日志。VS Code 插件断开后，agent 在 VM 上继续运行，直到 VM 被回收。

### 6.3 抢占与续训

- 租约超时（Colab 会话结束、机器断电）后，Hub 把 job 标记为 `preempted`，保留最后一个检查点。
- 再次派发时，agent 会拿到检查点的下载地址，用新的 run id（例如 `...-r2`）和 `--resume-from` 接续。续训链记录在 `tags.resumed_from`。
- 对不支持续训的组合（不在 `capabilities.resume` 里），只能从头重跑，旧的运行目录保留但不参与比较。

### 6.4 断网兜底：Drive 邮箱模式

agent 连不上 Hub 时，把 events 和结果包写到 Drive 上的 `rs-eval-outbox/<bundle_id>/`。Hub 侧定时任务用 `rclone` 同步这个目录，再走同样的入库流程。这也是“Hub 定期抓取”的落地方式，并且不要求节点能被外部访问。

## 七、Hub

### 7.1 状态存储（SQLite，`/geodata/rs-eval/hub/hub.db`）

| 表 | 主要字段 |
|---|---|
| `experiments` | id、yaml 快照、实验哈希、创建时间、状态（draft / active / done / archived） |
| `jobs` | id、experiment_id、run_id、cell、seed、env、状态、priority、`pinned_node`、`assigned_node`、租约到期时间、尝试次数、最近检查点、bundle_id |
| `nodes` | id、kind、能力（GPU、显存、磁盘、labels）、最后心跳、当前 job、启用或停用 |
| `events` | job_id、时间、level、event、message、progress（只保留最近 N 条，完整日志在运行目录里） |
| `bundles` | id、experiment_id、target、token 哈希、是否吊销 |
| `uploads` | 分块上传会话：job_id、类型（checkpoint / result）、总大小、已收块、sha256 |

### 7.2 Job 状态机

```
planned ──► queued ──► assigned ──► running ──► uploading ──► ingested
   │           ▲           │           │  ▲          │
   │           │           ▼           ▼  │          ▼
   └─► held    └──── preempted ◄── 租约超时   failed（可重新排队）
                                    cancelled（人工）
```

- `held`：人工暂停，调度器跳过。
- `failed` 记录 worker 的退出码和最后 50 行日志；尝试次数超过上限（默认 2）后不再自动重试。

### 7.3 调度：自动分配加人工调整

- **自动**：agent 调用 `claim` 时，调度器在 `queued` 的 job 中筛选：`placement` 包含节点类型、显存和磁盘满足 `requires`、所需资源在该节点已就绪或允许 fetch。然后按 `priority`、实验创建时间、cell 顺序挑选。
- **人工**（Web 和 CLI `rs-eval hub ...` 都能操作）：
  - 把 job **钉到**某个节点（设置 `pinned_node`；自动调度只把它派给这个节点）；
  - 调整 priority 和顺序，暂停或恢复，取消，失败后重新排队；
  - **迁移**：把 running 或 preempted 状态的 job 迁移到另一个节点，带上最近的检查点续训（仅限支持续训的组合）；
  - 停用节点：节点上当前 job 跑完后不再派新 job。
- 人工设置的优先级始终高于自动调度。

### 7.4 API

现有 API（`/api/runs`、`/api/matrix`、`/api/report.md`、`/api/downloads/*`）保持不变。新增：

**Agent 接口**（通过 tunnel 对外发布，用 bundle token 或节点 token 鉴权）：

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/agent/register` | 节点注册或更新能力 |
| POST | `/api/agent/heartbeat` | 心跳、续租，返回需要执行的指令（取消、暂停） |
| POST | `/api/agent/claim` | 领取 job，返回 job、env、检查点地址 |
| POST | `/api/agent/jobs/{id}/events` | 批量上报 event 和 `run.json` 快照 |
| POST | `/api/agent/uploads` | 创建分块上传会话 |
| PUT | `/api/agent/uploads/{uid}/{n}` | 上传第 n 块（约 32MB；Cloudflare 免费版单个请求体上限约 100MB） |
| POST | `/api/agent/uploads/{uid}/complete` | 合并、校验 sha256，触发入库 |
| GET | `/api/agent/checkpoints/{job}` | 下载续训用的检查点（支持 Range） |

**管理接口**（不经过 tunnel 对外发布，只能通过本机或 SSH 隧道访问；以后如果要公开，加 Cloudflare Access）：
`/api/hub/experiments`（列出、导入 yaml、激活）、`/api/hub/jobs`（列出、钉住、优先级、暂停、取消、重新排队、迁移）、`/api/hub/nodes`、`/api/hub/bundles`（导出、吊销 token）、`/api/hub/report?experiment=T5B2`。

### 7.5 入库校验

结果包上传完成后，Hub 依次检查：

1. tar 解包使用安全模式：拒绝绝对路径、`..`、符号链接；
2. `run.json` 符合 `RunRecord` schema，`run_id` 与 job 一致，没有覆盖已存在的运行目录；
3. `git_commit` 与 bundle 的 commit 一致；`tags.experiment_sha` 与实验哈希一致；
4. smoke 运行或缩减预算的运行写到 `runs_smoke/`，不进入正式 `runs/`；
5. 移动到 `runs_root/<run_id>/`，写入 `tags.node_id`、`tags.gpu_name`、`tags.bundle_id`。

任何一步失败时，结果包保存在 `hub/quarantine/`，job 标为 `failed`，并记录原因。入库后 `build_matrix` 和 `build_markdown` 不需要修改就能看到新结果；`tags.gpu_name` 方便在报告里注明跨设备的数值差异。

### 7.6 Web

在现有 `web/index.html`（无构建步骤）的基础上增加四个视图：

- **实验**：每个实验一张 cell 网格（模型 × 种子），格子颜色表示状态，显示 epoch 进度条和当前主指标；点开可看实时曲线、event 流、所在节点、续训链。
- **节点**：在线状态、GPU 和显存、当前 job、最后心跳时间、启用开关。
- **队列**：job 列表，可以拖拽排序，也有钉住、暂停、取消、迁移按钮。
- **导出**：选实验、选节点类型，导出部署包；按实验或全局导出报告（复用 `report.build_markdown`，增加实验过滤参数）。

进度刷新先用 5 秒轮询，以后有需要再换成 SSE。

## 八、网络与安全

- Hub 运行在 4090 主机（`home9950x`，外网通过 `superderp.cn:6001` SSH 进入），继续监听 `127.0.0.1:8080`。
- `cloudflared` 使用命名 tunnel，把 `hub.<域名>` 只路由到 `/api/agent/*`，其他路径一律返回 404。前提是有一个 DNS 托管在 Cloudflare 的域名；测试阶段可以先用 `trycloudflare` 的临时域名。
- 访问管理界面：继续使用 SSH 隧道（`portal_client.sh` 改为默认连接 6001 端口）。原来的单会话限制保留。
- Token：bundle token 只能访问本包里的 job，节点 token 只能访问该节点自己的 job。数据库里只存哈希，可以随时吊销。
- Colab 的 VS Code 插件和网页版在网络上没有区别，都是 VM 主动出站访问 Hub。区别在于插件里 `drive.mount()`、Colab Secrets（`userdata.get`）、`auth` 链接这些功能曾经不可用或不稳定，所以设计上：
  - 检查点主路径走 Hub 上传，Drive 只作为可选副本；
  - token 放在部署包里，不依赖 Colab Secrets；
  - agent 放到后台运行，不依赖 notebook 格子一直保持运行。

## 九、容器化

| 组件 | 方式 | 说明 |
|---|---|---|
| Hub | `deploy/docker-compose.yml`：`hub` 和 `cloudflared` 两个服务 | 卷：`/geodata/rs-eval/{runs,data,weights,hub}`；端口只绑定 `127.0.0.1:8080` |
| 本地 GPU 节点 | 节点镜像 + nvidia runtime（4090 主机已安装 Docker 29.0.2 和 nvidia runtime） | 镜像里有 agent 和 anysat、clay、prithvi、olmoearth 四套 venv（按各自的 `uv.lock` 构建），以及固定版本的上游源码；数据、权重、缓存、运行目录都挂卷 |
| Colab | 不支持（没有 Docker 守护进程） | 使用部署包；和镜像从同一个 commit、同一份 `uv.lock` 构建 |

- **版本**：镜像 label 里写入 commit，启动时注入 `RS_EVAL_CODE_VERSION`。Hub 入库时，容器节点和部署包节点按同一规则校验。
- **镜像体积**：四套 PyTorch 环境大约 15–20GB。第一版用单镜像，agent 在容器内直接调用 worker；以后体积成为问题时，再拆成每个 worker 一个镜像，由 agent 通过 docker socket 启动兄弟容器。
- **缓存卷**：T5 两阶段的 dense 缓存约 428GB，节点的 `cache_root` 卷必须放在空间足够的 NVMe 上（4090 主机上就是 `/` 所在盘）。

## 十、迁移与兼容

1. 现有 `runs/` 原样保留。Hub 第一版只读 `runs/`，功能和现在的 Portal 一样，可以直接替换现在 8080 端口上的进程。
2. 4090 上正在运行的 `rerun_baselines.sh` 队列不受影响。等它跑完，再启用 node-4090 agent，由 Hub 接管调度。
3. 历史运行没有 `experiment_sha` 和 `bundle_id`，matrix 和 report 照常统计；实验视图里把它们标注为“历史（非 Hub 派发）”。
4. `rerun_baselines.sh` 的各个分支逐个迁移成 `configs/experiments/*.yaml`，每迁移一个都要对比展开后的 run 列表和环境变量，确保完全一致。

## 十一、分期

| 期 | 内容 | 验收标准 |
|---|---|---|
| P1 | 实验 yaml 和校验、`ModelSpec.capabilities`、`rs-eval experiment plan/run`、迁移 2–3 个 case | `experiment plan T5B2` 展开的 run 和 env 与原 bash 分支完全一致；dispatch 不再有硬编码的模型判断 |
| P2 | `bundle export`（local/colab，可选带权重和数据）、`rs-eval fetch`、`git_source` 资源、版本注入、`rs-eval ingest <tar>` 手动入库 | 在 Colab 上用导出的包跑完一条 smoke 运行，手动入库后能在 matrix 中看到，审计不排除它 |
| P3 | Hub 的 SQLite、agent API、分块上传、租约和抢占续训、node agent、cloudflared | Colab 断线后，在新会话里自动续训，结果自动入库 |
| P4 | Web：实验、节点、队列、导出视图；手动钉住和迁移 | 在 Web 上完成一次手动迁移，并按实验导出报告 |
| P5 | 容器化：Hub 的 compose、节点镜像 | node-4090 用容器跑完一条运行，结果和宿主机运行一致（同一 commit、同一协议） |

P1 和 P2 不依赖网络暴露，单独做完也有用，建议先做。

## 十二、待定事项

1. Cloudflare 托管的域名（`superderp.cn` 的 DNS 是否在 Cloudflare 上）。
2. 3080 机器是否也接入为节点，节点名和路径是什么。
3. Colab VS Code 插件下 VM 的空闲回收时间、`drive.mount()` 是否可用，需要用当前插件版本实测。
4. Hub 上检查点的保留策略：按 job 只保留最近一个，还是按磁盘配额清理。
5. 跨设备运行（4090、A100）是否在报告中分开列，还是只标注不分开。目前倾向只标注，因为协议相同。
