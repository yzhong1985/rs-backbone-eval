# Docker 镜像（2026-11-16）

一条命令把代码和 **7 个 uv 环境**（根环境 + 6 个 worker）打成一个镜像；数据、权重、缓存、
账本、队列**全部挂载**，不进镜像。

## 为什么镜像这么大

torch 占了几乎全部体积，而 6 个 worker 钉的是 4 个不同 build：

| worker | torch | venv |
|---|---|---:|
| aef | 2.15.0+cpu | 0.9 GB |
| clay | 2.4.0+cu121 | 5.7 GB |
| lim | 2.9.1+cu128 | 6.8 GB |
| olmoearth | 2.9.1+cu128 | 6.8 GB |
| anysat | 2.14.1+cu130 | 5.5 GB |
| prithvi | 2.14.1+cu130 | 6.5 GB |

同版本的才能共用字节，所以镜像里实际是 4 份 torch：
`2.9.1`（lim+olmoearth 共享）、`2.14.1`（anysat+prithvi 共享）、`2.4.0`、`2.15.0+cpu`。
去重靠 **`UV_LINK_MODE=hardlink` + 单一 RUN 层**：uv 从 `/opt/uv-cache` 把文件硬链接进各
venv，Docker 导出的 tar 保留硬链接，所以每一份文件只存一次。

**不要**把这个 RUN 拆开，也不要把 venv 用 `COPY` 拷进来 —— 两种情况都会退化成实体复制，
镜像从 ~20 GB 涨到 ~30 GB。

想进一步缩小只有两条路，代价都在模型侧：统一 torch 版本（要验证 claymodel / AnySat 的兼容性），
或把 6 个 worker 合成一个环境（违背现在的隔离约定）。

## 构建

```bash
cd /home/yzhong/projects/rs-backbone-eval
UID=$(id -u) GID=$(id -g) docker compose -f docker/compose.yaml build
```

首次构建要下 ~30 GB 的 wheel，10–30 分钟取决于网速。之后改代码只重建最后几层。

## 冒烟

```bash
docker compose -f docker/compose.yaml run --rm rs-eval uv run rs-eval doctor
docker compose -f docker/compose.yaml run --rm rs-eval nvidia-smi
docker compose -f docker/compose.yaml run --rm rs-eval uv run pytest -q tests/test_queue.py
```

`doctor` 应该显示三个挂载目录存在；`nvidia-smi` 应该看到 4090。

## 跑实验

```bash
# 队列（推荐：与宿主上的 lane 共用同一把 GPU 锁）
docker compose -f docker/compose.yaml run --rm rs-eval \
  uv run rs-eval queue add --task T11D --model clay_v1_5 \
  --protocol clay_rgbn_probe_s64_e80 --seed 17 --run-id <run_id>
docker compose -f docker/compose.yaml run --rm rs-eval uv run rs-eval queue run --once

# 或直接跑一条 lane
docker compose -f docker/compose.yaml run --rm rs-eval \
  bash scripts/rerun_baselines.sh T11DE80
```

## 注意

- **GPU 驱动**：anysat/prithvi 的 `+cu130` wheel 需要宿主驱动 ≥ 580；驱动太旧就把这两个
  worker 的 torch 降一档（要改 `uv.lock`，属于模型侧决定）。
- **`/tmp` 必须挂载**：GPU 锁在 `/tmp/rs-eval-gpu.lock`，容器内外要看到同一把锁，否则宿主上
  跑着 lane 时容器还会再起一个 run。
- **上游源码**：`CLAY_SRC` / `ANYSAT_SRC` 指向 `/geodata/rs-eval/tools/`（只读挂载），
  镜像里不带。
- **`.env` 不烘焙**：镜像里的 `RS_EVAL_*` 都是容器路径；宿主路径由 compose 的 `volumes`
  和 `environment` 决定。
- 容器以宿主 UID 运行，写出的 `runs/` 文件属主与宿主一致，`audit_runs.py` 不受影响。
