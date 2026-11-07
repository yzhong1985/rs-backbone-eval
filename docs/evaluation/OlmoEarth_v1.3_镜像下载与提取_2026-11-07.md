# OlmoEarth v1.3 本地镜像下载与 T11O 提取（2026-11-07）

## 1. 为什么改成本地镜像

`data.source.coop` 由 Cloudflare 代理到 Source Cooperative 的 S3 数据代理。本机实测：

- 单连接 2–50 KB/s（同一时段 GCS 可达 3.3 MB/s，PyPI/GitHub 正常）；
- `aria2c -j4 -x8` 跑数小时后大量连接停在 99% 附近，聚合只有 50–90 KB/s；
- 135 个 shard、合计 **59.5 GB**，按该速率需要 10 天以上。

因此改为：**按需下载 135 个 shard 到本地 Zarr 镜像 → 本地 zarr 读取 → 提取 npz**。
镜像在仓库外：`/home/yzhong/rs-eval/data-olmoearth/cache/embeddings.zarr/`
（`/geodata` 只剩 32 GB，放不下；`/home` 有 1.1 TB）。

## 2. shard 集合是精确的

PASTIS-HD 的 2,433 个 patch 按 AEF sidecar 的 `grid_bbox` 映射到 OlmoEarth 的
UTM 30/31/32 网格。数组布局（`utm*/embeddings/zarr.json`）：

- shape `(9, 128, 1867776, 98304)` int8；
- 外层 shard `(1, 128, 2048, 2048)`，内层 chunk `(1, 64, 256, 256)`，zstd；
- 存储路径 `c/{year_index}/{dims_half}/{shard_y}/{shard_x}`。

按 patch 读取窗口（128×128）求 shard 并集：**135 个 shard**，每个 shard 含
1–54 个 patch（中位数 18）。没有可以安全跳过的 shard；128 维与 64 维共用同一
shard（外层 chunk 覆盖 128 维），所以下载一次即可支持两种维度。

## 3. 下载器

`/tmp/opencode/olmoearth_fetch.py`（一次性工具，不入库）：

- 2 MiB 定长 Range 请求，384–640 个并发 worker；
- 45 s 无字节即断开重连；每段最多重试 40 次；
- 已完成的段保存在 `<file>.parts/`，重启只补缺失段；
- 全部段到齐后顺序拼接为正式对象，再原子替换。

实测 640 并发稳定在 **2.4–2.6 MB/s**（每连接约 4 KB/s），59.5 GB 约需 7 h。
继续加并发不再提速，说明瓶颈在数据代理侧。

## 4. 本地读取校验

用已下载完成的 `utm30/195/30` 重取 patch 10000（`--zarr-root`）：

- 本地读取耗时 < 0.1 s（远程单 patch 需要数十秒到数分钟）；
- 与 2026-11-06 远程提取的 `features/2019/10000.npz` **逐元素完全一致**
  （features 与 valid 均 `array_equal`）。

说明本地镜像路径没有改变取窗、对齐或量化语义。

## 5. 编码口径（对方案文档的修正）

方案文档 §5.5 写的是 `olmoearth_int8_as_float16`。实际实现改为 **`olmoearth_int8`**：

- 提取器直接保存 int8（`np.savez`），不做 float16 转换；
- 64 维 npz 约 1.05 MB/patch，2,433 patch 约 2.6 GB（float16 会翻倍到 5.2 GB）；
- dequantize 与 AEF 完全一致：`sign(v) * (|v|/127.5)^2`，`-128` 为 NODATA；
- worker 通过 sidecar/`aux_paths.encoding` 选择预处理标签，不再依赖固定字符串。

这样 64 维与 128 维、AEF 与 OlmoEarth 四类输入共用同一份解码实现，只由声明字符串区分。

## 6. 当前状态

- 下载：进行中，约 2.4 MB/s；
- 已有 891 个 patch（2026-11-06 远程提取）可直接复用；
- T11O task/model/protocol、manifest builder、BOE/BOEM 轨与准入锚点已提交
  （`bcea012`）；
- 待下载完成后执行：

  ```bash
  uv run --python 3.12 --with 'zarr>=3' --with obstore --with numpy \
    python scripts/olmoearth_extract_pastis.py --dims 64 \
    --zarr-root /home/yzhong/rs-eval/data-olmoearth/cache/embeddings.zarr
  uv run rs-eval build-manifest --task T11O
  ```

  再按 `olmoearth_v13_probe_{mlp,upernet}` × seed 17/42/73 跑 8 epoch。
