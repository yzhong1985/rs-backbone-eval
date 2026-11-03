# AEF × PASTIS-HD 对齐抽查（2026-11-03）

目的：在正式提取 2,433 个 patch 之前，确认 AEF 2019 年度 embedding 的 10 m 网格
与 PASTIS-HD 的 10 m S2 网格、1 m SPOT 影像和地块边界对得上。

## 方法

- 取 3 个 patch：10000、10003（t30uxv / 30N）、20020（t31tfj / 31N）；
- 以 **SPOT 的仿射变换**为准，在 1 m 网格上取 `5 + 10k` 的像元中心，得到 128×128
  的 10 m 采样点（不是用 patch bbox 猜）；
- 把采样点转到 AEF COG 的 UTM，按最近像元读取 64 维特征；
- AEF 特征做 `sign(x)·(x/127.5)²` 后取 PCA 前三维做伪彩色；
- S2 取 2019 年最接近 07-15 的一景，`B4/B3/B2` 百分位拉伸；
- 把 PASTIS `ParcelIDs` 的边界叠加到 SPOT 1 m 和 AEF 10 m 图上；
- 定量：AEF 第一主成分与 S2 NDVI 的空间相关，并对 AEF 做 ±2 像元平移，
  看最佳相关是否落在 (0,0)。

## 结果

| patch | 区域 | corr(PC1, NDVI) @ (0,0) | 最佳平移 | 最佳 corr |
| --- | --- | ---: | --- | ---: |
| 20020 | 31N | **-0.590** | (0, 0) | -0.590 |
| 10003 | 30N | -0.416 | (0, -1) 即 10 m | -0.421 |
| 10000 | 30N | +0.120 | (+2, +2) 即 20 m | +0.133 |

图：

- `/home/yzhong/rs-eval/audits/aef-20261103/pastis_aef_alignment.png`
  （三行：S2 RGB / SPOT 1 m / AEF PCA / 标签）
- `/home/yzhong/rs-eval/audits/aef-20261103/pastis_aef_zoom_10003.png`
  （10003 的 512×512 放大，含地块边界叠加）

## 解读

1. **20020 完全对齐**（最佳平移 0，相关 -0.59），说明“SPOT 仿射 → 10 m 采样点 →
   AEF 最近像元”的路径是对的；
2. **10003 差 1 个像元（10 m）**，在 SPOT/S2 网格与 AEF COG 原点之间可能是半像元
   约定差异；对 10 m 任务可以接受，但正式提取前应统一到同一约定；
3. **10000 差 2 个像元、相关很弱（0.12）**，这一景需要单独查：可能是它的 COG 位于
   tile 边界、或该区域 AEF 年度特征与 7 月单景 NDVI 本来就关系弱；
4. 视觉上，AEF 伪彩色的色块变化与地块边界基本一致，SPOT 上的边界与田块边缘也一致；
   AEF 是 10 m 年度摘要，块状/混合像元是预期的，不能按“真彩色”标准比较。

## 对 T11 提取的影响

- **采样网格以 SPOT 仿射为准**，不要用 patch bbox；
- 正式提取时逐 patch 记录：COG 路径、窗口、仿射、`flipped`（COG 的 `transform.e>0`）、
  采样到的有效比例；
- 10000 的 2 像元偏移要在正式提取前定位：先检查它的 COG 是否跨 tile，再决定是统一
  平移还是换 COG；
- 32N（patch 40000）本次未做：对应 COG 还在下载，之后补一张。

## 复现

```
workers/lim/.venv/bin/python /tmp/opencode/align_fig.py
workers/lim/.venv/bin/python /tmp/opencode/zoom_fig.py
```

（临时脚本；正式提取器会把这些步骤固化到 `scripts/aef_extract_pastis.py`。）
