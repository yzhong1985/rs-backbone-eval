# OlmoEarth v1.3 × PASTIS 对齐 QC（2026-11-06）

> **2026-11-07 更正（P1 评审）**：本文的 `(0,+1)` 残差不是随机半像元不确定性，而是
> 提取器把 Zarr 仿射原点当作像元中心、少减 0.5 像元导致的系统性列偏移。实际影响
> 1,087/2,433 个 patch。提取器已改为 `floor((x-c)/a)`（最近像元中心），64/128 维已
> 重提、T11O 已重跑；旧结果标记 superseded。本文的“接受当前网格、不做平移修正”
> 结论作废，以 `T11O_OlmoEarth_v1.3_实验设置_2026-11-07.md` 为准。

- 数据：OlmoEarth v1.3 2019 年度 64 维 embedding，10 m
- 采样网格：AEF sidecar 的 SPOT affine 10 m 网格（`5 + 10k`）
- 提取：`scripts/olmoearth_extract_pastis.py`
- QC 图：`scripts/olmoearth_align_qc.py`
- 产物目录：`/home/yzhong/rs-eval/audits/olmoearth-20261106/`

## 结果

24 个 AEF QC patch 已全部提取；patch `10000` 后来补上。

24 个 patch：

- **valid share 全部为 1.0**；
- PC1 与 2019 年 7 月 S2 NDVI 的相关范围约 **-0.886 ～ 0.909**；
- 最佳平移：**9/24 在 (0,0)**，**20/24 在 1 像元内**，**24/24 在 2 像元内**；
- **8/24 的最佳平移是 `(0,+1)`**，集中在 30N/32N 的部分 patch；
- 但 24 patch 的 SPOT 亮度交叉检查给出**混合方向**（6/24 在 (0,0)，21/24 在 1 像元内，没有统一 +1），因此按“几何参考”标准不认为存在系统性偏移；AEF 的 NDVI (0,0) 严格标准不通过，但不再据此做平移修正；
- 30N 的 ymax 小数部分约 0.44、32N 约 0.45，31N 约 0.15/0.5；`(0,+1)` 与 ymax 的小数部分相关，提示 OlmoEarth Zarr 的 10 m 像元中心与 SPOT `5+10k` 采样点可能有半像元/取整约定差异；
- 在全量提取前必须复核这个像素中心约定；**不做逐 patch 平移修正**，也不把当前 877 个 patch 的全量提取继续下去。

## 图

- `olmoearth_pastis_alignment.png`：23 个 patch 的堆叠总图；
- `olmoearth_pastis_alignment_<patch>.png`：每个 patch 的四联图  
  **S2 RGB / SPOT 1 m / OlmoEarth PCA / S2 + 地块标签**；
- `olmoearth_alignment_batch.csv`：patch、tile、fold、日期、valid、PC1-NDVI 相关、最佳平移。

## 结论与限制

- 23 个 patch 的视觉图显示地块边界、道路、田块纹理基本对得上；
- 定量上没有系统性大偏移，最大 2 像元；
- 决定接受当前 10 m 网格，报告写“±1 像元残余不确定性”；不做逐 patch/全局平移修正；
- 全量 64 维提取恢复；当前 877 个 patch 作为全量提取的前缀继续使用；
- 这只是**对齐 QC**，不是 OlmoEarth vs AEF 的模型分数；训练对比尚未开始。

## 复现

```bash
cd /home/yzhong/projects/rs-backbone-eval

# 1) 提取（24 patch；已成功的会跳过）
uv run --python 3.12 --with 'zarr>=3' --with obstore --with numpy \
  python scripts/olmoearth_extract_pastis.py --dims 64 --workers 1 \
  --patch-ids 10000,10107,10213,10323,10432,10540,20000,20128,20256,20387,20522,20657,30000,30146,30292,30440,30589,30737,40000,40111,40223,40337,40451,40562 \
  --out /home/yzhong/rs-eval/data-olmoearth/features/2019

# 2) QC 图
PYTHONPATH=src:workers/lim/src workers/lim/.venv/bin/python \
  scripts/olmoearth_align_qc.py
```

## 相关

- `docs/evaluation/T11_OlmoEarth_v1.3_embedding对齐与比较方案_2026-11-06.md`
- `docs/evaluation/AEF_PASTIS对齐抽查_2026-11-03.md`

## 像素中心复核

见 `docs/evaluation/OlmoEarth_像素中心复核_2026-11-06.md`：NDVI 的 (0,+1) 与 SPOT/AEF 的混合结果不一致，暂不支持统一平移修正。
