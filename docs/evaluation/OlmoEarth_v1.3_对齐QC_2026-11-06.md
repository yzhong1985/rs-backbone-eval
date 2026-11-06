# OlmoEarth v1.3 × PASTIS 对齐 QC（2026-11-06）

- 数据：OlmoEarth v1.3 2019 年度 64 维 embedding，10 m
- 采样网格：AEF sidecar 的 SPOT affine 10 m 网格（`5 + 10k`）
- 提取：`scripts/olmoearth_extract_pastis.py`
- QC 图：`scripts/olmoearth_align_qc.py`
- 产物目录：`/home/yzhong/rs-eval/audits/olmoearth-20261106/`

## 结果

24 个 AEF QC patch 中，**23 个成功提取**；patch `10000` 在 Source Cooperative 读取时连续超时，暂缺（此前人工小窗口读取 valid=1.0，说明是网络/分块问题，不是无数据）。

23 个成功 patch：

- **valid share 全部为 1.0**；
- PC1 与 2019 年 7 月 S2 NDVI 的相关范围约 **-0.886 ～ 0.909**；
- 最佳平移：9/23 在 `(0,0)`，19/23 在 1 像元内，23/23 在 2 像元内；
- 没有发现统一的全局偏移方向；部分 30N/32N patch 的最佳平移是 `(0,+1)`，但相关系数提升很小（通常 <0.02），更像半像元/取整噪声，不做逐 patch 平移修正。

## 图

- `olmoearth_pastis_alignment.png`：23 个 patch 的堆叠总图；
- `olmoearth_pastis_alignment_<patch>.png`：每个 patch 的四联图  
  **S2 RGB / SPOT 1 m / OlmoEarth PCA / S2 + 地块标签**；
- `olmoearth_alignment_batch.csv`：patch、tile、fold、日期、valid、PC1-NDVI 相关、最佳平移。

## 结论与限制

- 23 个 patch 的视觉图显示地块边界、道路、田块纹理基本对得上；
- 定量上没有系统性大偏移，最大 2 像元；
- `(0,+1)` 的小偏移需要在大规模提取前复核一次像素中心约定（OlmoEarth Zarr 的 10 m 像元中心 vs SPOT `5+10k`），但当前证据不足以支持做全局平移修正；
- patch 10000 缺失是网络/分块问题，正式全量提取前需补上并复测；
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
