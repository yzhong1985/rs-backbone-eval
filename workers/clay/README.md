# Clay worker

已实现入口：T1/T1S/T2/T3/T4/T5/T8/T9/T11（S2/Landsat/光学/SAR 与预计算 embedding）
以及 T11D（PASTIS-HD RGB+NIR 四波段学生，教师蒸馏轨）。

必须保存传感器元数据哈希，尤其是 band order、波长、均值、标准差、GSD和数值缩放。

## T11D：RGB+NIR 四波段学生（教师蒸馏）

`src/clay_worker/rgbn.py`，由 `runner.py` 按 `task_id == "T11D"` 分派。

- 输入：PASTIS-HD 原始 S2 栈的 B02/B03/B04/B08（第 0/1/2/6 列），raw DN，
  Clay `sentinel-2-l2a` 四个对应波段的均值/标准差；只取 2019 年获取日期，
  按日期均匀选至多 20 期（`select_2019_frames`，与 manifest builder 共用）。
- 主干：Clay v1.5 `SegmentEncoder`，**冻结**，stride 8 -> 16×16 token（名义 80 m），
  1024 维；四波段由 `DynamicEmbedding` 的波长条件化权重支持，不改结构。
- 蒸馏（`clay_rgbn_distill_{a64,o64,o128}`）：时间均值 F -> E(1024→256→64/128)
  -> 与教师（AEF/OlmoEarth 2019，按 8×8 有效像元均值聚合）算 `mean(1-cos)`。
  教师只在训练期读取，推理路径不加载 AEF/OlmoEarth。
- 探针（`clay_rgbn_probe_*`）：冻结 F 或 F+E，训练 1×1 MLP 头；token logits
  双线性上采样到 128×128，对 10 m 标签算 CE/mIoU。80 m 不是 10 m 观测，
  报告必须写明该口径。
- 特征缓存：`RS_EVAL_RGBN_CACHE`（默认 `RS_EVAL_CACHE_ROOT/pastis-rgbn`），
  键为 checkpoint/manifest/波段/帧规则/网格/预处理/代码版本的指纹；原子写入，
  指纹或 `.npy` 头不符即重算。缓存 2433 景约 24 GB。
- 蒸馏投影发布到 `<cache>/projector/<target>/seed<seed>/<fingerprint>.pt`；
  探针按同一键加载并记录来源 run id。
