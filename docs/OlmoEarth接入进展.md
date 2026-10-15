# OlmoEarth v1.2 Tiny 接入进展

日期：2026-10-15。状态：接入进行中，**尚未产生任何成绩**。关联：[外部模型可比性评估](./外部模型可比性评估.md)、[模型扩展与低成本评测计划](./模型扩展与低成本评测计划.md)。

## 1. 已完成

| 项 | 位置 | 说明 |
|---|---|---|
| 模型注册 | `configs/models/olmoearth.yaml` | `enabled: false`，待 worker 落地后启用 |
| 下载条目 | `configs/downloads/catalog.yaml` | `allenai/OlmoEarth-v1_2-Tiny`，revision `12a9fdbf`，`weights.pth` sha256 `835c0b21…7697d` |
| worker 环境 | `workers/olmoearth/` | `olmoearth_pretrain==0.1.2` + torch 2.9.1，已 `uv sync` 成功 |
| 接口探测 | 本文件 §2 | 已确认模型类、参数量口径、输入契约 |

## 2. 接口探测结论（在 `workers/olmoearth/.venv` 实测）

- `load_model_from_id(ModelID.OLMOEARTH_V1_2_TINY, load_weights=False)` 返回 `LatentMIM`：
  - `encoder` **12.465M**、`decoder`（Predictor）1.857M、`target_encoder` **12.465M**（训练期自蒸馏副本）。
  - 合计 **26.79M**；**冻结探针只需 `encoder`**，报告容量时必须说明，不能把 26.79M 当"同量级"。
- 编码器前向：`encoder.forward(x: MaskedOlmoEarthSample, patch_size: int, input_res: int = BASE_GSD, ...) -> dict`。
- 输入样本 `OlmoEarthSample` 的关键字段（注意布局与其它 worker 不同）：
  - `sentinel2_l2a`：**`[B, H, W, T, C]`**（H/W 在 T 之前、通道在最后），而本项目数据是 `[T, C, H, W]`，需要转置。
  - `timestamps`：`[B, T, 3]`，三元组为 `[day, month, year]`。
  - `latlon`：`[B, 2]`（可选）。
  - 其它模态：`sentinel1`、`landsat`、`worldcover` 等，按需提供。
- 模态与波段定义在 `olmoearth_pretrain.data.constants.Modality`（`SENTINEL2`、`SENTINEL2_L2A`、`SENTINEL1`、`LANDSAT`…）。
- **波段不匹配（关键）**：`SENTINEL2_L2A` 需要 **12 个波段**、`SENTINEL2` 13 个、`SENTINEL1` 2 个（vv/vh）；编码器对输入通道做**固定位置的 `index_select`**。本项目 T1 是 **6 波段 S2**，直接传入会在 `flexi_vit.py` 的 `apply_embedding_to_modality` 抛 `IndexError`（见 `scripts/probe_olmoearth.py`）。
  - S2 L2A 的 12 波段按 bandset 分组：`[B02,B03,B04,B08]`、`[B05,B06,B07,B8A,B11,B12]`、`[B01,B09]`。
  - **项目波段对应关系**：`BLUE→B02`、`GREEN→B03`、`RED→B04`、**`NIR_NARROW→B8A`**（865nm 窄波，**不是 B08**）、`SWIR_1→B11`、`SWIR_2→B12`。
  - **没有任何 OlmoEarth bandset 被完整覆盖**：set 0 缺 B08、set 1 缺 B05/B06/B07、set 2 全缺。`bandset_coverage = (False, False, False)`。
  - **掩码按 bandset 而非按波段**：部分存在的 bandset 只能**带填充值送入**（`bandset_usable = (True, True, False)`，set 2 才标 MISSING）；若按"不完整就屏蔽"会把 SWIR 整组丢掉。
  - TerraMind 有 `bands={...}` 子集 API，OlmoEarth 没有等价入口；适配后该实验应标 `frozen_adapter`，不与纯冻结同表。
- **输出是密集 token，不是池化向量（更正）**：`encoder(...)` 返回 `TokensAndMasks`，其 `sentinel2_l2a` 形状为 **`(B, P_H, P_W, T, BS, D)`**。v1/v2 说"默认输出是池化向量、需要 token_exit_cfg"是**错误**——那是探测脚本用 `getattr(t, "shape")` 打印 NamedTuple 导致的假象。实测输入 64×64、`patch_size=4` 时：
  - `sentinel2_l2a: (1, 16, 16, 1, 1, 192)`、`sentinel2_l2a_mask: (1, 16, 16, 1, 1)`
  - `project_aggregated: (1, 192)`（另一供对比的池化输出）
  - 供分割头使用时 reshape 成 `(B, 192, P_H, P_W)` 即可接 `LiteSpatialHead`。

## 3. 待办

- [ ] **下载权重**：`hf_snapshot` 走项目代理（`RS_EVAL_PROXY`）或 `HF_ENDPOINT=https://hf-mirror.com`；下载后按 `weights.pth` 的 sha256 校验。
- [ ] **worker runner**：`workers/olmoearth/src/olmoearth_worker/runner.py`（probe + train 分发），复用 `RunWriter` 与既有契约。
- [ ] **probe**：构建 encoder（`load_weights` 按环境）、构造最小 `OlmoEarthSample`、跑一次前向、记录输出形状与显存；确认 token 网格能否 reshape 成 `[B, D, h, w]` 供分割头使用。
- [ ] **T1 冻结**（S2 六波段单期）：把项目 6 波段映射到 `sentinel2_l2a` 的子集，逐样本构造 `[B,H,W,T,C]` 与 `timestamps`。
- [ ] **T5 冻结**（HLS 三期六波段）：明确是走原生时序编码器读出，还是逐日期编码＋共享头；与 Clay/Prithvi 的"逐日期＋头"口径不同，**必须单列**。
- [ ] **短跑**：记录成本、显存、吞吐、缓存体积（stage A），再决定是否进入正式种子。
- [ ] **启用注册**：短跑通过后把 `configs/models/olmoearth.yaml` 的 `enabled` 改为 `true`。

## 4. 已知风险

- **许可**：权重与代码为 **OlmoEarth Artifact License**（含用途限制，非 Apache-2.0），公司使用前建议法务确认。
- **输入契约较重**：`[B,H,W,T,C]` + `[day,month,year]` 时间戳 + `MaskedOlmoEarthSample`，比 TerraMind/DOFA 的"张量＋波段"复杂，接入工作量高于另两家。
- **T5 读出不一致**：原生时序编码器与逐日期＋头不是同一融合口径，跨模型比较时必须分表。
- **HF 访问**：直连 `huggingface.co` 超时，需经项目代理或镜像。

## 5. 参考

- 仓库：<https://github.com/allenai/olmoearth_pretrain>
- 输入说明：<https://github.com/allenai/rslearn/blob/master/docs/foundation_models/OlmoEarth.md>
- 权重：<https://huggingface.co/allenai/OlmoEarth-v1_2-Tiny>
