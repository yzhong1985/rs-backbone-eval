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
