# Colab: LIM T11 三协议 × 三 seed（A100 40GB cell）

目的：正式出 `lim_s2_single` / `lim_s2_median` / `lim_s2_20bin`（LTAE2d）各
三 seed，全部落在同一 GPU class（A100 40GB），并顺带清掉 19437cf 之前的
旧代码 CPU run（按 tags 标 superseded，见 §14 归账）。

上传包：`~/rs-eval/drive-t11-pack-v4`（`pack_t11_drive.py` 产出）→ 放到
Drive `MyDrive/rs-t11-pack/`。**上传前请核对 commit**：包内 git bundle 的
pin commit 必须晚于 `3e98ec0`（encoder.to(device) 修复）。

文件清单：

- `lim/mit_huge_600.pth`（6.2 GB，LIM 权重）
- `manifest/cf092238…jsonl`（合同 manifest）
- `repo/rs-backbone-eval.bundle`（整仓库 bundle）
- `sidecars.zip` / `annotations.zip`（按 manifest 相对路径解包）
- `s2_sha256.json`（**全量 2,468 文件** hash；notebook 对 HF 下载
  逐文件验生前拒绝 dispatch）
- `PROVENANCE.json`

DATA_S2（~45 GB）**不上传**：notebook 在 Colab 内从 HuggingFace
`IGNF/PASTIS-HD` 快照直下到 Drive。

## 使用方式（Colab）

1. Runtime → A100 40GB（或 A100 40GB 换档）；
2. 挂载 Drive 后，逐 cell 从上到下执行本仓库的
   `colab/t11_bb_a100.py`（文件按 `# %%` 切 cell，可用
   [colab-code-from-file] 之类方式粘贴）；
3. notebook 会：验 sha（包完整性 48 项）→ clone bundle → sync uv 环境 →
   HF 直下 DATA_S2 → 逐文件 sha 校验 → `rs-eval dispatch` 三协议三 seed
   （run id 统一 `-a100` 后缀）→ run 目录回写 Drive。

## 审查点（供 reviewer 快速验证）

- **数据可信度**：`_verify_all()` 在任何 dispatch 之前执行，遇到空
  sha 清单/utf8 失配/缺文件即 `raise SystemExit`（不会再出现"空清单
  自动通过"，对应本轮 reviewer1 的建议）；
- **设备一致**：dispatch 前断言 `torch.cuda.get_device_name(0)` 包含
  `a100-40gb`，run 内 tags 由 worker 运行时写入（`host` 一致性由
  ledger 审数据时按 cell 惯例复核）；
- **归账流程**：每个 run 结束后把 `runs/<run_id>` 目录拷回 Drive；
  本地侧使用 `audit_runs.py` 常规路径入账本（git_commit 与
  bundle HEAD 一致）。

## 责任边界（reviewer 1 #2 的裁定）

- 本地 `T11BB` lane = **CPU cell**（lane 内 `CUDA_VISIBLE_DEVICES=""`）：
  只用于快速/诊断性复跑，不属于正式 A100 路径；
- 正式 BB 三协议 = A100 单 cell（本 notebook）。