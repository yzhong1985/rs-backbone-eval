# Colab: LIM T11 三协议 × 三 seed（A100 40GB cell）

目的：正式出 `lim_s2_single` / `lim_s2_median` / `lim_s2_20bin`（LTAE2d）各
三 seed，全部落在同一 GPU class（A100 40GB），并顺带清掉 19437cf 之前的
旧代码 CPU run（按 tags 标 superseded，见 §14 归账）。

上传包：`~/rs-eval/drive-t11-pack-v6`（`pack_t11_drive.py` 产出）→ 放到
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
`IGNF/PASTIS-HD` 快照直下到 Drive（作下载缓存 + 校验），校验通过后拷到
会话本地盘 `/content` 供训练读取（Drive FUSE 逐 epoch 读 45 GB 太慢，
review round 4）。

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
- **归账流程**：run 直接写在 Drive 的 `rs-t11-work/runs/`，完成后拷到
  `rs-t11-work/collected/<run_id>`；本地侧入账本（git_commit 与 bundle
  HEAD 一致，`uv sync --frozen` + 干净树断言保证不带 `-dirty`）；
- **数据落位**：`dispatch_lib.unpack_payload()` 解压到数据根（zip 内已带
  `derived/...` 前缀，解完校验目录结构，布局不符即报错——round 5 曾出现
  `derived/derived/...`）；
- **训练数据本地化**：HF 下载 → Drive 缓存 + sha 校验 → 拷到 `/content`
  本地盘重新校验，训练只读本地；仓库与 venv 也在 `/content`（Drive FUSE
  对符号链接/大包安装不可靠，round 5）；
- **失败 run 策略**：cell 中断后重跑会先读 `run.json`——`completed` 只补收，
  未完成则 **停下要求人工处理**（新 `-vN` 或 quarantine），绝不静默跳过
  （review round 4 #2）；
- **GPU 硬性要求**：dispatch 环境带 `RS_EVAL_REQUIRE_DEVICE=cuda`，worker
  端 cuda 不可用即报错，不会悄悄用 CPU 跑 A100 一格（review round 4 #1）。

## 回传后核对清单（reviewer 1）

收完 9 个 `-a100` run 后，本地：

```bash
uv run python scripts/audit_runs.py /geodata/rs-eval/runs .
```

逐项核对：`git_commit` == bundle HEAD；`tags.host` 均为
`nvidia-a100-sxm4-40gb`；`tags.test_samples_evaluated` == 496；
确认后才把旧 CPU 版 single/median run 标 `superseded`。

## 责任边界（reviewer 1 #2 的裁定）

- 本地 `T11BB` lane = **CPU cell**（lane 内 `CUDA_VISIBLE_DEVICES=""`）：
  只用于快速/诊断性复跑，不属于正式 A100 路径；
- 正式 BB 三协议 = A100 单 cell（本 notebook）。