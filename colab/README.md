# Colab: LIM T11 协议 × 三 seed（A100 40GB cell，路线 C）

目的：正式出 `lim_s2_single` / `lim_s2_20bin`（LTAE2d）各三 seed，全部落在
同一 GPU class（A100 40GB），并按 tags 把旧 CPU run 标 `superseded`
（见 `docs/reviews/审查响应_2026-11-04_第7轮.md`）。

> median 变体已停止新增（2026-11-04 决策：逐像素中值合成的是主干从未见过的
> 伪场景）；仓库里已有的 median 结果保留为历史观测，notebook 不再派发。

上传包：`~/rs-eval/drive-t11-pack-c-r8`（`pack_t11_drive.py --route-c` 产出）
→ 放到 Drive `MyDrive/rs-t11-pack/`。包内 git bundle 的 pin commit 必须与
`PROVENANCE.json.repo_commit` 一致，notebook 会断言。

文件清单（路线 C 只上传小包，数据在 Colab 内下载一次并复用）：

- `lim/mit_huge_600.pth`（6.2 GB，**用户单独上传**；包里不拷文件，但
  `PROVENANCE.checkpoint.sha256` 记录了它的 hash，notebook 派发前校验）
- `manifest/cf092238…jsonl`（合同 manifest）
- `repo/rs-backbone-eval.bundle`（整仓库 bundle）
- `sidecars.zip`（只含 `dates_s2` 的 json，~3 MB）
- `s2_sha256.json` / `annotations_sha256.json`（**全量** 2,468 / 1,455 文件）
- `PROVENANCE.json`

DATA_S2（~45 GB）与 ANNOTATIONS 不上传：notebook 从 `HF_ENDPOINT`
（默认 `https://hf-mirror.com`）快照直下到 Drive 作下载缓存，逐文件 sha
校验后拷到会话本地盘 `/content` 供训练读取。实际使用的 endpoint 写入
Drive 根目录 `hf_endpoint.json`，并通过 `RS_EVAL_HF_ENDPOINT` 记进每个
run 的 `tags.hf_endpoint`。

## 使用方式（Colab）

1. Runtime → A100 40GB；
2. 挂载 Drive 后，从上到下执行 `colab/t11_bb_a100.py`（按 `# %%` 切 cell）；
3. notebook 会：解包 → 验包内 sha → clone bundle → sync uv 环境 →
   `needs_download()` 判断 S2/ANNOTATIONS 是否齐全（缺任一即下载）→
   逐文件 sha 校验并写 marker → checkpoint sha 断言 → dispatch
   `lim_s2_{single,20bin}` × 3 seed → 回传前核对 run 的
   `model_checkpoint_sha256`。

## 审查点（供 reviewer 快速验证）

- **数据可信度**：所有 sha 校验在任何 dispatch 之前执行，空清单/缺文件/
  hash 不符即 `raise SystemExit`；Drive 侧的 marker 存的是清单文件本身的
  sha，换清单即失效。
- **下载判断**：`needs_download()` 同时数 DATA_S2 和 ANNOTATIONS，避免
  “S2 已缓存、ANNOTATIONS 从未下载”时跳过下载（round 7）。
- **checkpoint 可信度**：包里记录 6.2 GB 权重的 sha256（不拷文件），
  notebook 校验文件、回传时再校验 `run.json.model_checkpoint_sha256`；
  上传错版本会直接拒绝，不会污染正式 cell（round 7）。
- **设备一致**：dispatch 前断言 `torch.cuda.get_device_name(0)` 含
  `a100-40gb`；`RS_EVAL_REQUIRE_DEVICE=cuda` 让 worker 无法静默退回 CPU。
- **归账流程**：run 写在 Drive 的 `rs-t11-work/runs/`，完成后拷到
  `rs-t11-work/collected/<run_id>`；本地入账本前核对 `git_commit` ==
  bundle HEAD（干净树 + `uv sync --frozen`，不带 `-dirty`）。
- **数据落位**：`dispatch_lib.unpack_payload()` 解压到数据根（zip 内已带
  `derived/...` 前缀，布局不符即报错——round 5 曾出现 `derived/derived/...`）。
- **训练数据本地化**：HF 下载 → Drive 缓存 + sha 校验 → 拷到 `/content`
  重新校验；仓库与 venv 也在 `/content`（Drive FUSE 对符号链接/大包不可靠）。
- **失败 run 策略**：cell 中断后重跑先读 `run.json`——`completed` 只补收，
  未完成则停下要求人工处理（新 `-vN` 或 quarantine），绝不静默跳过。
- **时间位置编码**：20-bin 用距 2018-09-01 的天数（不是 day-of-year），
  与 AnySat 的 PASTIS loader / U-TAE 口径一致；run tags 记录
  `temporal_positions` 与 `ltae_T`（round 7）。

## 回传后核对清单

收完 6 个 `-a100` run 后，本地：

```bash
uv run python scripts/audit_runs.py /geodata/rs-eval/runs .
```

逐项核对：`git_commit` == bundle HEAD；`model_checkpoint_sha256` ==
`PROVENANCE.checkpoint.sha256`；`tags.host` 均为
`nvidia-a100-sxm4-40gb`；`tags.test_samples_evaluated` == 496；
`tags.validation_samples_evaluated` == 482；
确认后才把旧 CPU 版 single run 标 `superseded`。

**第一次跑时建议先只放行第一个 run**（single-17）：跑完核对
`tags.device=cuda`、`tags.host` 是 A100、`test_samples_evaluated=496`、
`validation_samples_evaluated=482` 四项都对，再让后面 5 个继续。

## 责任边界

- 本地 `T11BB` lane = **CPU cell**（lane 内 `CUDA_VISIBLE_DEVICES=""`）：
  只用于快速/诊断性复跑，不属于正式 A100 路径；
- 正式 BB 轨（LIM / AnySat / Clay / OlmoEarth）= A100 单 cell（本 notebook
  目前只派发 LIM 两协议；其余主干沿用同一数据与 manifest）。
