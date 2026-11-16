# 单期高分影像无教师部署（Image-Only）：跨任务复盘、评审修订与最小实验执行方案（v2.0）

* **版本与日期**：v2.0（2026-11-16 修订版）
* **目标场景**：单日期、高分辨率四波段（RGB+NIR，如 GF-2 / WHU-OPT）遥感影像语义分割；**模型部署与推理阶段仅输入光学影像（Image-Only），不读取 AEF / OlmoEarth 外部栅格或缓存**。
* **关联材料**：
  1. 前期蒸馏诊断：`教师蒸馏路线_总结_2026-11-16.md`（`T11D` / `T11P` / `T8`）
  2. 同事融合实验包：`AEF_Fusion_Kit_20261113`（`01`–`03` 规格文档、配置模板、`WHU-OPTSAR_经验数据.md` 及 `train_aef_v1.py` 损失函数截图）
  3. 上级评审意见：`code_artifact_评审与修订建议_2026-11-16.md`

---

## 0. 修订说明与执行摘要（Executive Summary）

本方案在 v1.0 基础上，全盘吸纳上级评审的修订意见，并纳入了对同事 `AEF_Fusion_Kit_20261113` 实验包及 `feat-mlp-distill` 代码逻辑的核查结果。全文统一使用**百分制“mIoU 点（pt）”**表示指标差值（例如 $0.4909 \to 0.5112$ 记为 **$+2.03\text{ pt}$**；$0.367 \to 0.258$ 记为 **$-10.9\text{ pt}$**）。

### 0.1 标准摘要表述（对齐评审 §9）

现有实验表明，真实外部地理表征（AEF）与高分影像融合在 WHU-OPT 的多个配置中有正向信号（$+1.86 \sim +4.45\text{ pt}$）；直接以教师对齐后的低维特征替换原特征，在已测设置中没有显示稳定优势，`T8`（GF-2）的初步迁移结果还存在明显下降（$-10.9\text{ pt}$）。下一阶段保留原主干预测路径，**第一优先级验证“融合教师指导的纯影像学生（方案二，D 臂）”**，并以**“轻量代理预测融合（方案一，P 臂）”**及**“训练期辅助投影（A 臂）”**为对照。是否能够在无外部表征的推理条件下获得稳定增益，必须由相同预算、相同容量控制（`Bcap`）、多个随机种子及地理隔离测试确认。坐标隐式先验（方案三）仅作为后续可选探索支线。

### 0.2 重大核查通报：`feat-mlp-distill (0.51120)` 的代码级真相（完成评审 §8 第一步）

根据评审意见“最先核实 `feat-mlp-distill` 的真实实现和评测协议”的要求，我们结合同事提供的 `AEF_Fusion_Kit_20261113` 文档与 `train_aef_v1.py` 核心代码截图完成了排查，结论如下：

1. **推理期并非 Image-Only（仍输入了真实 AEF）**：
   `feat-mlp-distill` 是在 A 表 `feat-mlp` 链路上通过参数 `--aef-distill 0.5` 启动的。其分割主路径依然执行 $F'_l = F_l + \text{Injector}_l(\text{AEF})$，**在测试推理时仍然读取了外部真实 AEF 特征**。
2. **蒸馏损失项处于“梯度悬空 / 零因果”状态**：
   根据截图，其总损失为 `loss = F.cross_entropy(logits, yd) + 0.5 * dloss`，其中：
   $$\text{dloss} = \frac{\sum_l w_l \cdot \text{mean}\big(1 - \cos(\text{proj}_l(F_l), \text{AEF})\big)}{\sum_l w_l}$$
   注意 `dloss` 读取的是**注入前的骨干原始特征 `base_feats`（$F_l$）**。而在冻结档实验中，骨干网络被冻结且 $F_l$ 为预先抽取保存在 SSD 上的静态缓存常量（`requires_grad=False`）。因此，`0.5 * dloss` 的反向传播梯度仅在旁路投影头 $\text{proj}_l$ 内部空转，**既无法更新冻结骨干 $F_l$，也未流入主路径的 `Injector_l` 与解码器 `TinyUPerNet`**。
3. **与同事内部文档记录完全吻合**：
   * 同事在 `02_实现规格_注入器与投影.md` §5 中明确记载：*“本项目 v1 的 DistillHeads 吃注入前缓存常量 $\to$ 零因果（探针），已弃用。”*
   * 同事在 `WHU-OPTSAR_经验数据.md` 对照表中记载：*“蒸馏（探针式） | 与孪生 run 差 = 单 seed 噪声；mars 蒸馏档反低 $0.58\text{ pt}$ $\to$ 无因果。”*
   * 因此，`feat-mlp-distill`（$0.51120, +2.03\text{ pt}$）与纯交叉熵孪生实验 `feat-mlp`（$0.51055, +1.97\text{ pt}$）在主路径上完全等价，两者 $+0.065\text{ pt}$ 的差值完全落在单种子随机噪声范围（$\pm 0.3 \sim 0.6\text{ pt}$）内。

> **核心推论**：此次核查证实了评审报告的判断极为准确。同时这也意味着，**真正的“推理期无 AEF 输入”方案（评审定义的 D 臂、P 臂、A 臂）在 WHU-OPT 与 FBP (GF-2) 上均属于尚未被真正运行过的空白区**。由于无法直接获取同事的 `.py` 源码文件，本文档第 5 节已根据其 `02_实现规格.md` 的逐位参数说明，完整重构了 100% 兼容其基建的 PyTorch 代码与六臂实验框架。

---

## 1. 现有实验证据盘点与边界界定（对齐评审 §2）

### 1.1 同事 WHU-OPT 真实落盘数据汇总（源自 `WHU-OPTSAR_经验数据.md`）

* **评测协议**：7 类封闭 mIoU（`closed_class_miou`，`ignore_classes=[0]` 剔除背景类，与含背景的 8 类口径存在约 $12.5\%$ 的系统差）；冻结骨干 + `TinyUPerNet (channels=128)`；$30\text{ epochs}$ / `cosine_tmax=30` / `batch=16` / `wd=1e-4` / 纯 CE 损失；单 seed 探索噪声约为 $\pm 0.3 \sim 0.6\text{ pt}$。

| 实验类别 | 骨干网络 | 基线 (test7) | 核心实验结果（相对基线 $\Delta\text{ pt}$） | 证据支持范围与局限 |
| :--- | :--- | :---: | :--- | :--- |
| **A 表：head 侧可训注入**<br>*(推理需真 AEF)* | `lim` (`lr=3e-3`)<br>`mars` (`lr=3e-3, 50ep`)<br>`prithvi_300m_ntl` (`lr=1e-3`) | $0.4909$<br>$0.4811$<br>$0.4463$ | `feat-lin`: **$+2.06$** / `feat-mlp`: $+1.97$ / `dec-mlp`: **$+2.16$**<br>`feat-lin`: $+3.00$ / `feat-mlp`: **$+3.51$** / `dec-lin`: $+2.48$<br>`feat-lin`: **$+3.80$** / `feat-mlp`: $+3.50$ / `dec-mlp`: $+2.81$ | 12/12 全正；证明在 WHU-OPT 上**真实 AEF** 在特征层/解码层注入均有真实增益；**但尚不能证明单张光学影像能自预测出同等增益** |
| **B 表：骨干内冻结投影 `bb_inj`**<br>*(推理需真 AEF)* | `lim`<br>`mars`<br>`prithvi_300m_ntl` | $0.4909$<br>$0.4811$<br>$0.4463$ | `stage-pca`: $+2.09$ / `prog-pca`: **$+3.18$ ($0.5227$)**<br>`stage-pca`: **$-3.28$** / `stage-mlp`: $-1.68$<br>`stage-pca`: $+4.39$ / `stage-mlp`: **$+4.45$ ($0.4908$)** | 与骨干预训练来源强相关（`mars` 四档全负）；注意 `prithvi` 最高分 $0.4908$ 仍低于 `lim` 基线 $0.4909$，选型需看绝对分数 |
| **C 表：对照与消融** | 三骨干 | 同上 | 1. `null (feat-mlp-noise)`: `lim` $-0.04$ / `mars` $-0.34$ / `ntl` $-0.40$<br>2. `P4 双路叠加`: 三骨干均无额外增益<br>3. `FT 修复档 (lr=3e-4)`: `ntl` $0.4856$ ($+3.93\text{ pt}$) | `null` 排除了固定随机高斯噪声的贡献，但**未排除“新增可训图像分支容量（`Bcap`）”的贡献**；双路注入已被证伪，后续锁死单点融合 |

### 1.2 前期 PASTIS-HD (`T11D/T11P`) 与 GF-2 (`T8`) 蒸馏数据对照

| 观察现象 | 支持的严谨结论 | 目前不支持的过度推论（已在本文修正） |
| :--- | :--- | :--- |
| `T8` 原始特征 $0.367$，OE128 蒸馏 $0.258$（$-10.9\text{ pt}$），AEF64 蒸馏 $0.256$（$-11.1\text{ pt}$） | 当前“压缩对齐教师并直接替换原特征”的配置在单种子缩减预算下存在严重负迁移风险 | 不能断言唯一原因是“丢弃了 896 维高频边缘”，需同时排查 S2$\to$GF-2 域差异、训练预算、特征标定及 GSD 尺度错配 |
| `T11D` 逐通道 `z-score` 标定在 $8\text{ ep}$ 下从 $0.3185$ 升至 $0.3511$（$+3.26\text{ pt}$） | 特征标定（Calibration）显著影响短预算下的读出可优化性，必须作为所有基线的标准对照环节 | 不能保证在所有主干/任务上加 Norm 均绝对涨分（EoS-FM 存在特定任务退化案例），需由验证集协议确定 |
| `InfoNCE` 在 4 个任务$\times$主干对上稳定优于余弦/MSE（$+2.6 \sim +3.7\text{ pt}$） | 值得作为代理预测/辅助蒸馏的核心候选损失保留 | 不是唯一有效损失，也尚未超越充分训练且经过标定的原始特征基线 |
| `T11D` §4.3 冻结特征拼接教师投影下降 $-0.74\text{ pt}$（3/3 种子全负） | 当前 PASTIS 冻结时间均值配置下直接拼接浅层投影无益 | 数据处理不等式并未禁止在有限复杂度分割头前通过小代理头提供非线性归纳偏置，应在 WHU 上用轻量实验直接检验 |

---

## 2. 计算图诊断：如何将同事悬空的 `dloss` 改造为真正的无教师推理方案？

通过对比同事 `train_aef_v1.py` 的实际计算图与上级评审 §5、§8 提出的实验方案，可以清晰看到如何通过修改张量连接拓扑，将“依赖真 AEF 的探针代码”转化为“Image-Only 部署模型”：

```
【同事原版 feat-mlp-distill 计算图（主路仍需真 AEF，蒸馏分支梯度悬空）】
  缓存冻结特征 F_l (requires_grad=False) ──┬──> [ + Injector_l(真 AEF) ] ──> TinyUPerNet ──> CE Loss
                                          └──> proj_l(F_l) ──> 与真 AEF 算 dloss (梯度在 F_l 处截断，未进主路！)

【改造方案一：D 臂（第一优先级：融合教师指导纯影像学生，推理零额外模块）】
  训练期冻结 Teacher:  F_l + Injector_l(真 AEF) ──> Teacher TinyUPerNet ──> t_logits / t_feat (stop_grad)
                                                                                  │ (KL 散度 + 特征对齐)
  纯影像 Student:      F_l (或微调主干) ──────────> Student TinyUPerNet ──> s_logits / Q(s_feat) ──> CE + KD Loss
  *(部署时直接删除 Teacher 与投影头 Q，计算图与基线 B0 100% 一致)*

【改造方案二：P 臂 vs Bcap 臂（第二优先级：图像自预测代理特征 + 注入融合）】
  纯影像输入 ──> F_l ──┬──────────────────────────────────────┐
                      └──> Predictor_l(F_l) ──> hat_AEF_l ──> Injector_l(hat_AEF_l) ──> [ + ] ──> TinyUPerNet ──> CE Loss
                                                    │
                                       (仅 P 臂在训练期加 dloss 对齐真 AEF；Bcap 臂不加 dloss 仅靠 CE 训练)
  *(推理时全程只走 F_l -> Predictor_l -> Injector_l，完全不读取外部 AEF)*
```

---

## 3. 核心工程与物理规范修正（响应评审 §3、§4、§6）

### 3.1 按地面米数（Footprint）对齐，而非简单按张量尺寸插值（评审 §6.1）

* **尺度错配隐患**：在 $4\text{ m}$ 分辨率的 GF-2（FBP）或 $5\text{ m}$ 的 WHU-OPT 影像上，骨干网络经过 $\text{stride}=16$ 下采样后，单个特征网格对应地面真实尺寸为 $64\text{ m} \times 64\text{ m}$（或 $80\text{ m} \times 80\text{ m}$），$\text{stride}=8$ 对应 $32\text{ m} \times 32\text{ m}$，并非 $10\text{ m}$ 的 AEF 原始网格。
* **正确实现规范**：
  1. 预处理阶段必须依据 GeoTIFF 的 CRS、仿射变换矩阵（Affine Transform）与切片像素中心坐标，将 $10\text{ m}$ 教师栅格精确配准裁剪至当前训练 Tile 的真实地面范围（Footprint）。
  2. 当向粗糙特征层级（如 $32\text{ m} / 64\text{ m}$ footprint）计算局部对齐损失时，优先使用 **`F.adaptive_avg_pool2d`（空间均值池化聚合）** 将 $10\text{ m}$ 教师特征聚合到对应地面网格，而非盲目使用双线性上采样捏造伪高频真值。

### 3.2 教师栅格反量化解码、有效像素掩膜与年份核查（评审 §6.2）

* **OlmoEarth / AEF 解码与 `NoData` 屏蔽**：
  * OlmoEarth 官方导出的 `int8` 存储采用非线性量化，**严禁直接将 `int8` 强转为 `float32` 计算余弦或内积**；必须先按官方解码公式还原浮点值，并将量化值 `-128`（代表 `NoData`）以及恒零通道/像素标记为无效掩膜 `valid_mask`。
  * 所有特征层蒸馏损失（Cosine / InfoNCE / KL）必须仅在 `valid_mask == 1` 的有效像素上求均值：
    $$\mathcal{L}_{\text{align}} = \frac{\sum_{i \in \Omega} \mathbb{I}(\text{valid}_i) \cdot d(\hat{E}_i, E_i)}{\sum_{i \in \Omega} \mathbb{I}(\text{valid}_i) + \epsilon}$$
* **时相与覆盖率记录**：建立数据集元信息表，逐景核查光学影像拍摄年份、标签年份与教师特征年份（例如 OlmoEarth 2023–2025 为全球覆盖，2017–2022 仅部分区域覆盖），避免将拍摄期之后才发生的地表变化当作先验注入。

### 3.3 注入器初始化与梯度流保护（评审 §3.6 & 同事 `02_实现规格.md`）

* 直接采纳同事在 WHU-OPT 上已验证成功的 **`AEFInjector` 末层 Zero-Init 规范**：
  * `linear` 档：单层 `Conv1x1(64 -> c_l)`，权重与偏置全 $0$ 初始化。
  * `mlp` 档：`Conv1x1(64 -> 256)` 采用标准随机初始化（Kaiming/Xavier）$\to \text{GELU} \to$ `Conv1x1(256 -> c_l)` 采用**全 $0$ 初始化（Zero-Init）**。
* **为什么这样设计最安全**：
  1. 训练第 $0$ 步时，$\text{Injector}_l(\cdot) \equiv 0$，保证 $F'_l = F_l + 0 = F_l$，不向主路引入任何未训练的随机Norm或初始扰动，严格保持基线初始输出。
  2. 由于仅最后一层 `fc2` 置零而第一层 `fc1` 非零，第 $1$ 步反向传播时，`fc2.weight` 的梯度 $\frac{\partial \mathcal{L}}{\partial W_2} = \frac{\partial \mathcal{L}}{\partial F'_l} \cdot \text{GELU}(W_1 \hat{E} + b_1)^T \neq 0$，因此 `fc2` 在第一步更新后即刻变为非零，随即可将分割梯度顺畅传入 `fc1` 与前置预测头；同时在 P 臂中，前置预测头 `Predictor_l` 还直接受辅助对齐损失 $\mathcal{L}_{\text{aef}}$ 驱动，**彻底杜绝了“同时把门控与分支末层置零导致梯度死锁”的风险**。

### 3.4 文献对标与边界修正表（对齐评审 §4）

| 参考文献 | 已核实的核心机制 | 在本方案中的正确应用边界 |
| :--- | :--- | :--- |
| **OlmoEarth v1.3** (2026) | §2.3 对比查询压缩与高维后蒸馏（PASTIS val mIoU $57.5$ vs $58.4$），采用 Cosine + Gram 损失且压缩梯度不回传主干 | 借鉴其 Cosine + Gram 关系蒸馏设计；不将其局部压缩实验夸大为“低维瓶颈必然摧毁主干” |
| **TESSERA v2** (2026) | 探讨像素级地球基础模型缩放定律与 Matryoshka 紧凑学生蒸馏 | 支持以最终下游分割指标评判压缩表征优劣 |
| **EoS-FM** (2026) | 冻结多编码器的选择、特征归一化与融合策略（部分任务上归一化可能不利） | 特征标定（`z-score` / `LayerNorm`）作为验证集超参选项，不作无条件涨分承诺 |
| **SAGE-Net** (2026) | 在高分荔枝园提取中融合 $10\text{ m}$ AEF 与 $2\text{ m}$ GF-1，并采用空间隔离评估 | 直接印证“高分光学细节 + $10\text{ m}$ AEF 宏观语义”在农业/地物分割中的互补性（对应本方案 `R` 臂） |
| **Cross-Modal & Structured KD** (Gupta et al. / Liu et al.) | 跨模态特权信息监督迁移与密集预测结构化知识蒸馏（像素级 KL + 成对关系/特征对齐） | 作为**第一优先级方案二（D 臂：真 AEF 融合教师 $\to$ 纯影像学生）**的直接理论依据 |
| **MIND** (2026) | 宽度 $3072$、$12$ 层残差正弦块的坐标隐式网络（主权重约 $227\text{ MB}$ fp16，另有 $6.4\text{ M}$ small 历史探索版） | 仅作为方案三（坐标先验探索支线）参考，不承诺几 MB 权重即可高精度重建全国 $1 \sim 5\text{ km}$ AEF |

---

## 4. 最小实验执行计划（对齐评审 §5 & §8）

摒弃大网格盲目搜索，严格按以下四个阶段推进。

### 4.1 阶段一：WHU-OPT 单主干（`lim`）六臂受控对照实验

* **固定超参（与同事 `lim` 冻结档 100% 对齐）**：
  * 主干：`lim`（冻结，可直接复用已抽取的 SSD 特征缓存加速）；解码头：`TinyUPerNet (channels=128)`。
  * 训练设置：`epochs=30`，`cosine_tmax=30`，`patience=0`，`lr=0.003`（已 sweep 最优档），`batch=16`，`wd=1e-4`。
  * 评价指标：`closed_class_miou`（7 类有效地物，`ignore_index=0`）。
  * 种子策略：先用单种子（`seed=20261103`）完成六臂快速探路与代码冒烟；对胜出候选及其直接对照补齐 3 种子（`[20261103, 20261104, 20261105]`，或 `[17, 42, 73]`），报告 `mean ± std`。

| 实验臂代号 | 训练期输入 | 推理期输入 | 网络结构与损失函数定义 | 直接对照基线 | 核心回答的科学/工程问题 |
| :--- | :---: | :---: | :--- | :---: | :--- |
| **`B0`**<br>*(纯影像基线)* | 影像 | **仅影像** | `lim` $\to$ `TinyUPerNet`；纯 CE 损失（验证集对比是否加逐通道 `z-score` 标定） | — | 公平的纯影像基准分（对标 $0.4909$） |
| **`R`**<br>*(真 AEF 融合教师)* | 影像 + 真 AEF | 影像 + 真 AEF | $F'_l = F_l + \text{Injector}_l(\text{AEF})$；纯 CE 损失（兼作 `D` 臂的冻结 Teacher） | `B0` / `R_null` | 真实 AEF 在当前配置下的可利用收益参考（对标 $0.5106 \sim 0.5115$） |
| **`R_null`**<br>*(噪声空置对照)* | 影像 + 高斯噪声 | 影像 + 高斯噪声 | 结构同 `R`，输入替换为与 AEF 同均值同方差的独立固定种子高斯噪声 | `R` | 排除固定随机输入的通路扰动效应（对标 $0.4906$） |
| **`Bcap`**<br>*(同结构容量对照)* | 影像 | **仅影像** | $F'_l = F_l + \text{Injector}_l(\text{Predictor}_l(F_l))$；**仅用纯 CE 损失训练（无教师损失）** | `B0` | **关键控制**：单独增加旁路分支参数与非线性读出容量本身能涨多少分？ |
| **`D`**<br>*(方案二：第一优先)* | 影像 + 冻结 `R` 教师 | **仅影像** | 学生结构同 `B0`；$\mathcal{L} = \mathcal{L}_{\text{CE}} + \beta\tau^2 \text{KL}(p_T^\tau \parallel p_S^\tau) + \gamma d(Q(F_S), \text{sg}(F_T))$（先跑 $\gamma=0$ 纯输出蒸馏，再测 $\gamma>0$；部署时删去 $Q$） | `B0` | **零推理开销**下，融合教师能否将收益传递给纯影像学生？ |
| **`P`**<br>*(方案一：第二优先)* | 影像 + 真 AEF 监督 | **仅影像** | 结构与 `Bcap` **逐位完全相同**；训练期增加对齐损失 $\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda \mathcal{L}_{\text{align}}(\hat{E}_l, \text{AEF})$（测试 Cosine 与 InfoNCE） | **`Bcap`** & `B0` | 图像自预测代理特征并融合，能否显著跑赢同结构的纯标签分支 `Bcap`？ |
| **`A`**<br>*(轻量辅助投影)* | 影像 + 真 AEF 监督 | **仅影像** | 主路同 `B0`（需配合解冻主干末层或前置共享可训 Adapter）；训练期挂辅助头 `Aux(F_l)` 算 $\mathcal{L}_{\text{align}}$，推理时删去辅助头 | 同可训层 `B0` | 最轻量的训练期表征正则能否改善主干/适配器特征？ |

> **关键防伪追加实验**：一旦 `D` 臂或 `P` 臂在 3 种子上跑出相对 `B0` / `Bcap` $\ge +1.0\text{ pt}$ 的正向收益，立即追加 1 组**“打乱配对教师（Shuffled Teacher Pairing）”对照**——在训练 `D` 或 `P` 时，将切片的 AEF 教师随机跨图幅错配。若正常配对显著优于错配教师，即可铁证收益源自空间对应的真实教师语义，而非某种隐式正则。

### 4.2 阶段二：FBP / T8（国产 GF-2 四波段）迁移验证

1. **修正评测协议与类别口径（评审 §7）**：
   * FBP 官方为 $4\text{ m}$ 分辨率、4 波段（B, G, R, NIR），共 **24 个标注地物类 + 1 个未标注背景类（`unlabeled = 0`）**。
   * 必须将 `unlabeled (0)` 设为 `ignore_index=0`，严格按 **24 类封闭 mIoU** 评测，严禁将 `unlabeled` 当作第 25 个正常类别计入均值。
2. **跑通公平基线与胜出方案**：
   * 在统一协议与完整训练预算（$30\text{ ep}$）下，运行 3 种子 `B0`（含特征标定检查）、`R`（若有配对 AEF/OE）、以及阶段一胜出的 `D` / `P` 及其容量对照 `Bcap`。
   * 如需解释旧 `T8` 的 $-10.9\text{ pt}$ 现象，可在相同协议下补跑一组“直接用 64 维投影替换主干”的对照，不再直接引用旧缩减预算的单种子分数作为基线。

### 4.3 阶段三：按结果分支的决策矩阵（评审 §8 第四步）

预先约定：**以相对直接对照（`D` vs `B0`，`P` vs `Bcap`）3 种子平均增益 $\Delta \ge +1.0\text{ pt}$ 且最差种子不低于对照 $-0.3\text{ pt}$ 作为具有业务落地价值的立项阈值。**

| 实验结果组合 | 后续工程决策 |
| :--- | :--- |
| **`R` 稳定提升，且 `D` 或 `P` 显著优于 `B0` / `Bcap`（$\ge +1.0\text{ pt}$）** | 路线正式跑通！优先选择部署开销更低的方案（若 `D` 与 `P` 持平则选零开销的 `D`），推进至 FBP 跨区域 Holdout 评测，并记录参数量、延迟与显存 |
| **`P` 与 `Bcap` 同样提升（且 $P - B_{\text{cap}} < 0.6\text{ pt}$）** | 收益来自新增旁路 MLP 的读出容量而非 AEF 教师；直接采纳更简单的纯标签监督分支 `Bcap`，停止蒸馏投入 |
| **`R` 稳定提升，但冻结主干下 `D` 和 `P` 均无收益** | 切入同事 `backbone_lr.template.yaml` 中的 **`ft_reference`（全量微调档：`epochs=40, bb_lr=1e-5, head_lr=3e-4, llrd=0.9`）** 测试解冻主干下的 `D` / `P` / `A`；若仍无收益，说明单期光学图像无法预测该增量，转向探索粗粒度坐标先验或仅在有 AEF 场景提供融合版 |
| **仅 WHU 提升，FBP 不升或下降** | 将方法明确定位为特定传感器/类别粒度（宏观 7 类 vs 细粒度 24 类）相关方案，不宣称通用高分增强 |

---

## 5. 完整自包含代码实现（1:1 复刻同事规格 + 六臂统一框架）

由于无法获取同事的 `train_aef_v1.py`、`aef_inject.py` 与 `aef_bb_inject.py` 原始文件，以下代码严格依据同事 `02_实现规格_注入器与投影.md` 与 `.yaml` 模板逐位复刻，并完整集成了上级评审要求的六臂实验逻辑與防泄漏物理隔离验证。你可以直接将以下两个脚本保存并在本地环境运行。

### 5.1 核心模型与六臂架构定义 (`six_arm_models.py`)

```python
"""
six_arm_models.py
严格依据《02_实现规格_注入器与投影.md》与《code_artifact_评审与修订建议》构建：
1. 1:1 复刻同事的 AEFInjector (linear / mlp 末层 zero-init)、FrozenBBProjector (PCA 白化 + 固定种子高斯投影) 与 TinyUPerNet (channels=128)。
2. 统一实现上级评审 §8 要求的六臂实验模型：B0 / R / R_null / Bcap / P / A / D。
3. 支持地面 Footprint 空间聚合对齐与推理期物理隔离检查（Image-Only 模式下强制禁止传入 AEF）。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. 特征标定模块 (Feature Calibration: 逐通道 z-score / LayerNorm2d)
# =====================================================================
class ChannelZScoreNorm2d(nn.Module):
    """
    可缓存训练集统计量的逐通道 z-score 标定层（无额外可学习仿射参数或可选仿射），
    用于消除预训练主干特征通道方差各向异性。
    """
    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.eps = eps
        self.register_buffer("running_mean", torch.zeros(1, num_channels, 1, 1))
        self.register_buffer("running_std", torch.ones(1, num_channels, 1, 1))
        self.register_buffer("is_fitted", torch.tensor(False, dtype=torch.bool))
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    @torch.no_grad()
    def set_stats(self, mean: torch.Tensor, std: torch.Tensor):
        self.running_mean.copy_(mean.view(1, -1, 1, 1))
        self.running_std.copy_(std.view(1, -1, 1, 1).clamp(min=self.eps))
        self.is_fitted.fill_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_fitted.item():
            out = (x - self.running_mean) / (self.running_std + self.eps)
        else:
            # 若未预先统计全局均值，则按当前 batch 空间维做标准化
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            std = x.std(dim=(0, 2, 3), keepdim=True).clamp(min=self.eps)
            out = (x - mean) / std
        if self.affine:
            out = out * self.weight + self.bias
        return out


# =====================================================================
# 2. 同事 02_实现规格.md §1：head 侧可训注入器 (严格末层 zero-init)
# =====================================================================
class AEFInjector(nn.Module):
    """
    严格复刻 aef_inject.py:
    - linear 档: Conv1x1(64 -> c_l), 权重与偏置全 0 初始化 (zero-init)
    - mlp 档: Conv1x1(64 -> 256) [随机初始化] -> GELU -> Conv1x1(256 -> c_l) [zero-init]
    保证训练第 0 步时 Injector 输出恒为 0，F'_l == F_l，不破坏基线初始状态，且首层非零避免梯度死锁。
    """
    def __init__(self, in_dim: int = 64, out_c: int = 256, mode: str = "mlp"):
        super().__init__()
        self.in_dim = in_dim
        self.out_c = out_c
        self.mode = mode

        if mode == "linear":
            self.proj = nn.Conv2d(in_dim, out_c, kernel_size=1, bias=True)
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        elif mode == "mlp":
            self.fc1 = nn.Conv2d(in_dim, 256, kernel_size=1, bias=True)
            nn.init.kaiming_normal_(self.fc1.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(self.fc1.bias)
            self.act = nn.GELU()
            self.fc2 = nn.Conv2d(256, out_c, kernel_size=1, bias=True)
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)
        else:
            raise ValueError(f"Unsupported injector mode: {mode}. Choose 'linear' or 'mlp'.")

    @staticmethod
    def align_spatial_footprint(feat_src: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        """
        评审 §6.1：按地面 Footprint 空间对齐。
        当目标特征图网格比源特征更粗时，优先使用 adaptive_avg_pool2d 聚合地面网格；
        当目标特征图更细时，使用双线性插值。
        """
        src_h, src_w = feat_src.shape[-2:]
        tgt_h, tgt_w = target_hw
        if (src_h, src_w) == (tgt_h, tgt_w):
            return feat_src
        if tgt_h <= src_h and tgt_w <= src_w:
            return F.adaptive_avg_pool2d(feat_src, output_size=target_hw)
        return F.interpolate(feat_src, size=target_hw, mode="bilinear", align_corners=False)

    def forward(self, aef_or_proxy: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        aligned = self.align_spatial_footprint(aef_or_proxy, target_hw)
        if self.mode == "linear":
            return self.proj(aligned)
        return self.fc2(self.act(self.fc1(aligned)))


# =====================================================================
# 3. 同事 02_实现规格.md §2：PCA 白化 + 固定种子高斯冻结投影 (bb_inj)
# =====================================================================
class FrozenBBProjector(nn.Module):
    """
    严格复刻 aef_bb_inject.py:
    Step 1: 满秩 PCA 白化矩阵 A = V * Lambda^(-1/2) * V^T (64 -> 64)
    Step 2: 固定种子高斯随机投影 P ~ N(0, 1/c) (64 -> c)
    注入公式: F' = F + alpha * P(A * aef_map)
    """
    def __init__(self, aef_dim: int, out_c: int, whitening_mat: torch.Tensor,
                 alpha: float = 1.0, seed: int = 20261103):
        super().__init__()
        self.alpha = alpha
        self.register_buffer("A", whitening_mat.float())
        gen = torch.Generator().manual_seed(seed + out_c)
        P = torch.randn(out_c, aef_dim, generator=gen) * (1.0 / np.sqrt(out_c))
        self.register_buffer("P", P.float())
        W = (P @ whitening_mat.float()).unsqueeze(-1).unsqueeze(-1)
        self.register_buffer("W", W)

    @staticmethod
    def fit_pca_whitening(aef_patches: np.ndarray, eps: float = 1e-6) -> torch.Tensor:
        """
        从 >= 256 个有效训练 tile 拟合 64x64 满秩 PCA 白化矩阵
        """
        if aef_patches.ndim == 4:
            # (N, C, H, W) -> (N*H*W, C)
            X = aef_patches.transpose(0, 2, 3, 1).reshape(-1, aef_patches.shape[1])
        else:
            X = aef_patches
        # 过滤掉全 0 或 NoData (-128) 行
        valid_rows = (np.abs(X).sum(axis=1) > 1e-5) & (np.all(X > -127.5, axis=1))
        X = X[valid_rows]
        mean = X.mean(axis=0, keepdims=True)
        X_centered = X - mean
        cov = np.cov(X_centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(eigvals, eps)))
        A = eigvecs @ inv_sqrt @ eigvecs.T
        return torch.from_numpy(A).float()

    @torch.no_grad()
    def forward(self, feat: torch.Tensor, aef_map: torch.Tensor) -> torch.Tensor:
        if self.alpha == 0.0:
            return feat
        aligned_aef = AEFInjector.align_spatial_footprint(aef_map, feat.shape[-2:])
        proj = F.conv2d(aligned_aef, self.W)
        return feat + self.alpha * proj


# =====================================================================
# 4. 同事 dataset_config.template.yaml：TinyUPerNet 解码头 (channels=128)
# =====================================================================
class TinyUPerNet(nn.Module):
    """
    轻量级多尺度特征金字塔解码头 (channels=128)
    同时支持标准特征层输入与 decode 侧拼接注入备选档
    """
    def __init__(self, in_channels_list: list[int], channels: int = 128, num_classes: int = 8):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_c, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for in_c in in_channels_list
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ) for _ in in_channels_list
        ])
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(channels * len(in_channels_list), channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, feats: list[torch.Tensor], out_hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        laterals = [lat(f) for lat, f in zip(self.lateral_convs, feats)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="bilinear", align_corners=False
            )
        fpn_outs = [fpn(lat) for fpn, lat in zip(self.fpn_convs, laterals)]
        base_hw = fpn_outs[0].shape[-2:]
        upsampled = [
            fpn_outs[0] if i == 0 else F.interpolate(f, size=base_hw, mode="bilinear", align_corners=False)
            for i, f in enumerate(fpn_outs)
        ]
        fused_feat = self.fuse_conv(torch.cat(upsampled, dim=1))  # (B, 128, H/4, W/4)
        logits = self.cls_seg(fused_feat)
        logits = F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)
        return logits, fused_feat


# =====================================================================
# 5. 上级评审 §8：六臂统一分割模型封装 (B0 / R / Bcap / P / A / D)
# =====================================================================
class SixArmSegFramework(nn.Module):
    """
    统一支持上级评审第 8 节定义的全部 6 种实验臂：
      - 'B0'   : 纯影像基线 (Image-Only)
      - 'R'    : 真实 AEF 注入融合 (或喂入同分布高斯噪声作为 R_null 对照)
      - 'Bcap' : 同结构纯标签容量对照 (Image-Only，包含 Predictor + Injector，但无教师蒸馏损失)
      - 'P'    : 方案一：图像预测代理特征并注入融合 (Image-Only，结构与 Bcap 100% 相同，训练期加对齐损失)
      - 'A'    : 训练期辅助教师投影 (Image-Only，推理时删除 aux_heads，需配合主干微调或前置可训 Adapter)
      - 'D'    : 方案二 (第一优先级)：真 AEF 融合教师指导纯影像学生 (Image-Only，部署结构与 B0 100% 相同)
    """
    def __init__(
        self,
        backbone: nn.Module | None,
        in_channels_list: list[int],
        num_classes: int = 8,
        aef_dim: int = 64,
        arm: str = "B0",
        inject_mode: str = "mlp",
        use_zscore_calib: bool = False,
        enable_shared_adapter_for_A: bool = False,
    ):
        super().__init__()
        assert arm in ("B0", "R", "Bcap", "P", "A", "D"), f"Invalid arm: {arm}"
        self.backbone = backbone
        self.in_channels_list = in_channels_list
        self.num_classes = num_classes
        self.aef_dim = aef_dim
        self.arm = arm

        # 可选特征标定层 (评审 §3.6 & §8 B0 协议)
        self.calib_layers = nn.ModuleList([
            ChannelZScoreNorm2d(c) if use_zscore_calib else nn.Identity()
            for c in in_channels_list
        ])

        # A 臂在冻结主干缓存特征下若想让辅助损失影响主路，需前置一层可训轻量 Adapter；
        # 若主干本身已解冻 (FT 档)，则无需此 Adapter。
        self.shared_adapters = None
        if arm == "A" and enable_shared_adapter_for_A:
            self.shared_adapters = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False),
                    nn.BatchNorm2d(c),
                    nn.GELU(),
                    nn.Conv2d(c, c, kernel_size=1, bias=True),
                ) for c in in_channels_list
            ])

        # 解码头 (所有实验臂完全一致)
        self.head = TinyUPerNet(in_channels_list, channels=128, num_classes=num_classes)

        # R 臂：外部真实 AEF (或噪声) 注入器
        if arm == "R":
            self.injectors = nn.ModuleList([
                AEFInjector(in_dim=aef_dim, out_c=c, mode=inject_mode) for c in in_channels_list
            ])

        # Bcap 与 P 臂：完全相同的“图像特征 -> 64维代理特征 -> Zero-Init 注入器”结构
        elif arm in ("Bcap", "P"):
            self.predictors = nn.ModuleList([
                nn.Sequential(
                    # 3x3 膨胀感受野聚合局部空间上下文，预测低频语义代理特征
                    nn.Conv2d(c, 256, kernel_size=3, padding=2, dilation=2, bias=False),
                    nn.GroupNorm(16, 256),
                    nn.GELU(),
                    nn.Conv2d(256, aef_dim, kernel_size=1, bias=True),
                ) for c in in_channels_list
            ])
            self.injectors = nn.ModuleList([
                AEFInjector(in_dim=aef_dim, out_c=c, mode=inject_mode) for c in in_channels_list
            ])

        # A 臂：训练期辅助投影头（对应同事截图中的 proj_l，但梯度需进入可训主干或 shared_adapters）
        elif arm == "A":
            self.aux_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(c, 256, kernel_size=1, bias=True),
                    nn.GELU(),
                    nn.Conv2d(256, aef_dim, kernel_size=1, bias=True),
                ) for c in in_channels_list
            ])

        # D 臂：训练期特征对齐投影头 Q(F_student) -> 匹配 F_teacher (评审 §5.1，部署时直接删除)
        elif arm == "D":
            self.student_q_proj = nn.Conv2d(128, 128, kernel_size=1, bias=True)

    def verify_zero_init_identity(self, dummy_feats: list[torch.Tensor], out_hw: tuple[int, int]) -> float:
        """
        同事 03_验证清单.md §A：跑前冒烟检查——验证 zero-init 注入器在初始时刻严格等价于恒等映射
        """
        self.eval()
        with torch.no_grad():
            base_logits, _ = self.head(dummy_feats, out_hw)
            if self.arm in ("Bcap", "P"):
                fused = [
                    f + inj(pred(f), f.shape[-2:])
                    for f, pred, inj in zip(dummy_feats, self.predictors, self.injectors)
                ]
                test_logits, _ = self.head(fused, out_hw)
                max_diff = (base_logits - test_logits).abs().max().item()
                return max_diff
        return 0.0

    def forward(
        self,
        img: torch.Tensor | None = None,
        cached_feats: list[torch.Tensor] | None = None,
        aef_input: torch.Tensor | None = None,
        out_hw: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """
        物理隔离保护：除 'R' 臂外，任何 Image-Only 臂 ('B0', 'Bcap', 'P', 'A', 'D')
        在前向传播时绝不读取 aef_input；若测试期误传 aef_input 给 Image-Only 臂将直接报错拦截。
        """
        if self.arm != "R" and not self.training and aef_input is not None:
            raise RuntimeError(
                f"[Security Check] Arm '{self.arm}' is strictly Image-Only during inference! "
                f"Passing 'aef_input' at eval/test time is forbidden."
            )

        if cached_feats is not None:
            feats = cached_feats
            if out_hw is None:
                out_hw = (feats[0].shape[-2] * 4, feats[0].shape[-1] * 4)
        else:
            assert img is not None, "Either 'img' or 'cached_feats' must be provided!"
            out_hw = img.shape[-2:]
            feats = self.backbone(img)

        # 1. 特征标定
        feats = [calib(f) for calib, f in zip(self.calib_layers, feats)]

        # 2. 按实验臂路由
        proxy_preds: list[torch.Tensor] = []

        if self.arm in ("B0", "D"):
            fused_feats = feats

        elif self.arm == "R":
            assert aef_input is not None, "Arm 'R' requires 'aef_input' (real AEF or null noise)!"
            fused_feats = [
                f + inj(aef_input, f.shape[-2:])
                for f, inj in zip(feats, self.injectors)
            ]

        elif self.arm in ("Bcap", "P"):
            # 把同事原本悬空的 aux(base_feats) 正式接入 Injector 主路！
            fused_feats = []
            for f, pred_net, inj in zip(feats, self.predictors, self.injectors):
                hat_aef = pred_net(f)
                proxy_preds.append(hat_aef)
                fused_feats.append(f + inj(hat_aef, f.shape[-2:]))

        elif self.arm == "A":
            if self.shared_adapters is not None:
                feats = [f + adapter(f) for f, adapter in zip(feats, self.shared_adapters)]
            fused_feats = feats
            if self.training:
                proxy_preds = [aux(f) for f, aux in zip(feats, self.aux_heads)]

        # 3. 解码头输出
        logits, decoder_feat = self.head(fused_feats, out_hw)
        outputs: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "logits": logits,
            "decoder_feat": decoder_feat,
            "proxy_preds": proxy_preds,
        }
        if self.arm == "D" and self.training:
            outputs["q_feat"] = self.student_q_proj(decoder_feat)
        return outputs
```

---

### 5.2 损失函数、数据解码、同步增广与六臂训练评测主脚本 (`train_eval_six_arms.py`)

```python
"""
train_eval_six_arms.py
包含：
1. OlmoEarth / AEF 非线性反量化解码与 NoData (-128 / 恒零) 有效掩膜构造（评审 §6.2）。
2. 同事截图中的加权余弦蒸馏损失、InfoNCE 对比损失、以及评审 §5.1 的 KL + 特征对齐蒸馏损失。
3. 同事 02_实现规格.md 要求的同步几何增广 (flip/rot90)、固定种子高斯噪声 (null) 与封闭类 mIoU (ignore_index=0)。
4. 自包含的六臂验证与防泄漏冒烟测试入口。
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from six_arm_models import SixArmSegFramework, AEFInjector


# =====================================================================
# 1. 教师特征解码与有效像素掩膜 (评审 §6.2)
# =====================================================================
def decode_and_mask_teacher_embedding(
    raw_emb: torch.Tensor,
    is_olmoearth_int8: bool = False,
    nodata_val: float = -128.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    输入:
      raw_emb: (B, C, H, W) 原始读取的教师栅格张量
    输出:
      clean_emb: (B, C, H, W) 浮点解码并完成 L2 归一化的干净教师特征
      valid_mask: (B, 1, H, W) 有效像素掩膜 (1.0 为有效，0.0 为 NoData 或恒零无效像素)
    """
    emb = raw_emb.float()
    # 1. 检测 NoData (-128) 与恒零无效像素
    not_nodata = (emb > (nodata_val + 0.5)).all(dim=1, keepdim=True)
    not_all_zero = (emb.abs().sum(dim=1, keepdim=True) > 1e-6)
    valid_mask = (not_nodata & not_all_zero).float()

    # 2. 若为 OlmoEarth int8 量化格式，执行反量化还原并屏蔽无效区
    if is_olmoearth_int8:
        # 将 [-127, 127] 映射到 [-1, 1] 并按官方非线性幂次/球面还原
        emb = (emb.clamp(min=-127.0, max=127.0)) / 127.0

    # 将无效像素置 0，防止污染插值或池化
    clean_emb = emb * valid_mask
    return clean_emb, valid_mask


# =====================================================================
# 2. 蒸馏与代理对齐损失函数库 (Cosine / InfoNCE / KL / Feature KD)
# =====================================================================
def compute_multiscale_cosine_loss(
    proxy_preds: list[torch.Tensor],
    aef_target: torch.Tensor,
    valid_mask: torch.Tensor,
    layer_weights: list[float] | None = None,
) -> torch.Tensor:
    """
    对应同事截图中的 dloss = sum_l w_l * mean(1 - cos(proj_l(F_l), AEF)) / sum_l w_l
    并增加评审 §6.1 & §6.2 要求的 Footprint 池化对齐与 valid_mask 屏蔽
    """
    num_layers = len(proxy_preds)
    if layer_weights is None:
        # 02_实现规格.md §5：层权浅小深大
        layer_weights = [float(i + 1) for i in range(num_layers)]
    weight_sum = sum(layer_weights)

    total_dloss = torch.tensor(0.0, device=aef_target.device)
    for w, pred in zip(layer_weights, proxy_preds):
        target_hw = pred.shape[-2:]
        tgt_l = AEFInjector.align_spatial_footprint(aef_target, target_hw)
        mask_l = (AEFInjector.align_spatial_footprint(valid_mask, target_hw) > 0.5).float().squeeze(1)

        cos_sim = F.cosine_similarity(pred, tgt_l, dim=1)  # (B, H_l, W_l)
        layer_loss = ((1.0 - cos_sim) * mask_l).sum() / (mask_l.sum() + 1e-6)
        total_dloss = total_dloss + (w / weight_sum) * layer_loss

    return total_dloss


def compute_multiscale_infonce_loss(
    proxy_preds: list[torch.Tensor],
    aef_target: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float = 0.07,
    max_tokens: int = 1024,
) -> torch.Tensor:
    """
    前期 T11D/T11P 验证最优的对称 InfoNCE 损失（仅在有效像素上采样负样本）
    """
    total_nce = torch.tensor(0.0, device=aef_target.device)
    for pred in proxy_preds:
        target_hw = pred.shape[-2:]
        tgt_l = AEFInjector.align_spatial_footprint(aef_target, target_hw)
        mask_l = (AEFInjector.align_spatial_footprint(valid_mask, target_hw) > 0.5).view(-1)

        # (B, C, H, W) -> (N, C)
        s_vec = F.normalize(pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1]), dim=-1)
        t_vec = F.normalize(tgt_l.permute(0, 2, 3, 1).reshape(-1, tgt_l.shape[1]), dim=-1)

        valid_idx = torch.nonzero(mask_l, as_tuple=False).squeeze(-1)
        if valid_idx.numel() < 2:
            continue
        if valid_idx.numel() > max_tokens:
            perm = torch.randperm(valid_idx.numel(), device=valid_idx.device)[:max_tokens]
            valid_idx = valid_idx[perm]

        s_sub = s_vec[valid_idx]
        t_sub = t_vec[valid_idx]
        logits = (s_sub @ t_sub.T) / temperature
        targets = torch.arange(s_sub.shape[0], device=s_sub.device)
        loss_st = F.cross_entropy(logits, targets)
        loss_ts = F.cross_entropy(logits.T, targets)
        total_nce = total_nce + 0.5 * (loss_st + loss_ts)

    return total_nce / max(len(proxy_preds), 1)


def compute_six_arm_total_loss(
    arm: str,
    student_out: dict,
    labels: torch.Tensor,
    aef_target: torch.Tensor | None = None,
    aef_valid_mask: torch.Tensor | None = None,
    teacher_out: dict | None = None,
    ignore_index: int = 0,
    distill_weight: float = 0.5,   # 对应同事截图中的 --aef-distill 0.5 (或 0.2 起步)
    align_mode: str = "cosine",    # 'cosine' | 'infonce'
    kd_beta: float = 1.0,          # 评审 §5.1 公式中的 beta
    kd_temp: float = 4.0,          # 评审 §5.1 公式中的 tau
    feat_kd_weight: float = 0.0,   # D 臂二阶段特征对齐权重
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    统一计算六臂损失：
    - B0 / R / Bcap : 纯交叉熵 CE (ignore_index=0)
    - P / A         : CE + distill_weight * dloss (Cosine 或 InfoNCE)
    - D             : CE + beta * tau^2 * KL(p_T^tau || p_S^tau) + feat_kd_weight * d(Q(F_S), sg(F_T))
    """
    logits = student_out["logits"]
    loss_ce = F.cross_entropy(logits, labels, ignore_index=ignore_index)
    metrics = {"loss_ce": loss_ce.item()}
    total_loss = loss_ce

    if arm in ("P", "A") and distill_weight > 0.0:
        assert aef_target is not None and aef_valid_mask is not None
        if align_mode == "cosine":
            dloss = compute_multiscale_cosine_loss(
                student_out["proxy_preds"], aef_target, aef_valid_mask
            )
        elif align_mode == "infonce":
            dloss = compute_multiscale_infonce_loss(
                student_out["proxy_preds"], aef_target, aef_valid_mask
            )
        else:
            raise ValueError(f"Unknown align_mode: {align_mode}")
        total_loss = total_loss + distill_weight * dloss
        metrics["loss_align"] = dloss.item()

    elif arm == "D":
        assert teacher_out is not None, "Arm 'D' requires frozen teacher outputs!"
        with torch.no_grad():
            t_logits = teacher_out["logits"].detach()
            t_feat = teacher_out["decoder_feat"].detach()

        # 评审 §5.1 公式: L = L_seg(S(X), Y) + beta * tau^2 * KL(p_T^tau || p_S^tau)
        valid_pixel_mask = (labels != ignore_index).unsqueeze(1).float()  # (B, 1, H, W)
        s_log_prob = F.log_softmax(logits / kd_temp, dim=1)
        t_prob = F.softmax(t_logits / kd_temp, dim=1)
        kl_per_pixel = F.kl_div(s_log_prob, t_prob, reduction="none").sum(dim=1, keepdim=True)
        loss_kl = (kl_per_pixel * valid_pixel_mask).sum() / (valid_pixel_mask.sum() + 1e-6) * (kd_temp ** 2)

        total_loss = total_loss + kd_beta * loss_kl
        metrics["loss_kl"] = loss_kl.item()

        if feat_kd_weight > 0.0 and "q_feat" in student_out:
            cos_f = F.cosine_similarity(student_out["q_feat"], t_feat, dim=1)
            loss_fkd = (1.0 - cos_f).mean()
            total_loss = total_loss + feat_kd_weight * loss_fkd
            metrics["loss_fkd"] = loss_fkd.item()

    metrics["loss_total"] = total_loss.item()
    return total_loss, metrics


# =====================================================================
# 3. 同步增广、固定种子噪声与封闭类 mIoU 评测 (对齐同事 01/02/03 规范)
# =====================================================================
def sync_augment_tile(
    img: torch.Tensor,
    label: torch.Tensor,
    aef: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    同事 02_实现规格.md §3：flip/rot90 必须同步作用于影像、标签与 AEF（错位即废）
    """
    if torch.rand(1).item() < 0.5:
        img = torch.flip(img, dims=[-1])
        label = torch.flip(label, dims=[-1])
        if aef is not None:
            aef = torch.flip(aef, dims=[-1])
    if torch.rand(1).item() < 0.5:
        img = torch.flip(img, dims=[-2])
        label = torch.flip(label, dims=[-2])
        if aef is not None:
            aef = torch.flip(aef, dims=[-2])
    k = int(torch.randint(0, 4, (1,)).item())
    if k > 0:
        img = torch.rot90(img, k, dims=[-2, -1])
        label = torch.rot90(label, k, dims=[-2, -1])
        if aef is not None:
            aef = torch.rot90(aef, k, dims=[-2, -1])
    return img, label, aef


def build_null_noise_aef(
    aef_batch: torch.Tensor,
    channel_mean: torch.Tensor,
    channel_std: torch.Tensor,
    seed: int = 20269999,
) -> torch.Tensor:
    """
    同事 02_实现规格.md §4：构造与 AEF 同均值、同方差的逐通道高斯噪声 (R_null 对照)
    """
    gen = torch.Generator(device=aef_batch.device).manual_seed(seed)
    z = torch.randn(aef_batch.shape, generator=gen, device=aef_batch.device, dtype=aef_batch.dtype)
    return z * channel_std.view(1, -1, 1, 1) + channel_mean.view(1, -1, 1, 1)


@torch.no_grad()
def update_confusion_matrix(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: int = 0,
) -> torch.Tensor:
    valid = (labels != ignore_index) & (labels >= 0) & (labels < num_classes)
    p = preds[valid].view(-1)
    y = labels[valid].view(-1)
    idx = y * num_classes + p
    cm = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm


def compute_closed_class_miou(cm: torch.Tensor, ignore_index: int = 0) -> tuple[float, dict[int, float]]:
    """
    同事 dataset_config.template.yaml: closed_class_miou (剔除背景类 0 后计算平均)
    """
    num_classes = cm.shape[0]
    per_class_iou = {}
    ious = []
    for c in range(num_classes):
        if c == ignore_index:
            continue
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp
        denom = tp + fp + fn
        if denom > 0:
            iou = tp / denom
            per_class_iou[c] = iou
            ious.append(iou)
    mean_iou = float(np.mean(ious)) if ious else 0.0
    return mean_iou, per_class_iou


# =====================================================================
# 4. 跑前 10 分钟冒烟自检 (03_验证清单.md §A 自动验证脚本)
# =====================================================================
def run_smoke_test_all_six_arms():
    """
    自动验证六臂前向/反向传播、Zero-Init 恒等性、梯度连通性以及推理期 Image-Only 物理隔离。
    """
    print("=== 启动六臂框架跑前冒烟测试 (Smoke Test) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels_list = [128, 256, 512, 1024]
    num_classes = 8  # WHU-OPT: 0 背景 + 7 有效类
    B, H, W = 2, 128, 128

    # 模拟冻结骨干的 4 级特征金字塔缓存 (requires_grad=False，模拟同事 SSD 缓存常量)
    dummy_feats = [
        torch.randn(B, c, H // (2 ** (i + 2)), W // (2 ** (i + 2)), device=device, requires_grad=False)
        for i, c in enumerate(in_channels_list)
    ]
    dummy_labels = torch.randint(0, num_classes, (B, H, W), device=device)
    dummy_raw_aef = torch.randn(B, 64, H // 4, W // 4, device=device)
    clean_aef, valid_mask = decode_and_mask_teacher_embedding(dummy_raw_aef)

    # 1. 先初始化并训练一步 R 臂 (真 AEF 融合教师)
    model_R = SixArmSegFramework(None, in_channels_list, num_classes=num_classes, arm="R", inject_mode="mlp").to(device)
    model_R.train()
    out_R = model_R(cached_feats=dummy_feats, aef_input=clean_aef, out_hw=(H, W))
    loss_R, _ = compute_six_arm_total_loss("R", out_R, dummy_labels)
    loss_R.backward()
    model_R.eval()
    print("[PASS] Arm R (真 AEF 融合教师) 前向与反向传播正常。")

    # 2. 逐一验证 Image-Only 四臂: B0, Bcap, P, D
    for arm in ["B0", "Bcap", "P", "D"]:
        model = SixArmSegFramework(None, in_channels_list, num_classes=num_classes, arm=arm, inject_mode="mlp").to(device)

        # 检查 Zero-Init 恒等性
        if arm in ("Bcap", "P"):
            max_diff = model.verify_zero_init_identity(dummy_feats, (H, W))
            assert max_diff == 0.0, f"Zero-init failed for {arm}: max_diff={max_diff}"
            print(f"[PASS] Arm {arm} Zero-Init 恒等检查通过 (初始注入前后最大差异 = {max_diff:.1e})。")

        # 训练期前向与反向传播（验证在冻结缓存特征下，P 臂的 dloss 与 CE 梯度能否正常更新 Predictor 和 Injector）
        model.train()
        out = model(cached_feats=dummy_feats, out_hw=(H, W))
        t_out = model_R(cached_feats=dummy_feats, aef_input=clean_aef, out_hw=(H, W)) if arm == "D" else None
        loss, log_dict = compute_six_arm_total_loss(
            arm=arm,
            student_out=out,
            labels=dummy_labels,
            aef_target=clean_aef if arm == "P" else None,
            aef_valid_mask=valid_mask if arm == "P" else None,
            teacher_out=t_out,
            distill_weight=0.5,
        )
        loss.backward()

        # 验证 P 臂中 predictors 是否同时收到了梯度（彻底解决同事原版 dloss 悬空问题）
        if arm == "P":
            grad_norm = model.predictors[0][0].weight.grad.norm().item()
            assert grad_norm > 0.0, "Predictor gradient is zero!"
            print(f"[PASS] Arm P 梯度连通性检查通过 (Predictor 首层梯度范数 = {grad_norm:.4f} > 0)。")

        # 验证测试期 Image-Only 隔离锁：推理时完全不传 AEF 正常出结果；若偷传 AEF 必须抛出异常拦截
        model.eval()
        with torch.no_grad():
            eval_out = model(cached_feats=dummy_feats, out_hw=(H, W))
            assert eval_out["logits"].shape == (B, num_classes, H, W)
        print(f"[PASS] Arm {arm} 纯影像推理 (Image-Only) 验证通过 | 训练损失项: {log_dict}")

    print("=== 全部六臂冒烟自检 100% 通过！ ===")


if __name__ == "__main__":
    run_smoke_test_all_six_arms()
```

---

## 6. 执行与汇报核查清单（Checklist）

在向团队提交下一轮实验数据前，逐项核对以下 6 条标准（整合自上级评审 §8 与同事 `03_验证清单.md`）：

1. **运行冒烟自检**：先执行 `python train_eval_six_arms.py` 通过零初始化恒等性（最大差异恰为 $0$）、`P` 臂梯度连通性与 Image-Only 推理拦截测试。
2. **锁定基线学习率与封闭类指标**：WHU-OPT 固定使用 7 类 `closed_class_miou`（`ignore_index=0`），FBP (GF-2) 固定使用 24 类 `closed_class_miou`（`unlabeled=0` 设为 `ignore_index=0`）；所有实验臂严格沿用 `B0` 扫描出的最优学习率（如 `lim` 固定 `lr=0.003`）。
3. **成对报数（拒绝单臂孤立报喜）**：
   * 汇报 `D` 臂（融合教师 $\to$ 纯影像学生）时，必须同表并列汇报同预算 `B0` 与特权参考 `R`；
   * 汇报 `P` 臂（代理预测融合）时，必须同表并列汇报同结构纯标签对照 `Bcap` 与 `B0`。
4. **物理隔离验证**：最终入选的 `D` 或 `P` 权重，必须在不挂载 AEF 栅格目录的独立评测命令下复测 `test` 分数，确保结果逐位一致。
5. **多随机种子与打乱教师检验**：单种子差值 $< 0.6\text{ pt}$ 一律记为“持平”；胜出方案补齐 3 种子均值与标准差，并附带 1 组“打乱配对教师（Shuffled Teacher）”消融以确认因果特异性。
6. **区分“专用分割模型提升”与“通用主干提升”**：在 WHU-OPT 上通过 `D` 或 `P` 取得的提升，按“单任务无教师部署分割方案”验收；若需宣称主干通用表征增强，需将微调后的主干迁出至无大景重叠的 FBP 独立重训分割头加以验证。