# AnySat 断点续训与 Colab 运行

支持 AnySat 的 T1、T3、T4、T5 的 `train` 动作，包括冻结主干与全量微调。Prithvi、Clay、OlmoEarth 和 AnySat 其他任务暂未接入；控制层会拒绝给它们传 `--resume-from`。

## 保存和恢复

在 Colab 挂载 Drive、解包新版代码、设置 `.env` 并准备好相同数据清单后，首次运行：

```bash
%%bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /content/rs-backbone-eval
set -a; source .env; set +a
export RS_EVAL_CHECKPOINT_ROOT=/content/drive/MyDrive/rs-eval-checkpoints
export RS_EVAL_CHECKPOINT_EVERY=1
uv run rs-eval dispatch --action train --task T5 --model anysat_base \
  --protocol frozen_probe --seed 42 --run-id anysat-T5-s42-colab-01
```

每次保存成功后会输出 `train.checkpoint: saved epoch N: .../latest.pt`。文件同时存在于本地 run 目录和：

```text
/content/drive/MyDrive/rs-eval-checkpoints/anysat-T5-s42-colab-01/latest.pt
```

新会话重新挂载 Drive、准备相同版本源码、依赖、数据和 manifest 后，以**新的 run ID** 恢复：

```bash
%%bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /content/rs-backbone-eval
set -a; source .env; set +a
export RS_EVAL_CHECKPOINT_ROOT=/content/drive/MyDrive/rs-eval-checkpoints
export RS_EVAL_CHECKPOINT_EVERY=1
uv run rs-eval dispatch --action train --task T5 --model anysat_base \
  --protocol frozen_probe --seed 42 --run-id anysat-T5-s42-colab-02 \
  --resume-from /content/drive/MyDrive/rs-eval-checkpoints/anysat-T5-s42-colab-01/latest.pt
```

后续检查点保存在 `.../anysat-T5-s42-colab-02/latest.pt`，再次中断时从这个新路径接着恢复。旧 run 和旧检查点保留；新 run 记录父 run ID、恢复位置以及完整的历史曲线。即使重建 VM 后本地旧 run 文件夹已经丢失，也能仅凭 Drive 的 `latest.pt` 恢复。

`epochs` 始终表示实验的**总轮数**。例如已保存 60/125 轮，恢复后执行余下 65 轮；无需把环境变量改成 65。原有 micro-batch、梯度累积、学习率、冻结方式、裁剪、验证频率和 smoke/评测限制等配置必须保持相同，数据清单、初始权重校验和、配置文件、代码版本、PyTorch 版本和 CPU/CUDA 设备类型也必须一致。不匹配会在继续训练前报错。

## 检查点内容及边界

- `latest.pt`：当前模型权重、AdamW 全部状态、下一轮位置、累计 optimizer steps、Python/NumPy/Torch CPU/CUDA RNG、训练曲线与累计时长，以及 B1 的历史最佳验证权重。
- 每轮在所有累积窗口更新完毕、验证完成后保存，下一轮从头创建 DataLoader。当前数据读取没有随机增强，恢复 RNG 可延续 shuffle 和 dropout 随机流。没有跨 GPU 类型逐位一致的保证。
- 不保存半个 epoch 的梯度；默认最多损失当前尚未完整保存的一轮。`RS_EVAL_CHECKPOINT_EVERY=N` 可降低写盘频率，但最多回退 N 轮。
- 先写本地临时文件并替换，再写外部临时副本并替换。Drive 复制失败会中止当前尝试并保留先前的完整外部检查点；不要使用 `.tmp` 文件恢复。Drive/FUSE 的实际同步可靠性仍取决于服务，保存成功日志是应用层完成信号。
- 若训练已完成、在最终测试阶段断开，可以从最后的检查点恢复；训练循环会跳过，只重新做最终选模落盘和评测。
- `best_checkpoint.pt` 只有验证选中的权重，不含优化器和 RNG，不能用作续训入口。旧代码启动且从未写出 `latest.pt` 的运行无法追溯恢复。
- 全量微调检查点还包含优化器和最佳模型，可能达到数 GB。预留本地及 Drive 空间，写入时还需要临时副本；训练数据继续放在 `/content` 本地磁盘。
- 完成后仍需把新 run 目录打包回传，检查点本身不是最终实验结果包。

smoke 使用单独的 `RS_EVAL_RUNS_ROOT` 和检查点目录。总 epochs 或 limit 不一致时不允许把 smoke 检查点续成正式实验。先对每个任务分别做保存→中断→恢复的 GPU smoke，再安排正式预算；旧 `5753749` 网盘包不包含此功能，需要更新后重新打包。

CPU 回归通过真实四个训练循环和小型可微模型检验：连续训练与第一轮后中断、重新创建模型/AdamW、恢复到第三轮，模型参数、优化器状态、RNG、最佳权重和曲线一致。尚未在 Colab A100 与真实 AnySat checkpoint 上实测吞吐、显存及 Drive 写入耗时。
