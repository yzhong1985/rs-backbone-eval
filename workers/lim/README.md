# LIM worker (MiT-Huge, MixMIM)

Isolated environment for the LIM encoder on iSAID (T9). The checkpoints and the
data live outside the repository; the control plane only ever sees `run.json`.

## Checkpoint

`LIM_CHECKPOINT` must point at `mit_huge_600.pth` (MixMIM, Million-AID,
epoch 150/300). The encoder half is loaded key-for-key (573 tensors, 0 missing,
0 unexpected); the reconstruction decoder and mask token (104 tensors, 27.9 M
parameters) are dropped. `absolute_pos_embed` is a 56x56 grid and is
bicubically interpolated to the tile's token grid at forward time.

## Tracks

| protocol | what trains | note |
|---|---|---|
| `upernet_frozen` (B2U) | UPerNet head only | first round; backbone frozen |
| `upernet_finetune` (B1U) | encoder + head | upper-bound reference (P0-b) |

Both use the shared `UPerNetHead` from `rs_backbone_eval.heads` (PPM + FPN,
GroupNorm). The effective batch is fixed at 8 (micro 1 x accumulation 8); the
worker refuses any other product.

## Protocol details

* training reads one manifest tile per row, pads the label with `255`, and never
  lets padding into the loss;
* test/validation run **whole-image sliding windows** through
  `rs_backbone_eval.evaluation.windows`: overlapping logits are averaged before
  the argmax and padding is dropped, so a 387x278 image contributes exactly
  107,466 pixels;
* the 16-colour iSAID palette is mapped to category ids (1 storage_tank ...
  15 Helicopter, 0 background) in `t9.PALETTE`.

## Environment knobs

| variable | meaning |
|---|---|
| `LIM_WORKER_EPOCHS` | epochs (official 50) |
| `LIM_WORKER_MICRO_BATCH` / `LIM_WORKER_GRAD_ACCUM` | must multiply to 8 |
| `LIM_WORKER_LR` | AdamW learning rate |
| `LIM_WORKER_LIMIT_BATCHES` | stop after N steps (smoke) |
| `LIM_WORKER_EVAL_IMAGES` | score only the first N images (smoke) |
| `LIM_WORKER_NORM_SAMPLES` | tiles used for the mean/std estimate |
| `LIM_WORKER_NUM_WORKERS` | dataloader workers |

## Smoke (P0-b)

The smoke follows the plan's two budgets: 512px and 1024px tiles, frozen and
full fine-tune, plus a timed whole-image evaluation. It is a throughput
measurement, not a result: `LIM_WORKER_LIMIT_BATCHES` keeps the run short and
the run record is marked `[SMOKE]`/`REDUCED-BUDGET`.
