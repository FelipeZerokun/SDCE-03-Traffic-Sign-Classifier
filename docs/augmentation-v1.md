# Position and scale augmentation experiment

## Hypothesis and controls

Modest random changes in position and scale may improve generalization.
This is a hypothesis, not an established explanation for baseline errors.
The baseline achieved 90.84% validation accuracy, macro F1 0.864, and
class-27 recall 21.7%, compared with 100% class-27 training recall.

Use the same split, architecture, seed 42, Adam learning rate 0.001,
batch size 128, and ten epochs. Start from fresh weights. Select the
checkpoint by highest validation accuracy, with the first epoch winning ties.
The test set remains reserved for final evaluation.

## Transformation

- Applied only to training tensors after RGB 32 × 32 preprocessing.
- RandomAffine with no rotation or shear.
- Horizontal and vertical translation up to 5% (rounded to pixel offsets).
- Scale sampled from 0.9 to 1.1.
- Bilinear interpolation and black fill for exposed borders.
- No flips or brightness changes.
- Validation and diagnostic training evaluation use no augmentation.

The preview of four training images showed modest changes without obvious
severe cropping. This sample does not guarantee all transformations retain
useful detail. Interpolation introduces smoothing and fill introduces borders.

## Commands

```powershell
uv run traffic-sign-classifier train --config configs/augmentation.toml
uv run python scripts/evaluate_validation.py --run-dir outputs/runs/augmentation-v1
```

Outputs are isolated in outputs/runs/augmentation-v1; existing runs are
never overwritten by training. Metadata records augmentation separately
from deterministic inference preprocessing. Evaluation reports are overwritten
on rerun. Paths passed to --run-dir resolve relative to the project root.

## Results

Completed ten epochs on the NVIDIA GeForce RTX 3060 Laptop GPU. Epoch 10
was selected; reloading the checkpoint reproduced its validation accuracy.

| Metric | Baseline (epoch 8) | Augmentation (epoch 10) |
| --- | --- | --- |
| Validation accuracy | 90.84% | 91.26% |
| Validation loss | 0.4219 | 0.3283 |
| Validation macro F1 | 0.864 | 0.880 |
| Validation weighted F1 | 0.906 | 0.911 |
| Class-27 validation recall | 13/60 (21.7%) | 43/60 (71.7%) |
| Class-27 track 00004 correct | 4/30 | 21/30 |
| Class-27 track 00005 correct | 9/30 | 22/30 |
| Unaugmented training accuracy | 99.29% | 99.32% |
| Unaugmented training macro F1 | 0.994 | 0.995 |
| Class-27 training recall | 100% | 100% |

Overall validation accuracy improved by approximately 0.42 percentage points;
class-27 recall improved by 50 percentage points on both tracks combined.
There are regressions: recall for class 16 fell from 96.7% to 74.4%, class 29
from 85.0% to 50.0%, class 34 from 88.9% to 65.6%, and class 40 from 90% to 60%.
The result is a modest aggregate improvement, not an improvement in every class.
Macro recall rounds to 0.871 in both reports despite the macro F1 gain.

A single split and seed provide preliminary evidence only. These results do
not isolate whether translation, scaling, interpolation, or border changes
contributed to the gain. The best result occurred at the epoch budget limit;
convergence has not been established. No further tuning or test evaluation
was performed.

Augmented training metrics are measured on changing transformed inputs;
the table uses fixed-checkpoint, unaugmented training evaluation. The
training–validation accuracy gap remains about 8.06 percentage points.

## Verification and artifacts

- 107 tests passed, including augmentation config validation, baseline default,
  repeatability under a fixed seed, and tensor/label preservation.
- Ruff lint/format checks and source mypy checks passed.
- best.pt, run.json, and history.json were saved in the new run directory.
- training_report.txt, validation_report.txt, both confusion matrices, and
  class_27_tracks.png were generated using the selected checkpoint.
- The baseline checkpoint and reports were preserved; the test set was unused.

## Remaining project work

This completes the first augmentation experiment. Dedicated evaluation and
prediction CLI commands, final test-set integrity checks and evaluation after
model selection, and the final project report remain to be implemented.
