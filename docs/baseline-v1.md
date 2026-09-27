# Baseline v1 experiment

## Purpose

Establish a reproducible CNN baseline and investigate its validation errors
before changing the architecture or training procedure.

## Setup

- Dataset: GTSRB, 43 classes.
- Training split: 31,379 images.
- Validation split: 7,830 images.
- Split: track-separated, seed 42.
- Test set: not used for training, model selection, or this analysis.
- Input: full RGB images resized to 32 × 32 using bilinear interpolation.
- Pixel values: scaled to [0, 1].
- Augmentation: none.
- Model: two convolution/ReLU/pooling stages with 32 and 64 channels,
  followed by a 128-unit hidden layer and 43 output logits.
- Optimizer: Adam.
- Learning rate: 0.001.
- Batch size: 128.
- Epochs: 10.
- Training seed: 42.
- Device: NVIDIA GeForce RTX 3060 Laptop GPU.
- Checkpoint selection: highest validation accuracy; first epoch wins ties.

## Results

- Selected checkpoint: epoch 8.
- Best validation accuracy: 90.84%.
- Validation loss at epoch 8: 0.4219.
- Validation macro F1: 0.864.
- Validation weighted F1: 0.906.
- Reloaded checkpoint reproduces 90.84% validation accuracy.
- Fixed-checkpoint training accuracy: 99.29%.
- Fixed-checkpoint training macro F1: 0.994.
- Fixed-checkpoint class-27 training recall: 180/180 = 100%.

Training loss continued decreasing while validation performance fluctuated.
This is evidence of overfitting, although epoch 8 improved on earlier results.

## Error analysis

### Class 5

- 44 validation images were incorrectly predicted as class 3.
- These mistakes covered five tracks.
- Both correct and incorrect examples included dark images.
- Brightness alone does not explain the observed mistakes.
- Two neighboring frames from track 00006 produced different predictions:
  - Frame 00000: class 5 score 32.3%, class 3 score 67.6%.
  - Frame 00001: class 5 score 57.2%, class 3 score 40.2%.
- Their original dimensions were 30 × 31 and 32 × 32.
- For this pair, low detail was already present in the source images;
  downsampling from a larger source was not the explanation.

These observations do not establish the cause of all class-5 errors.

### Class 27

- Validation recall: 13/60 = 21.7%.
- Validation F1: 0.347.
- Track 00004: 4/30 correct.
- Track 00005: 9/30 correct.
- Errors were distributed across seven other classes.
- Both validation tracks performed poorly.
- The first, middle, and last frames were inspected for each track.
- Visually clearer examples were not always classified correctly.

Only two validation tracks are available, limiting conclusions about the
class as a whole. The grid is a diagnostic sample, not a causal experiment.

## Current interpretation

The baseline generalizes reasonably overall but has substantial weaknesses
in some classes. Overall accuracy hides these weaknesses.

The observed errors do not yet establish that brightness, resolution,
background, class imbalance, or architecture is the primary cause.

## Completed training–validation comparison

Both splits were evaluated using the same saved epoch-8 checkpoint on CPU,
with model.eval(), torch.inference_mode(), and no weight updates.

| Metric | Training | Validation |
| --- | --- | --- |
| Images | 31,379 | 7,830 |
| Accuracy | 99.29% | 90.84% |
| Macro F1 | 0.994 | 0.864 |
| Class-27 recall | 180/180 (100%) | 13/60 (21.7%) |

The overall accuracy gap is 8.45 percentage points. Class 27 is fitted
perfectly on the training images but generalizes poorly to both held-out
tracks. Other classes show similar gaps: class-21 recall is 100% versus
48.3%, and class-24 recall is 99.5% versus 50.0%.

These results support a generalization problem rather than an inability
to fit the available class-27 training examples. They do not identify
the visual features responsible or prove memorization as the sole cause.

Fixed-checkpoint training metrics differ from logged epoch training metrics:
the latter aggregate predictions while weights change during the epoch.

Training-set metrics are diagnostic and are not estimates of unseen accuracy.
The test set remains untouched. No architecture or training hyperparameters
were changed during this analysis.

## Follow-up experiment

The training-only position and scale augmentation experiment is implemented
and documented in [augmentation-v1.md](augmentation-v1.md). It tests a hypothesis
for improving generalization. Keep the split, architecture, seed, and other
training settings fixed, and use a separate output directory. Keep validation
preprocessing deterministic. Record the exact augmentation settings before
running the experiment.

Compare validation accuracy and macro F1, with class-27 recall as a secondary
diagnostic. An improvement on this single split and seed would be preliminary;
do not tune solely to the two class-27 validation tracks.

## Artifacts

Local artifacts are stored under outputs/runs/baseline-v1/:

- run.json
- history.json
- best.pt
- learning_curves.png
- validation_report.txt
- validation_confusion_matrix.npy
- class_27_tracks.png
- training_report.txt
- training_confusion_matrix.npy

Generated artifacts are ignored by Git. This document records the findings
in version-controlled documentation.

## Reproduction

From the project root, after the baseline training run:

```powershell
uv run python scripts/plot_training_history.py
uv run python scripts/evaluate_validation.py
```

The evaluation script checks annotation and split fingerprints against the
checkpoint, but does not fingerprint image contents. It is currently specific
to the baseline architecture and preprocessing. Reports and plots are replaced
when the script is rerun; the checkpoint is not modified.

