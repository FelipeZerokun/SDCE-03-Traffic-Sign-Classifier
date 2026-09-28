# Traffic Sign Classifier

A modern Python rebuild of the Udacity Self-Driving Car Engineer
traffic-sign classification project.

## Status

Training, evaluation, and cropped-image prediction commands are implemented.
The final model is the augmentation-v1 epoch-10 checkpoint, selected using
91.26% validation accuracy and 0.880 macro F1. Further training is deferred.
See the [final report](docs/final-report.md) for closure status, final results,
reproduction commands, and limitations. The original course's 93% validation
target is not met by this iteration on our track-separated split.

Final supplied-test accuracy is **90.19%** (macro F1 **0.8650**). Eight exact
training/validation overlaps were found; excluding them gives **90.18%** on
12,622 images. Near-duplicate independence has not been established.

## Evaluate and predict

Run setup first and provide the local dataset, split manifest, and checkpoint.
The example output directory already exists in the completed local project;
choose a new directory when repeating evaluation. Prediction accepts your own
cropped image path.

```powershell
uv run traffic-sign-classifier evaluate --checkpoint outputs/runs/augmentation-v1/best.pt --config configs/dataset.toml --manifest outputs/splits/baseline-v2.json --split test --output outputs/final/test
uv run traffic-sign-classifier predict --checkpoint outputs/runs/augmentation-v1/best.pt --image path/to/cropped-sign.png --top-k 5
```

Evaluation writes JSON metrics/per-image predictions and a CSV confusion matrix.
Use a new output directory for each evaluation; existing results are protected.
Both commands default to CPU and accept `--device cuda`. Prediction scores are
softmax outputs, not calibrated confidence. Local datasets, the split manifest,
and the saved checkpoint are required and remain outside Git.

## Goal

Train and evaluate a CNN that recognizes cropped traffic-sign images,
with reproducible experiments and a tested Python application.

See [project requirements](docs/requirements.md).

## Environment

- Python 3.12
- uv
- PyTorch and torchvision
- NumPy, Pillow, Matplotlib, and scikit-learn
- pytest, Ruff, and mypy

The current dependency configuration selects CUDA 12.8 PyTorch builds
on Windows and Linux.

## Setup

Install uv, then run:

    uv sync --locked

## Development checks

    uv run ruff format --check .
    uv run ruff check .
    uv run mypy

    uv run pytest

## Original implementation

The original project is available at:
https://github.com/FelipeZerokun/SDCE-03-Traffic-Sign-Classifier

The local legacy/ directory is an ignored reference copy.
Datasets belong in data/ and generated results in outputs/.

## Audit the training dataset

    uv run traffic-sign-classifier audit --config configs/dataset.toml

## Train the baseline

Create a split manifest if one does not already exist:

    uv run traffic-sign-classifier split --config configs/dataset.toml --output outputs/splits/baseline-v2.json

Then train using the settings in `configs/training.toml`:

    uv run traffic-sign-classifier train --config configs/training.toml

The default run uses ten epochs, batches of 128, Adam with a learning rate of
0.001, and seed 42. Device `auto` selects CUDA when available, otherwise CPU;
set `device = "cpu"` or `device = "cuda"` to require a particular device.
Configuration paths resolve relative to the training configuration file.

The CNN has two convolution/ReLU/pooling stages (32 and 64 channels), followed
by a 128-unit hidden layer and 43 output logits. It uses the existing full-image
RGB 32 × 32 preprocessing, scaling to [0, 1], without augmentation.

Outputs go to `outputs/runs/baseline-v1/`:

- `run.json`: settings, dependency versions, device, and source fingerprints.
- `history.json`: sample-weighted loss and accuracy for each completed epoch.
- `best.pt`: model weights from the highest validation-accuracy epoch, with
  architecture, ordered class IDs, preprocessing settings, and run metadata.
  The first epoch wins ties. Output index equals the GTSRB numeric class ID.

The supplied test set is never loaded during training. Set a new output path
for every run; existing directories are rejected to preserve earlier results.
An interrupted run may contain completed epoch results, but resume training is
not implemented. Checkpoints contain inference weights, not optimizer state.

Random seeds and deterministic PyTorch operations are configured. Identical
results are not guaranteed across hardware, operating systems, or dependency
versions. Fingerprints cover annotation and split files, not image bytes.

## Baseline analysis

From the project root:

    uv run python scripts/plot_training_history.py
    uv run python scripts/evaluate_validation.py

The scripts save learning curves, training and validation reports, confusion
matrices, and a class-27 validation image grid alongside the checkpoint.

The same epoch-8 checkpoint achieved 99.29% training accuracy and 90.84%
validation accuracy. Class-27 recall was 100% on training images but 21.7%
on validation images across only two tracks. These results indicate a
generalization gap; they do not establish its cause. Final test evaluation
is recorded separately in the final report.

## Augmentation experiment

Optional training-only augmentation is enabled by `augment = true` in the
training configuration; omitted settings default to false. The separate
experiment preserves the baseline output directory:

    uv run traffic-sign-classifier train --config configs/augmentation.toml
    uv run python scripts/evaluate_validation.py --run-dir outputs/runs/augmentation-v1

See [augmentation experiment notes](docs/augmentation-v1.md) for settings,
controls, and results. Evaluation always uses deterministic preprocessing.

The completed ten-epoch experiment reached 91.26% validation accuracy and
0.880 macro F1. Class-27 recall improved to 71.7%, but several other classes
regressed. These are single-seed validation results, not final test results.
