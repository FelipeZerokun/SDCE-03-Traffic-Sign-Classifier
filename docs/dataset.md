# Dataset

## Source

Dataset: German Traffic Sign Recognition Benchmark (GTSRB)

Downloaded distribution:
https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

Original benchmark documentation:
https://github.com/houbensebastian/GermanTrafficSignBenchmarks

Download date: 2026.09.24
Kaggle dataset version: Version 1 (314.36 MB)
License listed by Kaggle: CC0: Public Domain

## Local organization

- data/Train/: training images
- data/Train.csv: training annotations
- data/Test/: test images
- data/Test.csv: test annotations
- data/Meta/: reference sign images
- data/Meta.csv: reference metadata

The Udacity pickle files are not used by the new pipeline.

## Initial observations

- Train.csv contains 39,209 records.
- Test.csv contains 12,630 records.
- Training annotations contain 43 distinct class IDs.
- CSV columns include image paths, dimensions, bounding boxes, and class IDs.
- Training audit results and final test-audit status are recorded below.

## Split policy

Preserve the supplied test set for final evaluation.

Create validation data from the training set, keeping images
from the same physical-sign track together.

Track identifiers are parsed from the training filenames and validated
against the annotated class IDs.

Save split assignments and the random seed for reproducibility.

## Preprocessing

The implemented preprocessing is documented below; the augmentation experiment
is documented in [augmentation-v1.md](augmentation-v1.md).

## Training-image audit results

The implemented audit completed successfully:

- 39,209 annotation records loaded.
- All referenced training images decoded successfully.
- All image dimensions matched their annotations.
- All images used RGB mode.
- All 43 expected classes were represented.
- Class counts ranged from 210 to 2,250 images.

The training data is imbalanced. Evaluation includes per-class
metrics alongside overall accuracy.

Class 33 contains 689 images. The track-separated split was subsequently
implemented and validated, as documented below. The reason for incomplete
sequences was not established.

This initial image audit did not assess bounding-box validity, duplicate
content, or track grouping. Subsequent split validation and exact-duplicate
checks are documented below. Bounding-box validation remains deferred because
preprocessing uses full images.

## Baseline training/validation split

Settings:
- Requested validation fraction: 0.2
- Random seed: 42
- Selection performed separately within each class, by whole track

Results:
- Training: 31,379 images across 1,046 tracks
- Validation: 7,830 images across 261 tracks
- All 43 classes appear in both splits.
- No image paths or track identifiers overlap between splits.

Classes 0, 19, and 37 each have only one validation track.
Their per-class validation results therefore have limited coverage
of distinct physical signs.

The supplied test set was not used to create this split.

The subsequent exact image-content duplicate audit is recorded below.

## Baseline preprocessing

- Retain RGB color channels and the full image.
- Resize directly to 32 × 32 pixels using bilinear interpolation.
- Direct resizing may change the original aspect ratio.
- Convert to a float32 tensor with shape [3, 32, 32].
- Scale pixel values from [0, 255] to [0, 1].
- Use the same deterministic preprocessing for training, validation, and prediction.
- Do not apply augmentation in the initial baseline.
## Dataset preparation completion

Training and validation data can now be loaded as model-ready batches:

- Image tensors: float32, shape [batch_size, 3, 32, 32].
- Pixel values: [0, 1].
- Labels: int64, shape [batch_size].
- Training order is shuffled; validation order is fixed.
- Partial final batches are retained.

Real-batch smoke checks passed for both splits.

An exact duplicate audit checked all 39,209 training-source images using original dimensions and decoded RGB pixels. It found no duplicate groups.

Limitations:
- Near-duplicate content was not independently measured.
- Some classes have only one validation track.
- Bounding-box validation is deferred because preprocessing uses full images.
- Final test integrity checks were completed on 2026-09-28: all 12,630 images
  decoded as RGB, matched dimensions, and covered all 43 classes. No repeated
  test paths or within-test exact RGB duplicate groups were found. Eight test
  images exactly match training-source class-14 track 00023. Full supplied-set
  and exact-overlap-excluded metrics are reported separately in
  [the final report](final-report.md). Near-duplicate independence is unverified.
