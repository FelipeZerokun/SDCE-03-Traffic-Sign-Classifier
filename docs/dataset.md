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
- CSV columns include image paths, dimensions, bounding boxes,
  and class IDs.
- Image existence, readability, and annotation consistency
  have not yet been audited.

## Split policy

Preserve the supplied test set for final evaluation.

Create validation data from the training set, keeping images
from the same physical-sign track together.

Verify how track identifiers are encoded in this distribution
before implementing the split.

Save split assignments and the random seed for reproducibility.

## Preprocessing

Decisions about bounding-box cropping, resizing, normalization,
and augmentation will follow the training-data audit.

## Training-image audit results

The implemented audit completed successfully:

- 39,209 annotation records loaded.
- All referenced training images decoded successfully.
- All image dimensions matched their annotations.
- All images used RGB mode.
- All 43 expected classes were represented.
- Class counts ranged from 210 to 2,250 images.

The training data is imbalanced. Evaluation will include per-class
metrics alongside overall accuracy.

Class 33 contains 689 images. Track-level inspection is still needed
to establish group sizes and investigate incomplete sequences.

These checks do not yet establish bounding-box validity, absence of
duplicate images, or correctness of track grouping.

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

Image-content duplication across different filenames has not yet
been checked.