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