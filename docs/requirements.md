# Project requirements

## Objective

Build a reproducible Python application that trains, evaluates, and
runs a convolutional neural network to classify cropped traffic-sign images.

## Scope

- Use an image-based distribution of the German Traffic Sign
  Recognition Benchmark (GTSRB), with its supplied annotations.
- Document the exact dataset source, version where available,
  license, and local file organization.
- Verify image files, annotations, and the expected 43 classes.
- Preserve the supplied test set for final evaluation.
- Create reproducible training and validation splits, keeping
  related image sequences together where sequence identifiers
  are available.
- Implement and document resizing, normalization, and augmentation.
- Apply random augmentation only to training data.
- Support training, evaluation, and prediction through a
  command-line interface.
- Support CPU execution and optional NVIDIA GPU acceleration.

Traffic-sign detection in full road scenes, lane-finding integration,
vehicle control, and C++ deployment are outside the initial scope.

## Functional requirements

- FR-01: Load images, labels, and class names, rejecting invalid inputs
  with clear error messages.
- FR-02: Keep training, validation, and test data separate. Use only
  training data to fit preprocessing statistics and model parameters.
- FR-03: Train a CNN using settings loaded from a configuration file.
- FR-04: Save a checkpoint with the model information, class mapping,
  and preprocessing settings needed for prediction.
- FR-05: Evaluate a saved model using overall accuracy, per-class
  precision and recall, and a confusion matrix.
- FR-06: Predict a class and top-k class scores for a new cropped image.

## Engineering requirements

- ER-01: Manage dependencies with uv and commit uv.lock.
- ER-02: Keep reusable application code inside the src package.
- ER-03: Test important behavior using pytest.
- ER-04: Pass Ruff formatting, linting, and mypy checks.
- ER-05: Record experiment settings, random seeds, dependency versions,
  and evaluation results. Document limits to reproducibility.
- ER-06: Keep datasets, checkpoints, and generated outputs out of Git.
- ER-07: Document installation, commands, results, and known limitations.
- ER-08: Run automated checks in GitHub Actions.

## Evaluation policy

Use validation results to select models and tune settings.
Reserve the test set for final evaluation after those decisions.

Set a numerical performance target after verifying the dataset,
reviewing the course rubric, and measuring a baseline.