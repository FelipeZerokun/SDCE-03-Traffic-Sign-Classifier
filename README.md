# Traffic Sign Classifier

A modern Python rebuild of the Udacity Self-Driving Car Engineer
traffic-sign classification project.

## Status

Project foundation configured. Training, evaluation, and prediction
are not implemented yet.

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

Tests will be added alongside application behavior.

## Original implementation

The original project is available at:
https://github.com/FelipeZerokun/SDCE-03-Traffic-Sign-Classifier

The local legacy/ directory is an ignored reference copy.
Datasets belong in data/ and generated results in outputs/.

## Audit the training dataset

    uv run traffic-sign-classifier audit --config configs/dataset.toml