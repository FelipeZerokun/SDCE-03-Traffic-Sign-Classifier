"""Configured baseline training and validation-only checkpoint selection."""

import json
import math
import os
import platform
import random
import tomllib
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from traffic_sign_classifier.config import load_dataset_config
from traffic_sign_classifier.data_loading import create_data_loader
from traffic_sign_classifier.manifest import load_split_manifest
from traffic_sign_classifier.model import BaselineCNN
from traffic_sign_classifier.provenance import sha256_file
from traffic_sign_classifier.torch_dataset import TrafficSignDataset


@dataclass(frozen=True)
class TrainingConfig:
    dataset_config: Path
    manifest: Path
    output: Path
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int
    device: str
    augment: bool = False


def load_training_config(path: Path) -> TrainingConfig:
    """Validate training settings; resolve paths relative to the TOML file."""
    with path.open("rb") as file:
        settings = tomllib.load(file).get("training")
    if not isinstance(settings, dict):
        raise ValueError("Configuration requires a [training] section")
    paths: dict[str, Path] = {}
    for key in ("dataset_config", "manifest", "output"):
        value = settings.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"training.{key} must be a non-empty string")
        paths[key] = (path.resolve().parent / value).resolve()
    integers: dict[str, int] = {}
    for key in ("epochs", "batch_size", "seed"):
        value = settings.get(key)
        minimum = 0 if key == "seed" else 1
        if type(value) is not int or not minimum <= value <= 2**32 - 1:
            raise ValueError(f"training.{key} must be an integer >= {minimum}")
        integers[key] = value
    rate = settings.get("learning_rate")
    if (
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not math.isfinite(rate)
        or rate <= 0
    ):
        raise ValueError("training.learning_rate must be finite and positive")
    device = settings.get("device")
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("training.device must be auto, cpu, or cuda")
    augment = settings.get("augment", False)
    if not isinstance(augment, bool):
        raise ValueError("training.augment must be a boolean")
    return TrainingConfig(
        dataset_config=paths["dataset_config"],
        manifest=paths["manifest"],
        output=paths["output"],
        epochs=integers["epochs"],
        batch_size=integers["batch_size"],
        seed=integers["seed"],
        learning_rate=float(rate),
        device=str(device),
        augment=augment,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """Compute sample-weighted loss and accuracy, including partial batches."""
    model.train(optimizer is not None)
    total_loss = 0.0
    correct = 0
    count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss).item():
                raise ValueError("Training or validation loss is not finite")
            if optimizer is not None:
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
            size = labels.size(0)
            total_loss += loss.item() * size
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            count += size
    if count == 0:
        raise ValueError("Cannot run an epoch with an empty dataset")
    return {"loss": total_loss / count, "accuracy": correct / count}


def train(config: TrainingConfig) -> Path:
    """Train a baseline, preserving the best validation-accuracy checkpoint."""
    device_name = config.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    dataset_config = load_dataset_config(config.dataset_config)
    split = load_split_manifest(config.manifest, dataset_config.train_annotations)
    if {item.class_id for item in split.training} != set(range(43)):
        raise ValueError("Baseline training requires all 43 GTSRB classes")

    # Refuse to overwrite any previous or interrupted experiment.
    config.output.mkdir(parents=True, exist_ok=False)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    loaders = [
        create_data_loader(
            TrafficSignDataset(records, dataset_config.root, augment=augment),
            batch_size=config.batch_size,
            shuffle=shuffle,
            seed=config.seed,
        )
        for records, shuffle, augment in (
            (split.training, True, config.augment),
            (split.validation, False, False),
        )
    ]
    model = BaselineCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    metadata = {
        "schema_version": 1,
        "architecture": "baseline_cnn_v1",
        "class_ids": list(range(43)),
        "preprocessing": {
            "color_mode": "RGB",
            "size": [32, 32],
            "resize": "PIL bilinear",
            "scale": 1 / 255,
            "crop": False,
            "augmentation": False,
        },
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "training_augmentation": {
            "enabled": config.augment,
            "type": "RandomAffine",
            "degrees": 0,
            "translate": [0.05, 0.05],
            "scale": [0.9, 1.1],
            "interpolation": "bilinear",
            "fill": 0,
            "stage": "after deterministic preprocessing, training only",
        },
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0)
        if device.type == "cuda"
        else "CPU",
        "versions": {
            name: version(name) for name in ("torch", "torchvision", "numpy", "pillow")
        },
        "python": platform.python_version(),
        "annotations_sha256": sha256_file(dataset_config.train_annotations),
        "manifest_sha256": sha256_file(config.manifest),
        "training_images": len(split.training),
        "validation_images": len(split.validation),
        "selection_metric": "validation accuracy (first epoch wins ties)",
    }
    (config.output / "run.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    best_accuracy = -1.0
    history: list[dict[str, object]] = []
    checkpoint = config.output / "best.pt"
    print(f"Training on {metadata['device_name']}", flush=True)
    for epoch in range(1, config.epochs + 1):
        training = run_epoch(model, loaders[0], device, optimizer)
        validation = run_epoch(model, loaders[1], device)
        history.append({"epoch": epoch, "training": training, "validation": validation})
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            temporary = config.output / "best.tmp"
            torch.save(
                {
                    **metadata,
                    "epoch": epoch,
                    "validation": validation,
                    "model_state_dict": model.state_dict(),
                },
                temporary,
            )
            temporary.replace(checkpoint)
        (config.output / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(
            f"Epoch {epoch}/{config.epochs}: train loss={training['loss']:.4f} "
            f"accuracy={training['accuracy']:.2%}; validation loss={validation['loss']:.4f} "
            f"accuracy={validation['accuracy']:.2%}",
            flush=True,
        )
    print(
        f"Best checkpoint: {checkpoint} ({best_accuracy:.2%} validation accuracy)",
        flush=True,
    )
    return checkpoint
