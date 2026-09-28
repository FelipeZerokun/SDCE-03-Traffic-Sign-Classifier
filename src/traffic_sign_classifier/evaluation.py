"""Saved-model evaluation with test integrity checks and portable reports."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from traffic_sign_classifier.classes import CLASS_NAMES
from traffic_sign_classifier.config import load_dataset_config
from traffic_sign_classifier.data_loading import create_data_loader
from traffic_sign_classifier.dataset import Annotation, inspect_image, read_annotations
from traffic_sign_classifier.inference import load_model
from traffic_sign_classifier.manifest import load_split_manifest
from traffic_sign_classifier.provenance import sha256_file
from traffic_sign_classifier.torch_dataset import TrafficSignDataset


def fingerprint(path: Path) -> str:
    """Hash dimensions and decoded RGB pixels, not encoded file bytes."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return hashlib.sha256(
            rgb.width.to_bytes(8, "big") + rgb.height.to_bytes(8, "big") + rgb.tobytes()
        ).hexdigest()


def audit_test(
    training: list[Annotation], test: list[Annotation], root: Path
) -> dict[str, object]:
    """Audit the supplied test set and enumerate overlap with training source."""
    counts = Counter(item.class_id for item in test)
    if set(counts) != set(range(43)):
        raise ValueError("Final test evaluation requires all 43 classes")
    paths = [item.image_path for item in test]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate test annotation paths")
    if set(paths) & {item.image_path for item in training}:
        raise ValueError("Test paths overlap with training source")
    modes = Counter(inspect_image(item, root) for item in test)
    training_hashes: dict[str, list[str]] = {}
    for item in training:
        digest = fingerprint(root / item.image_path)
        training_hashes.setdefault(digest, []).append(item.image_path.as_posix())
    test_digests = [fingerprint(root / item.image_path) for item in test]
    test_hashes = Counter(test_digests)
    overlaps = [
        {
            "test_path": item.image_path.as_posix(),
            "training_source_paths": training_hashes[digest],
            "sha256": digest,
        }
        for item, digest in zip(test, test_digests, strict=True)
        if digest in training_hashes
    ]
    return {
        "images": len(test),
        "class_counts": dict(sorted(counts.items())),
        "image_modes": dict(modes),
        "training_source_images": len(training),
        "exact_cross_source_duplicate_groups": len(
            training_hashes.keys() & test_hashes.keys()
        ),
        "overlapping_test_images": len(overlaps),
        "overlaps": overlaps,
        "within_test_duplicate_groups": sum(
            count > 1 for count in test_hashes.values()
        ),
        "comparison": "original dimensions and decoded RGB pixels",
        "near_duplicates_checked": False,
    }


def metrics(actual: list[int], predicted: list[int]) -> dict[str, object]:
    """Compute metrics over all 43 classes with zero for undefined ratios."""
    if not actual or len(actual) != len(predicted):
        raise ValueError("Nonempty matching label lists are required")
    matrix = np.zeros((43, 43), dtype=np.int64)
    for truth, guess in zip(actual, predicted, strict=True):
        matrix[truth, guess] += 1
    per_class = []
    f1_total = 0.0
    for index, name in enumerate(CLASS_NAMES):
        correct = int(matrix[index, index])
        support = int(matrix[index].sum())
        guesses = int(matrix[:, index].sum())
        precision = correct / guesses if guesses else 0.0
        recall = correct / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        f1_total += f1
        per_class.append(
            {
                "class_id": index,
                "class_name": name,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return {
        "images": len(actual),
        "accuracy": float(matrix.trace() / matrix.sum()),
        "macro_f1": f1_total / 43,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def evaluate(
    checkpoint_path: Path,
    config_path: Path,
    manifest: Path,
    split_name: str,
    output: Path,
    device: str = "cpu",
) -> dict[str, object]:
    """Evaluate a frozen checkpoint without modifying weights or old reports."""
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    if split_name not in {"validation", "test"}:
        raise ValueError("Split must be validation or test")
    model, checkpoint = load_model(checkpoint_path, device)
    config = load_dataset_config(config_path)
    if sha256_file(config.train_annotations) != checkpoint.get("annotations_sha256"):
        raise ValueError("Training annotations do not match checkpoint")
    if sha256_file(manifest) != checkpoint.get("manifest_sha256"):
        raise ValueError("Split manifest does not match checkpoint")
    split = load_split_manifest(manifest, config.train_annotations)
    audit: dict[str, object] = {}
    annotations = list(split.validation)
    if split_name == "test":
        annotations = read_annotations(config.test_annotations)
        print(
            "Auditing test images and exact overlap with training/validation...",
            flush=True,
        )
        audit = audit_test(
            list(split.training) + list(split.validation), annotations, config.root
        )
    dataset = TrafficSignDataset(annotations, config.root)
    loader = create_data_loader(dataset, batch_size=128, shuffle=False, seed=42)
    actual: list[int] = []
    predicted: list[int] = []
    torch.set_num_threads(4)
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            if not torch.isfinite(logits).all():
                raise ValueError("Model produced non-finite scores")
            actual.extend(labels.tolist())
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
    report = {
        **metrics(actual, predicted),
        "split": split_name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "annotations_sha256": sha256_file(
            config.test_annotations
            if split_name == "test"
            else config.train_annotations
        ),
        "manifest_sha256": sha256_file(manifest),
        "device": device,
        "audit": audit,
        "predictions": [
            {"path": item.image_path.as_posix(), "actual": truth, "predicted": guess}
            for item, truth, guess in zip(annotations, actual, predicted, strict=True)
        ],
    }
    if split_name == "test":
        overlaps = audit["overlaps"]
        assert isinstance(overlaps, list)
        excluded = {row["test_path"] for row in overlaps}
        keep = [
            index
            for index, item in enumerate(annotations)
            if item.image_path.as_posix() not in excluded
        ]
        report["nonoverlapping_subset"] = (
            metrics(
                [actual[index] for index in keep], [predicted[index] for index in keep]
            )
            if keep
            else None
        )
        report["interpretation"] = (
            "Top-level metrics cover the full supplied test set. "
            "The nonoverlapping subset excludes exact RGB matches to training/validation; "
            "near-duplicate independence is not established."
        )
        print(f"Exact-overlap test images: {len(excluded)}", flush=True)
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    np.savetxt(
        output / "confusion_matrix.csv",
        np.array(report["confusion_matrix"]),
        fmt="%d",
        delimiter=",",
    )
    return report
