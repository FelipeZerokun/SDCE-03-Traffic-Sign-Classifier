"""Audit dataset annotations and image files."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from traffic_sign_classifier.dataset import (
    inspect_image,
    read_annotations,
    training_track_key,
)


@dataclass(frozen=True)
class AuditSummary:
    image_count: int
    class_counts: dict[int, int]
    mode_counts: dict[str, int]
    track_counts: dict[tuple[int, int], int]


def audit_dataset(csv_path: Path, data_root: Path) -> AuditSummary:
    """Inspect training images and summarize classes, modes, and tracks."""
    annotations = read_annotations(csv_path)
    class_counts: Counter[int] = Counter()
    mode_counts: Counter[str] = Counter()
    track_counts: Counter[tuple[int, int]] = Counter()

    for annotation in annotations:
        mode = inspect_image(annotation, data_root)
        class_counts[annotation.class_id] += 1
        mode_counts[mode] += 1
        track_counts[training_track_key(annotation)] += 1

    return AuditSummary(
        image_count=len(annotations),
        class_counts=dict(sorted(class_counts.items())),
        mode_counts=dict(sorted(mode_counts.items())),
        track_counts=dict(sorted(track_counts.items())),
    )
