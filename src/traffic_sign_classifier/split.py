"""Create reproducible training and validation track assignments."""

import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from traffic_sign_classifier.dataset import Annotation, training_track_key

TrackKey = tuple[int, int]


@dataclass(frozen=True)
class AnnotationSplit:
    training: tuple[Annotation, ...]
    validation: tuple[Annotation, ...]


def split_annotations(
    annotations: Iterable[Annotation],
    validation_fraction: float,
    seed: int,
) -> AnnotationSplit:
    """Assign annotations to splits without separating their tracks."""
    records = sorted(
        annotations,
        key=lambda annotation: annotation.image_path.as_posix(),
    )

    paths = [annotation.image_path for annotation in records]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate image paths in annotations")

    keyed_records = [
        (annotation, training_track_key(annotation)) for annotation in records
    ]

    training_tracks, validation_tracks = split_tracks(
        (key for _, key in keyed_records),
        validation_fraction,
        seed,
    )

    return AnnotationSplit(
        training=tuple(
            annotation for annotation, key in keyed_records if key in training_tracks
        ),
        validation=tuple(
            annotation for annotation, key in keyed_records if key in validation_tracks
        ),
    )


def split_tracks(
    track_keys: Iterable[TrackKey],
    validation_fraction: float,
    seed: int,
) -> tuple[set[TrackKey], set[TrackKey]]:
    """Return disjoint training and validation track sets."""
    if not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be between 0 and 1")

    tracks_by_class: dict[int, set[TrackKey]] = defaultdict(set)
    for key in track_keys:
        tracks_by_class[key[0]].add(key)

    if not tracks_by_class:
        raise ValueError("Cannot split an empty collection of tracks")

    generator = random.Random(seed)
    training: set[TrackKey] = set()
    validation: set[TrackKey] = set()

    for class_id in sorted(tracks_by_class):
        tracks = sorted(tracks_by_class[class_id])
        if len(tracks) < 2:
            raise ValueError(f"Class {class_id} needs at least two tracks")

        generator.shuffle(tracks)

        validation_count = math.floor(len(tracks) * validation_fraction + 0.5)
        validation_count = max(1, min(validation_count, len(tracks) - 1))

        validation.update(tracks[:validation_count])
        training.update(tracks[validation_count:])

    return training, validation


def save_split_manifest(
    split: AnnotationSplit,
    output_path: Path,
    validation_fraction: float,
    seed: int,
    *,
    annotations_sha256: str,
) -> None:
    """Save split assignments without overwriting an existing manifest."""

    def records(
        annotations: tuple[Annotation, ...],
    ) -> list[dict[str, str | int]]:
        return [
            {
                "path": annotation.image_path.as_posix(),
                "class_id": annotation.class_id,
                "track_id": training_track_key(annotation)[1],
            }
            for annotation in annotations
        ]

    manifest = {
        "schema_version": 2,
        "annotations_sha256": annotations_sha256,
        "validation_fraction": validation_fraction,
        "seed": seed,
        "training": records(split.training),
        "validation": records(split.validation),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("x", encoding="utf-8", newline="\n") as file:
        json.dump(manifest, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
