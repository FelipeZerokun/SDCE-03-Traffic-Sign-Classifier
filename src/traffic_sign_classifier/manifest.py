"""Validate saved dataset split manifests."""

import json
from collections.abc import Mapping
from pathlib import Path

from traffic_sign_classifier.dataset import (
    Annotation,
    read_annotations,
    training_track_key,
)
from traffic_sign_classifier.provenance import sha256_file
from traffic_sign_classifier.split import AnnotationSplit


def validate_manifest_source(
    manifest: Mapping[str, object],
    annotations_path: Path,
) -> None:
    """Check the manifest version and source annotation fingerprint."""
    version = manifest.get("schema_version")
    if type(version) is not int or version != 2:
        raise ValueError("Expected manifest schema version 2")

    expected_digest = manifest.get("annotations_sha256")
    if not isinstance(expected_digest, str):
        raise ValueError("Manifest is missing its annotation fingerprint")

    actual_digest = sha256_file(annotations_path)
    if expected_digest != actual_digest:
        raise ValueError("Manifest fingerprint does not match the annotation file")


def resolve_manifest_records(
    records: object,
    annotations_by_path: Mapping[str, Annotation],
) -> tuple[Annotation, ...]:
    """Validate manifest entries and resolve their source annotations."""
    if not isinstance(records, list) or not records:
        raise ValueError("Split records must be a non-empty list")

    resolved: list[Annotation] = []
    seen_paths: set[str] = set()

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index}: expected an object")

        path = record.get("path")
        if not isinstance(path, str):
            raise ValueError(f"Record {index}: expected a string path")

        if path in seen_paths:
            raise ValueError(f"Duplicate manifest path: {path}")

        if path not in annotations_by_path:
            raise ValueError(f"Unknown manifest path: {path}")

        annotation = annotations_by_path[path]
        class_id = record.get("class_id")
        track_id = record.get("track_id")

        if type(class_id) is not int or class_id != annotation.class_id:
            raise ValueError(f"{path}: manifest class does not match")

        expected_track = training_track_key(annotation)[1]
        if type(track_id) is not int or track_id != expected_track:
            raise ValueError(f"{path}: manifest track does not match")

        seen_paths.add(path)
        resolved.append(annotation)

    return tuple(resolved)


def load_split_manifest(
    manifest_path: Path,
    annotations_path: Path,
) -> AnnotationSplit:
    """Load a complete split with no shared image paths or tracks."""
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)

    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a JSON object")

    validate_manifest_source(manifest, annotations_path)

    annotations = read_annotations(annotations_path)
    annotations_by_path = {
        annotation.image_path.as_posix(): annotation for annotation in annotations
    }
    if len(annotations_by_path) != len(annotations):
        raise ValueError("Duplicate image paths in source annotations")

    training = resolve_manifest_records(manifest.get("training"), annotations_by_path)
    validation = resolve_manifest_records(
        manifest.get("validation"), annotations_by_path
    )

    training_paths = {item.image_path.as_posix() for item in training}
    validation_paths = {item.image_path.as_posix() for item in validation}

    if not training_paths.isdisjoint(validation_paths):
        raise ValueError("Training and validation share image paths")

    if training_paths | validation_paths != set(annotations_by_path):
        raise ValueError("Split does not cover all source annotations")

    training_tracks = {training_track_key(item) for item in training}
    validation_tracks = {training_track_key(item) for item in validation}

    if not training_tracks.isdisjoint(validation_tracks):
        raise ValueError("Training and validation share tracks")

    expected_classes = {item.class_id for item in annotations}
    if {item.class_id for item in training} != expected_classes or {
        item.class_id for item in validation
    } != expected_classes:
        raise ValueError("Every source class must appear in both splits")

    return AnnotationSplit(training=training, validation=validation)
