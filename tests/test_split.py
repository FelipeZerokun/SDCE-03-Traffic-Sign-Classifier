import json
from pathlib import Path

import pytest

from traffic_sign_classifier.dataset import Annotation, training_track_key
from traffic_sign_classifier.split import (
    save_split_manifest,
    split_annotations,
    split_tracks,
)


def test_split_preserves_tracks_and_separates_groups() -> None:
    tracks = {(class_id, track_id) for class_id in (0, 1) for track_id in range(10)}

    training, validation = split_tracks(tracks, 0.2, 42)

    assert training.isdisjoint(validation)
    assert training | validation == tracks

    for class_id in (0, 1):
        assert sum(key[0] == class_id for key in validation) == 2
        assert sum(key[0] == class_id for key in training) == 8


def test_split_is_reproducible_and_order_independent() -> None:
    tracks = [(0, track_id) for track_id in range(10)]

    expected = split_tracks(tracks, 0.2, 42)

    assert split_tracks(tracks, 0.2, 42) == expected
    assert split_tracks(reversed(tracks), 0.2, 42) == expected


def test_duplicate_tracks_do_not_change_split() -> None:
    tracks = [(0, track_id) for track_id in range(10)]

    assert split_tracks(tracks + tracks, 0.2, 42) == split_tracks(tracks, 0.2, 42)


@pytest.mark.parametrize("fraction", [0.01, 0.99])
def test_small_class_remains_in_both_splits(fraction: float) -> None:
    training, validation = split_tracks([(0, 0), (0, 1)], fraction, 42)

    assert len(training) == 1
    assert len(validation) == 1


def test_rejects_class_with_one_track() -> None:
    with pytest.raises(ValueError, match="at least two tracks"):
        split_tracks([(0, 0)], 0.2, 42)


def test_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty collection"):
        split_tracks([], 0.2, 42)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.1])
def test_rejects_invalid_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        split_tracks([(0, 0), (0, 1)], fraction, 42)


@pytest.fixture
def annotations() -> list[Annotation]:
    return [
        Annotation(
            image_path=Path(
                f"Train/{class_id}/{class_id:05d}_{track_id:05d}_{frame_id:05d}.png"
            ),
            class_id=class_id,
            width=32,
            height=32,
            roi_x1=5,
            roi_y1=5,
            roi_x2=26,
            roi_y2=26,
        )
        for class_id in (0, 1)
        for track_id in range(5)
        for frame_id in range(3)
    ]


def test_assigns_all_images_without_track_overlap(
    annotations: list[Annotation],
) -> None:
    result = split_annotations(annotations, 0.2, 42)

    assert len(result.training) == 24
    assert len(result.validation) == 6

    training_paths = {item.image_path for item in result.training}
    validation_paths = {item.image_path for item in result.validation}

    assert training_paths.isdisjoint(validation_paths)
    assert training_paths | validation_paths == {
        item.image_path for item in annotations
    }

    training_tracks = {training_track_key(item) for item in result.training}
    validation_tracks = {training_track_key(item) for item in result.validation}

    assert training_tracks.isdisjoint(validation_tracks)


def test_annotation_split_is_order_independent(
    annotations: list[Annotation],
) -> None:
    assert split_annotations(annotations, 0.2, 42) == split_annotations(
        reversed(annotations), 0.2, 42
    )


def test_rejects_duplicate_image_paths(
    annotations: list[Annotation],
) -> None:
    with pytest.raises(ValueError, match="Duplicate image paths"):
        split_annotations(annotations + [annotations[0]], 0.2, 42)


def test_saves_split_manifest(
    tmp_path: Path,
    annotations: list[Annotation],
) -> None:
    split = split_annotations(annotations, 0.2, 42)
    output_path = tmp_path / "splits" / "baseline.json"

    save_split_manifest(
        split,
        output_path,
        0.2,
        42,
        annotations_sha256="a" * 64,
    )

    manifest = json.loads(output_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 2
    assert manifest["annotations_sha256"] == "a" * 64
    assert manifest["validation_fraction"] == 0.2
    assert manifest["seed"] == 42

    for name, expected in (
        ("training", split.training),
        ("validation", split.validation),
    ):
        assert manifest[name] == [
            {
                "path": item.image_path.as_posix(),
                "class_id": item.class_id,
                "track_id": training_track_key(item)[1],
            }
            for item in expected
        ]


def test_manifest_does_not_overwrite_existing_file(
    tmp_path: Path,
    annotations: list[Annotation],
) -> None:
    split = split_annotations(annotations, 0.2, 42)
    output_path = tmp_path / "existing.json"
    output_path.write_text("original contents", encoding="utf-8")

    with pytest.raises(FileExistsError):
        save_split_manifest(
            split,
            output_path,
            0.2,
            42,
            annotations_sha256="a" * 64,
        )

    assert output_path.read_text(encoding="utf-8") == "original contents"
