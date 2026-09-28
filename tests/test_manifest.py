import json
from pathlib import Path

import pytest

from traffic_sign_classifier.dataset import Annotation
from traffic_sign_classifier.manifest import (
    load_split_manifest,
    resolve_manifest_records,
    validate_manifest_source,
)
from traffic_sign_classifier.provenance import sha256_file


def test_accepts_matching_source(tmp_path: Path) -> None:
    annotations_path = tmp_path / "Train.csv"
    annotations_path.write_bytes(b"original annotations")
    manifest = {
        "schema_version": 2,
        "annotations_sha256": sha256_file(annotations_path),
    }

    validate_manifest_source(manifest, annotations_path)


def test_rejects_changed_source(tmp_path: Path) -> None:
    annotations_path = tmp_path / "Train.csv"
    annotations_path.write_bytes(b"original annotations")
    manifest = {
        "schema_version": 2,
        "annotations_sha256": sha256_file(annotations_path),
    }
    annotations_path.write_bytes(b"modified annotations")

    with pytest.raises(ValueError, match="fingerprint does not match"):
        validate_manifest_source(manifest, annotations_path)


def test_rejects_unsupported_version(tmp_path: Path) -> None:
    manifest = {"schema_version": 1}

    with pytest.raises(ValueError, match="schema version 2"):
        validate_manifest_source(manifest, tmp_path / "Train.csv")


def test_rejects_missing_fingerprint(tmp_path: Path) -> None:
    manifest = {"schema_version": 2}

    with pytest.raises(ValueError, match="missing its annotation fingerprint"):
        validate_manifest_source(manifest, tmp_path / "Train.csv")


@pytest.fixture
def annotations_by_path() -> dict[str, Annotation]:
    path = "Train/20/00020_00007_00000.png"
    return {
        path: Annotation(
            image_path=Path(path),
            class_id=20,
            width=32,
            height=32,
            roi_x1=5,
            roi_y1=5,
            roi_x2=26,
            roi_y2=26,
        )
    }


def test_resolves_source_annotation(
    annotations_by_path: dict[str, Annotation],
) -> None:
    path = next(iter(annotations_by_path))
    records = [{"path": path, "class_id": 20, "track_id": 7}]

    resolved = resolve_manifest_records(records, annotations_by_path)

    assert resolved == (annotations_by_path[path],)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("path", "unknown.png", "Unknown manifest path"),
        ("class_id", 21, "manifest class does not match"),
        ("track_id", 8, "manifest track does not match"),
    ],
)
def test_rejects_inconsistent_record(
    annotations_by_path: dict[str, Annotation],
    field: str,
    value: str | int,
    message: str,
) -> None:
    record: dict[str, object] = {
        "path": next(iter(annotations_by_path)),
        "class_id": 20,
        "track_id": 7,
    }
    record[field] = value

    with pytest.raises(ValueError, match=message):
        resolve_manifest_records([record], annotations_by_path)


def test_rejects_duplicate_manifest_record(
    annotations_by_path: dict[str, Annotation],
) -> None:
    record = {
        "path": next(iter(annotations_by_path)),
        "class_id": 20,
        "track_id": 7,
    }

    with pytest.raises(ValueError, match="Duplicate manifest path"):
        resolve_manifest_records([record, record], annotations_by_path)


@pytest.mark.parametrize("records", [None, [], "invalid", [123]])
def test_rejects_invalid_record_structure(
    annotations_by_path: dict[str, Annotation],
    records: object,
) -> None:
    with pytest.raises(ValueError):
        resolve_manifest_records(records, annotations_by_path)


@pytest.fixture
def manifest_files(tmp_path: Path) -> tuple[Path, Path]:
    annotations_path = tmp_path / "Train.csv"
    paths = [
        f"Train/0/00000_{track:05d}_{frame:05d}.png"
        for track in range(2)
        for frame in range(2)
    ]
    annotations_path.write_text(
        "Width,Height,Roi.X1,Roi.Y1,Roi.X2,Roi.Y2,ClassId,Path\n"
        + "".join(f"32,32,5,5,26,26,0,{path}\n" for path in paths),
        encoding="utf-8",
    )

    records = [
        {"path": path, "class_id": 0, "track_id": index // 2}
        for index, path in enumerate(paths)
    ]
    manifest_path = tmp_path / "split.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "annotations_sha256": sha256_file(annotations_path),
                "validation_fraction": 0.5,
                "seed": 42,
                "training": records[:2],
                "validation": records[2:],
            }
        ),
        encoding="utf-8",
    )

    return manifest_path, annotations_path


def test_loads_complete_split(manifest_files: tuple[Path, Path]) -> None:
    manifest_path, annotations_path = manifest_files

    split = load_split_manifest(manifest_path, annotations_path)

    assert len(split.training) == 2
    assert len(split.validation) == 2
    assert split.training[0].image_path.name == "00000_00000_00000.png"
    assert split.validation[0].image_path.name == "00000_00001_00000.png"


@pytest.mark.parametrize(
    ("problem", "message"),
    [
        ("overlap", "share image paths"),
        ("missing", "does not cover"),
        ("track_leakage", "share tracks"),
    ],
)
def test_rejects_invalid_partition(
    manifest_files: tuple[Path, Path],
    problem: str,
    message: str,
) -> None:
    manifest_path, annotations_path = manifest_files
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if problem == "overlap":
        manifest["validation"].append(manifest["training"][0])
    elif problem == "missing":
        manifest["validation"].pop()
    else:
        manifest["training"][1], manifest["validation"][0] = (
            manifest["validation"][0],
            manifest["training"][1],
        )

    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_split_manifest(manifest_path, annotations_path)
