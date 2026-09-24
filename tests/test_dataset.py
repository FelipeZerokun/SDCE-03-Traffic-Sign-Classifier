from pathlib import Path

import pytest

from traffic_sign_classifier.dataset import (
    parse_annotation,
    read_annotations,
    validate_annotation_header,
)

VALID_HEADER = "Width,Height,Roi.X1,Roi.Y1,Roi.X2,Roi.Y2,ClassId,Path"


def test_accepts_valid_header(tmp_path: Path) -> None:
    csv_path = tmp_path / "annotations.csv"
    csv_path.write_text(VALID_HEADER + "\n", encoding="utf-8")

    validate_annotation_header(csv_path)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("", "missing CSV header"),
        ("Width,Height\n", "missing required columns"),
        (VALID_HEADER + ",Path\n", "duplicate column names"),
    ],
)
def test_rejects_invalid_header(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    csv_path = tmp_path / "annotations.csv"
    csv_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        validate_annotation_header(csv_path)


@pytest.fixture
def annotation_row() -> dict[str, str]:
    return {
        "Width": "27",
        "Height": "26",
        "Roi.X1": "5",
        "Roi.Y1": "5",
        "Roi.X2": "22",
        "Roi.Y2": "20",
        "ClassId": "20",
        "Path": "Train/20/00020_00000_00000.png",
    }


def test_parses_annotation(annotation_row: dict[str, str]) -> None:
    annotation = parse_annotation(annotation_row)

    assert annotation.image_path == Path("Train/20/00020_00000_00000.png")
    assert annotation.class_id == 20
    assert annotation.width == 27
    assert annotation.height == 26
    assert (
        annotation.roi_x1,
        annotation.roi_y1,
        annotation.roi_x2,
        annotation.roi_y2,
    ) == (5, 5, 22, 20)


def test_rejects_non_integer_width(annotation_row: dict[str, str]) -> None:
    annotation_row["Width"] = "not-a-number"

    with pytest.raises(ValueError, match="Invalid annotation"):
        parse_annotation(annotation_row)


def test_rejects_missing_field(annotation_row: dict[str, str]) -> None:
    del annotation_row["ClassId"]

    with pytest.raises(ValueError, match="Missing annotation field: ClassId"):
        parse_annotation(annotation_row)


def test_rejects_empty_path(annotation_row: dict[str, str]) -> None:
    annotation_row["Path"] = " "

    with pytest.raises(ValueError, match="Path must not be empty"):
        parse_annotation(annotation_row)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("Width", "0"),
        ("Width", "-1"),
        ("Height", "0"),
        ("Height", "-1"),
    ],
)
def test_rejects_non_positive_dimensions(
    annotation_row: dict[str, str],
    field: str,
    value: str,
) -> None:
    annotation_row[field] = value

    with pytest.raises(ValueError, match="width and height must be positive"):
        parse_annotation(annotation_row)


@pytest.mark.parametrize("class_id", ["-1", "43"])
def test_rejects_out_of_range_class_id(
    annotation_row: dict[str, str],
    class_id: str,
) -> None:
    annotation_row["ClassId"] = class_id

    with pytest.raises(ValueError, match="Class ID must be between 0 and 42"):
        parse_annotation(annotation_row)


@pytest.mark.parametrize("class_id", [0, 42])
def test_accepts_boundary_class_ids(
    annotation_row: dict[str, str],
    class_id: int,
) -> None:
    annotation_row["ClassId"] = str(class_id)

    annotation = parse_annotation(annotation_row)

    assert annotation.class_id == class_id


def test_reads_annotations(tmp_path: Path) -> None:
    csv_path = tmp_path / "annotations.csv"
    csv_path.write_text(
        VALID_HEADER + "\n"
        "27,26,5,5,22,20,20,Train/20/example.png\n"
        "30,28,4,4,25,23,0,Train/0/example.png\n",
        encoding="utf-8",
    )

    annotations = read_annotations(csv_path)

    assert len(annotations) == 2
    assert annotations[0].class_id == 20
    assert annotations[1].class_id == 0


def test_rejects_header_only_file(tmp_path: Path) -> None:
    csv_path = tmp_path / "annotations.csv"
    csv_path.write_text(VALID_HEADER + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no annotation records"):
        read_annotations(csv_path)


def test_reports_invalid_record_location(tmp_path: Path) -> None:
    csv_path = tmp_path / "annotations.csv"
    csv_path.write_text(
        VALID_HEADER + "\n0,26,5,5,22,20,20,Train/20/example.png\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 2") as error:
        read_annotations(csv_path)

    assert str(csv_path) in str(error.value)


@pytest.mark.parametrize(
    "record",
    [
        "27,26,5,5,22,20,20",
        "27,26,5,5,22,20,20,Train/20/example.png,extra",
    ],
)
def test_rejects_mismatched_row_length(
    tmp_path: Path,
    record: str,
) -> None:
    csv_path = tmp_path / "annotations.csv"
    csv_path.write_text(VALID_HEADER + "\n" + record + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="row length does not match"):
        read_annotations(csv_path)
