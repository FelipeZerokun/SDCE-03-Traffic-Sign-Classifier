from pathlib import Path

import pytest

from traffic_sign_classifier.dataset import validate_annotation_header

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
