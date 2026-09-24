from pathlib import Path

import pytest
from PIL import Image

from traffic_sign_classifier.audit import audit_dataset


@pytest.fixture
def small_dataset(tmp_path: Path) -> Path:
    Image.new("RGB", (27, 26)).save(tmp_path / "00020_00000_00000.png")
    Image.new("L", (30, 28)).save(tmp_path / "00000_00000_00000.png")

    csv_path = tmp_path / "Train.csv"
    csv_path.write_text(
        "Width,Height,Roi.X1,Roi.Y1,Roi.X2,Roi.Y2,ClassId,Path\n"
        "27,26,5,5,22,20,20,00020_00000_00000.png\n"
        "30,28,4,4,25,23,0,00000_00000_00000.png\n",
        encoding="utf-8",
    )

    return tmp_path


def test_summarizes_dataset(small_dataset: Path) -> None:
    summary = audit_dataset(
        small_dataset / "Train.csv",
        small_dataset,
    )

    assert summary.image_count == 2
    assert summary.class_counts == {0: 1, 20: 1}
    assert summary.mode_counts == {"L": 1, "RGB": 1}
    assert summary.track_counts == {(20, 0): 1, (0, 0): 1}


def test_audit_fails_when_image_is_missing(small_dataset: Path) -> None:
    (small_dataset / "00000_00000_00000.png").unlink()

    with pytest.raises(ValueError, match="cannot read image"):
        audit_dataset(
            small_dataset / "Train.csv",
            small_dataset,
        )
