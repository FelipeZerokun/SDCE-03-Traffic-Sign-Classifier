import json
from pathlib import Path

import pytest

from traffic_sign_classifier.cli import main
from traffic_sign_classifier.provenance import sha256_file


def test_help_exits_successfully(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--help"])

    assert error.value.code == 0
    assert "audit" in capsys.readouterr().out


def test_reports_missing_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_config = tmp_path / "missing.toml"

    exit_code = main(["audit", "--config", str(missing_config)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert repr(str(missing_config)) in captured.err
    assert captured.out == ""


def test_split_command_creates_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    csv_path = tmp_path / "Train.csv"
    csv_path.write_text(
        "Width,Height,Roi.X1,Roi.Y1,Roi.X2,Roi.Y2,ClassId,Path\n"
        "32,32,5,5,26,26,0,Train/0/00000_00000_00000.png\n"
        "32,32,5,5,26,26,0,Train/0/00000_00001_00000.png\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "dataset.toml"
    config_path.write_text(
        "[data]\n"
        'root = "."\n'
        'train_annotations = "Train.csv"\n'
        'test_annotations = "Test.csv"\n'
        "[split]\n"
        "validation_fraction = 0.2\n"
        "seed = 42\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "splits" / "baseline.json"

    exit_code = main(
        [
            "split",
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    manifest = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(manifest["training"]) == 1
    assert len(manifest["validation"]) == 1
    assert manifest["seed"] == 42
    assert manifest["validation_fraction"] == 0.2

    captured = capsys.readouterr()
    assert "Manifest saved:" in captured.out
    assert captured.err == ""

    original_contents = output_path.read_bytes()
    assert (
        main(
            [
                "split",
                "--config",
                str(config_path),
                "--output",
                str(output_path),
            ]
        )
        == 1
    )
    assert output_path.read_bytes() == original_contents
    assert "Error:" in capsys.readouterr().err
    assert manifest["schema_version"] == 2
    assert manifest["annotations_sha256"] == sha256_file(csv_path)
