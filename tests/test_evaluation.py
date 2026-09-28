import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from traffic_sign_classifier.dataset import Annotation
from traffic_sign_classifier.evaluation import audit_test, evaluate, metrics
from traffic_sign_classifier.inference import PREPROCESSING
from traffic_sign_classifier.model import BaselineCNN
from traffic_sign_classifier.provenance import sha256_file


def test_metrics_include_absent_classes_and_correct_denominators() -> None:
    result = metrics([0, 0, 1], [0, 1, 1])
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx((2 / 3 + 2 / 3) / 43)
    assert result["per_class"][0]["precision"] == 1
    assert result["per_class"][0]["recall"] == 0.5
    assert result["confusion_matrix"][0][1] == 1


def test_test_audit_detects_pixels_despite_different_filenames(tmp_path: Path) -> None:
    test = []
    for index in range(43):
        path = Path(f"test-{index}.png")
        Image.new("RGB", (4, 4), (index, 0, 0)).save(tmp_path / path)
        test.append(Annotation(path, index, 4, 4, 0, 0, 3, 3))
    train_path = Path("training.png")
    Image.new("RGB", (4, 4), (0, 0, 0)).save(tmp_path / train_path)
    training = [Annotation(train_path, 0, 4, 4, 0, 0, 3, 3)]
    overlapping = audit_test(training, test, tmp_path)
    assert overlapping["overlapping_test_images"] == 1
    assert overlapping["overlaps"][0]["test_path"] == "test-0.png"
    Image.new("RGB", (4, 4), (255, 0, 0)).save(tmp_path / train_path)
    result = audit_test(training, test, tmp_path)
    assert result["images"] == 43
    assert result["exact_cross_source_duplicate_groups"] == 0


def test_evaluate_preserves_checkpoint_and_refuses_stale_source(tmp_path: Path) -> None:
    annotations = tmp_path / "Train.csv"
    annotations.write_text(
        "Width,Height,Roi.X1,Roi.Y1,Roi.X2,Roi.Y2,ClassId,Path\n"
        "4,4,0,0,3,3,0,00000_00000_00000.png\n"
        "4,4,0,0,3,3,0,00000_00001_00000.png\n",
        encoding="utf-8",
    )
    Image.new("RGB", (4, 4)).save(tmp_path / "00000_00001_00000.png")
    config = tmp_path / "dataset.toml"
    config.write_text(
        '[data]\nroot = "."\ntrain_annotations = "Train.csv"\n'
        'test_annotations = "missing-test.csv"\n[split]\n'
        "validation_fraction = 0.5\nseed = 42\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "split.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "annotations_sha256": sha256_file(annotations),
                "training": [
                    {"path": "00000_00000_00000.png", "class_id": 0, "track_id": 0}
                ],
                "validation": [
                    {"path": "00000_00001_00000.png", "class_id": 0, "track_id": 1}
                ],
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "model.pt"
    model = BaselineCNN()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    torch.save(
        {
            "schema_version": 1,
            "architecture": "baseline_cnn_v1",
            "class_ids": list(range(43)),
            "preprocessing": PREPROCESSING,
            "model_state_dict": model.state_dict(),
            "epoch": 1,
            "annotations_sha256": sha256_file(annotations),
            "manifest_sha256": sha256_file(manifest),
        },
        checkpoint,
    )
    digest = sha256_file(checkpoint)
    output = tmp_path / "evaluation"
    report = evaluate(checkpoint, config, manifest, "validation", output)
    assert report["accuracy"] == 1
    assert sha256_file(checkpoint) == digest
    assert json.loads((output / "report.json").read_text())["images"] == 1
    assert (output / "confusion_matrix.csv").exists()
    with pytest.raises(ValueError, match="Output already exists"):
        evaluate(checkpoint, config, manifest, "validation", output)
    annotations.write_text(annotations.read_text() + "\n")
    with pytest.raises(ValueError, match="annotations do not match"):
        evaluate(checkpoint, config, manifest, "validation", tmp_path / "other")
