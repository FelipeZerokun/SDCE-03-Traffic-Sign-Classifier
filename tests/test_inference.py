from pathlib import Path

import pytest
import torch
from PIL import Image

from traffic_sign_classifier.cli import main
from traffic_sign_classifier.inference import PREPROCESSING, load_model, predict
from traffic_sign_classifier.model import BaselineCNN


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    model = BaselineCNN()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.layers[-1].bias[14] = 5
    path = tmp_path / "model.pt"
    torch.save(
        {
            "schema_version": 1,
            "architecture": "baseline_cnn_v1",
            "class_ids": list(range(43)),
            "preprocessing": PREPROCESSING,
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    return path


def test_prediction_uses_saved_weights_and_softmax(
    checkpoint: Path,
    tmp_path: Path,
) -> None:
    path = tmp_path / "sign.png"
    Image.new("L", (19, 27), 100).save(path)
    model, _ = load_model(checkpoint)
    first = predict(model, path, 43)
    assert first == predict(model, path, 43)
    assert first[0]["class_id"] == 14
    assert first[0]["class_name"] == "Stop"
    assert sum(float(row["score"]) for row in first) == pytest.approx(1)
    assert first[0]["score"] == pytest.approx(
        float(torch.tensor(5).exp() / (torch.tensor(5).exp() + 42))
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("architecture", "unknown"),
        ("class_ids", list(reversed(range(43)))),
        ("preprocessing", {**PREPROCESSING, "crop": True}),
    ],
)
def test_reject_incompatible_checkpoint(
    checkpoint: Path, key: str, value: object
) -> None:
    data = torch.load(checkpoint, weights_only=True)
    data[key] = value
    torch.save(data, checkpoint)
    with pytest.raises(ValueError, match=key):
        load_model(checkpoint)


def test_predict_invalid_k(checkpoint: Path, tmp_path: Path) -> None:
    model, _ = load_model(checkpoint)
    with pytest.raises(ValueError, match="top-k"):
        predict(model, tmp_path / "unused.png", 44)


def test_predict_cli_missing_image(checkpoint: Path, tmp_path: Path) -> None:
    assert (
        main(
            [
                "predict",
                "--checkpoint",
                str(checkpoint),
                "--image",
                str(tmp_path / "missing.png"),
            ]
        )
        == 1
    )
