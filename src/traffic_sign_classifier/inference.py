"""Validated checkpoint loading and deterministic cropped-image prediction."""

from pathlib import Path
from pickle import UnpicklingError
from typing import Any

import torch
from PIL import Image

from traffic_sign_classifier.classes import CLASS_NAMES
from traffic_sign_classifier.model import BaselineCNN
from traffic_sign_classifier.preprocessing import preprocess_image

PREPROCESSING = {
    "color_mode": "RGB",
    "size": [32, 32],
    "resize": "PIL bilinear",
    "scale": 1 / 255,
    "crop": False,
    "augmentation": False,
}


def load_model(path: Path, device: str = "cpu") -> tuple[BaselineCNN, dict[str, Any]]:
    """Reject unsupported metadata instead of silently changing inference."""
    if device not in {"cpu", "cuda"}:
        raise ValueError("Device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must contain a metadata dictionary")
        for key, expected in (
            ("schema_version", 1),
            ("architecture", "baseline_cnn_v1"),
            ("class_ids", list(range(43))),
            ("preprocessing", PREPROCESSING),
        ):
            if checkpoint.get(key) != expected:
                raise ValueError(f"Unsupported checkpoint {key}")
        model = BaselineCNN()
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device).eval()
    except (KeyError, RuntimeError, TypeError, EOFError, UnpicklingError) as error:
        raise ValueError(f"Invalid checkpoint: {error}") from error
    return model, checkpoint


def predict(model: BaselineCNN, path: Path, top_k: int = 5) -> list[dict[str, object]]:
    """Return ranked softmax scores, which are not calibrated confidence."""
    if not 1 <= top_k <= 43:
        raise ValueError("top-k must be between 1 and 43")
    with Image.open(path) as image:
        tensor = preprocess_image(image).unsqueeze(0)
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        logits = model(tensor.to(device))
        if not torch.isfinite(logits).all():
            raise ValueError("Model produced non-finite scores")
        scores, indices = logits.softmax(dim=1)[0].topk(top_k)
    return [
        {
            "class_id": int(index),
            "class_name": CLASS_NAMES[int(index)],
            "score": float(score),
        }
        for score, index in zip(scores.tolist(), indices.tolist(), strict=True)
    ]
