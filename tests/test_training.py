from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from traffic_sign_classifier.model import BaselineCNN
from traffic_sign_classifier.training import load_training_config, run_epoch


def test_training_updates_weights_and_validation_preserves_them(tmp_path: Path) -> None:
    torch.manual_seed(7)
    torch.set_num_threads(2)
    model = BaselineCNN()
    images = torch.rand(5, 3, 32, 32)
    labels = torch.tensor([0, 1, 2, 3, 4])
    loader = DataLoader(TensorDataset(images, labels), batch_size=3)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    run_epoch(model, loader, torch.device("cpu"), optimizer)
    assert any(
        not torch.equal(before[key], value) for key, value in model.state_dict().items()
    )
    trained = {key: value.clone() for key, value in model.state_dict().items()}
    metrics = run_epoch(model, loader, torch.device("cpu"))
    assert not model.training
    with torch.no_grad():
        logits = model(images)
        assert logits.shape == (5, 43)
        assert metrics["loss"] == pytest.approx(
            torch.nn.functional.cross_entropy(logits, labels).item()
        )
        assert metrics["accuracy"] == pytest.approx(
            (logits.argmax(1) == labels).float().mean().item()
        )
    assert all(
        torch.equal(trained[key], value) for key, value in model.state_dict().items()
    )
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)
    restored = BaselineCNN()
    restored.load_state_dict(
        torch.load(checkpoint, weights_only=True)["model_state_dict"]
    )
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(images), logits)


@pytest.mark.parametrize(
    "replacement",
    [
        "epochs = 0",
        "batch_size = true",
        "learning_rate = nan",
        "seed = -1",
        'device = "invalid"',
        'augment = "true"',
        "augment = 1",
    ],
)
def test_invalid_training_settings(tmp_path: Path, replacement: str) -> None:
    settings = {
        "epochs": "epochs = 1",
        "batch_size": "batch_size = 2",
        "learning_rate": "learning_rate = 0.001",
        "seed": "seed = 42",
        "device": 'device = "cpu"',
    }
    settings[replacement.split(" = ")[0]] = replacement
    path = tmp_path / "training.toml"
    path.write_text(
        '[training]\ndataset_config = "dataset.toml"\nmanifest = "split.json"\noutput = "run"\n'
        + "\n".join(settings.values()),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="training\\."):
        load_training_config(path)


def test_augmentation_config_and_baseline_default() -> None:
    root = Path(__file__).resolve().parents[1]
    assert load_training_config(root / "configs/augmentation.toml").augment is True
    assert load_training_config(root / "configs/training.toml").augment is False
