from pathlib import Path

import pytest

from traffic_sign_classifier.config import load_dataset_config


def test_resolves_paths_relative_to_config(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "dataset.toml"
    config_path.write_text(
        "[data]\n"
        'root = "../data"\n'
        'train_annotations = "Train.csv"\n'
        'test_annotations = "Test.csv"\n'
        "[split]\n"
        "validation_fraction = 0.2\n"
        "seed = 42\n",
        encoding="utf-8",
    )

    config = load_dataset_config(config_path)

    expected_root = (tmp_path / "data").resolve()
    assert config.root == expected_root
    assert config.train_annotations == expected_root / "Train.csv"
    assert config.test_annotations == expected_root / "Test.csv"
    assert config.split.validation_fraction == 0.2
    assert config.split.seed == 42


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("", r"requires a \[data\] section"),
        ('[data]\nroot = ""\n', "data.root"),
        ("[data]\nroot = 123\n", "data.root"),
        ('[data]\nroot = "../data"\n', "data.train_annotations"),
        (
            '[data]\nroot = "../data"\ntrain_annotations = "Train.csv"\n',
            "data.test_annotations",
        ),
    ],
)
def test_rejects_invalid_config(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    config_path = tmp_path / "dataset.toml"
    config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_dataset_config(config_path)


@pytest.mark.parametrize(
    ("split_settings", "message"),
    [
        ("", r"requires a \[split\] section"),
        (
            "[split]\nvalidation_fraction = 0.0\nseed = 42\n",
            "split.validation_fraction",
        ),
        (
            "[split]\nvalidation_fraction = 1.0\nseed = 42\n",
            "split.validation_fraction",
        ),
        (
            "[split]\nvalidation_fraction = true\nseed = 42\n",
            "split.validation_fraction",
        ),
        (
            "[split]\nvalidation_fraction = 0.2\nseed = true\n",
            "split.seed",
        ),
        (
            "[split]\nvalidation_fraction = 0.2\n",
            "split.seed",
        ),
    ],
)
def test_rejects_invalid_split_settings(
    tmp_path: Path,
    split_settings: str,
    message: str,
) -> None:
    config_path = tmp_path / "dataset.toml"
    config_path.write_text(
        "[data]\n"
        'root = "../data"\n'
        'train_annotations = "Train.csv"\n'
        'test_annotations = "Test.csv"\n' + split_settings,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_dataset_config(config_path)
