"""Load application configuration."""

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SplitConfig:
    validation_fraction: float
    seed: int


@dataclass(frozen=True)
class DatasetConfig:
    root: Path
    train_annotations: Path
    test_annotations: Path
    split: SplitConfig


def load_dataset_config(config_path: Path) -> DatasetConfig:
    """Load dataset paths, resolving them relative to the config file."""
    with config_path.open("rb") as file:
        contents = tomllib.load(file)

    data = contents.get("data")
    if not isinstance(data, dict):
        raise ValueError("Configuration requires a [data] section")

    paths: dict[str, Path] = {}
    for key in ("root", "train_annotations", "test_annotations"):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"data.{key} must be a non-empty string")
        paths[key] = Path(value)

    root = (config_path.resolve().parent / paths["root"]).resolve()

    split = contents.get("split")
    if not isinstance(split, dict):
        raise ValueError("Configuration requires a [split] section")

    fraction = split.get("validation_fraction")
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not 0 < fraction < 1
    ):
        raise ValueError("split.validation_fraction must be a number between 0 and 1")

    seed = split.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("split.seed must be an integer")

    split_config = SplitConfig(
        validation_fraction=float(fraction),
        seed=seed,
    )

    return DatasetConfig(
        root=root,
        train_annotations=(root / paths["train_annotations"]).resolve(),
        test_annotations=(root / paths["test_annotations"]).resolve(),
        split=split_config,
    )
