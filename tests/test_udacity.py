"""Verify that conversion preserves pixels, labels, and supplied ordering."""

import pickle
import runpy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from traffic_sign_classifier.manifest import load_split_manifest


def test_udacity_conversion_is_lossless_and_preserves_partition(tmp_path: Path) -> None:
    prepare = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/prepare_udacity.py")
    )["prepare"]
    source = tmp_path / "source"
    source.mkdir()
    arrays = {}
    rng = np.random.default_rng(7)
    for filename in ("train.p", "valid.p", "test.p"):
        images = rng.integers(0, 256, (43, 32, 32, 3), dtype=np.uint8)
        labels = rng.permutation(43).astype(np.uint8)
        arrays[filename] = (images, labels)
        with (source / filename).open("wb") as stream:
            pickle.dump({"features": images, "labels": labels}, stream)
    output = tmp_path / "converted"
    prepare(source, output)
    split = load_split_manifest(output / "split.json", output / "Train.csv")
    for records, filename in (
        (split.training, "train.p"),
        (split.validation, "valid.p"),
    ):
        images, labels = arrays[filename]
        assert [row.class_id for row in records] == labels.tolist()
        for row, expected in zip(records, images, strict=True):
            with Image.open(output / row.image_path) as actual:
                np.testing.assert_array_equal(np.asarray(actual), expected)
    with pytest.raises(ValueError, match="already exists"):
        prepare(source, output)
