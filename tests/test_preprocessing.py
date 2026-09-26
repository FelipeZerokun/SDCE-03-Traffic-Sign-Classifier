import torch
from PIL import Image

from traffic_sign_classifier.preprocessing import preprocess_image


def test_returns_expected_shape_dtype_and_rane() -> None:
    image = Image.new("RGB", (50, 20), color=(0, 128, 255))

    tensor = preprocess_image(image)

    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype == torch.float32
    assert tensor.device.type == "cpu"
    assert tensor.min().item() >= 0.0
    assert tensor.max().item() <= 1.0


def test_preservers_rgb_channel_order_and_scales_pixels() -> None:
    image = Image.new("RGB", (50, 20), color=(0, 128, 255))

    tensor = preprocess_image(image)

    expected = (
        torch.tensor(
            [0.0, 128.0 / 255.0, 1.0],
            dtype=torch.float32,
        )
        .view(3, 1, 1)
        .expand(3, 32, 32)
    )

    torch.testing.assert_close(tensor, expected)


def test_converts_grayscale_to_three_channels() -> None:
    image = Image.new("L", (50, 20), color=128)

    tensor = preprocess_image(image)

    expected = torch.full((3, 32, 32), 128.0 / 255.0)
    torch.testing.assert_close(tensor, expected)


def test_is_repeatable_and_preserves_input() -> None:
    image = Image.new("RGB", (50, 20), color=(10, 20, 30))
    original_pixels = image.tobytes()

    first = preprocess_image(image)
    second = preprocess_image(image)

    assert torch.equal(first, second)
    assert image.size == (50, 20)
    assert image.mode == "RGB"
    assert image.tobytes() == original_pixels
