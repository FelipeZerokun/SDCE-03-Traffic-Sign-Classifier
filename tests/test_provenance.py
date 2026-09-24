from pathlib import Path

from traffic_sign_classifier.provenance import sha256_file


def test_sha256_matches_known_value(tmp_path: Path) -> None:
    path = tmp_path / "example.bin"
    path.write_bytes(b"abc")

    assert sha256_file(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_sha256_depends_on_contents_not_filename(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_bytes(b"same contents")
    second.write_bytes(b"same contents")

    assert sha256_file(first) == sha256_file(second)

    second.write_bytes(b"changed contents")

    assert sha256_file(first) != sha256_file(second)
