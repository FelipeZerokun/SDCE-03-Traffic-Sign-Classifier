"""Calculate fingerprints for dataset provenance."""

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file's contents."""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()
