"""File I/O helpers for the experimental SSD weight offloader."""

from pathlib import Path


def read_exact_into(path: Path, destination: memoryview) -> None:
    """Read into an existing slot without allocating a layer-sized bytes object."""
    with path.open("rb", buffering=0) as source:
        offset = 0
        while offset < len(destination):
            count = source.readinto(destination[offset:])
            if not count:
                raise EOFError(f"Short SSD weight file: {path}, read {offset}/{len(destination)} bytes")
            offset += count
