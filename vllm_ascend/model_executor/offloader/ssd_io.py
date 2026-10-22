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


def storage_numel(shape: tuple[int, ...], stride: tuple[int, ...]) -> int:
    """Elements needed for a nonnegative-stride view, including storage gaps."""
    if len(shape) != len(stride) or any(step < 0 for step in stride):
        raise ValueError("SSD weight storage requires matching shapes and nonnegative strides")
    if any(dimension == 0 for dimension in shape):
        return 0
    return 1 + sum((dimension - 1) * step for dimension, step in zip(shape, stride))


def build_layer_layout(parameters, alignment=64):
    """Build one layer's byte offsets from (shape, stride, element_size)."""
    layout = []
    total = 0
    for shape, stride, element_size in parameters:
        total = (total + alignment - 1) // alignment * alignment
        size = storage_numel(shape, stride) * element_size
        layout.append((total, size))
        total += size
    return layout, total
