# SPDX-License-Identifier: Apache-2.0
"""Source checks for the 310P MRV2 MRoPE host preparation path."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ROPE = ROOT / "vllm_ascend" / "_310p" / "worker" / "v2" / "rope.py"


def test_mrope_positions_use_mrv1_pinned_buffer_and_writer() -> None:
    source = ROPE.read_text(encoding="utf-8")

    assert "pin_memory=is_pin_memory_available()" in source
    assert "self.positions_np = self.positions_cpu.numpy()" in source
    assert "MRotaryEmbedding.get_next_input_positions_tensor(" in source
    assert "decode_positions = torch.arange(" not in source
    assert "self.positions.copy_(self.positions_cpu, non_blocking=True)" in source
    assert "self.positions[:, :num_tokens_after_padding].copy_(" not in source
    assert "non_blocking=True" in source
