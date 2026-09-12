# SPDX-License-Identifier: Apache-2.0
"""Source-level checks for the 310P Mamba align preprocess custom op."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODEL_STATE = ROOT / "vllm_ascend" / "_310p" / "worker" / "v2" / "model_state.py"
KERNEL = ROOT / "csrc" / "moe" / "preprocess_mamba_align_v310" / "op_kernel" / "preprocess_mamba_align_v310.cpp"


def test_preprocess_state_dispatches_ascendc_without_torch_small_ops() -> None:
    source = MODEL_STATE.read_text()
    start = source.index("    def preprocess_state(")
    end = source.index("    def postprocess_state(", start)
    preprocess = source[start:end]

    assert "torch.ops._C_ascend.preprocess_mamba_align_310(" in preprocess
    assert "torch.ops._C_ascend.precopy_mamba_align_310(" in preprocess
    assert "_ensure_align_ctx(" in preprocess
    assert "masked_select" not in preprocess
    assert "index_select" not in preprocess
    assert ".item()" not in preprocess


def test_preprocess_kernel_matches_upstream_state_transition() -> None:
    source = KERNEL.read_text()

    assert "srcColGm.SetValue(reqIdx, oldStateIdx)" in source
    assert "numAccepted > 1 ? numAccepted - 1 : 0" in source
    assert "computedAfter + tilingData.mambaBlockSize - 1" in source
    assert "oldStateIdx >= 0 && oldStateIdx != newStateIdx" in source
    assert "numAcceptedTokensGm.SetValue(reqIdx, 1)" in source
