# SPDX-License-Identifier: Apache-2.0
"""Source-level checks for the 310P Mamba align preprocess custom op."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODEL_STATE = ROOT / "vllm_ascend" / "_310p" / "worker" / "v2" / "model_state.py"
KERNEL = ROOT / "csrc" / "moe" / "preprocess_mamba_align_v310" / "op_kernel" / "preprocess_mamba_align_v310.cpp"
POSTPROCESS_KERNEL = (
    ROOT
    / "csrc"
    / "moe"
    / "postprocess_mamba_align_v310"
    / "op_kernel"
    / "postprocess_mamba_align_v310.cpp"
)


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


def test_postprocess_state_dispatches_fused_ascendc_align_op() -> None:
    source = MODEL_STATE.read_text()
    start = source.index("    def postprocess_state(")
    postprocess = source[start:]

    assert "torch.ops._C_ascend.postprocess_mamba_align_310(" in postprocess
    assert "torch.ops._C_ascend.update_mamba_num_accepted_310(" in postprocess
    assert "self.recoverssm.commit_step(" in postprocess
    assert postprocess.index("update_mamba_num_accepted_310") < postprocess.index(
        "self.recoverssm.commit_step("
    ) < postprocess.index("postprocess_mamba_align_310")
    assert "masked_select" not in postprocess
    assert ".item()" not in postprocess


def test_postprocess_kernel_matches_upstream_align_decisions() -> None:
    source = POSTPROCESS_KERNEL.read_text()

    assert "runningTokens = newComputed - accepted + 1" in source
    assert "alignedComputed < runningTokens" in source
    assert "bias = alignedComputed - runningTokens" in source
    assert "dstColumn = alignedComputed / tilingData.blockSize - 1" in source
    assert "numAcceptedGm.SetValue(req, 1)" in source
    assert "srcColumn == dstColumn && bias == 0" in source
