# SPDX-License-Identifier: Apache-2.0
"""Source checks for 310P CPU-first Mamba state preparation."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODEL_STATE = ROOT / "vllm_ascend" / "_310p" / "worker" / "v2" / "model_state.py"
KERNEL = ROOT / "csrc" / "moe" / "preprocess_mamba_align_v310" / "op_kernel" / "preprocess_mamba_align_v310.cpp"
POSTPROCESS_KERNEL = (
    ROOT / "csrc" / "moe" / "postprocess_mamba_align_v310" / "op_kernel" / "postprocess_mamba_align_v310.cpp"
)


def test_preprocess_state_plans_on_cpu_without_custom_ops() -> None:
    source = MODEL_STATE.read_text(encoding="utf-8")
    start = source.index("    def preprocess_state(")
    end = source.index("    def postprocess_state(", start)
    preprocess = source[start:end]

    assert "_copy_mamba_state_from_cpu_plan(" in preprocess
    assert "input_batch.block_tables_np" in preprocess
    assert "input_batch.idx_mapping_np" in preprocess
    assert "torch.ops._C_ascend" not in preprocess
    assert "masked_select" not in preprocess
    assert "index_select" not in preprocess
    assert ".item()" not in preprocess


def test_mamba_copy_api_matches_vllm_b2f685834a() -> None:
    source = MODEL_STATE.read_text(encoding="utf-8")

    assert "from vllm.v1.worker.mamba_utils import get_mamba_groups" in source
    assert "self.model.get_mamba_state_copy_func()" in source
    assert "_get_mamba_spec_for_layer" not in source
    assert "validate_mamba_state_copy_funcs" not in source
    assert "get_mamba_state_copy_funcs" not in source


def test_preprocess_kernel_matches_upstream_state_transition() -> None:
    source = KERNEL.read_text(encoding="utf-8")

    assert "srcColGm.SetValue(reqIdx, oldStateIdx)" in source
    assert "numAccepted > 1 ? numAccepted - 1 : 0" in source
    assert "computedAfter + tilingData.mambaBlockSize - 1" in source
    assert "oldStateIdx >= 0 && oldStateIdx != newStateIdx" in source
    assert "numAcceptedTokensGm.SetValue(reqIdx, 1)" in source


def test_postprocess_state_keeps_upstream_order_on_cpu() -> None:
    source = MODEL_STATE.read_text(encoding="utf-8")
    start = source.index("    def postprocess_state(")
    postprocess = source[start:]

    assert "torch.ops._C_ascend" not in postprocess
    assert "self._num_accepted_tokens_cpu" in postprocess
    assert "self.recoverssm.commit_step(" in postprocess
    assert (
        postprocess.index("self._num_accepted_tokens_cpu")
        < postprocess.index("self.recoverssm.commit_step(")
        < postprocess.index("aligned_computed")
    )
    assert "masked_select" not in postprocess
    assert ".item()" not in postprocess


def test_postprocess_kernel_matches_upstream_align_decisions() -> None:
    source = POSTPROCESS_KERNEL.read_text(encoding="utf-8")

    assert "runningTokens = newComputed - accepted + 1" in source
    assert "alignedComputed < runningTokens" in source
    assert "bias = alignedComputed - runningTokens" in source
    assert "dstColumn = alignedComputed / tilingData.blockSize - 1" in source
    assert "numAcceptedGm.SetValue(req, 1)" in source
    assert "srcColumn == dstColumn && bias == 0" in source
