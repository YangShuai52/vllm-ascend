# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""310P-specific speculator for MTP eager mode.

310P does not have Triton, so the upstream Triton kernels used by
AutoRegressiveSpeculator (prepare_prefill_inputs, prepare_decode_inputs,
update_draft_inputs) are replaced with CPU/NumPy fallbacks via monkey-patch.

Additionally:
- hf_overrides may not be a dict for W8A8 models without quantization_config
  in config.json; _create_draft_vllm_config handles this.
- enforce_eager means no cudagraph_manager; propose forces the eager path.
"""

import logging

import numpy as np
import torch
from vllm.config import VllmConfig, replace
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import (
    AscendAutoRegressiveSpeculator,
)
from vllm_ascend.worker.v2.spec_decode.mtp.speculator import AscendMTPSpeculator

logger = logging.getLogger(__name__)

# TODO: Refactor these CPU fallbacks to use Triton Dispatcher after vLLM
# RFC #45133 lands. 310P does not install Triton, so the upstream
# @triton.jit kernels (prepare_prefill_inputs, prepare_decode_inputs,
# update_draft_inputs) cannot be dispatched via _kernel[(num_reqs,)].
# These functions replicate the same logic with NumPy and are monkey-
# patched onto the upstream module when HAS_TRITON is False.


def _ascend_prepare_prefill_inputs(
    last_token_indices: torch.Tensor,
    current_draft_step: torch.Tensor,
    input_buffers: InputBuffers,
    input_batch: InputBatch,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    max_num_reqs,
) -> torch.Tensor:
    """CPU fallback for prepare_prefill_inputs (no Triton on 310P)."""
    num_reqs = input_batch.num_reqs
    device = input_buffers.input_ids.device

    idx_mapping = input_batch.idx_mapping
    query_start_loc_target = input_batch.query_start_loc
    seq_lens_target = input_batch.seq_lens

    draft_input_ids = input_buffers.input_ids
    draft_positions = input_buffers.positions
    draft_query_start_loc = input_buffers.query_start_loc
    draft_seq_lens = input_buffers.seq_lens

    num_sampled_np = num_sampled.cpu().numpy() if num_sampled is not None else None
    num_rejected_np = num_rejected.cpu().numpy() if num_rejected is not None else None
    last_sampled_np = last_sampled.cpu().numpy()
    next_prefill_np = next_prefill_tokens.cpu().numpy()
    idx_mapping_np = idx_mapping.cpu().numpy()
    query_start_loc_np = query_start_loc_target.cpu().numpy()
    seq_lens_np = seq_lens_target.cpu().numpy()
    target_positions_np = input_batch.positions.cpu().numpy()
    target_input_ids_np = input_batch.input_ids.cpu().numpy()

    last_token_indices_np = np.empty(num_reqs, dtype=np.int32)
    for req_idx in range(num_reqs):
        req_state_idx = int(idx_mapping_np[req_idx])
        query_start = int(query_start_loc_np[req_idx])
        query_end = int(query_start_loc_np[req_idx + 1])
        query_len = query_end - query_start
        seq_len = int(seq_lens_np[req_idx])

        num_rej = int(num_rejected_np[req_idx]) if num_rejected_np is not None else 0
        query_len -= num_rej

        num_samp = int(num_sampled_np[req_idx]) if num_sampled_np is not None else 0
        if num_samp > 0:
            next_token = int(last_sampled_np[req_state_idx])
        else:
            next_token = int(next_prefill_np[req_state_idx])

        if query_len > 1:
            draft_input_ids_np = target_input_ids_np[query_start + 1 : query_start + query_len]
            draft_input_ids[query_start : query_start + query_len - 1] = torch.from_numpy(
                draft_input_ids_np
            ).to(device)

        last_token_index = query_start + query_len - 1
        last_token_indices_np[req_idx] = last_token_index
        draft_input_ids[last_token_index] = next_token

        draft_positions[query_start : query_start + query_len] = input_batch.positions[
            query_start : query_start + query_len
        ]

        draft_query_start_loc[req_idx] = query_start
        draft_seq_lens[req_idx] = seq_len

    current_draft_step.fill_(0)

    if num_reqs > 0:
        last_query_end = int(query_start_loc_np[num_reqs])
        draft_query_start_loc[num_reqs : max_num_reqs + 1] = last_query_end
        draft_seq_lens[num_reqs:max_num_reqs] = 0
        last_token_indices[num_reqs:max_num_reqs] = 0

    last_token_indices[:num_reqs] = torch.from_numpy(last_token_indices_np).to(device)
    return last_token_indices


def _ascend_prepare_decode_inputs(
    draft_tokens: torch.Tensor,
    target_seq_lens: torch.Tensor,
    num_rejected: torch.Tensor,
    input_buffers: InputBuffers,
    max_model_len: int,
    max_num_reqs: int,
    advance_draft_positions: bool = True,
):
    """CPU fallback for prepare_decode_inputs (no Triton on 310P)."""
    num_reqs = draft_tokens.shape[0]

    draft_tokens_np = draft_tokens.cpu().numpy()
    target_seq_lens_np = target_seq_lens.cpu().numpy()
    num_rejected_np = num_rejected.cpu().numpy()

    input_ids = input_buffers.input_ids
    positions = input_buffers.positions
    query_start_loc = input_buffers.query_start_loc
    seq_lens = input_buffers.seq_lens

    for req_idx in range(num_reqs):
        input_ids[req_idx] = int(draft_tokens_np[req_idx])

        target_seq_len = int(target_seq_lens_np[req_idx])
        num_rej = int(num_rejected_np[req_idx])
        seq_len = target_seq_len - num_rej
        if advance_draft_positions:
            pos_val = positions[req_idx].item()
            pos_val = min(pos_val + 1, max_model_len - 1)
            positions[req_idx] = pos_val
            seq_len = min(seq_len + 1, max_model_len)
        seq_lens[req_idx] = seq_len

    for i in range(num_reqs, max_num_reqs + 1):
        query_start_loc[i] = num_reqs


def _ascend_update_draft_inputs(
    draft_tokens: torch.Tensor,
    current_draft_step: torch.Tensor,
    hidden_states: torch.Tensor,
    output_draft_tokens: torch.Tensor,
    next_input_hidden_states: torch.Tensor,
    input_buffers: InputBuffers,
    num_reqs: int,
    max_model_len: int,
    num_speculative_steps: int,
    advance_draft_positions: bool = True,
):
    """CPU fallback for update_draft_inputs (no Triton on 310P)."""
    step = current_draft_step.item()

    output_draft_tokens[:num_reqs, step] = draft_tokens[:num_reqs]

    if step >= num_speculative_steps - 1:
        return

    input_buffers.input_ids[:num_reqs] = draft_tokens[:num_reqs]
    next_input_hidden_states[:num_reqs] = hidden_states[:num_reqs]

    if advance_draft_positions:
        for req_idx in range(num_reqs):
            pos_val = input_buffers.positions[req_idx].item()
            input_buffers.positions[req_idx] = min(pos_val + 1, max_model_len - 1)
            seq_val = input_buffers.seq_lens[req_idx].item()
            input_buffers.seq_lens[req_idx] = min(seq_val + 1, max_model_len)

# Monkey-patch the upstream Triton kernels with the CPU fallbacks above
# when Triton is not available (310P).
from vllm.triton_utils import HAS_TRITON

if not HAS_TRITON:
    import vllm.v1.worker.gpu.spec_decode.autoregressive.speculator as _upstream_spec

    _upstream_spec.prepare_prefill_inputs = _ascend_prepare_prefill_inputs
    _upstream_spec.prepare_decode_inputs = _ascend_prepare_decode_inputs
    _upstream_spec.update_draft_inputs = _ascend_update_draft_inputs
    logger.info(
        "310P: Triton not available, using CPU fallbacks for spec_decode kernels"
    )


class Ascend310PAutoRegressiveSpeculator(AscendAutoRegressiveSpeculator):
    """310P-specific speculator base.

    310P differs from standard Ascend (A2/A3) in ways that affect the
    speculator:

    1. W8A8 checkpoints store quantization metadata outside config.json,
       so draft_model_config.hf_overrides may not be a dict. Upstream
       get_quant_config() raises ValueError in that case.
    2. In eager mode, no cudagraph_manager is created, so
       dispatch_cg_and_sync_dp() must take the need_eager=True path.
    3. In FULL_DECODE_ONLY mode, prefill_cudagraph_manager is None
       because only decode graphs are captured; the upstream capture()
       and propose() must skip prefill-related assertions.

    Overrides:
    - _create_draft_vllm_config: ensure hf_overrides is a dict.
    - propose: force is_profile=True when prefill_cudagraph_manager is
      None to trigger the eager dispatch path.
    - capture: skip prefill graph capture when prefill_cudagraph_manager
      is None.
    """

    def _create_draft_vllm_config(self) -> VllmConfig:
        """Build the runtime config used while executing the draft model."""
        parallel_config = replace(
            self.vllm_config.parallel_config,
            pipeline_parallel_size=1,
        )
        draft_model_config = self.draft_model_config
        if not isinstance(getattr(draft_model_config, "hf_overrides", None), dict):
            draft_model_config.hf_overrides = {}
        return replace(
            self.vllm_config,
            model_config=draft_model_config,
            parallel_config=parallel_config,
        )

    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict,
        slot_mappings: dict,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple | None = None,
        is_profile=None,
        dp_sync=None,
    ):
        """Override propose to force eager dispatch when no prefill graph manager.

        When prefill_cudagraph_manager is None (eager mode or FULL_DECODE_ONLY
        without prefill capture), dispatch_cg_and_sync_dp must take the
        need_eager=True path. We set is_profile=True to trigger this in the
        upstream propose.
        """
        self.input_batch = input_batch
        sync_state = dp_sync
        if self.prefill_cudagraph_manager is None:
            is_profile = True

        from vllm_ascend._310p.ops.rotary_embedding import AscendRotaryEmbedding310
        from vllm_ascend.worker.v2.attn_utils import build_attn_metadata_wrapper
        from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import (
            torch_gather_wrapper,
        )
        from vllm_ascend.worker.v2.spec_decode.pcp_utils import (
            disable_target_pcp_for_replicated_draft,
        )

        # 310P: preprocess Mamba/GDN recurrent state before draft forward.
        # Without this, the draft model's hybrid recurrent state is stale,
        # causing draft predictions to diverge after a few steps.
        try:
            _ms = self.model_state
            if _ms is not None and self.input_batch is not None:
                _ms.preprocess_state(
                    self.input_batch,
                    self.block_tables,
                    self.kv_cache_config,
                    self.input_batch.num_computed_tokens_gpu
                    if hasattr(self.input_batch, 'num_computed_tokens_gpu')
                    else _ms.num_computed_tokens.gpu,
                )
        except Exception:
            pass

        AscendRotaryEmbedding310.set_rope_position_flag_310p(True)
        try:
            with (
                disable_target_pcp_for_replicated_draft(self),
                build_attn_metadata_wrapper(),
                torch_gather_wrapper(),
            ):
                result = super().propose(
                    input_batch,
                    attn_metadata,
                    slot_mappings,
                    last_hidden_states,
                    aux_hidden_states,
                    num_sampled,
                    num_rejected,
                    last_sampled,
                    next_prefill_tokens,
                    temperature,
                    seeds,
                    sync_state,
                    dummy_run,
                    skip_attn_for_dummy_run,
                    mm_inputs,
                    is_profile=is_profile,
                )
            # 310P: postprocess Mamba/GDN recurrent state after draft forward.
            try:
                _ms = self.model_state
                if _ms is not None:
                    num_reqs = input_batch.num_reqs
                    idx_mapping = input_batch.idx_mapping[:num_reqs]
                    _ms.postprocess_state(idx_mapping, 1)
            except Exception:
                pass
            return result
        finally:
            AscendRotaryEmbedding310.set_rope_position_flag_310p(False)

    def capture(self) -> None:
        """Override capture to skip prefill graph when prefill_cudagraph_manager
        is None (eager mode or FULL_DECODE_ONLY without prefill capture).
        """
        logger.info("Capturing model for speculator...")
        self.last_token_indices.zero_()

        from vllm_ascend.worker.v2.attn_utils import build_attn_metadata_wrapper
        from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import (
            torch_gather_wrapper,
        )
        from vllm_ascend.worker.v2.spec_decode.pcp_utils import (
            disable_target_pcp_for_replicated_draft,
        )

        if self.prefill_cudagraph_manager is not None:
            if self.prefill_cudagraph_manager.use_breakable_cg:
                self.prefill_cudagraph_manager.init_breakable_cg_runner(self.model)
            with disable_target_pcp_for_replicated_draft(self):
                self.prefill_cudagraph_manager.capture(
                    self._prefill,
                    self.model_state,
                    self.target_input_buffers,
                    self.block_tables,
                    self.draft_prefill_attn_groups,
                    self.kv_cache_config,
                    progress_bar_desc="Capturing prefill CUDA graphs",
                )

        if self.num_speculative_steps == 1:
            return

        assert self.decode_cudagraph_manager is not None
        with (
            disable_target_pcp_for_replicated_draft(self),
            build_attn_metadata_wrapper(),
        ):
            self.decode_cudagraph_manager.capture(
                self._multi_step_decode,
                self.model_state,
                self.input_buffers,
                self.block_tables,
                self.attn_groups,
                self.kv_cache_config,
                progress_bar_desc="Capturing decode CUDA graphs",
            )


class Ascend310PMTPSpeculator(Ascend310PAutoRegressiveSpeculator, AscendMTPSpeculator):
    """310P MTP speculator. Combines 310P base with Ascend MTP."""

    pass


def init_310p_speculator(
    vllm_config: VllmConfig,
    device: torch.device,
):
    """Create a 310P-specific speculator for MTP eager mode."""
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    if (
        speculative_config.method == "mtp"
        and not speculative_config.use_gemma4_mtp()
        and not speculative_config.use_step3p5_mtp()
    ):
        return Ascend310PMTPSpeculator(vllm_config, device)
    # Fall back to standard Ascend speculators for other methods.
    from vllm_ascend.worker.v2.spec_decode import init_speculator

    return init_speculator(vllm_config, device)
