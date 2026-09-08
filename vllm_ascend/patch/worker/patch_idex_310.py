import torch
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
from vllm.third_party.flash_linear_attention.ops import index as fla_index

from vllm_ascend._310p.ops.fla.gdn_310 import AscendGatedDeltaNetAttention310
from vllm_ascend._310p.ops.fla.idex import (
    prepare_chunk_indices_310,
    prepare_chunk_offsets_310,
)
from vllm_ascend._310p.spec_decode.llm_base_proposer_310 import AscendSpecDecodeBaseProposer310
from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer
from vllm_ascend.utils import is_rc_device

fla_index.prepare_chunk_indices = prepare_chunk_indices_310
fla_index.prepare_chunk_offsets = prepare_chunk_offsets_310

# 310P: protect tail slot during MTP input_ids shift to avoid GatherV2 corruption
# caused by the NPU slice-assign writing one element past the intended range
# on the persistent drafter input_ids buffer.
AscendSpecDecodeBaseProposer.set_inputs_first_pass = (  # type: ignore[method-assign]
    AscendSpecDecodeBaseProposer310.set_inputs_first_pass
)
AscendSpecDecodeBaseProposer._run_merged_draft = (  # type: ignore[method-assign]
    AscendSpecDecodeBaseProposer310._run_merged_draft
)

# 310P: Apply the same Qwen3.5 MTP forward patch as the standard worker path
# (patch_qwen3_5.py). 310P does not have STANDARD_WORKER_PATCHES, so the
# patch is not loaded by the default __init__. Without this, the MTP draft
# model uses the upstream forward which does not combine token embeddings
# with target hidden states, causing 0% draft acceptance.
try:
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MultiTokenPredictor
    from vllm.sequence import IntermediateTensors
    from vllm.distributed import tensor_model_parallel_all_gather
    from vllm.distributed.parallel_state import get_pp_group

    def qwen3_5_mtp_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        assert hidden_states.shape[-1] == inputs_embeds.shape[-1]
        inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
        hidden_states = self.fc(hidden_states)
        residual = None

        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[current_step_idx]
        if mtp_layer.use_attn_reduce_scatter_for_moe:
            from vllm.model_executor.models.utils import sequence_parallel_chunk

            assert hidden_states.shape[0] == positions.shape[-1]
            hidden_states = sequence_parallel_chunk(hidden_states)
            assert residual is None
        hidden_states, residual = mtp_layer(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        if mtp_layer.use_attn_reduce_scatter_for_moe:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            hidden_states = hidden_states[: positions.shape[-1]]
        return hidden_states

    Qwen3_5MultiTokenPredictor.forward = qwen3_5_mtp_forward
    import logging
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"310P: MTP forward NOT patched: {e}")

# Patch _warmup_prefill_kernels to no-op on 310P: triton.next_power_of_2 does
# not exist in the triton version used on 310P CI, and NPU does not use these
# CUDA warmup kernel anyway.
QwenGatedDeltaNetAttention._warmup_prefill_kernels = lambda self, qkv_or_qkvz, v_dim: None  # type: ignore[method-assign]
QwenGatedDeltaNetAttention._split_ba_for_tp = AscendGatedDeltaNetAttention._split_ba_for_tp
QwenGatedDeltaNetAttention.get_state_shape = AscendGatedDeltaNetAttention.get_state_shape
QwenGatedDeltaNetAttention._forward_core = AscendGatedDeltaNetAttention310._forward_core
QwenGatedDeltaNetAttention.get_state_dtype = AscendGatedDeltaNetAttention310.get_state_dtype

# 310P: make Qwen GDN use the 310P attention backend, including the
# MTP ACL graph padding replay fixes provided by gdn_attn_builder_310.py.
QwenGatedDeltaNetAttention.get_attn_backend = AscendGatedDeltaNetAttention310.get_attn_backend

# Vision pos-embed: 310P images do not install Triton, so upstream
# ``HAS_TRITON=False`` already selects ``pos_embed_interpolate_native``.
# No ``fast_pos_embed_interpolate`` rewrite is required.

if is_rc_device():
    from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend

    from vllm_ascend._310p.ops.gdn_attn_builder_310 import GDNAttentionMetadataBuilder310
    from vllm_ascend._310p.ops.qwen3vl_310 import rot_pos_emb_310

    # 310P RC: use blocking H2D in rot_pos_emb to avoid race with subsequent indexing.
    Qwen3_VisionTransformer.rot_pos_emb = rot_pos_emb_310  # type: ignore[method-assign]

    # Qwen3.5 on 310P RC uses upstream GDNAttentionBackend via MambaBase.get_attn_backend().
    GDNAttentionBackend.get_builder_cls = staticmethod(  # type: ignore[method-assign]
        lambda: GDNAttentionMetadataBuilder310
    )
