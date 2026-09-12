// SPDX-License-Identifier: Apache-2.0
#ifndef VLLM_ASCEND_POSTPROCESS_MAMBA_ALIGN_310_TORCH_ADPT_H
#define VLLM_ASCEND_POSTPROCESS_MAMBA_ALIGN_310_TORCH_ADPT_H

namespace vllm_ascend {
at::Tensor postprocess_mamba_align_310(
    const at::Tensor& idx_mapping, at::Tensor num_accepted_tokens,
    const at::Tensor& state_idx,
    const at::Tensor& num_computed_tokens, const at::Tensor& block_table_ptrs,
    const at::Tensor& state_base_addrs, const at::Tensor& state_block_strides,
    const at::Tensor& state_elem_sizes, const at::Tensor& state_inner_sizes,
    const at::Tensor& state_conv_widths, const at::Tensor& state_group_indices,
    const at::Tensor& state_dim_row_count, const at::Tensor& state_dim_row_stride,
    int64_t num_reqs, int64_t block_size, int64_t block_table_stride_req,
    bool conv_state_dim_first)
{
    EXEC_NPU_CMD(aclnnPostprocessMambaAlignV310, idx_mapping,
                 num_accepted_tokens, state_idx, num_computed_tokens,
                 block_table_ptrs, state_base_addrs, state_block_strides,
                 state_elem_sizes, state_inner_sizes, state_conv_widths,
                 state_group_indices, state_dim_row_count, state_dim_row_stride,
                 num_reqs, block_size, block_table_stride_req,
                 conv_state_dim_first, num_accepted_tokens);
    return num_accepted_tokens;
}
}  // namespace vllm_ascend
#endif
