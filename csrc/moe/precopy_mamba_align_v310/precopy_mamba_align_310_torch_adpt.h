// SPDX-License-Identifier: Apache-2.0
#ifndef VLLM_ASCEND_PRECOPY_MAMBA_ALIGN_310_TORCH_ADPT_H
#define VLLM_ASCEND_PRECOPY_MAMBA_ALIGN_310_TORCH_ADPT_H

namespace vllm_ascend {

at::Tensor precopy_mamba_align_310(
    at::Tensor state_idx,
    const at::Tensor& src_col,
    const at::Tensor& token_bias,
    const at::Tensor& block_table_ptrs,
    const at::Tensor& state_base_addrs,
    const at::Tensor& state_block_strides,
    const at::Tensor& state_elem_sizes,
    const at::Tensor& state_inner_sizes,
    const at::Tensor& state_conv_widths,
    const at::Tensor& state_group_indices,
    const at::Tensor& state_dim_row_count,
    const at::Tensor& state_dim_row_stride,
    const at::Tensor& idx_mapping,
    int64_t num_reqs,
    int64_t block_table_stride_req,
    bool conv_state_dim_first)
{
    EXEC_NPU_CMD(aclnnPrecopyMambaAlignV310,
                 state_idx,
                 src_col,
                 token_bias,
                 block_table_ptrs,
                 state_base_addrs,
                 state_block_strides,
                 state_elem_sizes,
                 state_inner_sizes,
                 state_conv_widths,
                 state_group_indices,
                 state_dim_row_count,
                 state_dim_row_stride,
                 idx_mapping,
                 num_reqs,
                 block_table_stride_req,
                 conv_state_dim_first,
                 state_idx);
    return state_idx;
}

}  // namespace vllm_ascend

#endif
