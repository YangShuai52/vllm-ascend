// SPDX-License-Identifier: Apache-2.0
#ifndef VLLM_ASCEND_PREPROCESS_MAMBA_ALIGN_310_TORCH_ADPT_H
#define VLLM_ASCEND_PREPROCESS_MAMBA_ALIGN_310_TORCH_ADPT_H

namespace vllm_ascend {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> preprocess_mamba_align_310(
    const at::Tensor& idx_mapping,
    at::Tensor state_idx,
    const at::Tensor& num_computed_tokens,
    const at::Tensor& query_start_loc,
    at::Tensor num_accepted_tokens,
    at::Tensor src_col,
    at::Tensor src_off,
    int64_t mamba_block_size)
{
    EXEC_NPU_CMD(aclnnPreprocessMambaAlignV310,
                 idx_mapping,
                 state_idx,
                 num_computed_tokens,
                 query_start_loc,
                 num_accepted_tokens,
                 src_col,
                 src_off,
                 mamba_block_size,
                 state_idx,
                 num_accepted_tokens,
                 src_col,
                 src_off);
    return {state_idx, num_accepted_tokens, src_col, src_off};
}

}  // namespace vllm_ascend

#endif
