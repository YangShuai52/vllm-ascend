// SPDX-License-Identifier: Apache-2.0
#ifndef VLLM_ASCEND_POSTPROCESS_SAMPLED_310_TORCH_ADPT_H
#define VLLM_ASCEND_POSTPROCESS_SAMPLED_310_TORCH_ADPT_H

namespace vllm_ascend {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> post_update_310(
    const at::Tensor& idx_mapping,
    at::Tensor num_computed_tokens,
    at::Tensor last_sampled_tokens,
    at::Tensor output_bin_counts,
    const at::Tensor& sampled_tokens,
    const at::Tensor& num_sampled,
    const at::Tensor& num_rejected,
    const at::Tensor& query_start_loc,
    at::Tensor all_token_ids,
    at::Tensor total_len,
    bool has_output_bin_counts,
    bool has_query_start_loc)
{
    EXEC_NPU_CMD(aclnnPostprocessSampledV310,
                 idx_mapping,
                 num_computed_tokens,
                 last_sampled_tokens,
                 output_bin_counts,
                 sampled_tokens,
                 num_sampled,
                 num_rejected,
                 query_start_loc,
                 all_token_ids,
                 total_len,
                 has_output_bin_counts,
                 has_query_start_loc,
                 num_computed_tokens,
                 last_sampled_tokens,
                 all_token_ids,
                 total_len);
    return {num_computed_tokens, last_sampled_tokens, all_token_ids, total_len};
}

}  // namespace vllm_ascend

#endif
