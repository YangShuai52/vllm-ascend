// SPDX-License-Identifier: Apache-2.0
#ifndef VLLM_ASCEND_UPDATE_MAMBA_NUM_ACCEPTED_310_TORCH_ADPT_H
#define VLLM_ASCEND_UPDATE_MAMBA_NUM_ACCEPTED_310_TORCH_ADPT_H
namespace vllm_ascend {
at::Tensor update_mamba_num_accepted_310(
    const at::Tensor& idx_mapping, const at::Tensor& num_sampled,
    at::Tensor num_accepted_tokens, int64_t num_reqs,
    int64_t scalar_num_sampled, bool has_tensor_num_sampled)
{
    EXEC_NPU_CMD(aclnnUpdateMambaNumAcceptedV310, idx_mapping, num_sampled,
                 num_accepted_tokens, num_reqs, scalar_num_sampled,
                 has_tensor_num_sampled, num_accepted_tokens);
    return num_accepted_tokens;
}
}  // namespace vllm_ascend
#endif
