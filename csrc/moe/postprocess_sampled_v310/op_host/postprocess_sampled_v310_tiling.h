// SPDX-License-Identifier: Apache-2.0
#ifndef VLLM_ASCEND_POSTPROCESS_SAMPLED_V310_TILING_H
#define VLLM_ASCEND_POSTPROCESS_SAMPLED_V310_TILING_H

#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(PostprocessSampledV310TilingData)
    TILING_DATA_FIELD_DEF(int64_t, numReqs);
    TILING_DATA_FIELD_DEF(int64_t, sampledStride);
    TILING_DATA_FIELD_DEF(int64_t, tokenStride);
    TILING_DATA_FIELD_DEF(int64_t, outputBinCountsStride);
    TILING_DATA_FIELD_DEF(int64_t, hasOutputBinCounts);
    TILING_DATA_FIELD_DEF(int64_t, hasQueryStartLoc);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(PostprocessSampledV310, PostprocessSampledV310TilingData)

}  // namespace optiling

#endif
