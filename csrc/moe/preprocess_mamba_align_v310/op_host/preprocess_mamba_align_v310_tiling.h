// SPDX-License-Identifier: Apache-2.0
#ifndef VLLM_ASCEND_PREPROCESS_MAMBA_ALIGN_V310_TILING_H
#define VLLM_ASCEND_PREPROCESS_MAMBA_ALIGN_V310_TILING_H

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(PreprocessMambaAlignV310TilingData)
    TILING_DATA_FIELD_DEF(int64_t, numReqs);
    TILING_DATA_FIELD_DEF(int64_t, mambaBlockSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(PreprocessMambaAlignV310, PreprocessMambaAlignV310TilingData)

}  // namespace optiling

#endif
