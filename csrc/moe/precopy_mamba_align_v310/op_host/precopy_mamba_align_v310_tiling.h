// SPDX-License-Identifier: Apache-2.0
#ifndef VLLM_ASCEND_PRECOPY_MAMBA_ALIGN_V310_TILING_H
#define VLLM_ASCEND_PRECOPY_MAMBA_ALIGN_V310_TILING_H

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(PrecopyMambaAlignV310TilingData)
    TILING_DATA_FIELD_DEF(int64_t, numReqs);
    TILING_DATA_FIELD_DEF(int64_t, numStates);
    TILING_DATA_FIELD_DEF(int64_t, temporalTiles);
    TILING_DATA_FIELD_DEF(int64_t, blockTableStrideReq);
    TILING_DATA_FIELD_DEF(int64_t, convStateDimFirst);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(PrecopyMambaAlignV310, PrecopyMambaAlignV310TilingData)

}  // namespace optiling

#endif
