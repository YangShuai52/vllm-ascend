// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void update_mamba_num_accepted_v310(
    GM_ADDR idxMapping, GM_ADDR numSampled, GM_ADDR numAcceptedTokens,
    GM_ADDR numAcceptedTokensOut, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    GlobalTensor<int32_t> mappingGm, sampledGm, acceptedGm;
    mappingGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(idxMapping));
    sampledGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(numSampled));
    acceptedGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(numAcceptedTokensOut));
    for (int64_t row = GetBlockIdx(); row < tilingData.numReqs; row += GetBlockNum()) {
        const int32_t req = mappingGm.GetValue(row);
        if (req < 0) continue;
        int32_t value = tilingData.hasTensorNumSampled != 0
                            ? sampledGm.GetValue(row)
                            : static_cast<int32_t>(tilingData.scalarNumSampled);
        if (value < 1) value = 1;
        acceptedGm.SetValue(req, value);
    }
}
