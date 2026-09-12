// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void preprocess_mamba_align_v310(
    GM_ADDR idxMapping, GM_ADDR stateIdx, GM_ADDR numComputedTokens,
    GM_ADDR queryStartLoc, GM_ADDR numAcceptedTokens, GM_ADDR srcCol,
    GM_ADDR srcOff, GM_ADDR stateIdxOut, GM_ADDR numAcceptedTokensOut,
    GM_ADDR srcColOut, GM_ADDR srcOffOut, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    GlobalTensor<int32_t> idxMappingGm;
    GlobalTensor<int32_t> stateIdxGm;
    GlobalTensor<int32_t> numComputedTokensGm;
    GlobalTensor<int32_t> queryStartLocGm;
    GlobalTensor<int32_t> numAcceptedTokensGm;
    GlobalTensor<int32_t> srcColGm;
    GlobalTensor<int32_t> srcOffGm;

    idxMappingGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(idxMapping));
    stateIdxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(stateIdxOut));
    numComputedTokensGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(numComputedTokens));
    queryStartLocGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(queryStartLoc));
    numAcceptedTokensGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(numAcceptedTokensOut));
    srcColGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(srcColOut));
    srcOffGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(srcOffOut));

    const int64_t blockIdx = GetBlockIdx();
    const int64_t blockNum = GetBlockNum();
    for (int64_t batchIdx = blockIdx; batchIdx < tilingData.numReqs; batchIdx += blockNum) {
        const int32_t reqIdx = idxMappingGm.GetValue(batchIdx);
        if (reqIdx < 0) {
            continue;
        }
        const int32_t oldStateIdx = stateIdxGm.GetValue(reqIdx);
        const int32_t numAccepted = numAcceptedTokensGm.GetValue(reqIdx);
        srcColGm.SetValue(reqIdx, oldStateIdx);
        srcOffGm.SetValue(reqIdx, numAccepted > 1 ? numAccepted - 1 : 0);

        const int32_t queryLen = queryStartLocGm.GetValue(batchIdx + 1) -
                                 queryStartLocGm.GetValue(batchIdx);
        const int64_t computedAfter = static_cast<int64_t>(numComputedTokensGm.GetValue(reqIdx)) + queryLen;
        const int32_t newStateIdx = static_cast<int32_t>(
            (computedAfter + tilingData.mambaBlockSize - 1) / tilingData.mambaBlockSize - 1);
        stateIdxGm.SetValue(reqIdx, newStateIdx);
        if (oldStateIdx >= 0 && oldStateIdx != newStateIdx) {
            numAcceptedTokensGm.SetValue(reqIdx, 1);
        }
    }
}
