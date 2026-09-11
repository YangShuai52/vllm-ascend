// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void postprocess_sampled_v310(
    GM_ADDR idxMapping, GM_ADDR numComputedTokens, GM_ADDR lastSampledTokens,
    GM_ADDR outputBinCounts, GM_ADDR sampledTokens, GM_ADDR numSampled,
    GM_ADDR numRejected, GM_ADDR queryStartLoc, GM_ADDR allTokenIds,
    GM_ADDR totalLen, GM_ADDR numComputedTokensOut,
    GM_ADDR lastSampledTokensOut, GM_ADDR allTokenIdsOut,
    GM_ADDR totalLenOut, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    GlobalTensor<int32_t> idxMappingGm;
    GlobalTensor<int32_t> numComputedTokensGm;
    GlobalTensor<int64_t> lastSampledTokensGm;
    GlobalTensor<int32_t> outputBinCountsGm;
    GlobalTensor<int32_t> sampledTokensGm;
    GlobalTensor<int32_t> numSampledGm;
    GlobalTensor<int32_t> numRejectedGm;
    GlobalTensor<int32_t> queryStartLocGm;
    GlobalTensor<int32_t> allTokenIdsGm;
    GlobalTensor<int32_t> totalLenGm;

    idxMappingGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(idxMapping));
    numComputedTokensGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(numComputedTokensOut));
    lastSampledTokensGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(lastSampledTokensOut));
    outputBinCountsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(outputBinCounts));
    sampledTokensGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sampledTokens));
    numSampledGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(numSampled));
    numRejectedGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(numRejected));
    queryStartLocGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(queryStartLoc));
    allTokenIdsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(allTokenIdsOut));
    totalLenGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(totalLenOut));

    const int64_t blockIdx = GetBlockIdx();
    const int64_t blockNum = GetBlockNum();
    for (int64_t batchIdx = blockIdx; batchIdx < tilingData.numReqs; batchIdx += blockNum) {
        const int32_t reqIdx = idxMappingGm.GetValue(batchIdx);
        if (reqIdx < 0) {
            continue;
        }

        const int32_t total = totalLenGm.GetValue(reqIdx);
        int32_t count = numSampledGm.GetValue(batchIdx);
        if (count < 0) {
            count = 0;
        } else if (count > tilingData.sampledStride) {
            count = static_cast<int32_t>(tilingData.sampledStride);
        }

        for (int32_t i = 0; i < count; ++i) {
            const int32_t token = sampledTokensGm.GetValue(batchIdx * tilingData.sampledStride + i);
            allTokenIdsGm.SetValue(static_cast<int64_t>(reqIdx) * tilingData.tokenStride + total + i, token);
            if (tilingData.hasOutputBinCounts != 0) {
                const int64_t binOffset = static_cast<int64_t>(reqIdx) * tilingData.outputBinCountsStride + token;
                outputBinCountsGm.SetValue(binOffset, outputBinCountsGm.GetValue(binOffset) + 1);
            }
        }

        if (count > 0) {
            const int32_t last = sampledTokensGm.GetValue(batchIdx * tilingData.sampledStride + count - 1);
            lastSampledTokensGm.SetValue(reqIdx, static_cast<int64_t>(last));
            totalLenGm.SetValue(reqIdx, total + count);
        }

        int32_t queryLen = 0;
        if (tilingData.hasQueryStartLoc != 0) {
            queryLen = queryStartLocGm.GetValue(batchIdx + 1) - queryStartLocGm.GetValue(batchIdx);
        }
        const int32_t computedDelta = queryLen - numRejectedGm.GetValue(batchIdx);
        if (computedDelta != 0) {
            numComputedTokensGm.SetValue(reqIdx, numComputedTokensGm.GetValue(reqIdx) + computedDelta);
        }
    }
}
