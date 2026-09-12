// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
using namespace AscendC;

__aicore__ inline void CopyByteRange(uint64_t srcAddr, uint64_t dstAddr,
                                     int64_t size, int64_t tileIdx,
                                     int64_t numTiles)
{
    if (size <= 0) return;
    GlobalTensor<uint8_t> src;
    GlobalTensor<uint8_t> dst;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(srcAddr));
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(dstAddr));
    const int64_t begin = size * tileIdx / numTiles;
    const int64_t end = size * (tileIdx + 1) / numTiles;
    for (int64_t i = begin; i < end; ++i) dst.SetValue(i, src.GetValue(i));
}

extern "C" __global__ __aicore__ void postprocess_mamba_align_v310(
    GM_ADDR idxMapping, GM_ADDR numAcceptedTokens,
    GM_ADDR stateIdx, GM_ADDR numComputedTokens, GM_ADDR blockTablePtrs,
    GM_ADDR stateBaseAddrs, GM_ADDR stateBlockStrides, GM_ADDR stateElemSizes,
    GM_ADDR stateInnerSizes, GM_ADDR stateConvWidths, GM_ADDR stateGroupIndices,
    GM_ADDR stateDimRowCount, GM_ADDR stateDimRowStride,
    GM_ADDR numAcceptedTokensOut, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    GlobalTensor<int32_t> idxMappingGm, numAcceptedGm;
    GlobalTensor<int32_t> stateIdxGm, computedGm, elemSizesGm, convWidthsGm;
    GlobalTensor<int32_t> groupIndicesGm, rowCountsGm;
    GlobalTensor<int64_t> tablePtrsGm, baseAddrsGm, blockStridesGm;
    GlobalTensor<int64_t> innerSizesGm, rowStridesGm;
    idxMappingGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(idxMapping));
    numAcceptedGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(numAcceptedTokensOut));
    stateIdxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(stateIdx));
    computedGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(numComputedTokens));
    tablePtrsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(blockTablePtrs));
    baseAddrsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(stateBaseAddrs));
    blockStridesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(stateBlockStrides));
    elemSizesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(stateElemSizes));
    innerSizesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(stateInnerSizes));
    convWidthsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(stateConvWidths));
    groupIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(stateGroupIndices));
    rowCountsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(stateDimRowCount));
    rowStridesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(stateDimRowStride));

    const int64_t tasks = tilingData.numReqs * tilingData.numStates * tilingData.temporalTiles;
    for (int64_t task = GetBlockIdx(); task < tasks; task += GetBlockNum()) {
        const int64_t tile = task % tilingData.temporalTiles;
        const int64_t state = (task / tilingData.temporalTiles) % tilingData.numStates;
        const int64_t batch = task / (tilingData.temporalTiles * tilingData.numStates);
        const int32_t req = idxMappingGm.GetValue(batch);
        if (req < 0) continue;
        const int32_t accepted = numAcceptedGm.GetValue(req);
        const int32_t srcColumn = stateIdxGm.GetValue(req);
        const int32_t newComputed = computedGm.GetValue(req);
        const int32_t runningTokens = newComputed - accepted + 1;
        const int32_t alignedComputed = newComputed / tilingData.blockSize * tilingData.blockSize;

        if (alignedComputed < runningTokens) continue;
        const int32_t bias = alignedComputed - runningTokens;
        const int32_t dstColumn = alignedComputed / tilingData.blockSize - 1;
        if (srcColumn == dstColumn && state == 0 && tile == 0)
            numAcceptedGm.SetValue(req, 1);
        if (srcColumn == dstColumn && bias == 0) continue;

        const int32_t convWidth = convWidthsGm.GetValue(state);
        if (convWidth > 0 && tile != 0) continue;
        const int32_t group = groupIndicesGm.GetValue(state);
        const uint64_t tableAddr = static_cast<uint64_t>(tablePtrsGm.GetValue(group));
        GlobalTensor<int32_t> tableGm;
        tableGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tableAddr));
        const int64_t rowBase = batch * tilingData.blockTableStrideReq;
        const int64_t base = baseAddrsGm.GetValue(state);
        const int64_t stride = blockStridesGm.GetValue(state);
        const int64_t elemSize = elemSizesGm.GetValue(state);
        const int64_t innerSize = innerSizesGm.GetValue(state);
        const int64_t srcBlock = tableGm.GetValue(rowBase + srcColumn);
        const int64_t dstBlock = tableGm.GetValue(rowBase + dstColumn);
        const uint64_t srcAddr = static_cast<uint64_t>(base + srcBlock * stride);
        const uint64_t dstAddr = static_cast<uint64_t>(base + dstBlock * stride);

        if (convWidth > 0) {
            const int32_t numDstTokens = convWidth - bias;
            if (numDstTokens <= 0) continue;
            if (tilingData.convStateDimFirst != 0) {
                const int32_t rows = rowCountsGm.GetValue(state);
                const int64_t rowStride = rowStridesGm.GetValue(state);
                for (int32_t token = 0; token < numDstTokens; ++token)
                    for (int32_t row = 0; row < rows; ++row)
                        CopyByteRange(srcAddr + row * rowStride + (token + bias) * elemSize,
                                      dstAddr + row * rowStride + token * elemSize,
                                      elemSize, 0, 1);
            } else {
                const int64_t tokenBytes = innerSize * elemSize;
                if (srcBlock == dstBlock && bias != 0) {
                    for (int32_t token = 0; token < numDstTokens; ++token)
                        CopyByteRange(srcAddr + (token + bias) * tokenBytes,
                                      dstAddr + token * tokenBytes, tokenBytes, 0, 1);
                } else {
                    CopyByteRange(srcAddr + static_cast<int64_t>(bias) * tokenBytes,
                                  dstAddr, static_cast<int64_t>(numDstTokens) * tokenBytes, 0, 1);
                }
            }
        } else {
            const int64_t actualSrcBlock = tableGm.GetValue(rowBase + srcColumn + bias);
            const uint64_t actualSrcAddr = static_cast<uint64_t>(base + actualSrcBlock * stride);
            CopyByteRange(actualSrcAddr, dstAddr, innerSize * elemSize,
                          tile, tilingData.temporalTiles);
        }
    }
}
