// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"

using namespace AscendC;

__aicore__ inline void CopyByteRange(uint64_t srcAddr, uint64_t dstAddr, int64_t size,
                                     int64_t tileIdx, int64_t numTiles)
{
    if (size <= 0) {
        return;
    }
    GlobalTensor<uint8_t> src;
    GlobalTensor<uint8_t> dst;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(srcAddr));
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(dstAddr));
    const int64_t begin = size * tileIdx / numTiles;
    const int64_t end = size * (tileIdx + 1) / numTiles;
    for (int64_t i = begin; i < end; ++i) {
        dst.SetValue(i, src.GetValue(i));
    }
}

extern "C" __global__ __aicore__ void precopy_mamba_align_v310(
    GM_ADDR stateIdx, GM_ADDR srcCol, GM_ADDR tokenBias,
    GM_ADDR blockTablePtrs, GM_ADDR stateBaseAddrs,
    GM_ADDR stateBlockStrides, GM_ADDR stateElemSizes,
    GM_ADDR stateInnerSizes, GM_ADDR stateConvWidths,
    GM_ADDR stateGroupIndices, GM_ADDR stateDimRowCount,
    GM_ADDR stateDimRowStride, GM_ADDR idxMapping, GM_ADDR stateIdxOut,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    GlobalTensor<int32_t> stateIdxGm;
    GlobalTensor<int32_t> srcColGm;
    GlobalTensor<int32_t> tokenBiasGm;
    GlobalTensor<int64_t> blockTablePtrsGm;
    GlobalTensor<int64_t> stateBaseAddrsGm;
    GlobalTensor<int64_t> stateBlockStridesGm;
    GlobalTensor<int32_t> stateElemSizesGm;
    GlobalTensor<int64_t> stateInnerSizesGm;
    GlobalTensor<int32_t> stateConvWidthsGm;
    GlobalTensor<int32_t> stateGroupIndicesGm;
    GlobalTensor<int32_t> stateDimRowCountGm;
    GlobalTensor<int64_t> stateDimRowStrideGm;
    GlobalTensor<int32_t> idxMappingGm;

    stateIdxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(stateIdxOut));
    srcColGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(srcCol));
    tokenBiasGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tokenBias));
    blockTablePtrsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(blockTablePtrs));
    stateBaseAddrsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(stateBaseAddrs));
    stateBlockStridesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(stateBlockStrides));
    stateElemSizesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(stateElemSizes));
    stateInnerSizesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(stateInnerSizes));
    stateConvWidthsGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(stateConvWidths));
    stateGroupIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(stateGroupIndices));
    stateDimRowCountGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(stateDimRowCount));
    stateDimRowStrideGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(stateDimRowStride));
    idxMappingGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(idxMapping));

    const int64_t totalTasks = tilingData.numReqs * tilingData.numStates * tilingData.temporalTiles;
    for (int64_t task = GetBlockIdx(); task < totalTasks; task += GetBlockNum()) {
        const int64_t tileIdx = task % tilingData.temporalTiles;
        const int64_t state = (task / tilingData.temporalTiles) % tilingData.numStates;
        const int64_t batch = task / (tilingData.temporalTiles * tilingData.numStates);
        const int32_t reqIdx = idxMappingGm.GetValue(batch);
        if (reqIdx < 0) {
            continue;
        }
        const int32_t srcColumn = srcColGm.GetValue(reqIdx);
        const int32_t dstColumn = stateIdxGm.GetValue(reqIdx);
        if (srcColumn < 0 || dstColumn < 0 || srcColumn == dstColumn) {
            continue;
        }

        const int32_t bias = tokenBiasGm.GetValue(reqIdx);
        const int32_t convWidth = stateConvWidthsGm.GetValue(state);
        if (convWidth > 0 && tileIdx != 0) {
            continue;
        }
        const int32_t group = stateGroupIndicesGm.GetValue(state);
        const uint64_t blockTableAddr = static_cast<uint64_t>(blockTablePtrsGm.GetValue(group));
        GlobalTensor<int32_t> blockTableGm;
        blockTableGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(blockTableAddr));
        const int64_t rowBase = batch * tilingData.blockTableStrideReq;
        const int64_t stateBase = stateBaseAddrsGm.GetValue(state);
        const int64_t blockStride = stateBlockStridesGm.GetValue(state);
        const int64_t elemSize = stateElemSizesGm.GetValue(state);
        const int64_t innerSize = stateInnerSizesGm.GetValue(state);
        const int64_t dstBlock = blockTableGm.GetValue(rowBase + dstColumn);
        const uint64_t dstAddr = static_cast<uint64_t>(stateBase + dstBlock * blockStride);

        if (convWidth > 0) {
            const int32_t numDstTokens = convWidth - bias;
            if (numDstTokens <= 0) {
                continue;
            }
            const int64_t srcBlock = blockTableGm.GetValue(rowBase + srcColumn);
            const uint64_t srcBlockAddr = static_cast<uint64_t>(stateBase + srcBlock * blockStride);
            if (tilingData.convStateDimFirst != 0) {
                const int32_t rows = stateDimRowCountGm.GetValue(state);
                const int64_t rowStride = stateDimRowStrideGm.GetValue(state);
                for (int32_t token = 0; token < numDstTokens; ++token) {
                    for (int32_t row = 0; row < rows; ++row) {
                        CopyByteRange(srcBlockAddr + row * rowStride + (token + bias) * elemSize,
                                      dstAddr + row * rowStride + token * elemSize,
                                      elemSize, 0, 1);
                    }
                }
            } else {
                const int64_t tokenBytes = innerSize * elemSize;
                if (srcBlock == dstBlock && bias != 0) {
                    for (int32_t token = 0; token < numDstTokens; ++token) {
                        CopyByteRange(srcBlockAddr + (token + bias) * tokenBytes,
                                      dstAddr + token * tokenBytes, tokenBytes, 0, 1);
                    }
                } else {
                    CopyByteRange(srcBlockAddr + static_cast<int64_t>(bias) * tokenBytes,
                                  dstAddr, static_cast<int64_t>(numDstTokens) * tokenBytes, 0, 1);
                }
            }
            continue;
        }

        const int64_t actualSrcBlock = blockTableGm.GetValue(rowBase + srcColumn + bias);
        const uint64_t srcAddr = static_cast<uint64_t>(stateBase + actualSrcBlock * blockStride);
        CopyByteRange(srcAddr, dstAddr, innerSize * elemSize, tileIdx, tilingData.temporalTiles);
    }
}
