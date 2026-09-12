// SPDX-License-Identifier: Apache-2.0
#include "precopy_mamba_align_v310_tiling.h"

#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/error_log.h"

namespace optiling {

constexpr int64_t TEMPORAL_TILES = 16;
struct PrecopyMambaAlignV310CompileInfo {};

static ge::graphStatus Tiling(gert::TilingContext* context)
{
    auto* stateMetaShape = context->GetInputShape(4);
    OP_CHECK_NULL_WITH_CONTEXT(context, stateMetaShape);
    const auto stateMetaStorage = stateMetaShape->GetStorageShape();
    OP_CHECK_IF(stateMetaStorage.GetDimNum() != 1,
                OP_LOGE(context, "stateBaseAddrs must be 1D"),
                return ge::GRAPH_FAILED);

    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* numReqs = attrs->GetAttrPointer<int64_t>(0);
    const int64_t* blockTableStrideReq = attrs->GetAttrPointer<int64_t>(1);
    const bool* convStateDimFirst = attrs->GetAttrPointer<bool>(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, numReqs);
    OP_CHECK_NULL_WITH_CONTEXT(context, blockTableStrideReq);
    OP_CHECK_NULL_WITH_CONTEXT(context, convStateDimFirst);
    OP_CHECK_IF(*numReqs < 0 || *blockTableStrideReq <= 0,
                OP_LOGE(context, "invalid numReqs or blockTableStrideReq"),
                return ge::GRAPH_FAILED);

    const int64_t numStates = stateMetaStorage.GetDim(0);
    PrecopyMambaAlignV310TilingData tiling;
    tiling.set_numReqs(*numReqs);
    tiling.set_numStates(numStates);
    tiling.set_temporalTiles(TEMPORAL_TILES);
    tiling.set_blockTableStrideReq(*blockTableStrideReq);
    tiling.set_convStateDimFirst(*convStateDimFirst ? 1 : 0);
    auto* rawTilingData = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTilingData);
    tiling.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tiling.GetDataSize());

    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    platform_ascendc::PlatformAscendC platform(platformInfo);
    const uint32_t coreNum = platform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "AIV core count must be positive"),
                return ge::GRAPH_FAILED);
    const int64_t tasks = *numReqs * numStates * TEMPORAL_TILES;
    uint32_t blockDim = tasks <= 0 ? 1 : static_cast<uint32_t>(tasks > coreNum ? coreNum : tasks);
    context->SetBlockDim(blockDim);
    auto* workspaceSizes = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaceSizes);
    workspaceSizes[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParse(gert::TilingParseContext*)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(PrecopyMambaAlignV310)
    .Tiling(Tiling)
    .TilingParse<PrecopyMambaAlignV310CompileInfo>(TilingParse);

}  // namespace optiling
