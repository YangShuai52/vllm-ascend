// SPDX-License-Identifier: Apache-2.0
#include "postprocess_sampled_v310_tiling.h"

#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/error_log.h"

namespace optiling {

struct PostprocessSampledV310CompileInfo {};

static ge::graphStatus Tiling(gert::TilingContext* context)
{
    auto* idxShape = context->GetInputShape(0);
    auto* sampledShape = context->GetInputShape(4);
    auto* tokenShape = context->GetInputShape(8);
    auto* queryStartLocShape = context->GetInputShape(7);
    auto* outputBinCountsShape = context->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, idxShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, sampledShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, tokenShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryStartLocShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputBinCountsShape);

    auto idxStorage = idxShape->GetStorageShape();
    auto sampledStorage = sampledShape->GetStorageShape();
    auto tokenStorage = tokenShape->GetStorageShape();
    auto queryStartLocStorage = queryStartLocShape->GetStorageShape();
    auto outputBinCountsStorage = outputBinCountsShape->GetStorageShape();
    OP_CHECK_IF(idxStorage.GetDimNum() != 1, OP_LOGE(context, "idxMapping must be 1D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(sampledStorage.GetDimNum() != 2, OP_LOGE(context, "sampledTokens must be 2D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tokenStorage.GetDimNum() != 2, OP_LOGE(context, "allTokenIds must be 2D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(queryStartLocStorage.GetDimNum() != 1,
                OP_LOGE(context, "queryStartLoc must be 1D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(sampledStorage.GetDim(0) != idxStorage.GetDim(0),
                OP_LOGE(context, "sampledTokens first dim must equal numReqs"),
                return ge::GRAPH_FAILED);

    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* hasOutputBinCounts = attrs->GetAttrPointer<bool>(0);
    const bool* hasQueryStartLoc = attrs->GetAttrPointer<bool>(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, hasOutputBinCounts);
    OP_CHECK_NULL_WITH_CONTEXT(context, hasQueryStartLoc);
    OP_CHECK_IF(*hasQueryStartLoc && queryStartLocStorage.GetDim(0) < idxStorage.GetDim(0) + 1,
                OP_LOGE(context, "queryStartLoc must have at least numReqs + 1 elements"),
                return ge::GRAPH_FAILED);

    auto* tiling = context->GetTilingData<PostprocessSampledV310TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->set_numReqs(idxStorage.GetDim(0));
    tiling->set_sampledStride(sampledStorage.GetDim(1));
    tiling->set_tokenStride(tokenStorage.GetDim(1));
    tiling->set_outputBinCountsStride(
        outputBinCountsStorage.GetDimNum() == 2 ? outputBinCountsStorage.GetDim(1) : 0);
    tiling->set_hasOutputBinCounts(*hasOutputBinCounts ? 1 : 0);
    tiling->set_hasQueryStartLoc(*hasQueryStartLoc ? 1 : 0);

    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    platform_ascendc::PlatformAscendC platform(platformInfo);
    uint32_t coreNum = platform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "AIV core count must be positive"),
                return ge::GRAPH_FAILED);
    uint32_t blockDim = static_cast<uint32_t>(idxStorage.GetDim(0));
    if (blockDim == 0) {
        blockDim = 1;
    }
    if (blockDim > coreNum) {
        blockDim = coreNum;
    }
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

IMPL_OP_OPTILING(PostprocessSampledV310)
    .Tiling(Tiling)
    .TilingParse<PostprocessSampledV310CompileInfo>(TilingParse);

}  // namespace optiling
