// SPDX-License-Identifier: Apache-2.0
#include "preprocess_mamba_align_v310_tiling.h"

#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/error_log.h"

namespace optiling {

struct PreprocessMambaAlignV310CompileInfo {};

static ge::graphStatus Tiling(gert::TilingContext* context)
{
    auto* idxShape = context->GetInputShape(0);
    auto* queryShape = context->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, idxShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const auto idxStorage = idxShape->GetStorageShape();
    const auto queryStorage = queryShape->GetStorageShape();
    OP_CHECK_IF(idxStorage.GetDimNum() != 1, OP_LOGE(context, "idxMapping must be 1D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(queryStorage.GetDimNum() != 1 || queryStorage.GetDim(0) < idxStorage.GetDim(0) + 1,
                OP_LOGE(context, "queryStartLoc must contain numReqs + 1 elements"),
                return ge::GRAPH_FAILED);

    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* blockSize = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, blockSize);
    OP_CHECK_IF(*blockSize <= 0, OP_LOGE(context, "mambaBlockSize must be positive"),
                return ge::GRAPH_FAILED);

    PreprocessMambaAlignV310TilingData tiling;
    tiling.set_numReqs(idxStorage.GetDim(0));
    tiling.set_mambaBlockSize(*blockSize);
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
    uint32_t blockDim = static_cast<uint32_t>(idxStorage.GetDim(0));
    blockDim = blockDim == 0 ? 1 : (blockDim > coreNum ? coreNum : blockDim);
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

IMPL_OP_OPTILING(PreprocessMambaAlignV310)
    .Tiling(Tiling)
    .TilingParse<PreprocessMambaAlignV310CompileInfo>(TilingParse);

}  // namespace optiling
