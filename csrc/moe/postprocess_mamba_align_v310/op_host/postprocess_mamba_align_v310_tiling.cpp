// SPDX-License-Identifier: Apache-2.0
#include "postprocess_mamba_align_v310_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/error_log.h"
namespace optiling {
constexpr int64_t TEMPORAL_TILES = 16;
struct PostprocessMambaAlignV310CompileInfo {};
static ge::graphStatus Tiling(gert::TilingContext* context)
{
    auto* metaShape = context->GetInputShape(5);
    OP_CHECK_NULL_WITH_CONTEXT(context, metaShape);
    const auto storage = metaShape->GetStorageShape();
    OP_CHECK_IF(storage.GetDimNum() != 1,
                OP_LOGE(context, "stateBaseAddrs must be 1D"), return ge::GRAPH_FAILED);
    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* numReqs = attrs->GetAttrPointer<int64_t>(0);
    const int64_t* blockSize = attrs->GetAttrPointer<int64_t>(1);
    const int64_t* tableStride = attrs->GetAttrPointer<int64_t>(2);
    const bool* dimFirst = attrs->GetAttrPointer<bool>(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, numReqs); OP_CHECK_NULL_WITH_CONTEXT(context, blockSize);
    OP_CHECK_NULL_WITH_CONTEXT(context, tableStride); OP_CHECK_NULL_WITH_CONTEXT(context, dimFirst);
    OP_CHECK_IF(*numReqs < 0 || *blockSize <= 0 || *tableStride <= 0,
                OP_LOGE(context, "invalid postprocess attributes"), return ge::GRAPH_FAILED);
    PostprocessMambaAlignV310TilingData data;
    data.set_numReqs(*numReqs); data.set_numStates(storage.GetDim(0));
    data.set_temporalTiles(TEMPORAL_TILES); data.set_blockSize(*blockSize);
    data.set_blockTableStrideReq(*tableStride);
    data.set_convStateDimFirst(*dimFirst ? 1 : 0);
    auto* raw = context->GetRawTilingData(); OP_CHECK_NULL_WITH_CONTEXT(context, raw);
    data.SaveToBuffer(raw->GetData(), raw->GetCapacity()); raw->SetDataSize(data.GetDataSize());
    fe::PlatFormInfos* info = context->GetPlatformInfo(); OP_CHECK_NULL_WITH_CONTEXT(context, info);
    platform_ascendc::PlatformAscendC platform(info);
    const uint32_t cores = platform.GetCoreNumAiv();
    OP_CHECK_IF(cores == 0, OP_LOGE(context, "AIV core count must be positive"), return ge::GRAPH_FAILED);
    const int64_t tasks = *numReqs * storage.GetDim(0) * TEMPORAL_TILES;
    context->SetBlockDim(tasks <= 0 ? 1 : static_cast<uint32_t>(tasks > cores ? cores : tasks));
    auto* sizes = context->GetWorkspaceSizes(1); OP_CHECK_NULL_WITH_CONTEXT(context, sizes); sizes[0] = 0;
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus TilingParse(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(PostprocessMambaAlignV310).Tiling(Tiling)
    .TilingParse<PostprocessMambaAlignV310CompileInfo>(TilingParse);
}  // namespace optiling
