// SPDX-License-Identifier: Apache-2.0
#include "update_mamba_num_accepted_v310_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/error_log.h"
namespace optiling {
struct UpdateMambaNumAcceptedV310CompileInfo {};
static ge::graphStatus Tiling(gert::TilingContext* context)
{
    auto* attrs = context->GetAttrs(); OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* numReqs = attrs->GetAttrPointer<int64_t>(0);
    const int64_t* scalarSampled = attrs->GetAttrPointer<int64_t>(1);
    const bool* tensorSampled = attrs->GetAttrPointer<bool>(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, numReqs);
    OP_CHECK_NULL_WITH_CONTEXT(context, scalarSampled);
    OP_CHECK_NULL_WITH_CONTEXT(context, tensorSampled);
    OP_CHECK_IF(*numReqs < 0, OP_LOGE(context, "numReqs must be non-negative"),
                return ge::GRAPH_FAILED);
    UpdateMambaNumAcceptedV310TilingData data;
    data.set_numReqs(*numReqs); data.set_scalarNumSampled(*scalarSampled);
    data.set_hasTensorNumSampled(*tensorSampled ? 1 : 0);
    auto* raw = context->GetRawTilingData(); OP_CHECK_NULL_WITH_CONTEXT(context, raw);
    data.SaveToBuffer(raw->GetData(), raw->GetCapacity()); raw->SetDataSize(data.GetDataSize());
    fe::PlatFormInfos* info = context->GetPlatformInfo(); OP_CHECK_NULL_WITH_CONTEXT(context, info);
    platform_ascendc::PlatformAscendC platform(info);
    const uint32_t cores = platform.GetCoreNumAiv();
    OP_CHECK_IF(cores == 0, OP_LOGE(context, "AIV core count must be positive"), return ge::GRAPH_FAILED);
    context->SetBlockDim(*numReqs <= 0 ? 1 : static_cast<uint32_t>(*numReqs > cores ? cores : *numReqs));
    auto* sizes = context->GetWorkspaceSizes(1); OP_CHECK_NULL_WITH_CONTEXT(context, sizes); sizes[0] = 0;
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus TilingParse(gert::TilingParseContext*) { return ge::GRAPH_SUCCESS; }
IMPL_OP_OPTILING(UpdateMambaNumAcceptedV310).Tiling(Tiling)
    .TilingParse<UpdateMambaNumAcceptedV310CompileInfo>(TilingParse);
}  // namespace optiling
