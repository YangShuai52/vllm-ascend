// SPDX-License-Identifier: Apache-2.0
#include "register/op_def_registry.h"
namespace ops {
class PostprocessMambaAlignV310 : public OpDef {
public:
    explicit PostprocessMambaAlignV310(const char* name) : OpDef(name)
    {
        this->Input("idxMapping").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("numAcceptedTokens").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("stateIdx").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("numComputedTokens").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("blockTablePtrs").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("stateBaseAddrs").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("stateBlockStrides").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("stateElemSizes").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("stateInnerSizes").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("stateConvWidths").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("stateGroupIndices").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("stateDimRowCount").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("stateDimRowStride").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Output("numAcceptedTokensOut").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Attr("numReqs").AttrType(REQUIRED).Int();
        this->Attr("blockSize").AttrType(REQUIRED).Int();
        this->Attr("blockTableStrideReq").AttrType(REQUIRED).Int();
        this->Attr("convStateDimFirst").AttrType(REQUIRED).Bool();
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(false)
            .DynamicRankSupportFlag(false).DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false).PrecisionReduceFlag(false)
            .ExtendCfgInfo("coreType.value", "AiCore");
        this->AICore().AddConfig("ascend310p", config);
    }
};
OP_ADD(PostprocessMambaAlignV310);
}  // namespace ops
