// SPDX-License-Identifier: Apache-2.0
#include "register/op_def_registry.h"

namespace ops {

class PostprocessSampledV310 : public OpDef {
public:
    explicit PostprocessSampledV310(const char* name) : OpDef(name)
    {
        this->Input("idxMapping").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("numComputedTokens").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("lastSampledTokens").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("outputBinCounts").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("sampledTokens").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("numSampled").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("numRejected").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("queryStartLoc").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("allTokenIds").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("totalLen").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});

        this->Output("numComputedTokensOut").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("lastSampledTokensOut").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Output("allTokenIdsOut").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("totalLenOut").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});

        this->Attr("hasOutputBinCounts").AttrType(REQUIRED).Bool();
        this->Attr("hasQueryStartLoc").AttrType(REQUIRED).Bool();

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(false)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("coreType.value", "AiCore");
        this->AICore().AddConfig("ascend310p", config);
    }
};

OP_ADD(PostprocessSampledV310);

}  // namespace ops
