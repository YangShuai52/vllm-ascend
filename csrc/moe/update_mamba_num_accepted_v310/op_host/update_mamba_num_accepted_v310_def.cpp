// SPDX-License-Identifier: Apache-2.0
#include "register/op_def_registry.h"
namespace ops {
class UpdateMambaNumAcceptedV310 : public OpDef {
public:
    explicit UpdateMambaNumAcceptedV310(const char* name) : OpDef(name)
    {
        this->Input("idxMapping").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("numSampled").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Input("numAcceptedTokens").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("numAcceptedTokensOut").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Attr("numReqs").AttrType(REQUIRED).Int();
        this->Attr("scalarNumSampled").AttrType(REQUIRED).Int();
        this->Attr("hasTensorNumSampled").AttrType(REQUIRED).Bool();
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(false)
            .DynamicRankSupportFlag(false).DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false).PrecisionReduceFlag(false)
            .ExtendCfgInfo("coreType.value", "AiCore");
        this->AICore().AddConfig("ascend310p", config);
    }
};
OP_ADD(UpdateMambaNumAcceptedV310);
}  // namespace ops
