// SPDX-License-Identifier: Apache-2.0
#include "register/op_def_registry.h"

namespace ops {

class PreprocessMambaAlignV310 : public OpDef {
public:
    explicit PreprocessMambaAlignV310(const char* name) : OpDef(name)
    {
        for (const char* input : {"idxMapping", "stateIdx", "numComputedTokens", "queryStartLoc",
                                  "numAcceptedTokens", "srcCol", "srcOff"}) {
            this->Input(input).ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        }
        for (const char* output : {"stateIdxOut", "numAcceptedTokensOut", "srcColOut", "srcOffOut"}) {
            this->Output(output).ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        }
        this->Attr("mambaBlockSize").AttrType(REQUIRED).Int();

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

OP_ADD(PreprocessMambaAlignV310);

}  // namespace ops
