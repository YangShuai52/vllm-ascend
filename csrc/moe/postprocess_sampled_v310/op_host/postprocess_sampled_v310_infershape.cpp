// SPDX-License-Identifier: Apache-2.0
#include "register/op_impl_registry.h"

namespace ops {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    constexpr size_t inputIndices[] = {1, 2, 8, 9};
    for (size_t i = 0; i < 4; ++i) {
        const gert::Shape* inputShape = context->GetInputShape(inputIndices[i]);
        gert::Shape* outputShape = context->GetOutputShape(i);
        if (inputShape == nullptr || outputShape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        *outputShape = *inputShape;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    constexpr size_t inputIndices[] = {1, 2, 8, 9};
    for (size_t i = 0; i < 4; ++i) {
        context->SetOutputDataType(i, context->GetInputDataType(inputIndices[i]));
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(PostprocessSampledV310)
    .InferShape(InferShape)
    .InferDataType(InferDataType);

}  // namespace ops
