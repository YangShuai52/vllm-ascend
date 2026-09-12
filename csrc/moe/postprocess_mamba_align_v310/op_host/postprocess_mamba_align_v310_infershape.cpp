// SPDX-License-Identifier: Apache-2.0
#include "register/op_impl_registry.h"
namespace ops {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(2);
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (inputShape == nullptr || outputShape == nullptr) return ge::GRAPH_FAILED;
    *outputShape = *inputShape;
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(PostprocessMambaAlignV310).InferShape(InferShape);
}  // namespace ops
