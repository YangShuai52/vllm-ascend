/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*!
 * \file fused_gdn_gating.cpp
 * \brief L0-level API for FusedGdnGating.
 */

#include "fused_gdn_gating.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(FusedGdnGating);

static constexpr FusedGdnGatingOutput kNullOutput{nullptr, nullptr};

FusedGdnGatingOutput FusedGdnGating(const aclTensor *aLog, const aclTensor *a,
                                    const aclTensor *b, const aclTensor *dtBias,
                                    float beta, float threshold,
                                    aclTensor *g, aclTensor *betaOutput,
                                    aclOpExecutor *executor)
{
    L0_DFX(FusedGdnGating, aLog, a, b, dtBias, beta, threshold);

    OP_CHECK(g != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "g output tensor is null."),
             return kNullOutput);
    OP_CHECK(betaOutput != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "beta_output tensor is null."),
             return kNullOutput);

    auto ret = INFER_SHAPE(FusedGdnGating,
                           OP_INPUT(aLog, a, b, dtBias),
                           OP_OUTPUT(g, betaOutput),
                           OP_ATTR(beta, threshold));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return kNullOutput,
                        "FusedGdnGating InferShape failed.");

    ret = ADD_TO_LAUNCHER_LIST_AICORE(FusedGdnGating,
                                      OP_INPUT(aLog, a, b, dtBias),
                                      OP_OUTPUT(g, betaOutput),
                                      OP_ATTR(beta, threshold));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return kNullOutput,
        "FusedGdnGating ADD_TO_LAUNCHER_LIST_AICORE failed.");

    return FusedGdnGatingOutput{g, betaOutput};
}

} // namespace l0op
