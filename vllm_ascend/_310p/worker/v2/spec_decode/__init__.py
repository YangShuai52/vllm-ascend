# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""310P spec_decode package: Triton-free speculator for MTP eager mode."""
from vllm_ascend._310p.worker.v2.spec_decode.speculator import (
    Ascend310PMTPSpeculator,
    init_310p_speculator,
)
