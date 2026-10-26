# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.ops.fused_moe import moe_comm_method as comm


@pytest.mark.parametrize("device_type", [AscendDeviceType._310P, AscendDeviceType.A2, AscendDeviceType.A3])
@pytest.mark.parametrize("ep_size", [1, 4])
def test_setup_only_initializes_supported_backends(device_type, ep_size):
    with (
        patch.object(comm, "get_current_hardware_profile", return_value=get_hardware_profile(device_type)),
        patch.object(comm, "_MoECommMethods", {}),
        patch.object(comm, "AlltoAllCommImpl") as alltoall,
        patch.object(comm, "AllGatherCommImpl") as allgather,
        patch.object(comm, "MC2CommImpl") as mc2,
        patch.object(comm, "FusedMC2CommImpl") as fused_mc2,
    ):
        config = SimpleNamespace(ep_size=ep_size, num_experts=256, num_local_experts=256 // ep_size)
        comm.setup_moe_comm_method(config)
        allgather.assert_called_once_with(config)
        if ep_size > 1:
            alltoall.assert_called_once_with(config)
        else:
            alltoall.assert_not_called()
        if device_type is not AscendDeviceType._310P and ep_size > 1:
            mc2.assert_called_once_with(config)
            fused_mc2.assert_called_once_with(config)
        else:
            mc2.assert_not_called()
            fused_mc2.assert_not_called()
