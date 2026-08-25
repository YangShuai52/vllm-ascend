# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
from vllm.sequence import IntermediateTensors

import vllm_ascend.patch.worker.patch_deepseek_v2 as patch_deepseek_v2
from vllm_ascend.patch.worker.patch_deepseek_v2 import (
    _PP_TOPK_INDICES_KEY,
    _patched_forward,
    _should_skip_indexer_init,
)


def _config(**overrides) -> SimpleNamespace:
    values = {"num_hidden_layers": 80}
    values.update(overrides)
    return SimpleNamespace(**values)


def test_glm51_skip_topk_keeps_per_layer_indexer():
    assert not _should_skip_indexer_init(
        _config(),
        "model.layers.2.self_attn",
        skip_topk=True,
    )


def test_glm52_shared_layer_skips_indexer_init():
    assert _should_skip_indexer_init(
        _config(indexer_types=["full", "full", "shared"]),
        "model.layers.2.self_attn",
        skip_topk=True,
    )


def test_mtp_layer_keeps_indexer():
    indexer_types = ["full"] * 80 + ["shared"]
    assert not _should_skip_indexer_init(
        _config(indexer_types=indexer_types),
        "model.layers.80.self_attn",
        skip_topk=True,
    )


def test_pp_forward_restores_and_propagates_topk_indices(monkeypatch):
    pp_group = SimpleNamespace(is_first_rank=False, is_last_rank=False)
    monkeypatch.setattr(patch_deepseek_v2, "get_pp_group", lambda: pp_group)

    received_topk_indices = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    model = SimpleNamespace(
        receive_pp_topk_indices=True,
        send_pp_topk_indices=True,
        topk_indices_buffer=torch.zeros((4, 2), dtype=torch.int32),
        config=SimpleNamespace(llama_4_scaling=None),
        layers=[lambda positions, hidden_states, residual, scaling: (hidden_states, residual)],
        start_layer=0,
        end_layer=1,
        aux_hidden_state_layers=(),
    )
    intermediate_tensors = IntermediateTensors(
        {
            "hidden_states": torch.ones((2, 4)),
            "residual": torch.zeros((2, 4)),
            _PP_TOPK_INDICES_KEY: received_topk_indices,
        }
    )

    output = _patched_forward(
        model,
        input_ids=None,
        positions=torch.arange(2),
        intermediate_tensors=intermediate_tensors,
    )

    assert isinstance(output, IntermediateTensors)
    torch.testing.assert_close(
        model.topk_indices_buffer[:2],
        received_topk_indices,
    )
    torch.testing.assert_close(
        output[_PP_TOPK_INDICES_KEY],
        received_topk_indices,
    )
