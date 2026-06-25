import torch

from vllm_ascend._310p.ops.fla import gdn_310
from vllm_ascend.ascend_forward_context import _EXTRA_CTX


def test_fused_gdn_gating_310_uses_pytorch_during_capture(monkeypatch):
    called = {"custom": 0, "pytorch": 0}

    def fake_custom_op(*args, **kwargs):
        called["custom"] += 1
        raise AssertionError("custom op should not run during ACL graph capture")

    def fake_pytorch(*args, **kwargs):
        called["pytorch"] += 1
        batch, num_heads = args[1].shape
        g = torch.zeros(1, batch, num_heads, dtype=torch.float32, device=args[1].device)
        beta = torch.zeros(1, batch, num_heads, dtype=args[1].dtype, device=args[1].device)
        return g, beta

    monkeypatch.setattr(_EXTRA_CTX, "capturing", True)
    monkeypatch.setattr(
        torch.ops._C_ascend,
        "npu_fused_gdn_gating",
        fake_custom_op,
    )
    monkeypatch.setattr(gdn_310, "fused_gdn_gating_pytorch", fake_pytorch)

    a_log = torch.randn(4, dtype=torch.float16)
    dt_bias = torch.randn(4, dtype=torch.float16)
    a = torch.randn(3, 4, dtype=torch.float16)
    b = torch.randn(3, 4, dtype=torch.float16)

    g, beta = gdn_310._fused_gdn_gating_310(a_log, a, b, dt_bias)

    assert called["pytorch"] == 1
    assert called["custom"] == 0
    assert g.shape == (1, 3, 4)
    assert beta.shape == (1, 3, 4)
