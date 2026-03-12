from aoae import runtime_checks as rc


def test_seed_helper_is_deterministic():
    import torch

    rc.set_global_seed(123, deterministic=False)
    a = torch.rand(5)
    rc.set_global_seed(123, deterministic=False)
    b = torch.rand(5)
    assert torch.allclose(a, b)


def test_moe_runtime_fallback_patch(monkeypatch):
    import sys
    import types

    calls = {"count": 0}

    def _fake_inspect():
        calls["count"] += 1
        return {
            "namespace_present": False,
            "available_ops": [],
            "missing_required_ops": ["moe_align_block_size", "moe_sum"],
        }

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__path__ = []  # mark as package for submodule imports
    fake_custom_ops = types.ModuleType("vllm._custom_ops")
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm._custom_ops", fake_custom_ops)
    monkeypatch.setattr(rc, "inspect_vllm_moe_ops", _fake_inspect)
    report = rc.ensure_vllm_moe_runtime(strict=False, verbose=False)
    assert set(report["patched_fallback_ops"]) == {"moe_align_block_size", "moe_sum"}
    assert report["missing_required_ops"] == []
