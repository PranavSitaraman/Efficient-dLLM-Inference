import pytest
import torch

from aoae.agreement_signals import compute_reuse_signal


def _cfg(method: str, threshold: float = 0.0, **extra):
    return {
        "inference": {
            "reuse_signal": {
                "method": method,
                "threshold": threshold,
                "top_k": int(extra.get("top_k", 2)),
                "min_overlap": int(extra.get("min_overlap", 1)),
                "min_streak": int(extra.get("min_streak", 2)),
            }
        }
    }


def test_argmax_match_signal():
    p = torch.tensor([[[6.0, 1.0, 0.0], [0.1, 5.0, 0.0]]])
    q = torch.tensor([[[5.0, 0.5, 0.0], [3.0, 2.0, 0.0]]])  # second position mismatch
    safe, _, _ = compute_reuse_signal(p, q, _cfg("argmax_match"))
    assert safe.tolist() == [[1.0, 0.0]]


def test_topk_overlap_signal_threshold():
    p = torch.tensor([[[6.0, 5.0, 1.0], [5.0, 4.0, 3.0]]])
    q = torch.tensor([[[7.0, 2.0, 6.0], [1.0, 0.0, 2.0]]])  # second argmax differs
    safe, _, _ = compute_reuse_signal(p, q, _cfg("topk_overlap", top_k=1, min_overlap=1))
    assert safe.tolist() == [[1.0, 0.0]]


def test_min_confidence_signal():
    p = torch.tensor([[[10.0, 0.0, 0.0], [1.0, 0.9, 0.8]]])
    q = torch.tensor([[[9.5, 0.1, 0.0], [1.1, 1.0, 0.9]]])
    safe, _, _ = compute_reuse_signal(p, q, _cfg("min_confidence", threshold=0.7))
    # first confident match passes, second low-confidence match fails.
    assert safe.tolist() == [[1.0, 0.0]]


def test_min_margin_signal():
    p = torch.tensor([[[8.0, 0.0, -1.0], [2.0, 1.95, 0.0]]])
    q = torch.tensor([[[7.5, 0.0, -1.0], [2.2, 2.1, 0.0]]])
    safe, _, _ = compute_reuse_signal(p, q, _cfg("min_margin", threshold=0.3))
    assert safe.tolist() == [[1.0, 0.0]]


def test_js_divergence_signal():
    p = torch.tensor([[[5.0, 0.1, 0.0], [5.0, 0.1, 0.0]]])
    q = torch.tensor([[[5.0, 0.1, 0.0], [0.0, 0.1, 5.0]]])
    safe, _, _ = compute_reuse_signal(p, q, _cfg("js_divergence", threshold=0.05))
    assert safe.tolist() == [[1.0, 0.0]]


def test_temporal_confidence_signal_requires_streak():
    p = torch.tensor([[[8.0, 0.1, 0.0]]])
    q = torch.tensor([[[7.0, 0.1, 0.0]]])
    cfg = _cfg("temporal_confidence", threshold=0.6, min_streak=2)
    safe1, state, _ = compute_reuse_signal(p, q, cfg, state=None)
    safe2, _, _ = compute_reuse_signal(p, q, cfg, state=state)
    assert safe1.tolist() == [[0.0]]
    assert safe2.tolist() == [[1.0]]


def test_unknown_signal_method_raises():
    p = torch.zeros(1, 1, 3)
    q = torch.zeros(1, 1, 3)
    with pytest.raises(ValueError):
        compute_reuse_signal(p, q, _cfg("not_a_method"))
