import math

import pytest
import torch

from aoae.evaluate import _aggregate_kv_dynamics
from aoae.kv_dynamics import SpeculativeDynamicsTracker


def _base_inputs():
    return {
        "max_prob": torch.tensor([[0.9, 0.8]], dtype=torch.float32),
        "mask_ind": torch.tensor([[True, True]]),
        "agreement": torch.ones(1, 2, dtype=torch.float32),
        "u_t": torch.zeros(1, 2, dtype=torch.float32),
        "r_t": torch.zeros(1, 2, dtype=torch.float32),
        "kappa_t": torch.zeros(1, 2, dtype=torch.float32),
        "q_t": torch.zeros(1, 2, dtype=torch.float32),
    }


def test_tracker_prefers_exact_kv_drift_and_tracks_attention_deviation():
    tracker = SpeculativeDynamicsTracker({"analysis": {"track_attention_deviation": True}})
    inputs = _base_inputs()

    hidden_1 = [torch.zeros(1, 2, 3)]
    hidden_2 = [torch.full((1, 2, 3), 10.0)]
    kv_1 = [(
        torch.zeros(1, 1, 2, 2),
        torch.zeros(1, 1, 2, 2),
    )]
    kv_2 = [(
        torch.ones(1, 1, 2, 2),
        torch.ones(1, 1, 2, 2),
    )]
    attn_1 = [torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])]
    attn_2 = [torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])]

    tracker.observe_step(
        layer_hiddens=hidden_1,
        layer_kv=kv_1,
        layer_attentions=attn_1,
        **inputs,
    )
    tracker.observe_step(
        layer_hiddens=hidden_2,
        layer_kv=kv_2,
        layer_attentions=attn_2,
        **inputs,
    )

    out = tracker.summarize()
    summary = out["summary"]
    per_layer = out["per_layer"]
    per_layer_attn = out["per_layer_attention_deviation"]

    assert summary["layer_drift_measure"] == "exact_kv"
    assert summary["attention_deviation_available"] is True
    assert summary["exact_kv_drift_steps"] == 1
    assert per_layer[0]["mean_drift"] == pytest.approx(2.0)
    assert per_layer_attn[0]["mean_attention_deviation"] == pytest.approx(math.sqrt(2.0))


def test_tracker_falls_back_to_hidden_state_proxy_without_kv():
    tracker = SpeculativeDynamicsTracker({"analysis": {"track_attention_deviation": False}})
    inputs = _base_inputs()

    tracker.observe_step(layer_hiddens=[torch.zeros(1, 2, 2)], **inputs)
    tracker.observe_step(layer_hiddens=[torch.ones(1, 2, 2)], **inputs)

    summary = tracker.summarize()["summary"]
    assert summary["layer_drift_measure"] == "hidden_state_proxy"
    assert summary["exact_kv_drift_steps"] == 0
    assert summary["hidden_state_proxy_steps"] == 1


def test_aggregate_kv_dynamics_preserves_new_summary_fields():
    records = [
        {
            "kv_dynamics": {
                "summary": {
                    "mean_agreement": 0.5,
                    "mean_access": 0.3,
                    "layer_drift_measure": "exact_kv",
                    "exact_kv_drift_steps": 2,
                    "hidden_state_proxy_steps": 0,
                    "layer_drift_slope": 0.2,
                    "attention_deviation_available": True,
                    "attention_deviation_measure": "mean_l2_delta",
                    "attention_deviation_slope": 0.1,
                },
                "per_layer": [
                    {"layer_idx": 0, "mean_drift": 1.0, "mean_hidden_drift": 1.0},
                ],
                "per_layer_attention_deviation": [
                    {"layer_idx": 0, "mean_attention_deviation": 0.4},
                ],
            }
        }
    ]

    summary = _aggregate_kv_dynamics(records)
    assert summary["layer_drift_measure"] == "exact_kv"
    assert summary["attention_deviation_available"] is True
    assert summary["mean_attention_deviation_slope"] == pytest.approx(0.1)
    assert summary["per_layer_drift"][0]["mean_drift"] == pytest.approx(1.0)
    assert summary["per_layer_attention_deviation"][0]["mean_attention_deviation"] == pytest.approx(0.4)
