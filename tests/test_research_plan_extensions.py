import torch


def test_preflight_report_contains_runtime_and_backend():
    from aoae.preflight import run_preflight

    report = run_preflight("configs/paper.yaml", strict_moe=False)
    assert "runtime" in report
    assert "backend" in report
    assert report["config_path"] == "configs/paper.yaml"
    assert "torch_version" in report["runtime"]


def _base_pos_cache_cfg(candidate_policy: str):
    return {
        "inference": {
            "positional_cache": {
                "enabled": True,
                "horizon": 4,
                "refresh_budget": 2,
                "candidate_policy": candidate_policy,
                "window_radius": 1,
                "use_topb_from_probs": True,
                "force_mandatory": True,
                "age_cap": 64,
            }
        }
    }


def test_positional_cache_candidate_policies_respect_budget_and_mandatory():
    from aoae.positional_cache import build_access_set

    B, L = 1, 8
    u_t = torch.zeros(B, L)
    r_t = torch.zeros(B, L)
    u_t[0, 2] = 1.0
    actions = {
        "u_t": u_t,
        "r_t": r_t,
        "kappa_t": torch.zeros(B, L),
        "q_t": torch.ones(B, L),
    }
    policy_out = {"access_probs": torch.linspace(0.0, 1.0, L).unsqueeze(0)}
    confidence = torch.linspace(1.0, 0.0, L).unsqueeze(0)

    for policy in ("learned_topb", "sliding_window", "confidence_topb"):
        cfg = _base_pos_cache_cfg(policy)
        q_exec, mandatory, _diag = build_access_set(actions, policy_out, cfg, confidence=confidence)
        # Mandatory must always be included
        assert q_exec[0, 2].item() == 1.0
        assert mandatory[0, 2].item() == 1.0
        # Optional selected positions cannot exceed budget
        optional = (q_exec.bool() & ~mandatory.bool()).sum().item()
        assert optional <= 2


def test_boundary_head_policy_outputs_and_actions():
    from aoae.models.policy import AOAEPolicy

    cfg = {
        "policy": {
            "d_model": 64,
            "n_layers": 1,
            "n_heads": 4,
            "dropout": 0.0,
            "use_positional_features": False,
            "boundary_head": {"enabled": True, "num_bins": 6},
        }
    }
    B, L, D = 2, 10, 32
    policy = AOAEPolicy(cfg, input_dim=D)
    H = torch.randn(B, L, D)
    mask = torch.randint(0, 2, (B, L)).bool()
    out = policy(H, mask, step_frac=0.5, temperature=1.0)
    assert "boundary_probs" in out
    assert out["boundary_probs"].shape == (B, 6)
    actions = policy.sample_actions(out, mask)
    assert "ell_t" in actions
    assert actions["ell_t"].shape == (B,)
    lp = policy.log_prob(out, actions)
    assert torch.isfinite(lp).all()


def test_policy_zero_temperature_remains_finite():
    from aoae.models.policy import AOAEPolicy

    cfg = {
        "policy": {
            "d_model": 32,
            "n_layers": 1,
            "n_heads": 4,
            "dropout": 0.0,
            "use_positional_features": False,
        }
    }
    B, L, D = 1, 6, 16
    policy = AOAEPolicy(cfg, input_dim=D)
    H = torch.randn(B, L, D)
    mask = torch.randint(0, 2, (B, L)).bool()
    out = policy(H, mask, step_frac=0.5, temperature=0.0)
    assert torch.isfinite(out["unmask_probs"]).all()
    assert torch.isfinite(out["remask_probs"]).all()
    assert torch.isfinite(out["cache_probs"]).all()
    assert torch.isfinite(out["access_probs"]).all()


def test_policy_can_disable_agreement_feature():
    from aoae.models.policy import AOAEPolicy

    cfg = {
        "policy": {
            "d_model": 32,
            "n_layers": 1,
            "n_heads": 4,
            "dropout": 0.0,
            "use_agreement_feature": False,
        }
    }
    policy = AOAEPolicy(cfg, input_dim=16)
    H = torch.randn(2, 6, 16)
    mask = torch.randint(0, 2, (2, 6)).bool()
    agreement = torch.ones(2, 6)
    out = policy(H, mask, step_frac=0.5, agreement=agreement)
    assert out["unmask_probs"].shape == (2, 6)


def test_compute_reward_prefers_faster_completion_given_equal_correctness():
    from aoae.inference import AOAETrajectory
    from aoae.train_grpo import compute_reward

    class _Tok:
        def decode(self, *_args, **_kwargs):
            return "#### 1"

    cfg = {"grpo": {"alpha": 1.0, "beta": 0.0, "access_reward_weight": 0.0}}
    gen = torch.tensor([[1, 2, 3]], dtype=torch.long)
    refs = ["#### 1"]
    tok = _Tok()

    traj_fast = AOAETrajectory()
    traj_fast.actions = [{}]
    traj_fast.thrash_counts = [torch.zeros(1)]
    traj_fast.completion_step = torch.tensor([2])  # faster

    traj_slow = AOAETrajectory()
    traj_slow.actions = [{}]
    traj_slow.thrash_counts = [torch.zeros(1)]
    traj_slow.completion_step = torch.tensor([8])  # slower

    r_fast, _ = compute_reward(gen, refs, tok, traj_fast, cfg, T=8)
    r_slow, _ = compute_reward(gen, refs, tok, traj_slow, cfg, T=8)
    assert r_fast.item() > r_slow.item()


def test_routing_tradeoff_annotation_adds_frontier_and_deltas():
    from aoae.paper import _annotate_tradeoff

    rows = [
        {"routing_mode": "hard", "accuracy": "0.60", "tps": "100.0"},
        {"routing_mode": "soft", "accuracy": "0.62", "tps": "95.0"},
        {"routing_mode": "soft", "accuracy": "0.58", "tps": "130.0"},
    ]
    _annotate_tradeoff(rows)
    for row in rows:
        assert "frontier_index" in row
        assert "delta_accuracy_vs_hard" in row
        assert "delta_tps_vs_hard" in row


def test_reuse_decision_table_emits_expected_constraints():
    from aoae.paper import _decision_table

    rows = [
        {"reuse_signal_method": "argmax_match", "reuse_signal_threshold": "0.0", "accuracy": "0.60", "tps": "100.0", "thrash_rate_given_cached": "0.10"},
        {"reuse_signal_method": "js_divergence", "reuse_signal_threshold": "0.05", "accuracy": "0.59", "tps": "120.0", "thrash_rate_given_cached": "0.30"},
        {"reuse_signal_method": "min_confidence", "reuse_signal_threshold": "0.70", "accuracy": "0.60", "tps": "110.0", "thrash_rate_given_cached": "0.05"},
    ]
    decisions = _decision_table(rows, argmax_acc=0.60)
    assert len(decisions) == 3
    assert all("constraint" in r for r in decisions)
