import torch


def _cfg(feature_mode="scalar_only"):
    return {
        "phase_a_v2": True,
        "phase_a_v2_config": {
            "feature_mode": feature_mode,
            "hidden_dim": 8,
            "target_u_rate": 0.10,
            "target_r_rate": 0.02,
        },
        "policy": {
            "d_model": 16,
            "n_layers": 1,
            "n_heads": 4,
            "dropout": 0.0,
        },
        "grpo": {
            "train_heads": ["unmask", "remask"],
            "include_heads_in_logprob": ["unmask", "remask"],
        },
    }


def test_v5_hybrid_accepts_bf16_model_hidden_states():
    from aoae.phase_a_v2 import PhaseAV2Policy

    policy = PhaseAV2Policy(_cfg("v5_hybrid"), input_dim=8)
    H_t = torch.randn(1, 4, 8)
    mask = torch.tensor([[True, False, True, False]])
    confidence = torch.rand(1, 4)
    agreement = torch.rand(1, 4)
    frontier = torch.zeros(1, 4)
    hidden = torch.randn(1, 4, 8, dtype=torch.bfloat16)

    out = policy(
        H_t,
        mask,
        0.25,
        confidence=confidence,
        agreement=agreement,
        frontier_membership=frontier,
        aux_h_final=hidden,
        pri_h_final=hidden,
    )

    assert torch.isfinite(out["unmask_probs"]).all()
    assert torch.isfinite(out["remask_probs"]).all()


def test_candidate_extraction_scopes_u_and_r_domains():
    from aoae.phase_a_v2 import extract_remask_candidates, extract_unmask_candidates

    mask = torch.tensor([[True, False, True, False]])
    eligible = torch.tensor([[True, True, False, False]])

    assert torch.equal(extract_unmask_candidates(mask), torch.tensor([[True, False, True, False]]))
    assert torch.equal(extract_remask_candidates(mask, eligible), torch.tensor([[False, True, False, False]]))


def test_forced_reject_safety_remasks_regardless_of_learned_r():
    from aoae.phase_a_v2 import apply_safe_remask

    tokens = torch.tensor([[10, 11, 12, 13]])
    forced = torch.tensor([[False, True, False, False]])
    learned = torch.tensor([[False, False, True, True]])
    exec_candidates = torch.tensor([[False, False, True, False]])

    out = apply_safe_remask(tokens, 99, forced, learned, exec_candidates)

    assert torch.equal(out, torch.tensor([[10, 99, 99, 13]]))


def test_feature_construction_marks_u_agreement_unavailable_and_r_available():
    from aoae.phase_a_v2 import build_phase_a_features

    confidence = torch.tensor([[0.2, 0.8]])
    agreement = torch.tensor([[1.0, 0.0]])
    u_feats, u_avail = build_phase_a_features(
        head="u",
        confidence=confidence,
        step_frac=0.25,
        agreement=agreement,
    )
    r_feats, r_avail = build_phase_a_features(
        head="r",
        confidence=confidence,
        step_frac=0.25,
        agreement=agreement,
        frontier_membership=torch.tensor([[0.0, 1.0]]),
    )

    assert torch.allclose(u_feats[..., 0], confidence)
    assert torch.allclose(u_feats[..., 2], torch.full_like(confidence, 0.25))
    assert torch.equal(u_feats[..., 1], torch.zeros_like(confidence))
    assert u_avail["agreement"] is False
    assert r_avail["agreement"] is True
    assert r_avail["frontier_membership"] is True
    assert torch.allclose(r_feats[..., 1], agreement)


def test_warmstart_label_construction_uses_partial_labels():
    from aoae.phase_a_v2 import build_remask_labels, build_unmask_labels

    candidates = torch.tensor([[True, True, True, True]])
    selected = torch.tensor([[True, True, False, False]])
    accepted = torch.tensor([[True, False, False, False]])
    confidence = torch.tensor([[0.9, 0.9, 0.8, 0.01]])

    u = build_unmask_labels(
        candidate_mask=candidates,
        heuristic_selected=selected,
        verifier_accepted=accepted,
        confidence=confidence,
    )

    assert u.labels[0, 0].item() == 1.0
    assert u.labels[0, 1].item() == 0.0
    assert u.labels[0, 2].item() < 0.0
    assert u.labels[0, 3].item() == 0.0

    r = build_remask_labels(
        candidate_mask=candidates,
        forced_rejects=torch.tensor([[True, False, False, False]]),
        accepted_frontier=torch.tensor([[False, True, False, False]]),
    )

    assert r.labels[0, 0].item() == 1.0
    assert r.labels[0, 1].item() == 0.0


def test_weighted_bce_ignores_unlabeled_examples():
    from aoae.phase_a_v2 import weighted_bce_ignore_unlabeled

    logits = torch.tensor([[0.0, 100.0]], requires_grad=True)
    labels = torch.tensor([[1.0, -1.0]])
    weights = torch.tensor([[2.0, 99.0]])

    loss = weighted_bce_ignore_unlabeled(logits, labels, weights)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, :1],
        labels[:, :1],
    )

    assert torch.allclose(loss, expected)


def test_phase_a_policy_forward_hidden_residual_starts_near_scalar_behavior():
    from aoae.phase_a_v2 import PhaseAV2Policy

    torch.manual_seed(0)
    scalar = PhaseAV2Policy(_cfg("scalar_only"), input_dim=8)
    hidden = PhaseAV2Policy(_cfg("hidden_residual"), input_dim=8)
    hidden.load_state_dict(scalar.state_dict(), strict=False)
    H = torch.randn(1, 4, 8)
    mask = torch.tensor([[True, False, True, False]])
    confidence = torch.tensor([[0.1, 0.2, 0.9, 0.8]])

    out_scalar = scalar(H, mask, 0.5, confidence=confidence)
    out_hidden = hidden(H, mask, 0.5, confidence=confidence)

    assert torch.allclose(out_scalar["unmask_logits"], out_hidden["unmask_logits"], atol=1e-6)
    assert "hidden/delta_logit_norm_u" in out_hidden


def test_tiny_supervised_overfit_reduces_phase_a_loss():
    from aoae.phase_a_v2 import (
        PhaseAV2Policy,
        WarmStartLabels,
        phase_a_supervised_loss,
    )

    torch.manual_seed(0)
    policy = PhaseAV2Policy(_cfg("scalar_only"), input_dim=8)
    opt = torch.optim.AdamW(policy.parameters(), lr=5e-3)
    H = torch.randn(2, 4, 8)
    mask = torch.tensor([[True, True, False, False], [True, False, True, False]])
    conf = torch.tensor([[0.9, 0.1, 0.2, 0.4], [0.8, 0.2, 0.7, 0.1]])
    u = WarmStartLabels(
        labels=torch.tensor([[1.0, 0.0, -1.0, -1.0], [1.0, -1.0, 1.0, -1.0]]),
        weights=torch.tensor([[3.0, 1.0, 0.0, 0.0], [3.0, 0.0, 3.0, 0.0]]),
    )
    r = WarmStartLabels(
        labels=torch.tensor([[-1.0, -1.0, 1.0, 0.0], [-1.0, 0.0, -1.0, 1.0]]),
        weights=torch.tensor([[0.0, 0.0, 3.0, 3.0], [0.0, 3.0, 0.0, 3.0]]),
    )

    losses = []
    for _ in range(40):
        out = policy(H, mask, 0.5, confidence=conf)
        loss, _ = phase_a_supervised_loss(out, u, r)
        losses.append(float(loss.item()))
        opt.zero_grad()
        loss.backward()
        opt.step()

    assert losses[-1] < losses[0] * 0.75


def test_phase_a_grpo_logprob_only_includes_scoped_u_r_actions():
    from aoae.phase_a_v2 import PhaseAV2Policy

    policy = PhaseAV2Policy(_cfg("scalar_only"), input_dim=8)
    H = torch.randn(1, 4, 8)
    mask = torch.tensor([[True, False, True, False]])
    conf = torch.ones(1, 4) * 0.5
    out = policy(H, mask, 0.5, confidence=conf)
    actions = {
        "u_t": torch.tensor([[1.0, 1.0, 0.0, 1.0]]),
        "r_t": torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
        "kappa_t": torch.ones(1, 4),
        "q_t": torch.ones(1, 4),
    }

    lp = policy.log_prob(out, actions, include_heads={"unmask", "remask"})
    manual = (
        torch.log(out["unmask_probs"][0, 0].clamp(1e-7, 1 - 1e-7))
        + torch.log1p(-out["unmask_probs"][0, 2].clamp(1e-7, 1 - 1e-7))
        + torch.log(out["remask_probs"][0, 1].clamp(1e-7, 1 - 1e-7))
        + torch.log1p(-out["remask_probs"][0, 3].clamp(1e-7, 1 - 1e-7))
    )

    assert torch.allclose(lp.squeeze(), manual)


def test_synthetic_grpo_update_from_warmstart_checkpoint(tmp_path):
    from aoae.phase_a_v2 import PhaseAV2Policy
    from aoae.train_grpo import compute_grpo_loss

    policy = PhaseAV2Policy(_cfg("scalar_only"), input_dim=8)
    ckpt = tmp_path / "warm.pt"
    torch.save({"policy": policy.state_dict()}, ckpt)

    loaded = PhaseAV2Policy(_cfg("scalar_only"), input_dim=8)
    loaded.load_state_dict(torch.load(ckpt, map_location="cpu")["policy"])
    opt = torch.optim.AdamW(loaded.parameters(), lr=1e-3)

    H = torch.randn(1, 4, 8)
    mask = torch.tensor([[True, False, True, False]])
    conf = torch.ones(1, 4) * 0.5
    out = loaded(H, mask, 0.5, confidence=conf)
    actions = loaded.sample_actions(out, mask)
    old_lp = loaded.log_prob(out, actions, include_heads={"unmask", "remask"}).detach()
    traj = {
        "actions_list": [{k: v.detach() for k, v in actions.items()}],
        "old_log_probs": [old_lp],
        "H_t_list": [H.detach()],
        "mask_ind_list": [mask],
        "confidence_list": [conf],
        "quality_scores_list": [None],
        "step_fracs": [0.5],
    }

    loss = compute_grpo_loss(
        loaded,
        soft_mask_module=object(),
        trajectories=[traj],
        advantages=torch.tensor([1.0]),
        clip_eps=0.2,
        include_heads_in_logprob={"unmask", "remask"},
    )
    opt.zero_grad()
    loss.backward()
    opt.step()

    assert torch.isfinite(loss)


def test_warmstart_step_labels_map_future_verifier_outcomes():
    from types import SimpleNamespace
    from aoae.train_warmstart import _make_step_labels

    traj = SimpleNamespace(
        mask_ind_list=[
            torch.tensor([[True, True, False]]),
            torch.tensor([[False, True, False]]),
        ],
        confidence_list=[
            torch.tensor([[0.9, 0.2, 0.1]]),
            torch.tensor([[0.9, 0.2, 0.1]]),
        ],
        actions=[
            {"u_t": torch.tensor([[1.0, 1.0, 0.0]])},
            {"u_t": torch.zeros(1, 3)},
        ],
        run_primary_list=[False, True],
        frontier_accept_mask_list=[
            torch.zeros(1, 3, dtype=torch.bool),
            torch.tensor([[True, False, False]]),
        ],
        frontier_reject_mask_list=[
            torch.zeros(1, 3, dtype=torch.bool),
            torch.tensor([[False, True, False]]),
        ],
    )

    u_labels, _, _, _ = _make_step_labels(traj, 0, low_confidence_threshold=0.15)
    _, r_labels, _, r_train = _make_step_labels(traj, 1, low_confidence_threshold=0.15)

    assert u_labels.labels[0, 0].item() == 1.0
    assert u_labels.labels[0, 1].item() == 0.0
    assert r_labels.labels[0, 1].item() == 1.0
    assert r_train[0, 1].item() is True


def test_expert_steering_enabled_adds_expert_count_and_expert_actions_respect_domains():
    from aoae.phase_a_v2 import PhaseAHeuristicExpert

    expert = PhaseAHeuristicExpert(draft_threshold=0.5)
    mask = torch.tensor([[True, False, True, False]])
    conf = torch.tensor([[0.6, 0.9, 0.4, 0.9]])
    remask_candidates = torch.tensor([[False, True, False, False]])
    actions = expert.actions(
        confidence=conf,
        mask_indicator=mask,
        remask_candidate_mask=remask_candidates,
        forced_rejects=torch.tensor([[False, True, False, True]]),
    )

    assert torch.equal(actions["u_t"].bool(), torch.tensor([[True, False, False, False]]))
    assert torch.equal(actions["r_t"].bool(), torch.tensor([[False, True, False, False]]))
