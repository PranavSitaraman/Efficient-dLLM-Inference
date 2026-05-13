import torch

from aoae.positional_cache import build_access_set


def test_boundary_action_scales_optional_budget_per_sample():
    cfg = {
        "inference": {
            "positional_cache": {
                "enabled": True,
                "horizon": 4,
                "refresh_budget": 4,
                "candidate_policy": "learned_topb",
                "window_radius": 4,
                "use_topb_from_probs": True,
                "force_mandatory": True,
                "age_cap": 64,
            }
        }
    }
    B, L = 2, 10
    actions = {
        "u_t": torch.zeros(B, L),
        "r_t": torch.zeros(B, L),
        "kappa_t": torch.zeros(B, L),
        "q_t": torch.ones(B, L),  # allow optional selection everywhere
        "ell_t": torch.tensor([0, 3]),  # shallow and deep boundary
    }
    policy_out = {"access_probs": torch.linspace(0.0, 1.0, L).repeat(B, 1)}

    q_exec, mandatory, diag = build_access_set(
        actions=actions,
        policy_out=policy_out,
        cfg=cfg,
        confidence=None,
        boundary_action=actions["ell_t"],
        boundary_num_bins=4,
    )

    optional = (q_exec.bool() & ~mandatory.bool()).sum(dim=1)
    # budget for ell=0 is round(1/4 * 4)=1, for ell=3 is round(4/4 * 4)=4
    assert int(optional[0].item()) <= 1
    assert int(optional[1].item()) <= 4
    assert float(diag["access_effective_budget"]) > 0.0

