import torch
import pytest
from types import SimpleNamespace


def test_normalize_group_advantages_matches_paper_default_centering():
    from aoae.train_grpo import normalize_group_advantages

    rewards = torch.tensor([1.0, 2.0, 5.0], dtype=torch.float32)
    advantages = normalize_group_advantages(rewards, normalize_std=False)

    assert torch.allclose(advantages, rewards - rewards.mean())


def test_normalize_group_advantages_can_optionally_standardize():
    from aoae.train_grpo import normalize_group_advantages

    rewards = torch.tensor([1.0, 2.0, 5.0], dtype=torch.float32)
    advantages = normalize_group_advantages(rewards, normalize_std=True)

    assert torch.isclose(advantages.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(advantages.std(unbiased=False), torch.tensor(1.0), atol=1e-6)


def test_build_rollout_cfg_applies_training_overrides_without_mutating_source():
    from aoae.train_grpo import build_rollout_cfg

    cfg = {
        "inference": {"steps": 64, "gen_length": 256},
        "grpo": {"rollout_steps": 16, "rollout_gen_length": 128},
    }

    rollout_cfg = build_rollout_cfg(cfg)

    assert rollout_cfg["inference"]["steps"] == 16
    assert rollout_cfg["inference"]["gen_length"] == 128
    assert cfg["inference"]["steps"] == 64
    assert cfg["inference"]["gen_length"] == 256


def test_include_heads_for_logprob_parses_configured_head_subset():
    from aoae.train_grpo import _include_heads_for_logprob

    cfg = {
        "grpo": {
            "train_heads": ["unmask", "remask", "cache", "access"],
            "include_heads_in_logprob": ["cache", "access"],
        }
    }

    assert _include_heads_for_logprob(cfg) == {"cache", "access"}


def test_split_group_trajectory_returns_per_sample_views():
    from aoae.train_grpo import split_group_trajectory

    trajectory = SimpleNamespace(
        actions=[{"u_t": torch.tensor([[1.0], [0.0]])}],
        log_probs=[torch.tensor([0.1, 0.2])],
        H_t_list=[torch.randn(2, 3, 4)],
        mask_ind_list=[torch.tensor([[True, False, True], [False, True, False]])],
        quality_scores_list=[torch.randn(2, 3)],
        age_feature_list=[torch.randn(2, 3)],
        last_action_feature_list=[torch.randn(2, 3)],
        agreement_list=[torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])],
        step_fracs=[0.5],
        access_metrics={"access_next_h_spec_f1": 0.7},
    )

    split = split_group_trajectory(trajectory, 2)

    assert len(split) == 2
    assert split[0]["actions_list"][0]["u_t"].shape == (1, 1)
    assert split[1]["old_log_probs"][0].shape == (1,)
    assert split[0]["H_t_list"][0].shape == (1, 3, 4)
    assert split[1]["agreement_list"][0].shape == (1, 3)
    assert split[0]["access_metrics"]["access_next_h_spec_f1"] == 0.7


def test_collect_rollout_group_batches_group_rollouts(monkeypatch):
    import aoae.train_grpo as mod

    called = {}

    def fake_aoae_inference(
        base_model,
        policy,
        soft_mask_module,
        prism_adapter,
        prompt_ids,
        cfg,
        record_trajectory=False,
        policy_temperature=1.0,
    ):
        del base_model, policy, soft_mask_module, prism_adapter, record_trajectory, policy_temperature
        called["shape"] = tuple(prompt_ids.shape)
        called["steps"] = cfg["inference"]["steps"]
        called["gen_length"] = cfg["inference"]["gen_length"]
        batch = prompt_ids.shape[0]
        total_len = prompt_ids.shape[1] + cfg["inference"]["gen_length"]
        output_ids = torch.zeros(batch, total_len, dtype=torch.long)
        trajectory = SimpleNamespace(
            actions=[{"u_t": torch.zeros(batch, cfg["inference"]["gen_length"])}],
            log_probs=[torch.zeros(batch)],
            H_t_list=[torch.zeros(batch, cfg["inference"]["gen_length"], 4)],
            weighted_embeds_list=[torch.zeros(batch, cfg["inference"]["gen_length"], 4)],
            entropy_list=[torch.zeros(batch, cfg["inference"]["gen_length"])],
            mask_ind_list=[torch.zeros(batch, cfg["inference"]["gen_length"], dtype=torch.bool)],
            quality_scores_list=[torch.zeros(batch, cfg["inference"]["gen_length"])],
            age_feature_list=[],
            last_action_feature_list=[],
            step_fracs=[1.0],
            access_metrics={},
        )
        return output_ids, trajectory

    def fake_compute_reward(generated_tokens, reference_answer, tokenizer, trajectory, cfg, T):
        del generated_tokens, reference_answer, tokenizer, trajectory, cfg, T
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        components = {
            "correctness": torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
            "speed_factor": torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
        }
        return rewards, components

    monkeypatch.setattr(mod, "aoae_inference", fake_aoae_inference)
    monkeypatch.setattr(mod, "compute_reward", fake_compute_reward)

    cfg = {
        "grpo": {
            "group_size": 4,
            "policy_temperature": 1.0,
            "normalize_advantage_std": False,
            "rollout_steps": 16,
            "rollout_gen_length": 128,
        },
        "inference": {"steps": 64, "gen_length": 256},
    }

    trajectories, rewards, advantages, reward_components = mod.collect_rollout_group(
        base_model=object(),
        policy=object(),
        soft_mask_module=object(),
        prism_adapter=None,
        prompt_ids=torch.ones(1, 5, dtype=torch.long),
        reference_answers=["42"],
        cfg=cfg,
        tokenizer=object(),
        dual_model=None,
    )

    assert called["shape"] == (4, 5)
    assert called["steps"] == 16
    assert called["gen_length"] == 128
    assert len(trajectories) == 4
    assert torch.allclose(rewards, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.allclose(advantages, torch.tensor([-1.5, -0.5, 0.5, 1.5]))
    assert torch.allclose(
        reward_components["speed_factor"],
        torch.tensor([1.0, 1.0, 1.0, 1.0]),
    )


def test_soft_mask_state_dict_excludes_static_embedding_weight():
    from aoae.models.soft_mask import SoftMaskedState

    cfg = {
        "soft_mask": {
            "top_k": 3,
            "omega_s_init": 0.8,
            "omega_a_init": 1.0,
            "omega_b_init": 2.0,
        }
    }
    module = SoftMaskedState(cfg, torch.randn(17, 9))
    module.set_mask_embedding(0)

    state = module.state_dict()

    assert "embedding_weight" not in state
    assert "mask_embed" in state


def test_compute_reward_penalizes_unresolved_masks():
    from aoae.train_grpo import compute_reward
    from aoae.inference import AOAETrajectory
    from unittest.mock import MagicMock

    tokenizer = MagicMock()
    tokenizer.decode.return_value = "I don't know."

    traj = AOAETrajectory()
    traj.completion_step = torch.tensor([8.0])
    traj.thrash_counts = [torch.tensor([0.0])]
    traj.actions = [{"u_t": torch.zeros(1, 4)}]
    traj.final_tokens = torch.tensor([[99, 99, 99, 99]])

    cfg = {
        "base_model": {"mask_token_id": 99},
        "grpo": {
            "alpha": 1.0,
            "beta": 0.0,
            "access_reward_weight": 0.0,
            "unresolved_penalty_weight": 0.25,
        },
    }

    reward = compute_reward(
        generated_tokens=torch.randint(0, 10, (1, 4)),
        reference_answer=["42"],
        tokenizer=tokenizer,
        trajectory=traj,
        cfg=cfg,
        T=16,
    )

    assert reward[0].item() < 0.0


def test_compute_reward_cache_quality_f1_adds_positive_reward():
    """cache_quality_weight > 0 should increase reward when cache_quality_f1 is high."""
    from aoae.train_grpo import compute_reward
    from aoae.inference import AOAETrajectory
    from unittest.mock import MagicMock

    tokenizer = MagicMock()
    tokenizer.decode.return_value = "I don't know."

    cfg = {
        "base_model": {"mask_token_id": 99},
        "grpo": {
            "alpha": 1.0,
            "beta": 0.0,
            "access_reward_weight": 0.0,
            "unresolved_penalty_weight": 0.0,
            "cache_quality_weight": 0.1,
        },
    }

    # Trajectory WITHOUT cache_quality_f1 (baseline)
    traj_no_f1 = AOAETrajectory()
    traj_no_f1.completion_step = torch.tensor([8.0])
    traj_no_f1.thrash_counts = [torch.tensor([0.0])]
    traj_no_f1.actions = [{"u_t": torch.zeros(1, 4)}]

    reward_no_f1 = compute_reward(
        generated_tokens=torch.randint(0, 10, (1, 4)),
        reference_answer=["42"],
        tokenizer=tokenizer,
        trajectory=traj_no_f1,
        cfg=cfg,
        T=16,
    )

    # Trajectory WITH high cache_quality_f1
    traj_with_f1 = AOAETrajectory()
    traj_with_f1.completion_step = torch.tensor([8.0])
    traj_with_f1.thrash_counts = [torch.tensor([0.0])]
    traj_with_f1.actions = [{"u_t": torch.zeros(1, 4)}]
    traj_with_f1.cache_quality_f1 = [
        torch.tensor([0.9]),
        torch.tensor([0.8]),
        torch.tensor([0.85]),
    ]

    reward_with_f1 = compute_reward(
        generated_tokens=torch.randint(0, 10, (1, 4)),
        reference_answer=["42"],
        tokenizer=tokenizer,
        trajectory=traj_with_f1,
        cfg=cfg,
        T=16,
    )

    # The F1 signal should boost the reward
    assert reward_with_f1[0].item() > reward_no_f1[0].item()
    # Verify the boost is approximately cache_quality_weight * mean_f1
    expected_boost = 0.1 * ((0.9 + 0.8 + 0.85) / 3.0)
    actual_boost = reward_with_f1[0].item() - reward_no_f1[0].item()
    assert abs(actual_boost - expected_boost) < 1e-5


def test_compute_reward_access_f1_adds_dense_access_reward():
    from aoae.train_grpo import compute_reward
    from aoae.inference import AOAETrajectory
    from unittest.mock import MagicMock

    tokenizer = MagicMock()
    tokenizer.decode.return_value = "I don't know."
    traj = AOAETrajectory()
    traj.completion_step = torch.tensor([8.0])
    traj.thrash_counts = [torch.tensor([0.0])]
    traj.actions = [{"u_t": torch.zeros(1, 4)}]
    traj.access_metrics = {"access_next_h_spec_f1": 0.75}

    cfg = {
        "base_model": {"mask_token_id": 99},
        "grpo": {
            "alpha": 1.0,
            "beta": 0.0,
            "access_reward_weight": 0.2,
            "unresolved_penalty_weight": 0.0,
            "cache_quality_weight": 0.0,
        },
    }

    reward, components = compute_reward(
        generated_tokens=torch.randint(0, 10, (1, 4)),
        reference_answer=["42"],
        tokenizer=tokenizer,
        trajectory=traj,
        cfg=cfg,
        T=16,
    )

    assert components["access_f1"].item() == pytest.approx(0.75)
    assert components["access_reward"].item() == pytest.approx(0.15)
    assert reward.item() == pytest.approx(0.15)


def test_compute_reward_uses_per_sample_access_f1_when_available():
    from aoae.train_grpo import compute_reward
    from unittest.mock import MagicMock

    tokenizer = MagicMock()
    tokenizer.decode.return_value = "I don't know."
    traj = SimpleNamespace(
        completion_step=torch.tensor([8.0, 8.0]),
        effective_flops=torch.tensor([0.0, 0.0]),
        aux_compute_units=torch.tensor([0.0, 0.0]),
        verifier_compute_units=torch.tensor([0.0, 0.0]),
        baseline_compute_units=torch.tensor([16.0, 16.0]),
        thrash_counts=[],
        cached_fractions=[],
        stable_cached_fractions=[],
        spec_cached_fractions=[],
        actions=[],
        final_tokens=torch.tensor([[1, 2], [1, 2]]),
        cache_quality_f1=[],
        access_metrics={"access_next_h_spec_f1": 0.5},
        access_metric_tensors={"access_next_h_spec_f1": torch.tensor([0.0, 1.0])},
    )
    cfg = {
        "base_model": {"mask_token_id": 99},
        "grpo": {
            "alpha": 1.0,
            "beta": 0.0,
            "access_reward_weight": 0.2,
            "unresolved_penalty_weight": 0.0,
            "cache_quality_weight": 0.0,
        },
    }

    reward, components = compute_reward(
        generated_tokens=torch.randint(0, 10, (2, 2)),
        reference_answer=["42", "42"],
        tokenizer=tokenizer,
        trajectory=traj,
        cfg=cfg,
        T=16,
    )

    assert components["access_reward"].tolist() == pytest.approx([0.0, 0.2])
    assert reward.tolist() == pytest.approx([0.0, 0.2])


def test_compute_reward_prefers_trajectory_effective_flops():
    from aoae.train_grpo import compute_reward
    from unittest.mock import MagicMock

    tokenizer = MagicMock()
    tokenizer.decode.return_value = "42"
    traj = SimpleNamespace(
        completion_step=torch.tensor([16.0]),
        effective_flops=torch.tensor([0.25]),
        aux_compute_units=torch.tensor([1.0]),
        verifier_compute_units=torch.tensor([3.0]),
        baseline_compute_units=torch.tensor([16.0]),
        thrash_counts=[],
        cached_fractions=[],
        stable_cached_fractions=[torch.tensor([0.9])],
        spec_cached_fractions=[],
        actions=[],
        final_tokens=torch.tensor([[1, 2, 3, 4]]),
        access_metrics={},
        cache_quality_f1=[],
    )
    cfg = {
        "base_model": {"mask_token_id": 99},
        "grpo": {
            "alpha": 1.0,
            "beta": 0.0,
            "cache_speed_source": "none",
            "access_reward_weight": 0.0,
            "cache_quality_weight": 0.0,
            "unresolved_penalty_weight": 0.0,
        },
    }

    reward, components = compute_reward(
        generated_tokens=torch.tensor([[1, 2, 3, 4]]),
        reference_answer=["42"],
        tokenizer=tokenizer,
        trajectory=traj,
        cfg=cfg,
        T=16,
    )

    assert abs(components["effective_flops"].item() - 0.25) < 1e-6
    assert abs(components["speed_factor"].item() - 0.75) < 1e-6
    assert abs(reward.item() - 0.75) < 1e-6


def test_compute_reward_uses_same_gsm8k_rule_as_eval():
    """Training reward and eval evaluator share the same flexible GSM8K extractor.

    The flexible extractor recovers numerical answers from prose like
    "The answer is 42", which masked-diffusion models often produce.
    Both training (compute_reward) and eval (build_evaluator) go through
    build_evaluator → MathEvaluator → check_gsm8k_correctness_llada.
    """
    from aoae.train_grpo import compute_reward
    from unittest.mock import MagicMock

    tokenizer = MagicMock()
    tokenizer.decode.return_value = "The answer is 42"
    traj = SimpleNamespace(
        completion_step=torch.tensor([1.0]),
        effective_flops=torch.tensor([0.0]),
        aux_compute_units=torch.tensor([0.0]),
        verifier_compute_units=torch.tensor([0.0]),
        baseline_compute_units=torch.tensor([1.0]),
        thrash_counts=[],
        cached_fractions=[],
        stable_cached_fractions=[],
        spec_cached_fractions=[],
        actions=[],
        final_tokens=torch.tensor([[1, 2, 3]]),
        access_metrics={},
        cache_quality_f1=[],
    )
    cfg = {
        "base_model": {"mask_token_id": 99},
        "data": {"eval_dataset": "openai/gsm8k"},
        "evaluation": {"task_type": "math"},
        "grpo": {
            "alpha": 1.0,
            "beta": 0.0,
            "access_reward_weight": 0.0,
            "cache_quality_weight": 0.0,
            "unresolved_penalty_weight": 0.0,
        },
    }

    reward, components = compute_reward(
        generated_tokens=torch.tensor([[1, 2, 3]]),
        reference_answer=["#### 42"],
        tokenizer=tokenizer,
        trajectory=traj,
        cfg=cfg,
        T=1,
    )

    # Flexible extractor finds "42" in "The answer is 42" — correctness = 1.
    assert components["correctness"].item() == 1.0
    assert reward.item() == 1.0


def test_compute_reward_uses_train_dataset_answer_format_when_different_from_eval():
    from aoae.train_grpo import compute_reward
    from unittest.mock import MagicMock

    tokenizer = MagicMock()
    tokenizer.decode.return_value = "The answer is 42"
    traj = SimpleNamespace(
        completion_step=torch.tensor([1.0]),
        effective_flops=torch.tensor([0.0]),
        aux_compute_units=torch.tensor([0.0]),
        verifier_compute_units=torch.tensor([0.0]),
        baseline_compute_units=torch.tensor([1.0]),
        thrash_counts=[],
        cached_fractions=[],
        stable_cached_fractions=[],
        spec_cached_fractions=[],
        actions=[],
        final_tokens=torch.tensor([[1, 2, 3]]),
        access_metrics={},
        cache_quality_f1=[],
    )
    cfg = {
        "base_model": {"mask_token_id": 99},
        "data": {
            "train_dataset": "nvidia/OpenMathInstruct-2",
            "eval_dataset": "openai/gsm8k",
        },
        "evaluation": {"task_type": "math"},
        "grpo": {
            "alpha": 1.0,
            "beta": 0.0,
            "access_reward_weight": 0.0,
            "cache_quality_weight": 0.0,
            "unresolved_penalty_weight": 0.0,
        },
    }

    reward, components = compute_reward(
        generated_tokens=torch.tensor([[1, 2, 3]]),
        reference_answer=["42"],
        tokenizer=tokenizer,
        trajectory=traj,
        cfg=cfg,
        T=1,
    )

    assert components["correctness"].item() == 1.0
    assert reward.item() == 1.0


def test_configure_grpo_trainability_freezes_unmask_remask_and_soft_mask():
    from aoae.models.policy import AOAEPolicy
    from aoae.models.soft_mask import SoftMaskedState
    from aoae.train_grpo import configure_grpo_trainability

    cfg = {
        "policy": {
            "d_model": 16,
            "n_layers": 1,
            "n_heads": 4,
            "dropout": 0.0,
            "use_positional_features": False,
            "use_age_feature": False,
            "use_last_action_feature": False,
            "boundary_head": {"enabled": False},
        },
        "soft_mask": {
            "top_k": 2,
            "omega_s_init": 0.8,
            "omega_a_init": 1.0,
            "omega_b_init": 2.0,
        },
        "grpo": {
            "train_heads": ["cache", "access"],
            "include_heads_in_logprob": ["cache", "access"],
            "train_soft_mask": False,
        },
    }
    policy = AOAEPolicy(cfg, input_dim=8)
    soft_mask = SoftMaskedState(cfg, torch.randn(11, 8))

    trainable = configure_grpo_trainability(policy, soft_mask, cfg)

    assert trainable
    assert not any(p.requires_grad for p in policy.head_unmask.parameters())
    assert not any(p.requires_grad for p in policy.head_remask.parameters())
    assert all(p.requires_grad for p in policy.head_cache.parameters())
    assert all(p.requires_grad for p in policy.head_access.parameters())
    assert not any(p.requires_grad for p in soft_mask.parameters())
