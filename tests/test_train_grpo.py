import torch


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
