from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: str):
    with (ROOT / path).open() as f:
        return yaml.safe_load(f)


def test_llada21_soft_config_uses_explicit_widened_router_budget():
    cfg = _load_yaml("configs/llada21_soft.yaml")

    assert cfg["base_model"]["backend"] == "soft_moe"
    assert cfg["base_model"]["soft_topk"] == 16
    assert cfg["evaluation"]["baseline_methods"] == [
        "llada21_speed_mode",
        "llada21_quality_mode",
    ]


def test_poc1_config_matches_routing_only_paper_setup():
    cfg = _load_yaml("configs/poc1.yaml")

    assert cfg["base_model"]["backend"] == "dual"
    assert cfg["base_model"]["soft_topk"] == 16
    assert cfg["inference"]["speculative_schedule"] == "llada21_block"
    assert cfg["inference"]["disable_remask"] is True


def test_paper_config_enables_full_aoae_stack():
    cfg = _load_yaml("configs/paper.yaml")

    assert cfg["base_model"]["backend"] == "dual"
    assert cfg["grpo"]["enabled"] is True
    assert cfg["data"]["use_chat_template"] == "auto"
    assert cfg["data"]["math_prompt_style"] == "auto"
    assert cfg["data"]["max_answer_len"] == 256
    assert cfg["evaluation"]["baseline_methods"] == [
        "llada21_speed_mode",
        "llada21_quality_mode",
        "fast_dllm",
    ]
    assert cfg["evaluation"]["save_predictions"] is True
    assert cfg["grpo"]["thrash_normalization"] == "response_length"
    assert cfg["grpo"]["cache_speed_source"] == "none"
    assert cfg["grpo"]["cache_quality_weight"] == 0.02
    assert cfg["grpo"]["access_reward_weight"] == 0.0
    assert cfg["grpo"]["min_checkpoint_reward"] < 0.0
    assert cfg["grpo"]["train_heads"] == ["cache", "access"]
    assert cfg["grpo"]["include_heads_in_logprob"] == ["cache", "access"]
    assert cfg["grpo"]["train_soft_mask"] is False
    assert cfg["policy"]["init_unmask_bias"] < 0
    assert cfg["policy"]["init_remask_bias"] < 0
    assert cfg["policy"]["init_cache_bias"] < 0
    assert cfg["inference"]["max_unmask_fraction_per_step"] <= 0.125
    assert cfg["inference"]["llada21_official"]["eos_early_stop"] is False
    assert cfg["inference"]["llada21_official"]["gen_length"] == 512
    assert cfg["base_model"]["soft_topk"] == 16
    assert cfg["cache"]["stable_kv_cache"] is False
    assert cfg["inference"]["positional_cache"]["enabled"] is True
    # The canonical paper config disables composition; gamma=0 returns the
    # verifier distribution exactly and matches the lossless-verification
    # sanity-check operating point.
    assert cfg["inference"]["compose_gamma"] == 0
    assert cfg["inference"]["verifier_schedule"]["mode"] == "candidate_budget"
    assert cfg["inference"]["verifier_schedule"]["draft_token_budget"] == 1
    assert cfg["inference"]["verifier"]["acceptance_mode"] == "argmax_match"
    assert cfg["inference"]["verifier"]["rejection_action"] == "remask"
    assert cfg["inference"]["verifier"]["recompute_after_reject"] is True
    assert cfg["inference"]["drafter"]["confidence_threshold"] == 0.7
    assert cfg["inference"]["drafter"]["aux_compute_ratio"] == 0.35
