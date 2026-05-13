from pathlib import Path

from aoae.cli import _load_config
from aoae.evaluate import _build_speculative_eval_points, _get_baseline_methods


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CONFIGS = {
    "paper.yaml",
    "paper_smoke.yaml",
    "eval_gsm8k.yaml",
    "eval_math500.yaml",
    "eval_humaneval.yaml",
    "ablation.yaml",
}


def _cfg(name: str) -> dict:
    return _load_config(str(ROOT / "configs" / name))


def test_configs_directory_is_submission_surface_only():
    observed = {path.name for path in (ROOT / "configs").glob("*.yaml")}
    assert observed == EXPECTED_CONFIGS


def test_all_canonical_configs_load_through_public_loader():
    for name in sorted(EXPECTED_CONFIGS):
        cfg = _cfg(name)
        assert cfg["base_model"]["backend"] == "dual"
        assert cfg["data"]["eval_dataset"]
        assert cfg["logging"]["output_dir"].startswith("outputs/")


def test_paper_config_is_v4_quality_balanced_scalar_grpo_contract():
    cfg = _cfg("paper.yaml")

    assert cfg["phase_a_v2"] is True
    assert cfg["feature_mode"] == "scalar_only"
    assert cfg["phase_a_v2_config"]["target_u_rate"] == 0.10
    assert cfg["phase_a_v2_config"]["target_r_rate"] == 0.02
    assert cfg["cache"]["enabled"] is False
    assert cfg["cache"]["stable_kv_cache"] is False
    assert cfg["reward_cache_terms_enabled"] is False

    grpo = cfg["grpo"]
    assert grpo["enabled"] is True
    assert grpo["train_heads"] == ["unmask", "remask"]
    assert grpo["include_heads_in_logprob"] == ["unmask", "remask"]
    assert grpo["reward_cache_terms_enabled"] is False
    assert grpo["cache_speed_source"] == "none"
    assert grpo["cache_quality_weight"] == 0.0
    assert grpo["access_reward_weight"] == 0.0
    assert grpo["expert_steering"]["enabled"] is False
    assert grpo["warm_start_from"] == "outputs/paper_final/train/policy_final.pt"

    assert cfg["policy"]["feature_mode"] == "scalar_only"
    assert cfg["policy"]["use_hidden_state"] is False
    assert cfg["policy"]["use_max_confidence_feature"] is True
    assert cfg["policy"]["use_quality_score_feature"] is True
    assert cfg["inference"]["verifier_schedule"]["draft_token_budget"] == 8
    assert cfg["inference"]["verifier_schedule"]["max_draft_microsteps"] == 3
    assert cfg["inference"]["primary_agree_threshold"] == 0.92

    assert _get_baseline_methods(cfg) == [
        "llada21_speed_mode",
        "llada21_quality_mode",
        "llada21_speed_anyorder",
        "llada21_quality_anyorder",
        "fast_dllm",
    ]
    points = _build_speculative_eval_points(cfg, explicit_policy_temperatures=None)
    assert [point["name"] for point in points] == [
        "qbal_tau0.5_lossy",
        "qbal_tau1.0_lossy",
        "qbal_tau0.5_lossless",
        "qbal_tau1.0_lossless",
    ]
    assert all(point["generation_mode"] == "any_order" for point in points)


def test_eval_configs_share_final_sweep_and_dataset_specific_contracts():
    point_names = None
    expected = {
        "eval_gsm8k.yaml": ("openai/gsm8k", "math", 1319),
        "eval_math500.yaml": ("HuggingFaceH4/MATH-500", "math", 500),
        "eval_humaneval.yaml": ("openai/openai_humaneval", "code", 164),
    }

    for name, (dataset, task_type, samples) in expected.items():
        cfg = _cfg(name)
        assert cfg["grpo"]["enabled"] is False
        assert cfg["warmstart"]["enabled"] is False
        assert cfg["data"]["eval_dataset"] == dataset
        assert cfg["data"]["eval_max_samples"] == samples
        assert cfg["evaluation"]["task_type"] == task_type
        assert cfg["hardware"]["tp_size"] == 1
        names = [point["name"] for point in _build_speculative_eval_points(cfg, None)]
        point_names = point_names or names
        assert names == point_names

    human = _cfg("eval_humaneval.yaml")
    assert human["evaluation"]["code"]["timeout_sec"] == 3.0
    assert human["data"]["math_prompt_style"] == "none"


def test_smoke_and_ablation_configs_are_tiny_final_variants():
    smoke = _cfg("paper_smoke.yaml")
    assert smoke["grpo"]["max_steps"] == 2
    assert smoke["evaluation"]["speculative_sweep"]["enabled"] is False
    assert smoke["logging"]["use_wandb"] is False

    ablation = _cfg("ablation.yaml")
    points = _build_speculative_eval_points(ablation, None)
    names = [point["name"] for point in points]
    assert names == [
        "ablate_block_default",
        "ablate_any_order_soft",
        "ablate_any_order_hard",
        "ablate_any_order_lossless_tau1",
    ]
    assert {point["generation_mode"] for point in points} == {"block", "any_order"}
