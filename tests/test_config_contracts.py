from pathlib import Path
import copy

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
    # The canonical paper config now enumerates both block-mode and any-order
    # LLaDA2.1 baselines so the eval table includes a fair comparison against
    # the official block decoder *and* the any-order single-block variant the
    # paper analyses for AOAE.  Fast-dLLM remains the parallel-decode anchor.
    assert cfg["evaluation"]["baseline_methods"] == [
        "llada21_speed_mode",
        "llada21_quality_mode",
        "llada21_speed_anyorder",
        "llada21_quality_anyorder",
        "fast_dllm",
    ]
    assert cfg["evaluation"]["generation_mode_filter"] == "block"
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
    assert cfg["inference"]["verifier_schedule"]["draft_token_budget"] == 12
    assert cfg["inference"]["verifier_schedule"]["max_draft_microsteps"] == 4
    assert cfg["inference"]["verifier"]["acceptance_mode"] == "argmax_match"
    assert cfg["inference"]["verifier"]["rejection_action"] == "remask"
    assert cfg["inference"]["verifier"]["recompute_after_reject"] is True
    assert cfg["inference"]["drafter"]["confidence_threshold"] == 0.7
    assert cfg["inference"]["drafter"]["aux_compute_ratio"] == 0.35
    assert cfg["inference"]["drafter"]["run_on_verifier"] == "auto"
    sweep_points = cfg["evaluation"]["speculative_sweep"]["points"]
    assert any(
        point["overrides"].get("inference.speculative_schedule") == "aoae_block"
        for point in sweep_points
    )
    assert any(
        point["overrides"].get("inference.speculative_schedule") == "aoae_block_policy"
        and point.get("generation_mode") == "any_order"
        for point in sweep_points
    )
    assert any(
        point["overrides"].get("inference.block_speculative.verifier_mode") == "self_accept_lossless"
        for point in sweep_points
    )
    for point in sweep_points:
        overrides = point.get("overrides", {})
        if overrides.get("inference.speculative_schedule") == "aoae_block":
            # The block schedule supports two rejection actions.  ``replace``
            # substitutes the verifier's argmax for rejected drafts while
            # ``correct_confident`` only replaces drafts the verifier is
            # confident about and remasks the rest; both are valid operating
            # points that the canonical sweep exercises across the Pareto.
            assert overrides.get("inference.block_speculative.rejection_action") in (
                "replace",
                "correct_confident",
            )
    assert cfg["hardware"]["tp_size"] == 1


def test_paper_config_keeps_block_and_any_order_eval_tracks_separate():
    from aoae.evaluate import _build_speculative_eval_points, _get_baseline_methods

    cfg = _load_yaml("configs/paper.yaml")

    assert _get_baseline_methods(cfg) == [
        "llada21_speed_mode",
        "llada21_quality_mode",
    ]
    block_points = _build_speculative_eval_points(cfg, explicit_policy_temperatures=None)
    block_names = [point["name"] for point in block_points]
    assert block_names == [
        "speed_balanced",
        "speed_max",
        "speed_extreme",
        "aoae_llada_sq",
        "aoae_llada_sq_softver",
    ]
    assert all(
        point["overrides"].get("inference.speculative_schedule") != "aoae_block_policy"
        for point in block_points
    )

    any_order_cfg = copy.deepcopy(cfg)
    any_order_cfg["evaluation"]["generation_mode_filter"] = "any_order"
    assert _get_baseline_methods(any_order_cfg) == [
        "llada21_speed_anyorder",
        "llada21_quality_anyorder",
        "fast_dllm",
    ]
    any_order_points = _build_speculative_eval_points(
        any_order_cfg,
        explicit_policy_temperatures=None,
    )
    any_order_names = [point["name"] for point in any_order_points]
    assert any_order_names == [
        "quality_max",
        "quality_max_hardver",
        "quality_max_sq",
        "quality_max_sq_hardver",
        "quality_balanced",
        "quality_balanced_hardver",
        "quality_balanced_sq",
        "quality_balanced_sq_hardver",
        "aoae_llada_sq_anyorder",
    ]
