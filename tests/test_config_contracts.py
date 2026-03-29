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
