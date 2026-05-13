import torch


def test_llada21_official_decode_speed_uses_block_diffusion(monkeypatch):
    from aoae import inference as inf

    calls = {}

    def fake_block(
        base_model,
        prompt_ids,
        cfg,
        tau_mask,
        tau_edit,
        max_steps_per_block,
        enable_mbe,
        gen_length,
        eos_early_stop,
        stats=None,
    ):
        calls["tau_mask"] = tau_mask
        calls["tau_edit"] = tau_edit
        calls["max_post_steps"] = max_steps_per_block
        calls["enable_mbe"] = enable_mbe
        calls["gen_length"] = gen_length
        calls["eos_early_stop"] = eos_early_stop
        return prompt_ids

    monkeypatch.setattr(inf, "block_smode_decode", fake_block)

    cfg = {"inference": {"gen_length": 256, "llada21_official": {"use_block_diffusion": True}}}
    prompt_ids = torch.zeros((1, 4), dtype=torch.long)
    out = inf.llada21_official_decode(None, prompt_ids, cfg, mode="speed")

    assert out is prompt_ids
    assert calls == {
        "tau_mask": 0.5,
        "tau_edit": 0.0,
        "max_post_steps": 16,
        "enable_mbe": False,
        "gen_length": 512,
        "eos_early_stop": True,
    }


def test_llada21_official_decode_uses_mode_specific_thresholds(monkeypatch):
    from aoae import inference as inf

    calls = {}

    def fake_block(
        base_model,
        prompt_ids,
        cfg,
        tau_mask,
        tau_edit,
        max_steps_per_block,
        enable_mbe,
        gen_length,
        eos_early_stop,
        stats=None,
    ):
        calls["tau_mask"] = tau_mask
        calls["tau_edit"] = tau_edit
        return prompt_ids

    monkeypatch.setattr(inf, "block_smode_decode", fake_block)

    cfg = {
        "inference": {
            "llada21_official": {
                "use_block_diffusion": True,
                "speed": {"threshold": 0.5, "editing_threshold": 0.0},
                "quality": {"threshold": 0.7, "editing_threshold": 0.5},
            }
        }
    }
    prompt_ids = torch.zeros((1, 4), dtype=torch.long)

    inf.llada21_official_decode(None, prompt_ids, cfg, mode="speed")
    assert calls == {"tau_mask": 0.5, "tau_edit": 0.0}

    inf.llada21_official_decode(None, prompt_ids, cfg, mode="quality")
    assert calls == {"tau_mask": 0.7, "tau_edit": 0.5}


def test_resolve_llada21_official_settings_defaults_to_model_card_recipe():
    from aoae.inference import resolve_llada21_official_settings

    cfg = {"inference": {"gen_length": 256, "block_length": 32, "temperature": 0.0}}

    speed = resolve_llada21_official_settings(cfg, mode="speed")
    quality = resolve_llada21_official_settings(cfg, mode="quality")

    assert speed["threshold"] == 0.5
    assert speed["editing_threshold"] == 0.0
    assert quality["threshold"] == 0.7
    assert quality["editing_threshold"] == 0.5
    assert speed["max_post_steps"] == 16
    assert speed["gen_length"] == 512
    assert speed["eos_early_stop"] is True


def test_confidence_threshold_decode_reports_actual_iterations():
    # Honest NFE accounting: the iteration counter must reflect *actual* model
    # forward passes, not the upper-bound horizon T.  When every position can
    # be unmasked in a single step, the loop terminates immediately and only
    # one forward should be counted (plus zero force-complete passes).
    from aoae.inference import confidence_threshold_decode

    class AlwaysConfidentModel:
        vocab_size = 8

        def __init__(self):
            self.calls = 0

        def forward(self, input_ids):
            self.calls += 1
            B, L = input_ids.shape
            logits = torch.full((B, L, self.vocab_size), -10.0)
            logits[..., 1] = 10.0
            return logits

    cfg = {
        "inference": {"steps": 100, "gen_length": 4, "temperature": 0.0},
        "base_model": {"mask_token_id": 7},
    }
    prompt_ids = torch.tensor([[2, 3]], dtype=torch.long)
    stats = {}
    confidence_threshold_decode(
        AlwaysConfidentModel(),
        prompt_ids,
        cfg,
        tau_mask=0.5,
        tau_edit=1.0,
        enable_t2t=False,
        stats=stats,
    )
    assert stats["iterations"] == 1
    assert stats["force_complete_passes"] == 0


def test_get_baseline_methods_respects_config_override():
    from aoae.evaluate import _get_baseline_methods

    cfg = {"evaluation": {"baseline_methods": ["llada21_speed_mode"]}}
    assert _get_baseline_methods(cfg) == ["llada21_speed_mode"]


def test_get_baseline_methods_defaults_to_canonical_paper_set():
    from aoae.evaluate import _get_baseline_methods

    cfg = {"evaluation": {}}
    assert _get_baseline_methods(cfg) == [
        "llada21_speed_mode",
        "llada21_quality_mode",
        "fast_dllm",
    ]
