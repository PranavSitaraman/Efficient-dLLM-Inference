import torch


def test_llada21_official_decode_speed_uses_block_diffusion(monkeypatch):
    from aoae import inference as inf

    calls = {}

    def fake_block(base_model, prompt_ids, cfg, tau_mask, tau_edit, max_steps_per_block, enable_mbe):
        calls["tau_mask"] = tau_mask
        calls["tau_edit"] = tau_edit
        calls["max_post_steps"] = max_steps_per_block
        calls["enable_mbe"] = enable_mbe
        return prompt_ids

    monkeypatch.setattr(inf, "block_smode_decode", fake_block)

    cfg = {"inference": {"llada21_official": {"use_block_diffusion": True}}}
    prompt_ids = torch.zeros((1, 4), dtype=torch.long)
    out = inf.llada21_official_decode(None, prompt_ids, cfg, mode="speed")

    assert out is prompt_ids
    assert calls == {
        "tau_mask": 0.5,
        "tau_edit": 0.0,
        "max_post_steps": 16,
        "enable_mbe": False,
    }


def test_get_baseline_methods_respects_config_override():
    from aoae.evaluate import _get_baseline_methods

    cfg = {"evaluation": {"baseline_methods": ["llada21_speed_mode"]}}
    assert _get_baseline_methods(cfg) == ["llada21_speed_mode"]
