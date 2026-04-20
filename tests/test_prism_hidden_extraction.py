import types

import torch


def _make_base_model(hidden_dim=4096, vocab_size=126464):
    import aoae.models.base_model as mod

    base_model = object.__new__(mod.LLaDABaseModel)
    torch.nn.Module.__init__(base_model)
    base_model._backend = "hf"
    base_model._block_length = 32
    # forward_hidden_only uses the explicit vocab_size/hidden_dim attributes
    # below; allocating the full LLaDA embedding here would make this unit test
    # consume ~2 GiB for no behavioral coverage.
    base_model._embedding_weight = torch.zeros(1, hidden_dim)
    base_model.vocab_size = vocab_size
    base_model.hidden_dim = hidden_dim
    base_model.model = types.SimpleNamespace(model=object())
    base_model._make_attention_mask = lambda batch_size, seq_len, device, block_length=32: torch.zeros(
        batch_size, 1, seq_len, seq_len, device=device
    )
    return base_model


def test_forward_hidden_only_prefers_hidden_width_over_vocab_logits(monkeypatch):
    import aoae.models.base_model as mod

    batch_size, seq_len, hidden_dim, vocab_size = 2, 3, 4096, 126464
    base_model = _make_base_model(hidden_dim=hidden_dim, vocab_size=vocab_size)
    logits = torch.randn(batch_size, seq_len, vocab_size)
    hidden = torch.randn(batch_size, seq_len, hidden_dim)

    def fake_forward_hf_outputs(self, input_ids, attention_mask=None, **kwargs):
        del self, input_ids, attention_mask, kwargs
        return (logits, hidden)

    monkeypatch.setattr(
        mod.LLaDABaseModel,
        "_forward_hf_outputs",
        fake_forward_hf_outputs,
    )

    extracted = mod.LLaDABaseModel.forward_hidden_only(
        base_model,
        torch.ones(batch_size, seq_len, dtype=torch.long),
    )

    assert extracted.shape == (batch_size, seq_len, hidden_dim)
    assert extracted is hidden


def test_forward_hidden_only_uses_last_hidden_state_when_available(monkeypatch):
    import aoae.models.base_model as mod

    batch_size, seq_len, hidden_dim, vocab_size = 2, 3, 4096, 126464
    base_model = _make_base_model(hidden_dim=hidden_dim, vocab_size=vocab_size)
    last_hidden = torch.randn(batch_size, seq_len, hidden_dim)

    def fake_forward_hf_outputs(self, input_ids, attention_mask=None, **kwargs):
        del self, input_ids, attention_mask, kwargs
        return types.SimpleNamespace(last_hidden_state=last_hidden)

    monkeypatch.setattr(
        mod.LLaDABaseModel,
        "_forward_hf_outputs",
        fake_forward_hf_outputs,
    )

    extracted = mod.LLaDABaseModel.forward_hidden_only(
        base_model,
        torch.ones(batch_size, seq_len, dtype=torch.long),
    )

    assert extracted is last_hidden


def test_forward_hidden_only_passes_attention_mask_to_new_hf_backbone():
    # LLaDA-8B-Instruct exposes both attention_mask and attention_bias.
    # 4D float block masks must be routed only to attention_bias because the
    # model preprocesses attention_mask via .view(B,-1)[:, None, None, :],
    # which would flatten [B,1,L,L] → [B,1,1,L²] causing a shape mismatch.
    import aoae.models.base_model as mod

    batch_size, seq_len, hidden_dim, vocab_size = 2, 3, 4096, 126464
    base_model = _make_base_model(hidden_dim=hidden_dim, vocab_size=vocab_size)
    expected_mask = torch.zeros(batch_size, 1, seq_len, seq_len)
    base_model._make_attention_mask = lambda *args, **kwargs: expected_mask

    received = {}

    class DummyBackbone(torch.nn.Module):
        def forward(self, input_ids, attention_mask=None, attention_bias=None,
                    output_hidden_states=False):
            del input_ids, output_hidden_states
            received["mask"] = attention_mask
            received["bias"] = attention_bias
            return types.SimpleNamespace(
                last_hidden_state=torch.randn(batch_size, seq_len, hidden_dim)
            )

    base_model.model = types.SimpleNamespace(model=DummyBackbone())

    extracted = mod.LLaDABaseModel.forward_hidden_only(
        base_model,
        torch.ones(batch_size, seq_len, dtype=torch.long),
    )

    assert extracted.shape == (batch_size, seq_len, hidden_dim)
    # 4D mask must arrive only as attention_bias, not as attention_mask
    assert received["mask"] is None
    assert received["bias"] is expected_mask


def test_prism_adapter_raises_clear_error_on_hidden_dim_mismatch():
    from aoae.models.prism import PRISMAdapter

    cfg = {"prism": {"hidden_dim": 32, "threshold": 0.5}}
    adapter = PRISMAdapter(cfg, hidden_dim=64)

    with torch.no_grad():
        wrong_hidden = torch.randn(2, 3, 128)
        try:
            adapter(wrong_hidden)
        except RuntimeError as exc:
            message = str(exc)
        else:
            raise AssertionError("expected PRISMAdapter width mismatch to raise")

    assert "expected last dim 64, got 128" in message
