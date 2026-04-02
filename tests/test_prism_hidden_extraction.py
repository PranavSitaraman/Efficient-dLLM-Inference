import types

import torch


def _make_base_model(hidden_dim=4096, vocab_size=126464):
    import aoae.models.base_model as mod

    base_model = object.__new__(mod.LLaDABaseModel)
    torch.nn.Module.__init__(base_model)
    base_model._backend = "hf"
    base_model._block_length = 32
    base_model._embedding_weight = torch.zeros(vocab_size, hidden_dim)
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
    import aoae.models.base_model as mod

    batch_size, seq_len, hidden_dim, vocab_size = 2, 3, 4096, 126464
    base_model = _make_base_model(hidden_dim=hidden_dim, vocab_size=vocab_size)
    expected_mask = torch.zeros(batch_size, 1, seq_len, seq_len)
    base_model._make_attention_mask = lambda *args, **kwargs: expected_mask

    class DummyBackbone(torch.nn.Module):
        def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
            del input_ids, output_hidden_states
            assert attention_mask is expected_mask
            return types.SimpleNamespace(
                last_hidden_state=torch.randn(batch_size, seq_len, hidden_dim)
            )

    base_model.model = types.SimpleNamespace(model=DummyBackbone())

    extracted = mod.LLaDABaseModel.forward_hidden_only(
        base_model,
        torch.ones(batch_size, seq_len, dtype=torch.long),
    )

    assert extracted.shape == (batch_size, seq_len, hidden_dim)


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
