"""
Unit tests for K_spec KV skip implementation.

Tests _kspec_find_clusters and forward_with_kspec_cache using a mock that
implements forward_replace_with_cache without loading any real model weights.

Run with:
    ./dlm_env/bin/python -m pytest tests/test_kspec.py -v
"""

import sys
import os
import types

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aoae.models.base_model import _kspec_find_clusters


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_kv(tag: float) -> object:
    """Return a tiny placeholder 'KV cache' object distinguishable by tag."""
    kv = types.SimpleNamespace(tag=tag)
    return kv


def _make_logits(B, L, V, *, fill: float):
    return torch.full((B, L, V), fill)


# ---------------------------------------------------------------------------
# 1. _kspec_find_clusters
# ---------------------------------------------------------------------------

class TestKspecFindClusters:
    def test_all_false(self):
        non_agreed = torch.zeros(8, dtype=torch.bool)
        assert _kspec_find_clusters(non_agreed) == []

    def test_all_true(self):
        non_agreed = torch.ones(8, dtype=torch.bool)
        assert _kspec_find_clusters(non_agreed) == [(0, 8)]

    def test_single_true(self):
        non_agreed = torch.tensor([False, False, True, False, False], dtype=torch.bool)
        assert _kspec_find_clusters(non_agreed) == [(2, 3)]

    def test_two_clusters(self):
        # TT FF TT F
        non_agreed = torch.tensor([True, True, False, False, True, True, False], dtype=torch.bool)
        assert _kspec_find_clusters(non_agreed) == [(0, 2), (4, 6)]

    def test_alternating(self):
        non_agreed = torch.tensor([True, False, True, False, True], dtype=torch.bool)
        assert _kspec_find_clusters(non_agreed) == [(0, 1), (2, 3), (4, 5)]

    def test_trailing_true(self):
        non_agreed = torch.tensor([False, True, True], dtype=torch.bool)
        assert _kspec_find_clusters(non_agreed) == [(1, 3)]

    def test_length_one_true(self):
        non_agreed = torch.tensor([True], dtype=torch.bool)
        assert _kspec_find_clusters(non_agreed) == [(0, 1)]

    def test_length_one_false(self):
        non_agreed = torch.tensor([False], dtype=torch.bool)
        assert _kspec_find_clusters(non_agreed) == []

    def test_clusters_are_contiguous(self):
        # FFTTFFTTFF → clusters at [2,4) and [6,8)
        na = torch.tensor([F, F, T, T, F, F, T, T, F, F], dtype=torch.bool) if False else None
        na = torch.zeros(10, dtype=torch.bool)
        na[2], na[3], na[6], na[7] = True, True, True, True
        clusters = _kspec_find_clusters(na)
        assert clusters == [(2, 4), (6, 8)]
        # Clusters must be non-overlapping and in order
        for i in range(len(clusters) - 1):
            assert clusters[i][1] <= clusters[i + 1][0]

    def test_large_tensor(self):
        # 256 positions: first 128 agreed (False), last 128 non-agreed (True)
        na = torch.zeros(256, dtype=torch.bool)
        na[128:] = True
        assert _kspec_find_clusters(na) == [(128, 256)]


# ---------------------------------------------------------------------------
# 2. forward_with_kspec_cache via a stub LLaDABaseModel
# ---------------------------------------------------------------------------

class StubModel:
    """
    Minimal stub that implements forward_replace_with_cache.

    Each call records the replace_slice and the tag of the incoming KV,
    and returns logits filled with the call index + 1, plus a new KV
    whose tag encodes the call sequence.
    """

    def __init__(self, vocab_size: int, dtype=torch.float32):
        self.vocab_size = vocab_size
        self.dtype = dtype
        self._backend = "dinfer"  # triggers the kspec path
        self.calls: list = []     # list of slice objects passed in order

    @property
    def device(self):
        return torch.device("cpu")

    def forward_replace_with_cache(
        self,
        full_input_ids: torch.Tensor,
        replace_slice: slice,
        past_key_values,
    ):
        call_idx = len(self.calls)
        self.calls.append(replace_slice)
        B = full_input_ids.shape[0]
        span = replace_slice.stop - replace_slice.start
        # Logits filled with a distinct value per call so we can check them
        logits = torch.full((B, span, self.vocab_size), float(call_idx + 1), dtype=self.dtype)
        new_kv = _make_kv(tag=past_key_values.tag * 10 + (call_idx + 1))
        return logits, new_kv

    def forward_with_kspec_cache(
        self,
        full_input_ids,
        resp_slice,
        aux_past_kv,
        k_spec_mask,
    ):
        # Inline the method from base_model since StubModel ≠ LLaDABaseModel.
        # We replicate the logic to test the algorithm, not the import.
        from aoae.models.base_model import _kspec_find_clusters
        B = full_input_ids.shape[0]
        P = resp_slice.start
        L_gen = resp_slice.stop - P
        dev = full_input_ids.device

        if k_spec_mask.all():
            return (
                torch.zeros(B, L_gen, self.vocab_size, dtype=self.dtype, device=dev),
                aux_past_kv,
            )

        non_agreed_pos = ~k_spec_mask.all(dim=0)
        clusters = _kspec_find_clusters(non_agreed_pos)

        logits_out = torch.zeros(B, L_gen, self.vocab_size, dtype=self.dtype, device=dev)
        current_kv = aux_past_kv

        for c_start, c_end in clusters:
            span_logits, current_kv = self.forward_replace_with_cache(
                full_input_ids,
                slice(P + c_start, P + c_end),
                current_kv,
            )
            logits_out[:, c_start:c_end, :] = span_logits

        return logits_out, current_kv


def _make_stub(V=32):
    return StubModel(vocab_size=V)


class TestForwardWithKspecCache:
    """Test the cluster-dispatch logic (algorithm correctness, not dInfer internals)."""

    # ---- all positions agreed → zero calls, zero logits returned ----

    def test_all_agreed_no_forward_calls(self):
        stub = _make_stub()
        B, L_total, P, L_gen = 1, 10, 3, 7
        full_ids = torch.zeros(B, L_total, dtype=torch.long)
        k_spec_mask = torch.ones(B, L_gen, dtype=torch.bool)  # all agreed
        aux_kv = _make_kv(tag=1.0)

        logits, kv_out = stub.forward_with_kspec_cache(full_ids, slice(P, P + L_gen), aux_kv, k_spec_mask)

        assert stub.calls == [], "No forward calls expected when all positions agreed"
        assert logits.shape == (B, L_gen, stub.vocab_size)
        assert logits.abs().max().item() == 0.0, "Logits should be zeros (caller fills with aux_logits)"
        assert kv_out.tag == aux_kv.tag, "KV cache should be unchanged"

    # ---- all positions non-agreed → one cluster covering full response ----

    def test_all_non_agreed_one_cluster(self):
        stub = _make_stub()
        B, L_total, P, L_gen = 1, 10, 3, 7
        full_ids = torch.zeros(B, L_total, dtype=torch.long)
        k_spec_mask = torch.zeros(B, L_gen, dtype=torch.bool)  # all non-agreed
        aux_kv = _make_kv(tag=1.0)

        logits, kv_out = stub.forward_with_kspec_cache(full_ids, slice(P, P + L_gen), aux_kv, k_spec_mask)

        assert len(stub.calls) == 1
        assert stub.calls[0] == slice(P, P + L_gen), "Should forward over full response span"
        assert logits.shape == (B, L_gen, stub.vocab_size)
        # logits from call 0 → filled with 1.0
        assert (logits == 1.0).all(), f"Expected 1.0 everywhere, got {logits.min()}"

    # ---- two non-agreed clusters, agreed positions stay zero ----

    def test_two_clusters_correct_slices(self):
        stub = _make_stub()
        B, L_total, P, L_gen = 1, 20, 5, 10
        full_ids = torch.zeros(B, L_total, dtype=torch.long)
        # positions 0,1 non-agreed; 2,3,4 agreed; 5,6 non-agreed; 7,8,9 agreed
        k_spec_mask = torch.ones(B, L_gen, dtype=torch.bool)
        k_spec_mask[:, 0] = False
        k_spec_mask[:, 1] = False
        k_spec_mask[:, 5] = False
        k_spec_mask[:, 6] = False
        aux_kv = _make_kv(tag=1.0)

        logits, kv_out = stub.forward_with_kspec_cache(full_ids, slice(P, P + L_gen), aux_kv, k_spec_mask)

        # Exactly two clusters
        assert len(stub.calls) == 2
        assert stub.calls[0] == slice(P + 0, P + 2), f"First cluster wrong: {stub.calls[0]}"
        assert stub.calls[1] == slice(P + 5, P + 7), f"Second cluster wrong: {stub.calls[1]}"

        # Logits at non-agreed positions: call 1 → 1.0; call 2 → 2.0
        assert (logits[:, 0:2, :] == 1.0).all(), "Cluster 0 logits should be 1.0"
        assert (logits[:, 5:7, :] == 2.0).all(), "Cluster 1 logits should be 2.0"
        # Logits at agreed positions: zero (caller will fill with aux_logits)
        assert (logits[:, 2:5, :] == 0.0).all(), "Agreed positions should be zero"
        assert (logits[:, 7:10, :] == 0.0).all(), "Agreed positions should be zero"

    # ---- KV chain: each call receives the KV returned by the previous ----

    def test_kv_chain(self):
        stub = _make_stub()
        B, L_total, P, L_gen = 1, 15, 3, 9
        full_ids = torch.zeros(B, L_total, dtype=torch.long)
        # Three clusters: pos 0,1 | pos 4,5 | pos 8
        k_spec_mask = torch.ones(B, L_gen, dtype=torch.bool)
        for pos in (0, 1, 4, 5, 8):
            k_spec_mask[:, pos] = False
        aux_kv = _make_kv(tag=1.0)

        _, kv_out = stub.forward_with_kspec_cache(full_ids, slice(P, P + L_gen), aux_kv, k_spec_mask)

        # tag starts at 1.0; call 1 → 1.0*10+1=11; call 2 → 11*10+2=112; call 3 → 112*10+3=1123
        assert len(stub.calls) == 3
        assert abs(kv_out.tag - 1123.0) < 1e-6, f"KV chain broken, got tag={kv_out.tag}"

    # ---- batch: union of non-agreed across batch items ----

    def test_batch_union(self):
        """If item 0 agrees at pos 2 but item 1 disagrees, pos 2 must be forwarded."""
        stub = _make_stub()
        B, L_total, P, L_gen = 2, 10, 2, 6
        full_ids = torch.zeros(B, L_total, dtype=torch.long)
        # B=2; position 2: item 0 agrees (True), item 1 disagrees (False)
        k_spec_mask = torch.ones(B, L_gen, dtype=torch.bool)
        k_spec_mask[1, 2] = False  # item 1 disagrees at pos 2
        aux_kv = _make_kv(tag=1.0)

        _, _ = stub.forward_with_kspec_cache(full_ids, slice(P, P + L_gen), aux_kv, k_spec_mask)

        # position 2 is non-agreed for ANY batch item → must be in a cluster
        forwarded_positions = set()
        for sl in stub.calls:
            forwarded_positions.update(range(sl.start - P, sl.stop - P))
        assert 2 in forwarded_positions, "Position 2 must be forwarded (item 1 disagrees)"

    # ---- single non-agreed position (cluster of size 1) ----

    def test_single_non_agreed_position(self):
        stub = _make_stub()
        B, L_total, P, L_gen = 1, 12, 4, 6
        full_ids = torch.zeros(B, L_total, dtype=torch.long)
        k_spec_mask = torch.ones(B, L_gen, dtype=torch.bool)
        k_spec_mask[:, 3] = False  # only position 3 non-agreed
        aux_kv = _make_kv(tag=1.0)

        logits, _ = stub.forward_with_kspec_cache(full_ids, slice(P, P + L_gen), aux_kv, k_spec_mask)

        assert len(stub.calls) == 1
        assert stub.calls[0] == slice(P + 3, P + 4)
        assert (logits[:, 3, :] == 1.0).all()
        assert (logits[:, :3, :] == 0.0).all()
        assert (logits[:, 4:, :] == 0.0).all()


# ---------------------------------------------------------------------------
# 3. Merge logic (torch.where at agreed positions)
# ---------------------------------------------------------------------------

class TestKspecMerge:
    """Test the caller-side merge: where(k_spec, aux_logits, pri_fresh)."""

    def test_merge_agreed_uses_aux(self):
        B, L, V = 1, 6, 32
        k_spec = torch.tensor([[True, True, False, False, True, False]], dtype=torch.bool)
        aux_logits = torch.full((B, L, V), 5.0)
        pri_fresh = torch.full((B, L, V), 9.0)
        # pri_fresh at agreed positions would be 0.0 (zeros from kspec skip)
        pri_fresh_with_zeros = pri_fresh.clone()
        pri_fresh_with_zeros[:, k_spec[0]] = 0.0

        resp_logits = torch.where(k_spec.unsqueeze(-1).expand_as(pri_fresh_with_zeros),
                                  aux_logits, pri_fresh_with_zeros)

        # Agreed positions → aux (5.0)
        for pos in [0, 1, 4]:
            assert (resp_logits[:, pos, :] == 5.0).all(), f"pos {pos} should be aux_logits"
        # Non-agreed positions → pri_fresh (9.0)
        for pos in [2, 3, 5]:
            assert (resp_logits[:, pos, :] == 9.0).all(), f"pos {pos} should be pri_fresh"

    def test_merge_all_agreed(self):
        B, L, V = 2, 8, 16
        k_spec = torch.ones(B, L, dtype=torch.bool)
        aux_logits = torch.randn(B, L, V)
        pri_fresh = torch.zeros(B, L, V)  # zeros from all-agreed early exit

        resp_logits = torch.where(k_spec.unsqueeze(-1).expand_as(pri_fresh), aux_logits, pri_fresh)
        assert torch.allclose(resp_logits, aux_logits)

    def test_merge_none_agreed(self):
        B, L, V = 1, 5, 16
        k_spec = torch.zeros(B, L, dtype=torch.bool)
        aux_logits = torch.full((B, L, V), 7.0)
        pri_fresh = torch.randn(B, L, V)

        resp_logits = torch.where(k_spec.unsqueeze(-1).expand_as(pri_fresh), aux_logits, pri_fresh)
        assert torch.allclose(resp_logits, pri_fresh)


# ---------------------------------------------------------------------------
# 4. Step 1 behaviour: k_spec all-False → processes entire response as 1 cluster
# ---------------------------------------------------------------------------

class TestStep1Behaviour:
    def test_first_step_full_forward(self):
        """On step 1, cache_mgr.spec is all-False → one cluster = full response."""
        stub = _make_stub()
        B, P, L_gen = 1, 4, 8
        L_total = P + L_gen
        full_ids = torch.zeros(B, L_total, dtype=torch.long)
        k_spec_mask = torch.zeros(B, L_gen, dtype=torch.bool)  # step 1: no prior agreement
        aux_kv = _make_kv(tag=1.0)

        logits, _ = stub.forward_with_kspec_cache(full_ids, slice(P, P + L_gen), aux_kv, k_spec_mask)

        assert len(stub.calls) == 1
        assert stub.calls[0] == slice(P, P + L_gen), "Step 1 must forward the full response block"
        assert logits.shape == (B, L_gen, stub.vocab_size)


# ---------------------------------------------------------------------------
# 5. Trivial sanity: clusters cover exactly the non-agreed positions
# ---------------------------------------------------------------------------

class TestClusterCoverage:
    @pytest.mark.parametrize("seed", [0, 7, 42, 137])
    def test_clusters_cover_non_agreed_exactly(self, seed):
        torch.manual_seed(seed)
        L = 32
        non_agreed = torch.rand(L) > 0.5
        clusters = _kspec_find_clusters(non_agreed)

        covered = set()
        for c_start, c_end in clusters:
            for p in range(c_start, c_end):
                covered.add(p)

        expected = {i for i in range(L) if non_agreed[i].item()}
        assert covered == expected, (
            f"Clusters don't match non-agreed positions.\n"
            f"non_agreed={non_agreed.tolist()}\nclusters={clusters}\n"
            f"covered={sorted(covered)}\nexpected={sorted(expected)}"
        )
