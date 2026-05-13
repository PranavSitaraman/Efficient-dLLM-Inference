# DLM Order Variants: Architecture and Attention Patterns

This document precisely defines the attention patterns, input shapes, and
component-level "any-order vs. block" distinctions for every evaluation
variant in this codebase.

---

## 1. Corrected Definitions

### Block-causal attention

The attention mask (shape L × L, built by `_make_attention_mask(block_length=32)`) enforces:

- Positions within the same block (size `block_length=32`): **fully bidirectional** — every position attends to every other in the same block.
- Positions in **earlier** blocks: **fully visible** — the current block can attend to all previously committed blocks (causal across blocks, i.e. no look-ahead into future blocks).
- Positions in **later** blocks: **blocked** — a position cannot attend to tokens in blocks that come after it.

The mask rule is simply: position `i` can attend to position `j` if `j < ceil((i+1)/32)*32` (i.e. `j` is not in a future block relative to `i`).

**The decoding loop also crops the input**: `prefix_ids = y[:, :blk_end]` is passed to the model when processing block `blk_idx`. This means tokens from future (not-yet-generated) blocks are never present in the input — they are simply absent from the sequence, not masked out by the attention mask. The block-causal mask then applies over this cropped prefix, so the current block has full bidirectional attention within itself and full causal visibility into all previous blocks.

In other words: **previous block tokens are NOT fixed/hidden from the current block. They are fully visible to the current block via normal (causal) attention. What is fixed is that the decoding loop does not update them anymore.**

**Critical fact: LLaDA 2.1 was trained exclusively with block-causal
attention at `block_length=32`. Full bidirectional attention (`block_length=0`)
is explicitly documented in the code as out-of-distribution and producing
garbage outputs. There is no true fully-bidirectional base model path in
production.**

### "Any-order" in this codebase

The term **does not mean the base model runs full bidirectional attention**.
The base model (`base_model.forward()`) always calls
`forward_block_causal(block_length=self._block_length)` — where
`_block_length` comes from `cfg["inference"]["block_length"]` (default 32)
— regardless of the eval mode. The "any-order" label means two different
things depending on context:

1. **For LLaDA baselines** (`llada21_speed_anyorder`,
   `llada21_quality_anyorder`): Same `block_smode_decode` decoder as the
   block baseline, but with a **wider block length** (default 128, set by
   `data.any_order_block_length`). This gives 4 bidirectional windows of 128
   tokens over a 512-token response instead of 16 windows of 32. The code
   comments call this "semi-any-order" because it is a pragmatic approximation:
   wider windows give more within-window flexibility while staying closer to
   the trained distribution than a single all-512-token block would.

2. **For trained AOAE variants (speculation policy head)**: the
   `AOAEPolicy` network is a fully bidirectional transformer with no causal
   masking. In any-order eval mode, it receives the **full response region**
   as input and attends globally across all positions simultaneously.

So "any-order" = **global decoding decisions** (policy sees all positions
at once) + **wider base-model blocks** (128 vs 32 for baselines), not full
bidirectional base model attention.

---

## 2. LLaDA Baselines

### Block baseline (`llada21_speed_mode`, `llada21_quality_mode`)

| Component | Attention pattern | Input to each forward |
|-----------|------------------|----------------------|
| LLaDA base model | Block-causal, `block_length=32` (trained pattern) | `y[:, :blk_end]` — cropped prefix up to current block end |

- Decoded left-to-right one block at a time via `block_smode_decode`.
- Within each block: parallel threshold unmasking (M2T) and editing (T2T).
- All components are block-structured. No policy network involved.
- Speed mode: τ_mask=0.5, τ_edit=0.0. Quality mode: τ_mask=0.7, τ_edit=0.5.

### Any-order baseline (`llada21_speed_anyorder`, `llada21_quality_anyorder`)

| Component | Attention pattern | Input to each forward |
|-----------|------------------|----------------------|
| LLaDA base model | Block-causal, `block_length=128` (4× wider than trained) | `y[:, :blk_end]` — prefix up to current (128-token) block end |

- Uses **the same `block_smode_decode` function** as the block baseline.
- Block length is widened to 128 (or `data.any_order_block_length`), so
  there are only 4 blocks over a 512-token response.
- This is labeled "any-order" because within a 128-token block the model
  can unmask tokens in any order simultaneously — substantially more
  flexible than 32-token blocks but still block-structured.
- Still runs left-to-right across the 4 coarser blocks.
- `suppress_eos=True` is applied to prevent the first decode step from
  collapsing to EOS (a known artifact when using wider-than-trained blocks).
- No policy network. No speculative (drafter/verifier) machinery.

**Both LLaDA baselines use an identical base model and an identical decoder
function; the only difference is the block length (32 vs 128).**

---

## 3. Trained AOAE Variants

The trained variants use the **dual-model speculative backend**:
- **Auxiliary (drafter)**: same LLaDA model, hard top-k MoE routing (~1.4B
  active params). Cheap. Used for draft proposals.
- **Primary (verifier)**: same model, soft/widened MoE routing (soft_topk=16
  experts vs. native 8). More accurate. Used for acceptance.
- Both paths call `base_model.forward()` → always `block_causal_32`.

### Any-order eval (`speculative_inference`)

This is the path for configs with `generation_mode_filter: any_order` and
sweep points whose schedule is not in `_HEURISTIC_BLOCK_SCHEDULES`.

| Component | Attention pattern | Input to each forward |
|-----------|------------------|----------------------|
| LLaDA base (aux drafter) | Block-causal, `block_length=32` | Full sequence `y` (prompt + all L_gen response tokens including masked ones) |
| LLaDA base (primary verifier) | Block-causal, `block_length=32` | Full sequence `y` |
| AOAEPolicy | **Fully bidirectional** (no causal mask) | Full response region: `H_t` of shape `[B, L_gen, D]` |

Key properties:
- The base model runs a full-sequence forward at each diffusion step. All
  `L_gen` response positions (masked and unmasked) are present in the
  input simultaneously.
- The block-causal attention mask (32-token blocks) means the model sees
  earlier blocks when processing later blocks (causal across blocks), but
  cannot look ahead.
- The AOAEPolicy receives the **entire** soft-masked state `H_t` over the
  response and attends globally (no windowing). It outputs per-position
  action probabilities (u_t, r_t, κ_t, q_t) over all L_gen positions
  in one pass.
- "Any-order" refers to the policy's ability to unmask any position in the
  response at any step, not to the base model's attention pattern.

**In summary for any-order trained variants: base model = block-causal-32 on
full sequence; policy = globally bidirectional on full response.**

### Blockwise eval (`aoae_block_inference`)

This is the path for configs with `generation_mode_filter: block` and
sweep points whose schedule maps to `aoae_block_trained` or
`aoae_block_policy`.

| Component | Attention pattern | Input to each forward |
|-----------|------------------|----------------------|
| LLaDA base (aux drafter) | Block-causal-32 over cropped prefix: current block is fully bidirectional, all previous blocks are fully visible (causal), future blocks absent | `prefix_ids = y[:, :blk_end]` — prompt + all committed blocks + current (partially masked) block |
| LLaDA base (primary verifier) | Same as aux | Same `prefix_ids = y[:, :blk_end]` |
| AOAEPolicy | **Fully bidirectional within its input** (no causal mask) | Current block's H_t slice only: `H_blk = H_t[:, resp_b_s:resp_b_e]`, shape `[B, block_len, D]` |

Key properties:
- Decoding proceeds left-to-right, block by block (block_idx = 0, 1, …).
- **Base model** receives `prefix_ids = y[:, :blk_end]`: all tokens from
  position 0 through the end of the current block. This includes the prompt,
  all previously committed response blocks (now filled with real tokens), and
  the current block (still partially or fully masked). Future blocks are simply
  absent — not in the input. The block-causal-32 attention mask then gives:
  - Current block positions: bidirectional attention within the block + full
    causal attention into all previous blocks (they are entirely visible).
  - Previous block positions: only see up to their own block end (they cannot
    look ahead into the current block — this is the "causal across blocks"
    direction).
- **AOAEPolicy** receives only `H_blk`, the soft-masked hidden state for the
  current block (shape `[B, block_len, D]`). It attends globally across those
  `block_len` positions (fully bidirectional within its input), but has no
  direct access to previous blocks' H_t. Its information about the committed
  prefix comes only indirectly, via the base model's logits that produced H_blk.
- Draft/verify micro-steps happen within each block before moving to the next.

**In summary for blockwise trained variants: base model = block-causal-32 on
the prefix up to blk_end, so previous block tokens are fully visible to the
current block (causal attention, not hidden). Policy = globally bidirectional
but only over the current block's H_t slice — it has no direct view of
previous blocks' hidden states.**

---

## 4. Side-by-Side Component Summary

| Variant | Base model attention | Base model input scope | Policy attention | Policy input scope |
|---------|---------------------|------------------------|------------------|--------------------|
| Block baseline (LLaDA speed/quality) | Block-causal-32 (trained) | Prefix up to current block | — (no policy) | — |
| Any-order baseline (LLaDA speed/quality anyorder) | Block-causal-128 (wider, ~OOD) | Prefix up to current (128-tok) block | — (no policy) | — |
| Trained: any-order eval | Block-causal-32 (trained) | **Full sequence** every step | **Fully bidirectional** | **Full response** (all L_gen positions) |
| Trained: block eval | Block-causal-32: current block bidi + full causal view of prior blocks | Prefix up to current block end (`y[:, :blk_end]`) | **Fully bidirectional within its input** | **Current block's H_t only** (prior blocks not directly visible to policy) |

---

## 5. Key Takeaways

1. **The LLaDA base model is always block-causal-32** regardless of eval
   mode. There is no true full-bidirectional base model path. The "any-order"
   label for baselines means a wider block (128), not full attention.

2. **The "any-order" vs "block" distinction for trained variants is primarily
   about the decoding loop structure and policy input scope:**
   - Any-order: single global diffusion loop over full response; policy sees
     all positions simultaneously.
   - Block: left-to-right block loop; base model and policy both operate on
     the current block's prefix/slice.

3. **The AOAEPolicy is always a fully bidirectional transformer** (no causal
   masking). The difference is what it is _fed_: the full response (any-order)
   or just the current block (blockwise). The architecture does not change
   between modes; only the input slice changes.

4. **GRPO training** uses `speculative_inference` (the any-order loop) with
   `record_trajectory=True`. The policy is trained on the any-order task:
   global decisions over the full response, with a block-causal-32 base model
   serving logits for the full sequence at each diffusion step.

---

## 6. Is the Blockwise Policy Head Reasonable?

**Short answer: yes, by analogy with autoregressive LMs — and arguably more
expressive. But there is a real train/eval mismatch.**

### The AR analogy

In a standard autoregressive LM the "policy head" (the LM head deciding
the next token) operates on the hidden state at a single position. That
hidden state was computed with full causal attention over the entire
prefix, so:
- **Direct input to the head**: scalar hidden state at position k only.
- **Indirect knowledge of prefix**: fully encoded in that hidden state via attention.

The blockwise AOAEPolicy is strictly analogous:
- **Direct input**: H_t for the current block's positions (shape `[B, 32, D]`).
- **Indirect knowledge of committed prefix**: fully encoded in those H_t
  vectors, because the base model's causal attention over `y[:, :blk_end]`
  has already incorporated all prior blocks into the current block's
  representations.

So claiming "previous block information is missing" would be wrong — it is
present, just implicitly, exactly as it is in the AR case.

### Where the blockwise policy is strictly more expressive than AR

The policy attends **bidirectionally across all 32 positions in the current
block simultaneously**. An AR head only has a scalar at one position. This
allows the blockwise policy to coordinate its unmask/remask/cache decisions
jointly across the block — choosing which of the 32 positions to unlock
together, aware of which ones are still masked vs. already committed in
prior micro-steps within the block. There is no AR equivalent of this
joint within-block coordination.

### The genuine concern: train/eval distributional shift

The policy was GRPO-trained in **any-order mode**, where it receives H_t
over all L_gen=512 positions at once. At blockwise eval time it receives
H_blk over only 32 positions. There are three potential shifts to consider,
but they have very different severities once you examine the architecture.

**First, and most important: the policy transformer has no positional encoding.**

`AOAEPolicy` uses `nn.TransformerEncoderLayer` with no learned or sinusoidal
position embeddings added anywhere (confirmed by inspection of `policy.py`).
The only positional information available to the backbone is whatever is
implicit in `H_t` itself (which comes from the base model) and the scalar
per-position features (`m_feat`, `t_feat`, confidence, agreement, etc.).
Without explicit positional encoding, the policy transformer is
**permutation-equivariant** — it is a set function that produces the same
output for each position regardless of where that position sits in the
sequence. This has two consequences:

- **Sequence length is not a problem at all.** A permutation-equivariant
  transformer handles any input length without distributional shift. Feeding
  32 positions instead of 512 is not a mismatch — the backbone has no
  expectation of a particular length.

- **Position within the sequence is not directly encoded.** The policy
  cannot distinguish "position 5 of the full response" from "position 5 of
  the current block" by learned positional index. Any such information must
  come from the content of H_t (which the base model does encode positionally
  via its own rotary/block-causal attention) and from the scalar features
  passed in.

**How does the policy know where to unmask without positional encoding?**

It doesn't need to. The policy is not learning *where* (in the abstract,
positional sense) to unmask. It is learning *what kind of position* to
unmask — high confidence, high drafter/verifier agreement, low entropy,
still masked. These are purely **content-based, per-position scalar features**.
Absolute positional identity is irrelevant to that criterion.

Positional identity does matter, but it arrives **pre-baked in H_t[k]**:
the base model has full positional encoding (RoPE + block-causal attention),
so H_t[k] already encodes "I am position k, conditioned on everything to
my left." The policy just reads off confidence/agreement/mask status at
each position and decides comparatively. This is exactly what Jazbec et al.
do — no positional encoding in the policy, decisions driven by content
features derived from the diffusion model's own hidden states.

The policy's transformer backbone adds one thing: **comparative reasoning**.
Instead of independently thresholding each position ("unmask if confidence
> 0.7"), the backbone lets each position attend to all others and produce
a relative ranking ("unmask the top-K by confidence among all currently
masked positions"). This comparative ranking works correctly regardless of
sequence length, because it is always relative within whatever window is
fed in.

**The two remaining shifts that do matter:**

1. **H_t conditioning, not policy architecture**: the real mismatch is that
   H_t itself is differently conditioned between the two modes. In any-order
   training at step t, the base model sees ~(t/T)×512 already-committed
   positions scattered across the full 512-slot response — a rich, mixed
   context. In blockwise eval at block start, all 32 current-block positions
   are masked and the base model only sees the committed prefix (prior blocks)
   plus 32 fresh masks. The base model's logits — and thus H_t — reflect this
   different conditioning. The policy's input distribution is shifted not
   because the policy architecture is fragile, but because the base model is
   operating in a different regime. This is a real but **mild** effect: the
   base model was itself trained with block-causal-32 attention, so the
   blockwise conditioning is closer to its training distribution than the
   any-order full-sequence conditioning is.

2. **step_frac scalar**: in the any-order loop step_frac = t/T decreases
   smoothly at every diffusion step. In `aoae_block_inference` it is
   approximated as `1 - blk_idx/n_blocks`, constant within a block's
   micro-steps. Mild inaccuracy — same scalar feature, same range, same
   meaning, just coarser.

**Summary of mismatch severity:**

| Concern | Severity | Reason |
|---------|----------|--------|
| Sequence length 512 → 32 | **None** | Policy is permutation-equivariant; no positional encoding |
| Policy not knowing "where" to unmask | **None** | Policy learns content-based criteria (confidence, agreement); position is implicit in H_t from the base model |
| H_t conditioning (base model regime) | **Mild** | Base model itself was trained with block-causal-32, so blockwise is closer to its training distribution than any-order full-sequence is |
| step_frac approximation | **Mild** | Coarser scalar signal, not a structural shift |

The train/eval mismatch for blockwise evaluation of an any-order-trained
policy is less severe than it initially appears. The policy's decision
criterion (content-based, comparative, positional-encoding-free) transfers
cleanly to shorter windows. The main residual concern is the different
base-model conditioning, which is actually ameliorated by the fact that
block-causal-32 is the base model's native training regime.

---

## 7. Relevant Source Locations

| File | Key functions / classes |
|------|------------------------|
| `aoae/models/base_model.py` | `LLaDABaseModel.forward()` → always `forward_block_causal(block_length=32)` ; `_make_attention_mask()` |
| `aoae/models/dual_model.py` | `DualModelWrapper.auxiliary_forward()`, `primary_forward()` — both delegate to `_model.forward()` |
| `aoae/models/policy.py` | `AOAEPolicy` — bidirectional `TransformerEncoder` backbone; `call_policy_block()` — crops input to active block window |
| `aoae/inference.py` | `block_smode_decode()` — block baseline decoder (also used by any-order baselines with wider block_length); `_override_block_length()` context manager |
| `aoae/speculative_inference.py` | `speculative_inference()` — any-order trained eval; `aoae_block_inference()` — blockwise trained eval |
| `aoae/evaluate.py` | `_semi_any_order_block_length()` — returns default 128 for any-order baselines; `_BASELINE_GENERATION_MODES` — maps baseline names to "any_order"/"block" |
