# 2026-05-03 — Soft-gated verifier hurts at `quality_balanced`

## Setup

Eval against `policy_best.pt` (trained with `train_heads=["cache","access"]`, so
the trained policy contributes nothing to unmask/remask — accuracy comes from
the speculative loop's argmax_match self-correction with a fixed threshold-rule
drafter `(confidence ≥ 0.7) & mask`). 50 GSM8K samples, schedule=`aoae`
(any-order full speculative).

2×2 ablation around `quality_balanced`:

| | soft verifier (top-16) | hard verifier (top-8) |
|---|---|---|
| **q-drafter** (thresh 0.7) | `quality_balanced` (baseline) | `quality_balanced_hardver` |
| **s-drafter** (thresh 0.5, larger budget+microsteps+cap) | `quality_balanced_sq` | `quality_balanced_sq_hardver` |

`hardver` flips `base_model.lossless_verification: true`, which short-circuits
`primary_forward` to `auxiliary_forward` inside `DualModelWrapper`
(`aoae/models/dual_model.py:154-160`). `sq` flips:
`drafter.confidence_threshold: 0.5`, `verifier_schedule.draft_token_budget: 16`
(from 8), `max_draft_microsteps: 4` (from 3), `max_unmask_fraction_per_step:
0.25` (from 0.125). All other knobs match `quality_balanced` exactly.

SLURM: job 9834025, `outputs/qb_ablations/`.

## Results

| Sweep point | Acc | TPS |
|---|---|---|
| `quality_balanced` (soft, q-drafter) | 0.82 | 51.1 |
| **`quality_balanced_hardver` (hard, q-drafter)** | **0.86** | **54.7** |
| `quality_balanced_sq` (soft, s-drafter) | 0.72 | 48.3 |
| `quality_balanced_sq_hardver` (hard, s-drafter) | 0.70 | 50.5 |

Reference baselines (same eval run):

| Method | Acc | TPS |
|---|---|---|
| `llada21_speed_anyorder` | 0.44 | 160.7 |
| `llada21_quality_anyorder` | 0.22 | 156.3 |
| `fast_dllm` | 0.62 | 46.5 |

## Surprising finding

**Hard verifier Pareto-dominates soft verifier at `quality_balanced`.**
`quality_balanced_hardver` beats `quality_balanced` on *both* accuracy
(86 vs 82) and TPS (55 vs 51). Widening the verifier's expert routing from
top-8 to top-16 — the entire point of the dual-routing design — is
strictly hurting at this operating point.

This contradicts the prior assumption that soft top-16 routing helps accuracy
on speculative decoding. Two candidate explanations to investigate:

1. **Routing-vs-acceptance interaction.** The verifier's job here is
   `argmax_match` against the drafter. Widening to top-16 produces *more
   diverse* logits than the drafter (which is hard top-8 / native LLaDA
   routing). That diversity makes argmax disagreement more likely *even when
   the drafter is correct*, leading to spurious rejection-then-remask cycles.
   Hard routing makes the verifier closer to the drafter's distribution,
   which actually raises argmax_match acceptance rate of correct drafts —
   improving both correctness (fewer good tokens are rejected) and TPS
   (fewer wasted re-drafts).
2. **Top-16 is OOD for the model.** LLaDA 2.1 was trained at top-8 grouped
   top-k. Forcing top-16 at inference may be slightly off-distribution for
   the expert routing weights, manifesting as small accuracy degradations
   even when the soft logits look superficially better-calibrated.

Both are testable. (1) would predict that the gap widens at lower
`primary_agree_threshold` (more verifier calls, more chances to spuriously
reject). (2) would predict a similar gap appears in non-speculative settings
(e.g., direct LLaDA decode with `lossless_verification: true` vs `false`).

## Secondary finding

**s-mode drafter dynamics hurt.** Both sq variants regress to ~70%
accuracy. Lowering the drafter threshold (0.5 vs 0.7), bumping budget
(16 vs 8), microsteps (4 vs 3), and cap (0.25 vs 0.125) does not translate
into meaningful TPS gains over the q-drafter baselines (48–51 TPS in the
sq band, 51–55 in the q band) — and accuracy drops 12–16 points. The
classical LLaDA s/q split, when applied inside the AOAE-headed any-order
speculative loop, appears to be a Pareto regression rather than a tradeoff.

Hypothesis: at threshold 0.5 the drafter proposes far more low-confidence
tokens. The verifier (whether soft or hard) rejects most of them, and the
rejected positions get remasked and re-drafted, paying compute without
making progress. With the more aggressive cap (0.25), the policy commits
12.5% → 25% of remaining masks per microstep, but if rejection rates
double, the actual *committed* token rate doesn't change much — it just
costs more compute per committed token. Net: more compute per correct
token, hence no TPS win and an accuracy hit.

## Implications for paper / design

- Don't include "soft verifier" as a default in the published Pareto. The
  baseline pitch — that we widen experts at verification time for accuracy —
  doesn't hold at the operating point we care about.
- Don't include LLaDA-style s/q drafter dynamics as a free win in the
  speculative loop. They cost accuracy without buying speed in the
  AOAE-headed any-order regime.
- GRPO training target should likely be `quality_balanced_hardver`
  (or a similarly hardver-flipped variant), not `quality_balanced`.

## qmax replication (64 samples, 2026-05-03, job 9850268)

| Variant | Acc | TPS |
|---|---|---|
| `quality_max` (soft, q-drafter) | **0.859** | 42.84 |
| `quality_max_hardver` (hard, q-drafter) | 0.812 | **45.87** |
| `quality_max_sq` (soft, s-drafter) | 0.750 | 42.56 |
| `quality_max_sq_hardver` (hard, s-drafter) | 0.734 | 42.79 |

**Key contrast vs qbal**: at the more conservative `quality_max` operating
point (`primary_agree_threshold=0.98`, `draft_token_budget=4`,
`max_draft_microsteps=1`, `max_unmask_fraction_per_step=0.0625`), hardver
**does not Pareto-dominate**. It still wins on TPS (+3.0) but loses
accuracy (−4.7 points). So the soft-verifier-hurts effect is *operating-point
dependent*, not a general property.

Likely explanation: at qmax, the very high `primary_agree_threshold=0.98`
forces frequent verifier calls — the verifier dominates the trajectory,
and soft top-16 routing's expressiveness gives a real accuracy benefit
that outweighs the argmax-disagreement cost. At qbal (`0.92`, looser),
the verifier runs less often and the speculative argmax_match dynamics
matter more, where routing diversity → spurious disagreement → retry
loops dominates.

Implication for the AOAE Pareto: the verifier-routing choice may need to
vary along the Pareto curve, not be globally fixed. For the upcoming GRPO
run we keep `quality_balanced_hardver` as the training target (since
that's where hardver Pareto-dominates), but eval the trained policy at
both qbal-hardver and qmax-soft to see how the trained heads transfer
across the curve.

The s-drafter regression is consistent across both operating points: a
clean signal that s-mode dynamics in the AOAE-headed any-order loop are
not useful in either regime.

## Open follow-ups

- Direct hard-vs-soft test outside speculative loop (single-model decode
  with `lossless_verification` toggled) would isolate the routing-OOD
  hypothesis from the speculative-disagreement hypothesis.
- A Pareto-spanning sweep with `primary_agree_threshold ∈ {0.85, 0.92,
  0.96, 0.98}` × {soft, hard} verifier × q-drafter, to map where the
  hardver vs soft crossover sits.
