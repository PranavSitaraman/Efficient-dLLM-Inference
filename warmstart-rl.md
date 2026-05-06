# Warm-Start RL for Diffusion Language Model Sampling Policies

## Paper: "Learning Unmasking Policies for Diffusion Language Models" (Jazbec et al., 2512.09106v3)

### Paper Summary

Jazbec et al. formalize dLLM sampling as an MDP where the frozen dLLM is the *environment* and a lightweight standalone policy network (single-layer transformer, ~300K params) learns which masked positions to unmask at each step. They train via GRPO with a multiplicative reward: `R = correctness * (1 - steps_used/T)^alpha`. Key findings:

1. **Confidence-only input suffices in cold-start.** The policy takes only per-position confidence scalars c_t^k = max_v p(v|y_t) plus a binary mask vector. They explicitly tried conditioning on the dLLM's hidden state H_t (as a classification head on h_t^k in R^H) and found it (a) required a much larger policy (~300M vs 300K params), (b) performed worse, and (c) exhibited less stable training dynamics (Section 4.4, Figure 8d).

2. **Expert steering helps but destabilizes.** In the full-diffusion regime (BL=L=256, no semi-AR blocks), naive cold-start RL underperforms semi-AR training. Their fix: "expert steering" — mix 1 heuristic rollout (Fast-dLLM, BL=32) into each GRPO group of G policy rollouts. This biases the policy toward semi-AR-like left-to-right generation, closing the accuracy gap. But it introduces "significant instability during training, further reducing the controllability through alpha, with multiple values of alpha collapsing to near-identical policies" (Section 4.2).

3. **Multiplicative reward prevents reward hacking.** Additive reward `r - alpha*(steps/T)` causes collapse to "unmask everything immediately" when all samples in a group are incorrect (early training), because faster-but-wrong samples get positive advantage. Multiplicative reward gates speed bonus on correctness.

4. **Policy learns qualitatively different strategies than heuristics.** RL policies unmask tokens in scattered, non-adjacent patterns (only ~20% adjacent tokens vs ~65% for Fast-dLLM), distribute compute more uniformly across blocks, and in the fast regime (alpha=10) learn to slow down on the final numerical-answer block.

---

## Our Setup vs. Theirs: Key Structural Differences

| Dimension | Jazbec et al. | Our AOAE System |
|-----------|--------------|-----------------|
| **Training paradigm** | Cold-start RL only (random init -> GRPO) | Warm-start supervision -> RL (PRISM-like) |
| **Policy scope** | Unmask only (u_t) | Unmask (u_t) + Remask (r_t) + Cache (kappa_t) + Access (q_t) |
| **Architecture** | Single-model dLLM as environment | Dual-model speculative: drafter (hard MoE) + verifier (soft MoE) |
| **Policy input** | Confidence scalars + binary mask | Scalar features (confidence, agreement, quality) OR H_t embeddings |
| **Inference** | Standard masked diffusion steps | Speculative inference with draft-verify-accept/reject loop |
| **Policy size** | ~300K params | ~300K params (d_model=128, 1 layer, 4 heads) |
| **Expert guidance** | Post-hoc "expert steering" (mix heuristic rollouts) | V3.5: expert_steering in GRPO; V4: explicit supervised warmstart |

The critical difference: they never pre-train their policy. Everything is learned from scratch via GRPO. We, by contrast, have a supervised warmstart phase that teaches the policy the structure of good u_t/r_t decisions before RL refinement.

---

## Question 1: Should We Include Expert Steering in Post-Warmstart RL?

### What the paper finds about expert steering (cold-start)

Expert steering (Appendix F) mixes E=1 deterministic expert rollout (Fast-dLLM, lambda=0.9, BL=32) into each GRPO group of G policy rollouts. The mixture policy is:

```
pi_ES = (G/(G+E)) * pi_phi + (E/(G+E)) * sum(delta_e)
```

Effects observed:
- **Positive:** Almost closes the accuracy gap between full-diffusion and semi-AR training. The expert provides an implicit curriculum: when the policy is worse than the expert, the expert sample gets positive advantage, biasing the policy toward it. When better, the expert gets negative advantage and the policy moves beyond it.
- **Negative:** "Significant instability during training, further reducing controllability through alpha, with multiple values of alpha collapsing to near-identical policies." They leave stabilization to future work.

### Our V3.5 experience with expert steering (cold-start RL)

V3.5 used expert_steering in cold-start GRPO. Result: accuracy oscillated around ~0.10 with high variability. The policy never stabilized — consistent with the paper's warning about instability.

### Analysis: Expert steering after warmstart

**Argument for including expert steering post-warmstart:**
- The warm-started policy already knows the approximate structure of good u_t/r_t decisions. Expert rollouts would serve as a *diversity mechanism* rather than a *bootstrap mechanism*. In cold-start, the expert essentially teaches the policy *what* to do; after warmstart, it would teach the policy *alternative strategies* it might not explore on its own.
- The paper shows that in full-diffusion (BL=L), the policy struggles to discover left-to-right generation patterns from scratch. Expert steering's main value is injecting this structural prior. After warmstart, we already have this prior baked into the policy weights — the supervised labels explicitly teach which positions to unmask (high-confidence, drafter-favored) and which to remask (low-confidence, verifier-rejected).

**Argument against including expert steering post-warmstart:**
- The instability the paper reports is fundamentally about the expert rollout having very different likelihood under pi_phi than the policy's own rollouts. This creates large importance sampling corrections in the GRPO objective. After warmstart, this gap should be *smaller* (the policy is closer to expert-like behavior), but the issue doesn't vanish — any time the expert takes an action the policy assigns low probability, the ratio rho_t explodes or collapses.
- Our V3.5 instability was likely compounded by the fact that our policy has *more heads* (u_t + r_t + kappa_t + q_t) and operates in a speculative inference loop with draft-verify dynamics. The expert heuristic only controls unmasking; the remask/cache/access decisions remain unguided, creating a partial-expert problem where the advantage signal is noisy.
- After warmstart, the policy should have enough structure to explore productive strategies on its own via GRPO's group sampling. The G=8 group already provides exploration diversity. Adding expert rollouts may actually *constrain* exploration to semi-AR-like patterns, when our speculative system may benefit from genuinely different strategies (e.g., aggressive draft-then-verify).

**Recommendation:** **Disable expert steering for V4 post-warmstart GRPO.** The warm-started policy provides the structural prior that expert steering was designed to inject. The instability risk is not worth the marginal exploration benefit. If RL training shows signs of mode collapse or insufficient exploration (all G rollouts converging to identical behavior), *then* consider re-enabling expert steering with a tighter clipping epsilon (the paper uses epsilon=0.2 for expert steering vs 0.5 without) and perhaps E=1 with a *softer* expert (e.g., the warm-started policy's own greedy rollout rather than a heuristic).

**Key ML principle:** The purpose of expert guidance in RL is to provide a *curriculum* when the policy is too far from any good solution to discover one through random exploration. Warmstart explicitly addresses this by moving the policy close to a good solution before RL begins. Applying both is redundant at best, destabilizing at worst — analogous to using both pre-training and auxiliary imitation loss in a policy gradient method, which often causes conflicting gradients.

---

## Question 2: Does Warm-Start Enable Effective H_t Conditioning?

### What the paper finds about H_t (cold-start)

Section 4.4 ("Confidence-based policy input"): They tried parameterizing the policy as "an additional classification head on top of LLaDA's final hidden state h_t^k in R^H." Results:
- Required ~300M parameters (1000x larger than confidence-based policy)
- Performed *worse* than confidence-only (Figure 8d)
- Exhibited less stable training dynamics

Their explanation: "The unembedding matrix W in R^{H x V}, which maps hidden states to token logits, plays a vital role in enabling effective policy decisions via confidence signals." They cite the early-exit literature (Schuster et al., 2022) where confidence-based stopping criteria outperform hidden-state-based ones.

### Critical reading: Why H_t failed in their cold-start setting

Their result has a clear confound: **they trained from scratch with RL only**. Consider what happens when a randomly initialized policy head receives H_t as input:

1. **The H_t space is high-dimensional and unstructured from the policy's perspective.** H_t in R^H (H=4096 for LLaDA-8B) is a rich representation that encodes syntax, semantics, uncertainty, and generation state — but a randomly initialized linear head has no idea how to decode any of this information.

2. **RL provides only scalar reward signal.** GRPO gives the policy a single number (correct/incorrect * speed) at the end of generation. Learning to extract useful features from a 4096-dim hidden state using only this sparse signal is a classic credit assignment nightmare. The policy must simultaneously learn (a) what features of H_t matter for unmasking decisions and (b) what unmasking decisions lead to good outcomes.

3. **Confidence is a *pre-extracted* sufficient statistic for their task.** For pure unmasking decisions, the confidence c_t^k = max_v p(v|y_t) is already the unembedding matrix W applied to H_t followed by softmax and max. It's a dimensionality reduction from R^H to R^1 that preserves exactly the information a threshold-based heuristic would use. No wonder a cold-start RL policy can learn from 1D input faster than from 4096D input.

4. **The 300M parameter count is revealing.** They needed 1000x more parameters to get *worse* performance. This suggests the model was overfitting to spurious correlations in H_t space during RL training, not learning meaningful features.

### Why warm-start changes the calculus

Your intuition is well-founded. Consider the PRISM analogy:

**PRISM's supervised method successfully learned a quality predictor conditioned on H_t that outperformed confidence-only heuristics.** PRISM used supervised labels (e.g., "this position was verified correct by a stronger model") to train a linear probe on H_t. The supervised signal told the probe *exactly which features of H_t predict quality*, bypassing the credit assignment problem entirely.

Our warmstart phase does the same thing for the policy head:

1. **Supervised warmstart provides dense, per-position labels.** Unlike RL's sparse end-of-episode reward, warmstart gives per-position binary labels: "this position should be unmasked" (based on future verifier acceptance), "this position should be remasked" (based on future verifier rejection). These labels directly teach the policy what features of H_t (or scalar features) predict good unmasking/remasking decisions.

2. **After warmstart, the policy head has learned a meaningful projection of H_t.** The weights connecting H_t to the u_t/r_t heads are no longer random — they encode the structure that the supervised labels taught. When RL begins, the policy is not starting from scratch in a 4096-dim wilderness; it's starting from a feature representation that already captures the relevant aspects of H_t.

3. **RL then only needs to *refine* the learned features, not discover them.** The RL phase's job shifts from "learn what H_t features matter AND learn what actions are good" to just "learn better action policies given already-meaningful features." This is a much easier optimization problem.

### The formal argument

Let f_phi: R^H -> R^2 (for u_t, r_t heads) be the policy mapping from H_t.

**Cold-start RL:** f_phi is randomly initialized. The effective hypothesis class is all linear/nonlinear mappings from R^H to R^2. The RL signal must search this entire space. The manifold of "useful" mappings is a tiny subspace, and the probability of random initialization landing near it is negligible. Gradient steps from sparse reward are noisy and may push toward local optima that exploit superficial correlations (e.g., norm of H_t) rather than meaningful features (e.g., whether the token is syntactically complete).

**Warm-start then RL:** f_phi is initialized by supervised training on dense per-position labels. The supervised phase projects f_phi onto or near the manifold of useful mappings. RL starts from this good initialization and refines — equivalent to fine-tuning a pre-trained feature extractor, which is known to be dramatically more sample-efficient and stable than training from scratch.

**Analogy to vision/NLP transfer learning:** This is the exact same phenomenon as ImageNet pre-training enabling effective fine-tuning on small downstream tasks. A randomly initialized ResNet cannot learn to classify medical images from 100 examples; a pre-trained one can. The pre-training teaches the general feature hierarchy; fine-tuning adapts it. Our warmstart teaches the H_t feature hierarchy; GRPO adapts it.

### Counterarguments to consider

1. **Warmstart labels may be biased.** The supervised labels come from the default policy's trajectories. If the default policy makes systematic errors (e.g., always unmasking left-to-right), the warm-started H_t features may encode these biases. RL must then *unlearn* some features while refining others, which could be harder than learning from scratch if the biases are strong.

   *Counter-counterargument:* The labels are derived from *verifier outcomes*, not from the default policy's decisions. A position is labeled "unmask" if the verifier later accepts it, regardless of whether the default policy chose to unmask it. This grounds the labels in the environment's dynamics, not the policy's heuristics.

2. **The PRISM result may not generalize.** PRISM's quality predictor is a simpler function (scalar quality score) than our full policy (4 action heads with different scoping across drafter/verifier steps). The feature extraction requirements may be qualitatively different.

   *Counter-counterargument:* We're proposing warmstart specifically for u_t and r_t in V4, which are structurally similar to PRISM's quality predictor (binary decisions conditioned on position-level features). The cache (kappa_t) and access (q_t) heads are disabled in V4.

3. **Even with warmstart, H_t may add optimization overhead in RL.** More parameters = more gradients to compute per step, potentially slower convergence per wall-clock second.

   *Valid concern:* This is an empirical question. If confidence-only with warmstart already achieves good performance, the added cost of H_t may not be justified. The experiment to run: compare warmstart+RL with scalar_only features vs. warmstart+RL with H_t features, measuring both final performance and training stability.

### Recommendation

**The warm-start phase should make H_t conditioning viable and potentially beneficial for the RL phase, but this needs empirical validation.** The paper's negative result on H_t is specific to cold-start RL and should not be taken as evidence that H_t is uninformative — it's evidence that H_t is too hard to learn from scratch with sparse RL signal.

**Proposed experiment path:**
1. V4 (current): Warmstart with scalar_only features, then GRPO with scalar_only. This establishes the baseline.
2. V5 (next): Warmstart with H_t features (feature_mode: hidden), then GRPO with H_t features. Compare training stability and final performance.
3. If V5 is unstable: Try warmstart with H_t features, then GRPO with scalar_only. This tests whether the supervised phase can learn from H_t even if RL cannot use it effectively.

The key prediction: **V5 warmstart (supervised on H_t) should converge faster and to a better loss than V4 warmstart (supervised on scalars only), because the supervised labels provide enough signal to learn the H_t -> action mapping.** Whether this advantage carries through RL is the open question.

---

## Summary of Recommendations

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| Expert steering in post-warmstart GRPO? | **No** (disable) | Warmstart provides the structural prior that expert steering injects; adding both is redundant and risks destabilization |
| H_t conditioning with warmstart? | **Worth testing** (V5) | Paper's negative result is cold-start-specific; supervised warmstart should enable effective H_t feature learning |
| Immediate V4 plan | Scalar-only warmstart -> scalar-only GRPO (no expert steering) | Conservative baseline; evaluate before adding complexity |

---

## Paper: "PRISM — Fine-Tuning Masked Diffusion for Provable Self-Correction" (Kim, Kim, Lee et al., 2510.01384v3)

### Paper Summary

PRISM introduces a lightweight, plug-and-play supervised fine-tuning framework that teaches any pretrained MDM to *self-correct* by learning per-token quality scores. The key insight: given a partially masked sequence y, PRISM learns g*(y) = p(x^i = y^i | y ⊕ m_i) — the probability that the token at position i is correct, conditioned on the rest of the sequence with position i re-masked. This is trained via a binary cross-entropy loss:

```
L(phi) = E[BCE(1[x^i = y^i], g_phi^i(y))]
```

where x is the ground truth, y is a sequence obtained by unmasking some positions from a masked version z of x using the pretrained MDM. The minimizer provably recovers the true per-token quality without RL or a verifier.

### Key Architectural and Design Choices

1. **Shared backbone, separate heads.** PRISM adds a lightweight adapter head to the existing MDM backbone. The backbone computes hidden states h_theta, which are passed through (a) the original unmasking head (softmax → token distribution) and (b) a new quality head (sigmoid → scalar in [0,1]). Both heads share the same backbone forward pass — zero additional inference cost.

2. **The quality head conditions on H_t.** This is critical for our Q2 analysis. PRISM's g_theta operates on the MDM's final hidden state, not just on confidences. The quality score is fundamentally a learned function of H_t, projecting the hidden representation to a scalar quality prediction. For LLaDA-8B, they add ~250M trainable parameters (LoRA adapter + quality head).

3. **Supervised training, not RL.** PRISM's loss is a supervised BCE loss with dense per-position binary labels (1[x^i = y^i]). Each training sample provides labels for k positions simultaneously (they unmask k positions from z to get y, then supervise all k). This is dramatically more sample-efficient than RL: PRISM fine-tuning on LLaDA-8B uses only 0.1M data pairs and completes in ~30 hours on 12 H100s.

4. **Regularization preserves unmasking ability.** The total loss includes lambda * MDM_loss as a regularizer, ensuring the backbone's unmasking posterior doesn't degrade while learning quality scores.

5. **Theoretical guarantee is independent of f_theta's quality.** Even if the pretrained MDM is imperfect, the PRISM loss minimizer still converges to the true per-token quality. This contrasts with distillation approaches where imperfect teachers yield imperfect students.

### Results

- **Sudoku (30M MDM):** PRISM outperforms ReMDM and ReMDM-conf within a few fine-tuning epochs (~530x fewer than pretraining epochs). The learned quality scores correctly identify incorrect cells.
- **OpenWebText (170M MDM):** PRISM improves generative perplexity and MAUVE, especially at low NFE counts (N < 1024). Fine-tuned with 1600x fewer tokens than pretraining.
- **LLaDA-8B (code generation):** On MBPP, PRISM achieves 32.3% at 1024 steps vs 31.9% for vanilla MDM. Gains are largest at low step counts (21.8% vs 18.2% at 256 steps), confirming that self-correction is most valuable when parallel generation creates more dependency errors.
- **Calibration:** PRISM quality scores are well-calibrated — tokens binned by predicted quality have empirical likelihoods closely matching the predictions.

### PRISM's Explicit Demonstration: Supervised Learning on H_t Works

This paper provides the strongest direct evidence for our Q2 argument. Key observations:

1. **PRISM successfully trains a quality head on H_t using only supervised learning.** The quality head is a function g_theta: H_t → [0,1] that learns to predict per-token correctness. It conditions on the full hidden state, not just confidence. And it works — achieving state-of-the-art remasking quality across scales.

2. **The quality head is lightweight relative to the backbone.** For LLaDA-8B, ~250M trainable params (LoRA + head) vs 8B backbone. This is much more efficient than Jazbec et al.'s 300M H_t policy that failed in cold-start RL. The difference? PRISM uses supervised labels; Jazbec et al. used RL reward.

3. **Dense per-position labels are the key enabler.** PRISM's BCE loss provides a gradient signal for every unmasked position in every training sample. Each gradient step teaches the quality head "for this specific H_t configuration at position i, the token is correct/incorrect." This is exactly the kind of signal needed to learn the H_t → action mapping.

4. **PRISM is sample-efficient precisely because it conditions on H_t.** The paper notes: "since f_theta is already pretrained, its hidden states encode useful representations, likely accelerating g_theta's training." The hidden states are a *rich, pre-structured* input that makes supervised learning easy. The problem isn't H_t — it's trying to learn from H_t with only RL signal.

---

## Synthesis: PRISM + Jazbec et al. → Our Warm-Start Argument

### The core tension between the two papers

| Paper | H_t conditioning | Training method | Result |
|-------|-----------------|-----------------|--------|
| Jazbec et al. | Failed (worse than confidence-only) | Cold-start RL (GRPO) | H_t too hard to learn from |
| PRISM | Succeeded (state-of-the-art quality scores) | Supervised BCE loss | H_t is learnable with supervision |

These results are **not contradictory** — they demonstrate the same fundamental principle: **the training signal determines whether H_t is usable.** Sparse RL reward cannot navigate high-dimensional H_t space; dense supervised labels can.

### What this means for our system

Our warm-start phase is structurally analogous to PRISM fine-tuning:

| Aspect | PRISM | Our Warmstart |
|--------|-------|---------------|
| Labels | BCE(1[x^i = y^i], g(y)) — "is this token correct?" | BCE-like supervision on u_t (should unmask?) and r_t (should remask?) derived from future verifier outcomes |
| Input | H_t from MDM backbone | H_t from dual-model backbone (or scalar features in V4) |
| Signal density | Per-position, per-sample | Per-position, per-trajectory-step |
| Architecture | Lightweight head on shared backbone | Lightweight policy (1-layer transformer) on features extracted from backbone |

The supervised warmstart phase provides PRISM-like dense labels that teach the policy head to extract meaningful features from its input. When the input is H_t, this supervised phase should succeed for the same reasons PRISM succeeds: the labels directly teach the mapping from hidden state to action.

### Strengthened argument for H_t conditioning in V5

The PRISM evidence strengthens our Q2 argument considerably:

1. **PRISM proves H_t is learnable via supervision at the 8B scale.** Not just on toy tasks — on LLaDA-8B with real code generation. The same backbone architecture, the same hidden state dimensionality.

2. **PRISM's quality score is conceptually close to our u_t/r_t heads.** PRISM's g*(y) answers "how likely is this token correct given context?" Our u_t answers "should we commit this drafted token?" and r_t answers "should we remask this verified token?" These are all per-position binary decisions conditioned on local hidden state + global context.

3. **The failure mode in Jazbec et al. is specifically the cold-start RL → H_t combination.** PRISM eliminates the cold-start problem and H_t works. Our warmstart eliminates the cold-start problem. Therefore, we should expect H_t to work in our setting too.

4. **PRISM's own limitation points to where RL adds value.** PRISM's authors note: "our notion of per-token quality score has limits; as it is based on the per-position posterior marginals, it cannot fully capture global errors, e.g., reasoning." This is precisely where RL excels — the end-of-episode reward captures global correctness. The ideal pipeline is: supervised warmstart learns the local H_t → action mapping (like PRISM), then RL refines for global objectives (like Jazbec et al.'s correctness × speed). Our two-phase approach combines the strengths of both.

### Revised experimental roadmap

Given the PRISM evidence, the argument for V5 (H_t features) is stronger than before:

1. **V4 (current):** Warmstart scalar_only → GRPO scalar_only. Establishes baseline.
2. **V5 (high priority):** Warmstart with H_t → GRPO with H_t. PRISM strongly predicts this should work.
3. **V5-hybrid (if V5 RL is unstable):** Warmstart with H_t → GRPO with scalar_only. Tests whether the supervised H_t features provide a better initialization even if RL can't use H_t directly. The warm-started policy weights may encode H_t knowledge that transfers even when RL operates on scalar projections.

### An additional insight: PRISM-like regularization for our RL phase

PRISM uses lambda * MDM_loss as a regularizer during fine-tuning to prevent the backbone from forgetting. An analogous concern exists in our RL phase: GRPO updates may push the policy head away from the useful features learned during warmstart. Consider:

- **Stability regularizer:** Add a KL or L2 penalty between the current policy and the warm-started policy during GRPO: `L_total = L_GRPO + lambda * ||theta - theta_warmstart||^2`. This prevents catastrophic forgetting of the supervised features while allowing RL refinement.
- This is different from GRPO's own KL regularization (which Jazbec et al. set to 0 for cold-start). In our case, the warm-started policy is a meaningful reference point, not a random initialization.

---

## Updated Summary of Recommendations

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| Expert steering in post-warmstart GRPO? | **No** (disable) | Warmstart provides the structural prior; expert steering is redundant and destabilizing |
| H_t conditioning with warmstart? | **Strong yes** for V5 | PRISM demonstrates supervised H_t learning works at 8B scale; Jazbec et al.'s failure is cold-start-specific |
| Immediate V4 plan | Scalar-only warmstart → scalar-only GRPO (no expert steering) | Conservative baseline; evaluate before adding complexity |
| V5 plan | H_t warmstart → H_t GRPO, possibly with stability regularizer | PRISM evidence makes this high-priority; expected to outperform V4 |
| PRISM-like regularization in GRPO? | **Worth considering** | Prevents catastrophic forgetting of supervised features during RL |

---

## Regularization in Post-Warmstart RL: A Nuanced Problem

### Why Jazbec et al. disabled KL (beta=0)

In cold-start RL, there is no meaningful reference policy. The initial policy is random — regularizing toward it would actively harm training by pulling the policy back toward uniform actions. Setting beta=0 is correct in their setting.

### Why our setting is fundamentally different

We have a supervised reference that is *already good*. The warm-started policy encodes useful structure learned from dense per-position labels. This creates a meaningful anchor:
- The warm-started policy represents "locally correct" behavior (unmask what the verifier will accept, remask what it will reject)
- RL's job is to refine this for *global* objectives (correctness × speed) that supervision alone cannot capture
- But RL can overshoot — pushing the policy far from the supervised optimum in search of reward, potentially into unstable regions

### The regularization menu (ordered by priority)

1. **Clipping alone (V4 default, try first).** The clipped surrogate already prevents large single-step policy shifts. With clip_eps=0.5 (our current setting from paper_smoke.yaml), the ratio rho is bounded to [0.5, 1.5]. This may be sufficient if warmstart is good — the policy starts near a good solution and RL makes small refinements.

2. **KL penalty toward warmstart reference (add if instability observed).** Standard RLHF-style: `L = L_GRPO - beta_kl * KL(pi_warmstart || pi_current)`. This explicitly penalizes divergence from the supervised solution. The warm-started checkpoint serves as the SFT model analog in the RLHF pipeline. Start with a small beta_kl (e.g., 0.01) — we want to allow refinement, not freeze the policy.

3. **Functional regularizer / lambda * MDM_loss (if accuracy degrades in RL).** PRISM-style: add the supervised warmstart loss as an auxiliary objective during GRPO training. This is stronger than KL — it directly maintains the supervised behavior, not just proximity in parameter space. Useful if we observe that RL improves speed but degrades correctness (the policy "forgets" which tokens are safe to unmask).

### The key insight: cold-start has no reference; warm-start does

The cold-start paper's beta=0 is not a generalizable recommendation — it's a consequence of their setup. In any two-phase pipeline (supervised → RL), the supervised checkpoint is a natural reference:
- InstructGPT: SFT model → RLHF with KL penalty toward SFT
- Our system: Warmstart → GRPO with KL penalty toward warmstart
- The principle: RL should *refine* supervised behavior, not *replace* it

### Decision for V4

**Start without explicit KL regularization** (rely on clipping). Evaluate warmstart quality first — if the warm-started policy already outperforms heuristic baselines on accuracy, the RL phase has a good foundation. Then:
- If GRPO training is stable and accuracy improves: no regularization needed
- If GRPO shows accuracy degradation while speed improves: add KL penalty (beta_kl ~ 0.01-0.1)
- If GRPO completely destabilizes: add functional regularizer (lambda * warmstart_loss as auxiliary)

### Pre-condition: Warmstart must be sufficiently good

All of the above assumes the warm-started policy is better than heuristic baselines. If warmstart converges but the policy is *worse* than DefaultPolicy (confidence threshold), then RL has a bad foundation and no amount of regularization helps. **We must eval the warmstart checkpoint against baselines before proceeding to GRPO.** This is the next step after the current training run completes.

---

## References

- Jazbec et al. (2025). "Learning Unmasking Policies for Diffusion Language Models." arXiv:2512.09106v3.
- Kim, Kim, Lee et al. (2025). "Fine-Tuning Masked Diffusion for Provable Self-Correction (PRISM)." arXiv:2510.01384v3.
- PRISM quality adapter (supervised hidden-state quality predictor in our codebase)
- V3.5 GRPO with expert_steering: accuracy ~0.10, high variability (our experiments)
- V4 warmstart: supervised u_t/r_t training with verifier-derived labels (current run)
