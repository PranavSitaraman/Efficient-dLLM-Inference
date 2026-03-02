# AOAE — Any-Order Adaptive Editing for Masked Diffusion LLMs

Reference implementation for the paper *"Any-Order Adaptive Editing (AOAE): Decoupling Denoising and Steering in Masked Diffusion LLMs"*.

AOAE pairs a **frozen** masked diffusion LLM (LLaDA) with a **lightweight auxiliary policy** (<0.01% of base parameters) that jointly decides *where* to unmask, *where* to remask, and *which* positions to commit to a dKV-Cache — trained end-to-end via GRPO. The policy uses **composed prediction** to bias token selection toward cache-aligned orderings, and supports evaluation on a **soft-routed MoE** variant of LLaDA2.1-mini for controlled throughput measurement.

## Repository Structure

```
Efficient-dLLM-Inference/
├── configs/
│   ├── default.yaml            # Default config (LLaDA-8B, HF backend)
│   ├── default_large.yaml      # LLaDA2.1-flash (100B MoE, dInfer, 4× GPU TP)
│   ├── llada21_flash.yaml      # LLaDA2.1-flash (alternate)
│   ├── llada21_mini.yaml       # LLaDA2.1-mini (16B, 2× GPU)
│   ├── llada21_mini_soft.yaml  # LLaDA2.1-mini soft-routed MoE (all 16B active)
│   └── llada21_mini_hard.yaml  # LLaDA2.1-mini hard-routed MoE (baseline)
├── aoae/
│   ├── __init__.py
│   ├── models/
│   │   ├── base_model.py       # Multi-backend LLaDA wrapper (HF/dKV/dInfer/soft_moe)
│   │   ├── soft_mask.py        # Soft-masked state construction (Eq. 5-6)
│   │   ├── policy.py           # 1-layer transformer, 3 Bernoulli heads (unmask/remask/cache)
│   │   ├── prism.py            # PRISM quality adapter (§2.4)
│   │   ├── soft_moe.py         # Soft-routed MoE wrapper (§3.7)
│   │   └── composed_prediction.py  # Cache-aligned token selection (§3.6)
│   ├── cache.py                # Policy-controlled dKV-Cache tracking
│   ├── dinfer_integration.py   # dInfer integration with policy-guided caching
│   ├── inference.py            # AOAE inference loop (Algorithm 1) + baselines
│   ├── train_grpo.py           # GRPO training with multiplicative reward (§3.5)
│   ├── train_prism.py          # PRISM adapter training
│   ├── evaluate.py             # GSM8K evaluation + Pareto curve generation
│   └── plot_pareto.py          # Accuracy-vs-throughput plot generation
├── external/                   # Cloned by setup.sh (gitignored)
│   ├── dInfer/                 # inclusionAI/dInfer — SGLang backend for LLaDA2.X
│   └── dKV-Cache/              # horseee/dKV-Cache — delayed KV caching
├── slurm/                      # SLURM job scripts for Kempner H100 cluster
│   ├── train_prism.sh
│   ├── train_grpo.sh
│   ├── train_mini.sh           # LLaDA2.1-mini (2× GPU)
│   └── eval.sh
├── tests/
│   └── test_components.py      # Unit tests (runs with mock model, no GPU)
├── results/
│   └── expected_results.json   # Target numbers from paper
├── run_train.py                # Training entry (supports torchrun for multi-GPU)
├── run_eval.py                 # Evaluation entry
├── setup.sh                    # Environment setup (--minimal for core only)
├── reproduce.sh                # End-to-end reproduction (local or --slurm)
└── requirements.txt
```

## Components Implemented

| Paper Section | Component | File |
|---|---|---|
| §3.1 | Soft-masked state (Eq. 5–6) | `aoae/models/soft_mask.py` |
| §3.2 | Unified action space (unmask/remask/cache, validity constraints) | `aoae/models/policy.py` |
| §3.3 | AOAE inference loop (Algorithm 1) with composed prediction | `aoae/inference.py` |
| §3.4 | Lightweight policy architecture (PRISM quality scores as input) | `aoae/models/policy.py` |
| §3.5 | GRPO with multiplicative reward (Eq. 12–13) | `aoae/train_grpo.py` |
| §3.6 | Composed prediction for cache-aligned token selection | `aoae/models/composed_prediction.py` |
| §3.7 | Soft-routed MoE wrapper for controlled evaluation | `aoae/models/soft_moe.py` |
| §2.3 | Policy-controlled dKV-Cache tracking | `aoae/cache.py` |
| §2.4 | PRISM quality adapter | `aoae/models/prism.py` |
| — | dInfer integration with policy-guided caching | `aoae/dinfer_integration.py` |
| §3.8 | Baselines (Uniform, S-Mode, Q-Mode, Fast-dLLM) | `aoae/inference.py` |
| §3.8 | Ablations (remask-only, cache-only, no composed prediction, etc.) | configurable via YAML |

## Quick Start (< 30 minutes on GSM8K)

### 1. Install

```bash
bash setup.sh            # Full install (core + dInfer + SGLang)
# Or: bash setup.sh --minimal  # Core only (for LLaDA-8B)
```

### 2. Quick smoke test (baselines only, no training)

```bash
python3 run_eval.py --config configs/default.yaml --max_samples 10
```

### 3. Train PRISM adapter (~10 min on 1 GPU)

```bash
python3 run_train.py --config configs/default.yaml --stage prism
```

### 4. Train AOAE policy via GRPO

```bash
# Single GPU
python3 run_train.py --config configs/default.yaml --stage grpo

# Multi-GPU via torchrun
torchrun --nproc_per_node 4 run_train.py --config configs/default.yaml --stage grpo
```

### 5. Evaluate with Pareto sweep

```bash
python3 run_eval.py \
    --config configs/default.yaml \
    --checkpoint outputs/default/policy_final.pt
```

### 6. Full reproduction

```bash
# Local (single GPU)
bash reproduce.sh

# SLURM cluster (auto-chains PRISM → GRPO → Eval)
bash reproduce.sh --slurm

# With LLaDA2.1-mini on SLURM
bash reproduce.sh --slurm --config configs/llada21_mini.yaml
```

## Configuration

All hyperparameters live in YAML configs. Key settings from `configs/default.yaml`:

| Parameter | Default | Description |
|---|---|---|
| `base_model.name_or_path` | `GSAI-ML/LLaDA-8B-Instruct` | HuggingFace model ID |
| `base_model.backend` | `auto` | `auto` / `hf` / `dkv` / `dinfer` / `soft_moe` |
| `policy.d_model` | 128 | Policy hidden dim |
| `policy.n_layers` | 1 | Policy transformer layers |
| `soft_mask.top_k` | 5 | Top-K for soft masking |
| `prism.threshold` | 0.5 | PRISM remask threshold δ |
| `grpo.group_size` | 8 | GRPO group size G |
| `grpo.alpha` | 1.0 | Speed penalty exponent |
| `grpo.beta` | 0.1 | Cache-thrashing penalty |
| `grpo.clip_eps` | 0.2 | PPO/GRPO clipping ε |
| `inference.steps` | 64 | Diffusion steps T |
| `inference.compose_gamma` | 0.5 | Composed prediction strength (§3.6) |
| `inference.disable_remask` | `false` | Ablation switch to disable remask actions while keeping other settings fixed |
| `inference.reuse_signal.method` | `argmax_match` | Training-free safe-to-reuse gate (`argmax_match`, `topk_overlap`, `min_confidence`, `min_margin`, `js_divergence`, `temporal_confidence`) |
| `inference.positional_cache.enabled` | `false` | Enable next-H positional speculative caching (`q_t` access head) |
| `inference.positional_cache.horizon` | `4` | Next-H window for access prediction metrics |
| `inference.positional_cache.refresh_budget` | `32` | Top-B non-mandatory refresh positions per step |
| `policy.use_positional_features` | `false` | Adds age + last-access features to policy state (enable for POC2) |
| `grpo.access_reward_weight` | `0.0` | Optional dense reward coefficient for next-H speculative access F1 |
| `analysis.track_kv_dynamics` | `false` | Enable KV-dynamics proxy logging during speculative eval |
| `analysis.attention_proxy_top_frac` | `0.1` | Fraction of highest-confidence positions used for confident-token drift proxy |
| `base_model.routing_temperature` | 0.01 | Soft routing τ_r (soft_moe backend only) |

## Experiment Tracking Outputs

- `run_eval.py` now writes:
  - `outputs/<run>/eval_results.json`
  - `outputs/<run>/eval_metadata.json`
  - `outputs/<run>/eval_tps_vs_accuracy.png`
  - `outputs/<run>/kv_dynamics_records.json` (if `analysis.track_kv_dynamics=true`)
  - `outputs/<run>/kv_dynamics_summary.json` (if enabled)
  - `outputs/<run>/kv_dynamics_layer_drift.png` (if enabled and matplotlib available)
  - `results/experiment_manifest.jsonl` (append-only registry across runs)
- Build a consolidated table from saved artifacts:
  - `python3 scripts/build_comparison_table.py`
  - Outputs: `results/comparison_table.csv` and `results/comparison_table.md`
- Aggregate KV-dynamics summaries across runs:
  - `python3 scripts/summarize_kv_dynamics.py`
  - Outputs: `results/kv_dynamics_table.csv` and `results/kv_dynamics_table.md`

POC2-friendly eval override example (no retraining):

```bash
python3 run_eval.py \
  --config configs/dual_mini_tau01.yaml \
  --mode speculative \
  --reuse_signal_method js_divergence \
  --reuse_signal_threshold 0.05 \
  --disable_remask \
  --track_kv_dynamics \
  --enable_positional_cache \
  --positional_cache_horizon 4 \
  --positional_cache_refresh_budget 32
```

## Architecture & Design

1. **Multi-backend base model**: Automatically selects HuggingFace (LLaDA-8B), dKV-Cache patched model (real sparse attention), dInfer/SGLang (LLaDA2.X MoE), or soft-routed MoE (all experts active) based on model name. Set `base_model.backend` to override.

2. **Unmask + Remask (no T2T)**: The policy uses two primitives — unmasking and remasking — preserving the any-order property of masked diffusion models. Token-to-Token editing is replaced by pure remasking, which defers correction to the base model's well-trained denoising capability.

3. **Composed prediction**: The policy's cache stability signal sharpens the base model's token distribution at confident positions, increasing KV-cache hit rates without sacrificing quality at uncertain positions (§3.6).

4. **Soft-routed MoE**: For controlled evaluation, the hard top-k expert routing in LLaDA2.1-mini is replaced with temperature-controlled soft routing, activating all 16B parameters per forward pass. This isolates AOAE's throughput gains from MoE sparsity (§3.7).

5. **Policy-controlled dKV-Cache**: `DKVCacheManager` provides lightweight position tracking for training rollouts (thrash counting for reward, cache invalidation on remask). `PolicyGuidedCacheManager` wraps this for dInfer-integrated evaluation with detailed cache statistics.

6. **Multi-GPU training**: Supports `torchrun` with PyTorch DDP. The base model is frozen; only the tiny policy (~500K params) and soft-mask gating params are trained. Gradient sync overhead is negligible.

## Supported Base Models

| Model | Config | Notes |
|-------|--------|-------|
| `GSAI-ML/LLaDA-8B-Instruct` | `configs/default.yaml` | Single GPU, recommended for dev |
| `inclusionAI/LLaDA2.1-mini` | `configs/llada21_mini.yaml` | 16B, 2× GPU |
| `inclusionAI/LLaDA2.1-mini` (soft-routed) | `configs/llada21_mini_soft.yaml` | 16B all active, 2× GPU |
| `inclusionAI/LLaDA2.1-mini` (hard-routed) | `configs/llada21_mini_hard.yaml` | 16B (~1.4B active), 2× GPU |
| `inclusionAI/LLaDA2.1-flash` | `configs/llada21_flash.yaml` | 100B MoE, 4× GPU + dInfer |
| `inclusionAI/LLaDA2.0-flash` | — | 100B MoE, 4× GPU + dInfer |
| `inclusionAI/LLaDA2.0-flash-CAP` | — | 100B MoE, accelerated decoding |

## External Dependencies

| Repo | Purpose | License |
|------|---------|---------|
| [inclusionAI/dInfer](https://github.com/inclusionAI/dInfer) | SGLang inference engine for LLaDA2.X | Apache-2.0 |
| [horseee/dKV-Cache](https://github.com/horseee/dKV-Cache) | Delayed KV caching for dLLMs | MIT |

## Citation

```bibtex
@article{aoae2026,
  title={Any-Order Adaptive Editing: Decoupling Denoising and Steering in Masked Diffusion LLMs},
  year={2026}
}
```
