# 4/18 Baseline Fidelity Plan

## Goal
Make the `LLaDA2.1-mini` GSM8K baseline faithful to authoritative sources so repo accuracy reflects the model and inference setup rather than truncation or grading artifacts.

## Source Of Truth
- Official inference settings come from the `inclusionAI/LLaDA2.1-mini` Hugging Face model card.
- The baseline recipe used in this repo is:
  - `block_length=32`
  - `temperature=0.0`
  - Speed mode: `threshold=0.5`, `editing_threshold=0.0`
  - Quality mode: `threshold=0.7`, `editing_threshold=0.5`
  - `max_post_steps=16`
  - `gen_length=512`
  - `eos_early_stop=true`
- Faithful LLaDA GSM8K answer extraction comes from InclusionAI's public `dInfer` GSM8K task configs.
- The repo baseline defaults to `gsm8k_num_fewshot=0`; public `dInfer` few-shot prompting is available only as an explicit opt-in because the technical reports do not clearly disclose few-shot benchmarking.

## Problems Identified
- The prior `llada21_official` config shape allowed a single shared threshold pair, which could incorrectly force quality thresholds onto speed mode.
- The repo baseline was capped at `256` generated tokens, which caused visible truncation on GSM8K.
- The official baseline path did not support EOS early stop, so generations could continue to the cap even after a valid stopping point.
- GSM8K evaluation used a generic heuristic extractor that could mark correct answers wrong by taking the last number on an answer line.
- Eval artifacts did not surface whether a sample was truncated or whether it was scored with an official evaluator.

## Implemented Changes
- Added mode-specific official LLaDA 2.1 setting resolution in `aoae/inference.py`.
- Added `gen_length` and `eos_early_stop` support to the official decode path.
- Updated checked-in `llada21_hard` and `llada21_soft` configs to use the official model-card baseline recipe.
- Added the LLaDA-style GSM8K prompt format and public `dInfer` extraction rule in `aoae/tasks.py`.
- Kept the repo baseline default at 0-shot and made few-shot prompting an explicit config opt-in via `data.gsm8k_num_fewshot`.
- Routed GSM8K evaluation through the LLaDA-aligned scorer in `aoae/evaluators.py`.
- Kept the old scalar extractor only as a heuristic fallback for datasets without an official evaluator and documented it as such.
- Added truncation diagnostics and scorer metadata to eval results and saved prediction artifacts.

## Guidance
- For GSM8K, do not use the generic `extract_answer()` heuristic to judge correctness.
- Use the model-family's public benchmark recipe whenever it defines one.
- Treat `math_heuristic_fallback` as a convenience path for unsupported datasets, not as a benchmark-quality evaluator.
