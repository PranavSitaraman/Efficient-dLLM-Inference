"""
Profiling script for the speculative_inference loop.

Runs N warmup + M timed samples through the speed_max operating point and
logs a cProfile report + per-phase wall-time breakdown to profile_results.txt.

Usage:
    python profile_loop.py [--n_warmup 2] [--n_profile 5] [--point speed_max]

For tp_size > 1 configs the script auto-relaunches under torchrun.
Force single-GPU with --tp_size 1.
"""

import argparse
import cProfile
import io
import os
import pstats
import shutil
import subprocess
import sys
import time
import yaml
import torch

from aoae.models.dual_model import DualModelWrapper
from aoae.models.policy import AOAEPolicy
from aoae.models.soft_mask import SoftMaskedState
from aoae.evaluate import _load_eval_dataset
from aoae.tasks import build_prompt
from aoae.speculative_inference import speculative_inference
from aoae.checkpoints import load_state_dict_flexible
from aoae.experiment_utils import set_nested
from aoae.runtime_checks import set_global_seed

# ---------------------------------------------------------------------------
# Operating point overrides (mirrors paper.yaml sweep points)
# ---------------------------------------------------------------------------
POINTS = {
    "speed_max": {
        "base_model.lossless_verification": True,
        "inference.speculative_schedule": "aoae",
        "inference.verifier_schedule.mode": "candidate_budget",
        "inference.verifier_schedule.draft_token_budget": 16,
        "inference.verifier_schedule.max_draft_microsteps": 4,
        "inference.primary_agree_threshold": 0.75,
        "inference.max_unmask_fraction_per_step": 0.25,
        "inference.disable_remask": True,
        "cache.prefix_kv_cache": True,
    },
    "quality_max": {
        "base_model.lossless_verification": False,
        "inference.speculative_schedule": "aoae",
        "inference.verifier_schedule.mode": "candidate_budget",
        "inference.verifier_schedule.draft_token_budget": 4,
        "inference.verifier_schedule.max_draft_microsteps": 1,
        "inference.primary_agree_threshold": 0.98,
        "inference.max_unmask_fraction_per_step": 0.0625,
        "inference.disable_remask": False,
        "cache.prefix_kv_cache": True,
    },
}


def _apply_runtime_env_defaults(env=None):
    target = os.environ if env is None else env
    target.setdefault("HF_HUB_DISABLE_XET", "1")
    target.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    target.setdefault("MASTER_ADDR", "127.0.0.1")
    target.setdefault("MASTER_PORT", "29500")
    target.setdefault("NCCL_SOCKET_FAMILY", "AF_INET")
    target.setdefault("GLOO_SOCKET_FAMILY", "AF_INET")
    if target.get("MASTER_ADDR") == "localhost":
        target["MASTER_ADDR"] = "127.0.0.1"
    return target


def _maybe_relaunch_with_torchrun(tp_size: int) -> None:
    """Re-exec this script under torchrun when tp_size > 1.

    Mirrors cli._maybe_relaunch_with_torchrun but targets the script file
    directly rather than aoae.cli.
    """
    if os.environ.get("AOAE_DISABLE_AUTO_TORCHRUN") == "1":
        return
    if any(key in os.environ for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        return
    if tp_size <= 1:
        return

    env = _apply_runtime_env_defaults(dict(os.environ))
    env["AOAE_DISABLE_AUTO_TORCHRUN"] = "1"

    script = os.path.abspath(__file__)
    # Prefer the torchrun that lives alongside the current Python executable so
    # we stay in the same conda env (shutil.which may find a different env's binary).
    py_bin_dir = os.path.dirname(sys.executable)
    env_torchrun = os.path.join(py_bin_dir, "torchrun")
    torchrun_bin = env_torchrun if os.path.isfile(env_torchrun) else shutil.which("torchrun")
    launcher_args = [
        "--nproc_per_node", str(tp_size),
        "--nnodes", "1",
        "--node_rank", "0",
        "--master_addr", env["MASTER_ADDR"],
        "--master_port", env["MASTER_PORT"],
    ]
    if torchrun_bin:
        cmd = [torchrun_bin, *launcher_args, script, *sys.argv[1:]]
    else:
        cmd = [sys.executable, "-m", "torch.distributed.run", *launcher_args, script, *sys.argv[1:]]

    print(f"[profile_loop] Relaunching under torchrun with tp_size={tp_size}")
    sys.exit(subprocess.call(cmd, env=env))


def _setup_distributed():
    _apply_runtime_env_defaults()
    if "RANK" not in os.environ:
        return None
    import torch.distributed as dist
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return {"rank": rank, "local_rank": local_rank, "world_size": world_size}


def _cleanup_distributed(dist_info) -> None:
    if dist_info is None:
        return
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def _apply_overrides(cfg, overrides):
    for k, v in overrides.items():
        set_nested(cfg, k, v)


def _build_prompt_ids(tokenizer, question, cfg, device):
    prompt_text, add_special_tokens = build_prompt(tokenizer, question, cfg)
    ids = tokenizer.encode(
        prompt_text,
        add_special_tokens=add_special_tokens,
        max_length=cfg["data"]["max_prompt_len"],
        truncation=True,
        return_tensors="pt",
    ).to(device)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--checkpoint", default="outputs/paper/policy_best.pt")
    parser.add_argument("--point", default="speed_max", choices=list(POINTS))
    parser.add_argument("--n_warmup", type=int, default=2)
    parser.add_argument("--n_profile", type=int, default=5)
    parser.add_argument("--out", default="profile_results.txt")
    parser.add_argument(
        "--tp_size", type=int, default=None,
        help="Override hardware.tp_size from config (use 1 to force single-GPU)."
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    _apply_overrides(cfg, POINTS[args.point])

    tp_size = args.tp_size if args.tp_size is not None else int(
        cfg.get("hardware", {}).get("tp_size", 1) or 1
    )
    if args.tp_size is not None:
        set_nested(cfg, "hardware.tp_size", tp_size)

    # Re-exec under torchrun before any CUDA/model init when tp_size > 1.
    _maybe_relaunch_with_torchrun(tp_size)

    dist_info = _setup_distributed()
    local_rank = dist_info["local_rank"] if dist_info else 0
    rank = dist_info["rank"] if dist_info else 0
    is_main = rank == 0

    try:
        set_global_seed(int(cfg.get("hardware", {}).get("seed", 42)))
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

        if is_main:
            print(f"Loading dual model (tp_size={tp_size})...")
        dual_model = DualModelWrapper(cfg).to(device)
        tokenizer = dual_model.tokenizer
        mask_id = cfg["base_model"]["mask_token_id"]

        embed_w = dual_model.get_embedding_weight()
        embed_dim = embed_w.shape[1]

        soft_mask = SoftMaskedState(cfg, embed_w).to(device)
        soft_mask.set_mask_embedding(mask_id)
        soft_mask.eval()

        policy = AOAEPolicy(cfg, input_dim=embed_dim).to(device)
        if is_main:
            print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        load_state_dict_flexible(policy, ckpt["policy"], "policy")
        policy.eval()

        if is_main:
            print("Loading eval dataset...")
        eval_ds = _load_eval_dataset(cfg["data"])
        questions = [s["question"] for s in eval_ds][: args.n_warmup + args.n_profile]

        prompt_ids_list = [
            _build_prompt_ids(tokenizer, q, cfg, device) for q in questions
        ]

        def run_one(prompt_ids):
            with torch.no_grad():
                speculative_inference(
                    dual_model=dual_model,
                    policy=policy,
                    soft_mask_module=soft_mask,
                    prism_adapter=None,
                    prompt_ids=prompt_ids,
                    cfg=cfg,
                    policy_temperature=1.0,
                )

        # --- Warmup ---
        if is_main:
            print(f"Warmup ({args.n_warmup} samples)...")
        for i in range(args.n_warmup):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            run_one(prompt_ids_list[i])
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # --- cProfile run (rank 0 only; other ranks still execute for TP) ---
        if is_main:
            print(f"Profiling ({args.n_profile} samples, point={args.point})...")
        profile_prompts = prompt_ids_list[args.n_warmup : args.n_warmup + args.n_profile]

        pr = cProfile.Profile() if is_main else None
        t_start = time.perf_counter()
        if pr:
            pr.enable()
        for pid in profile_prompts:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            run_one(pid)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        if pr:
            pr.disable()
        t_total = time.perf_counter() - t_start

        if is_main:
            print(f"\nTotal wall time for {args.n_profile} samples: {t_total:.2f}s  "
                  f"({t_total / args.n_profile:.2f}s/sample)")

            # --- Write report ---
            stream = io.StringIO()
            ps = pstats.Stats(pr, stream=stream)
            ps.sort_stats("cumulative")
            ps.print_stats(50)
            report = stream.getvalue()

            stream2 = io.StringIO()
            ps2 = pstats.Stats(pr, stream=stream2)
            ps2.sort_stats("tottime")
            ps2.print_stats(20)

            with open(args.out, "w") as f:
                f.write(f"=== Profile: point={args.point}, n_profile={args.n_profile} ===\n")
                f.write(f"Total wall time: {t_total:.2f}s  ({t_total/args.n_profile:.2f}s/sample)\n\n")
                f.write("--- Top 50 by cumulative time ---\n")
                f.write(report)
                f.write("\n--- Top 20 by self time (tottime) ---\n")
                f.write(stream2.getvalue())

            print(f"\nProfile written to {args.out}")
            print("\n--- Top 20 by self time (tottime) ---")
            ps2b = pstats.Stats(pr)
            ps2b.sort_stats("tottime")
            ps2b.print_stats(20)
    finally:
        _cleanup_distributed(dist_info)


if __name__ == "__main__":
    main()
