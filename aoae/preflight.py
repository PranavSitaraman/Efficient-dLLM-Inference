"""
AOAE preflight checks for environment + runtime capability + config sanity.

Usage:
  python -m aoae.preflight --config configs/paper.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml

from .runtime_checks import collect_runtime_info, ensure_vllm_moe_runtime


def _deep_merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in (override or {}).items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge_config(out[key], value)
        else:
            out[key] = value
    return out


def _load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "extends" not in cfg:
        return cfg
    parent_path = Path(cfg["extends"])
    if not parent_path.is_absolute():
        parent_path = cfg_path.parent / parent_path
    child = dict(cfg)
    child.pop("extends", None)
    return _deep_merge_config(_load_config(str(parent_path)), child)


def run_preflight(config_path: str, strict_moe: bool = False) -> Dict[str, Any]:
    cfg = _load_config(config_path)
    out_dir = cfg.get("logging", {}).get("output_dir", "")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    backend = cfg.get("base_model", {}).get("backend", "auto")
    allow_python_fallback = bool(cfg.get("base_model", {}).get("allow_python_fallback_ops", True))
    moe_report = None
    if backend in {"dinfer", "soft_moe", "dual", "auto"}:
        try:
            moe_report = ensure_vllm_moe_runtime(
                strict=strict_moe,
                verbose=True,
                allow_python_fallback=allow_python_fallback,
            )
        except Exception as exc:
            moe_report = {"error": str(exc)}
            if strict_moe:
                raise

    report = {
        "runtime": collect_runtime_info(),
        "config_path": config_path,
        "output_dir": out_dir,
        "backend": backend,
        "moe_report": moe_report,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="AOAE preflight check")
    parser.add_argument("--config", default="configs/paper.yaml", help="YAML config path.")
    parser.add_argument(
        "--strict_moe",
        action="store_true",
        help="Fail if required vLLM MoE ops are missing and cannot be patched.",
    )
    args = parser.parse_args()

    report = run_preflight(args.config, strict_moe=args.strict_moe)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
