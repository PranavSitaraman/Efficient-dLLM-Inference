"""Verify per-head gradient flow via optimizer state.

Walks model parameters in canonical order, matches them to optimizer state
indices, then prints exp_avg / exp_avg_sq norms. exp_avg == 0 iff no gradient
ever flowed to that parameter.
"""
import torch
import yaml
from aoae.models.policy import AOAEPolicy

ckpt = torch.load("outputs/smoke_uvscope/policy_latest.pt", map_location="cpu", weights_only=False)
opt = ckpt["optimizer"]

with open("configs/paper_smoke.yaml") as f:
    cfg = yaml.safe_load(f)
input_dim = cfg["base_model"].get("hidden_size", 1024)
policy = AOAEPolicy(cfg, input_dim=input_dim)

# Match optimizer.state[i] to policy.parameters() in registration order.
# torch's optim builds state by iterating params in the order given to the
# optimizer ctor, which equals the order of `policy.parameters()`.
named_params = list(policy.named_parameters())  # ordered

state = opt["state"]
HEADS = ["head_unmask.weight", "head_unmask.bias",
         "head_remask.weight", "head_remask.bias",
         "head_cache.weight",  "head_cache.bias",
         "head_access.weight", "head_access.bias"]

print("Per-head optimizer state (10-step smoke):")
print("  exp_avg == 0 AND exp_avg_sq == 0  ⇒  no gradient flowed (FROZEN)")
print("")
print(f"  {'name':25s} {'||exp_avg||':>14s}  {'||exp_avg_sq||':>17s}  status")
print("  " + "-" * 65)

for idx, (name, _) in enumerate(named_params):
    if name not in HEADS:
        continue
    if idx not in state:
        print(f"  {name:25s} (no opt state at idx {idx})")
        continue
    st = state[idx]
    ea  = st["exp_avg"].float().norm().item()
    eas = st["exp_avg_sq"].float().norm().item()
    status = "TRAINED" if (ea > 0 or eas > 0) else "FROZEN"
    print(f"  {name:25s} {ea:>14.6e}  {eas:>17.6e}  [{status}]")
