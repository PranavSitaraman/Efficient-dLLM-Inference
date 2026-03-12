"""Minimal diagnostic: verify LLaDA2.1-mini generation on GSM8K with block-causal mask."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import yaml
from datasets import load_dataset


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def reference_generate_with_mask(model, prompt, attention_mask, steps=128,
                                  gen_length=128, block_length=32,
                                  temperature=0., remasking='low_confidence',
                                  mask_id=156895):
    """Reference generate with explicit attention mask (block-causal)."""
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id,
                   dtype=torch.long, device=prompt.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            seq_len = x.shape[1]
            attn = attention_mask[:, :, :seq_len, :seq_len]
            logits = model.forward(x, attention_mask=attn)
            if hasattr(logits, 'logits'):
                logits = logits.logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            else:
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


def main():
    import torch.distributed as dist
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    with open("configs/poc1_tau_sweep.yaml") as f:
        cfg = yaml.safe_load(f)

    if rank == 0:
        print("=" * 70)
        print("DIAGNOSTIC: Verify generation with block-causal attention mask")
        print("=" * 70)

    from aoae.models.base_model import LLaDABaseModel
    if rank == 0:
        print("\nLoading model (dInfer backend)...")
    base_cfg = dict(cfg)
    base_cfg["base_model"] = dict(cfg["base_model"])
    base_cfg["base_model"]["backend"] = "dinfer"
    model_wrapper = LLaDABaseModel(base_cfg)
    tokenizer = model_wrapper.tokenizer
    mask_id = model_wrapper.mask_id
    device = model_wrapper.device

    if rank == 0:
        print(f"  mask_id = {mask_id}")
        print(f"  device = {device}")
        print(f"  backend = {model_wrapper._backend}")
        print(f"  block_length = {model_wrapper._block_length}")

    ds = load_dataset("openai/gsm8k", "main", split="test")

    from aoae.train_grpo import extract_answer

    test_cases = [
        (ds[0]["question"], ds[0]["answer"]),
        (ds[1]["question"], ds[1]["answer"]),
        (ds[2]["question"], ds[2]["answer"]),
    ]

    if rank == 0:
        print(f"\n--- Test 1: model.forward() with block-causal mask via wrapper ---")
    from aoae.inference import block_smode_decode
    for qi, (q, ref) in enumerate(test_cases):
        messages = [{"role": "user", "content": q}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer.encode(
            prompt_text, add_special_tokens=False, return_tensors="pt",
        ).to(device)

        output = block_smode_decode(
            model_wrapper, prompt_ids, cfg,
            tau_mask=0.5, tau_edit=0.0, max_steps_per_block=32,
        )
        gen_tokens = output[0, prompt_ids.shape[1]:]
        n_mask = (gen_tokens == mask_id).sum().item()
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        if rank == 0:
            print(f"\n  Q{qi}: {q[:120]}...")
            print(f"  Ref: {ref[:100]}...")
            print(f"  Prompt tokens: {prompt_ids.shape[1]}, Gen length: {len(gen_tokens)}, "
                  f"Masks remaining: {n_mask}")
            print(f"  Generated:\n    {repr(gen_text[:500])}")
            gen_ans = extract_answer(gen_text)
            ref_ans = extract_answer(ref)
            print(f"  Extracted: {gen_ans}  Reference: {ref_ans}")
            try:
                match = abs(float(gen_ans) - float(ref_ans)) < 1e-3 if gen_ans and ref_ans else False
            except:
                match = False
            print(f"  Match: {match}")

    if rank == 0:
        print(f"\n--- Test 2: reference generate with explicit block-causal mask ---")
    for qi, (q, ref) in enumerate(test_cases[:1]):
        messages = [{"role": "user", "content": q}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer.encode(
            prompt_text, add_special_tokens=False, return_tensors="pt",
        ).to(device)

        gen_length = 256
        block_length = 32
        total_len = prompt_ids.shape[1] + gen_length
        num_blocks = (total_len + block_length - 1) // block_length
        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
        attn_mask = (
            block_mask.repeat_interleave(block_length, dim=0)
            .repeat_interleave(block_length, dim=1)
            .unsqueeze(0)
            .unsqueeze(0)
        )[:, :, :total_len, :total_len]

        from vllm.config import get_current_vllm_config
        from vllm.forward_context import set_forward_context
        vllm_config = get_current_vllm_config()

        attn_mask_bool = attn_mask.bool()

        x = torch.full((1, total_len), mask_id, dtype=torch.long, device=device)
        x[:, :prompt_ids.shape[1]] = prompt_ids.clone()

        steps = 128
        num_gen_blocks = gen_length // block_length
        steps_per_block = steps // num_gen_blocks

        for num_block in range(num_gen_blocks):
            block_start = prompt_ids.shape[1] + num_block * block_length
            block_end = prompt_ids.shape[1] + (num_block + 1) * block_length
            block_mask_idx = (x[:, block_start:block_end] == mask_id)
            ntt = get_num_transfer_tokens(block_mask_idx, steps_per_block)

            for si in range(steps_per_block):
                mi = (x == mask_id)
                with set_forward_context(None, vllm_config):
                    outputs = model_wrapper.model(x, attention_mask=attn_mask_bool)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                x0 = torch.argmax(logits, dim=-1)
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                x0_p[:, prompt_ids.shape[1] + (num_block + 1) * block_length:] = -np.inf
                x0 = torch.where(mi, x0, x)
                confidence = torch.where(mi, x0_p, -np.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    _, sel = torch.topk(confidence[j], k=ntt[j, si])
                    transfer_index[j, sel] = True
                x[transfer_index] = x0[transfer_index]

        gen_tokens = x[0, prompt_ids.shape[1]:]
        n_mask = (gen_tokens == mask_id).sum().item()
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        if rank == 0:
            print(f"\n  Q0 (128 steps, direct model call with block-causal mask):")
            print(f"  Masks remaining: {n_mask}/{len(gen_tokens)}")
            print(f"  Generated:\n    {repr(gen_text[:500])}")
            gen_ans = extract_answer(gen_text)
            ref_ans = extract_answer(ref)
            print(f"  Extracted: {gen_ans}  Reference: {ref_ans}")

    if rank == 0:
        print("\n" + "=" * 70)
        print("DIAGNOSTIC COMPLETE")
        print("=" * 70)


if __name__ == "__main__":
    main()
