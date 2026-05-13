"""AOAE — Any-Order Adaptive Editing for Masked Diffusion LLMs.

Speculative diffusion: dual-model architecture where a fast hard-routed
MoE auxiliary (~1.4B active) produces draft predictions for KV-cache
pre-warming, and a slow soft-routed MoE primary (all 16B active) verifies
and refines. See paper §3.3 and §3.7.
"""
