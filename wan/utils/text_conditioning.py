# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
from typing import List, Optional, Tuple

import torch


def print_cuda_mem_summary(device: torch.device) -> None:
    """Prints CUDA memory summary if available, guarded for safety."""
    if torch.cuda.is_available():
        try:
            logging.info(
                "CUDA memory summary:\n" + torch.cuda.memory_summary(device=device)
            )
        except Exception:
            # Avoid crashing on some driver/toolkit combos
            pass


def prepare_text_conditioning(
    pipe,
    prompts: List[str],
    negative_prompts: Optional[List[str]] = None,
    precision: str = "fp16",
    device: Optional[torch.device] = None,
    offload_strategy: str = "set_none",
    print_memory: bool = True,
) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
    """
    Compute text embeddings once and safely offload the T5 encoder from GPU.

    - Enforces eval + torch.inference_mode() to avoid grads.
    - Casts to fp16/bf16 as requested.
    - Moves embeddings to the UNet/video model device.
    - Immediately offloads or removes `pipe.text_encoder` and clears caches.

    Args:
        pipe: Diffusers-style pipeline object. Must have `text_encoder` and `tokenizer` attrs.
        prompts: List of prompts for conditioning.
        negative_prompts: List of negative prompts for CFG, optional.
        precision: "fp16" or "bf16".
        device: Target device for embeddings. Defaults to the DiT device if None.
        offload_strategy: One of {"cpu", "set_none"}. "set_none" breaks references.
        print_memory: Print torch.cuda.memory_summary() before/after offload.

    Returns:
        (prompt_embeds, negative_prompt_embeds)
    """
    assert hasattr(pipe, "text_encoder"), "pipe must expose .text_encoder"

    # Resolve target device to match the UNet/DiT device
    if device is None:
        # Heuristic: Wan pipelines expose .device on the class; fallback to current CUDA device
        device = getattr(
            pipe, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    # No global grad graph for encoder
    target_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    text_encoder = getattr(pipe, "text_encoder", None)
    if text_encoder is None:
        raise RuntimeError(
            "pipe.text_encoder is None; cannot compute embeddings. Provide them externally."
        )

    if print_memory:
        print_cuda_mem_summary(device)

    with torch.inference_mode():
        # Ensure the encoder is in eval() and placed appropriately for compute
        if hasattr(text_encoder, "model"):
            text_encoder.model.eval()
        # Place model on compute device for speed
        if hasattr(text_encoder, "model") and device.type == "cuda":
            text_encoder.model.to(device)

        # The Wan T5 wrapper accepts a python list of strings
        prompt_embeds = text_encoder(prompts, device)
        negative_prompt_embeds = None
        if negative_prompts is not None:
            negative_prompt_embeds = text_encoder(negative_prompts, device)

        # Cast + move to target device and dtype
        prompt_embeds = [t.to(dtype=target_dtype, device=device) for t in prompt_embeds]
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = [
                t.to(dtype=target_dtype, device=device) for t in negative_prompt_embeds
            ]

    # Immediately offload encoder to free VRAM and break references
    try:
        if hasattr(text_encoder, "model"):
            text_encoder.model.to("cpu")
    except Exception:
        pass

    # Break references if requested to allow GC to reclaim memory
    if offload_strategy == "set_none":
        try:
            pipe.text_encoder = None
            # Keep tokenizer for later tokenization (it is CPU resident)
        except Exception:
            pass

    # Clear caches and print memory summaries
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if print_memory:
        print_cuda_mem_summary(device)

    return prompt_embeds, negative_prompt_embeds
