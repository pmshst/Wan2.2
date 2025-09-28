# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import contextlib
import gc
import logging
from typing import Iterator, List, Tuple

import torch


def _bytes(n: int) -> str:
    """Format bytes into a compact human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.0f} PB"


def _collect_cuda_tensors() -> List[Tuple[int, str, str, str]]:
    """Return a list of CUDA tensors with their sizes.

    Each item: (num_bytes, shape_str, dtype_str, device_str)
    """
    tensors: List[Tuple[int, str, str, str]] = []
    try:
        for obj in gc.get_objects():
            try:
                if isinstance(obj, torch.Tensor) and obj.is_cuda:
                    nbytes = obj.element_size() * obj.nelement()
                    tensors.append(
                        (nbytes, str(tuple(obj.shape)), str(obj.dtype), str(obj.device))
                    )
            except Exception:
                # Be defensive: ignore objects we cannot inspect
                continue
    except Exception:
        # Fallback if gc.get_objects() is unavailable or restricted
        pass
    # Sort largest first
    tensors.sort(key=lambda x: x[0], reverse=True)
    return tensors


@contextlib.contextmanager
def track_cuda_memory(
    tag: str, show_processes: bool = False, top_k: int = 10
) -> Iterator[None]:
    """
    Context manager that reports CUDA memory before and after the block.

    - Prints allocated and reserved bytes.
    - Prints the top-K CUDA tensors by size (best-effort via gc).
    - Prints torch.cuda.memory_summary() for detailed allocator info.
    - Optionally prints torch.cuda.list_gpu_processes() if available.
    """
    if not torch.cuda.is_available():
        logging.info(
            f"[track_cuda_memory:{tag}] CUDA not available; skipping diagnostics."
        )
        yield
        return

    device = torch.device("cuda")
    torch.cuda.synchronize()
    alloc_before = torch.cuda.memory_allocated(device)
    reserv_before = torch.cuda.memory_reserved(device)

    logging.info(
        f"[track_cuda_memory:{tag}] BEFORE | allocated={_bytes(alloc_before)}, reserved={_bytes(reserv_before)}"
    )

    tensors_before = _collect_cuda_tensors()
    if tensors_before:
        logging.info(
            f"[track_cuda_memory:{tag}] Top CUDA tensors before (n={min(len(tensors_before), top_k)}):"
        )
        for nbytes, shape, dtype, dev in tensors_before[:top_k]:
            logging.info(f"  - {shape} {dtype} on {dev} | {_bytes(nbytes)}")

    try:
        logging.info(
            f"[track_cuda_memory:{tag}] memory_summary BEFORE:\n{torch.cuda.memory_summary(device=device)}"
        )
    except Exception:
        pass
    if show_processes and hasattr(torch.cuda, "list_gpu_processes"):
        try:
            logging.info(
                f"[track_cuda_memory:{tag}] GPU processes BEFORE:\n{torch.cuda.list_gpu_processes()} "
            )
        except Exception:
            pass

    try:
        yield
    finally:
        torch.cuda.synchronize()
        alloc_after = torch.cuda.memory_allocated(device)
        reserv_after = torch.cuda.memory_reserved(device)
        logging.info(
            f"[track_cuda_memory:{tag}] AFTER  | allocated={_bytes(alloc_after)}, reserved={_bytes(reserv_after)} (Δalloc={_bytes(alloc_after-alloc_before)}, Δres={_bytes(reserv_after-reserv_before)})"
        )

        tensors_after = _collect_cuda_tensors()
        if tensors_after:
            logging.info(
                f"[track_cuda_memory:{tag}] Top CUDA tensors after (n={min(len(tensors_after), top_k)}):"
            )
            for nbytes, shape, dtype, dev in tensors_after[:top_k]:
                logging.info(f"  - {shape} {dtype} on {dev} | {_bytes(nbytes)}")

        try:
            logging.info(
                f"[track_cuda_memory:{tag}] memory_summary AFTER:\n{torch.cuda.memory_summary(device=device)}"
            )
        except Exception:
            pass
        if show_processes and hasattr(torch.cuda, "list_gpu_processes"):
            try:
                logging.info(
                    f"[track_cuda_memory:{tag}] GPU processes AFTER:\n{torch.cuda.list_gpu_processes()} "
                )
            except Exception:
                pass


def assert_text_encoder_off_gpu(pipe) -> None:
    """Assert that the pipeline's text encoder is not resident on any CUDA device.

    Conditions:
    - pipe.text_encoder is None OR
    - All parameters/buffers of pipe.text_encoder (and its submodules / .model) are on CPU.
    Raises with a helpful message if violated.
    """
    te = getattr(pipe, "text_encoder", None)
    if te is None:
        return

    # Helper to iterate parameters/buffers from various wrappers
    def _iter_tensors(obj):
        try:
            if hasattr(obj, "parameters"):
                for p in obj.parameters(recurse=True):
                    yield p
            if hasattr(obj, "buffers"):
                for b in obj.buffers(recurse=True):
                    yield b
        except Exception:
            return

    # Check the encoder itself and a nested `.model` attribute commonly used in wrappers
    suspects = [te]
    if hasattr(te, "model"):
        suspects.append(te.model)

    on_cuda = []
    for s in suspects:
        for t in _iter_tensors(s):
            try:
                if t.is_cuda:
                    on_cuda.append(t)
            except Exception:
                continue

    if on_cuda:
        # Build a concise and actionable error message
        n = len(on_cuda)
        example = on_cuda[0]
        raise RuntimeError(
            "text_encoder tensors detected on CUDA: "
            f"found {n} tensor(s), example device={example.device}, shape={tuple(example.shape)}, dtype={example.dtype}.\n"
            "Ensure you compute embeddings once, then offload the encoder (e.g., move to CPU and set pipe.text_encoder=None)."
        )
