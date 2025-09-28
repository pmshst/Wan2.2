# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import binascii
import logging
import os
import os.path as osp
import subprocess
from pathlib import Path
from typing import Optional

import imageio
import torch
import torchvision

__all__ = [
    "save_video",
    "save_image",
    "str2bool",
    "download_cosyvoice_repo",
    "download_cosyvoice_model",
]


def rand_name(length=8, suffix=""):
    name = binascii.b2a_hex(os.urandom(length)).decode("utf-8")
    if suffix:
        if not suffix.startswith("."):
            suffix = "." + suffix
        name += suffix
    return name


def save_video(
    tensor,
    save_file=None,
    fps=30,
    suffix=".mp4",
    nrow=8,
    normalize=True,
    value_range=(-1, 1),
):
    # cache file
    cache_file = (
        osp.join("/tmp", rand_name(suffix=suffix)) if save_file is None else save_file
    )

    # save to cache
    try:
        # preprocess
        tensor = tensor.clamp(min(value_range), max(value_range))
        tensor = torch.stack(
            [
                torchvision.utils.make_grid(
                    u, nrow=nrow, normalize=normalize, value_range=value_range
                )
                for u in tensor.unbind(2)
            ],
            dim=1,
        ).permute(1, 2, 3, 0)
        tensor = (tensor * 255).type(torch.uint8).cpu()

        # write video
        writer = imageio.get_writer(cache_file, fps=fps, codec="libx264", quality=8)
        for frame in tensor.numpy():
            writer.append_data(frame)
        writer.close()
    except Exception as e:
        logging.info(f"save_video failed, error: {e}")


def save_image(tensor, save_file, nrow=8, normalize=True, value_range=(-1, 1)):
    # cache file
    suffix = osp.splitext(save_file)[1]
    if suffix.lower() not in [".jpg", ".jpeg", ".png", ".tiff", ".gif", ".webp"]:
        suffix = ".png"

    # save to cache
    try:
        tensor = tensor.clamp(min(value_range), max(value_range))
        torchvision.utils.save_image(
            tensor, save_file, nrow=nrow, normalize=normalize, value_range=value_range
        )
        return save_file
    except Exception as e:
        logging.info(f"save_image failed, error: {e}")


def str2bool(v):
    """
    Convert a string to a boolean.

    Supported true values: 'yes', 'true', 't', 'y', '1'
    Supported false values: 'no', 'false', 'f', 'n', '0'

    Args:
        v (str): String to convert.

    Returns:
        bool: Converted boolean value.

    Raises:
        argparse.ArgumentTypeError: If the value cannot be converted to boolean.
    """
    if isinstance(v, bool):
        return v
    v_lower = v.lower()
    if v_lower in ("yes", "true", "t", "y", "1"):
        return True
    elif v_lower in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (True/False)")


def masks_like(tensor, zero=False, generator=None, p=0.2):
    assert isinstance(tensor, list)
    out1 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]

    out2 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]

    if zero:
        if generator is not None:
            for u, v in zip(out1, out2):
                random_num = torch.rand(
                    1, generator=generator, device=generator.device
                ).item()
                if random_num < p:
                    u[:, 0] = (
                        torch.normal(
                            mean=-3.5,
                            std=0.5,
                            size=(1,),
                            device=u.device,
                            generator=generator,
                        )
                        .expand_as(u[:, 0])
                        .exp()
                    )
                    v[:, 0] = torch.zeros_like(v[:, 0])
                else:
                    u[:, 0] = u[:, 0]
                    v[:, 0] = v[:, 0]
        else:
            for u, v in zip(out1, out2):
                u[:, 0] = torch.zeros_like(u[:, 0])
                v[:, 0] = torch.zeros_like(v[:, 0])

    return out1, out2


def best_output_size(w, h, dw, dh, expected_area):
    # float output size
    ratio = w / h
    ow = (expected_area * ratio) ** 0.5
    oh = expected_area / ow

    # process width first
    ow1 = int(ow // dw * dw)
    oh1 = int(expected_area / ow1 // dh * dh)
    assert ow1 % dw == 0 and oh1 % dh == 0 and ow1 * oh1 <= expected_area
    ratio1 = ow1 / oh1

    # process height first
    oh2 = int(oh // dh * dh)
    ow2 = int(expected_area / oh2 // dw * dw)
    assert oh2 % dh == 0 and ow2 % dw == 0 and ow2 * oh2 <= expected_area
    ratio2 = ow2 / oh2

    # compare ratios
    if max(ratio / ratio1, ratio1 / ratio) < max(ratio / ratio2, ratio2 / ratio):
        return ow1, oh1
    else:
        return ow2, oh2


def download_cosyvoice_repo(target_dir: str, repo_url: Optional[str] = None) -> Path:
    """Clone the CosyVoice repository if it is not already present.

    Args:
        target_dir: Directory where the repository should live.
        repo_url: Optional override for the repository URL. Defaults to the
            official upstream but can be customised via the
            ``WAN_COSYVOICE_REPO_URL`` environment variable.

    Returns:
        Path to the checked-out repository.
    """

    destination = Path(target_dir)
    if destination.exists():
        return destination

    resolved_url = repo_url or os.getenv(
        "WAN_COSYVOICE_REPO_URL",
        "https://github.com/alibaba-damo-academy/CosyVoice.git",
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.check_call(
            ["git", "clone", "--depth", "1", resolved_url, str(destination)]
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"Failed to clone CosyVoice repository from {resolved_url}. "
            "Install git or set WAN_COSYVOICE_REPO_URL to a valid mirror, "
            "or download the repository manually."
        ) from exc

    return destination


def download_cosyvoice_model(
    model_name: str, target_dir: str, repo_id: Optional[str] = None
) -> Path:
    """Download CosyVoice model weights from Hugging Face if missing.

    Args:
        model_name: Name of the model folder to materialise locally.
        target_dir: Destination directory for the weights.
        repo_id: Optional Hugging Face repository id. Defaults to
            ``iic/<model_name>`` and can be overridden with the
            ``WAN_COSYVOICE_MODEL_REPO`` environment variable.

    Returns:
        Path to the directory containing the weights.
    """

    destination = Path(target_dir)
    if destination.exists() and any(destination.iterdir()):
        return destination

    resolved_repo = repo_id or os.getenv(
        "WAN_COSYVOICE_MODEL_REPO", f"iic/{model_name}"
    )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "huggingface-hub is required to download CosyVoice weights. "
            "Install it or provide the weights manually."
        ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=resolved_repo,
            local_dir=str(destination),
            local_dir_use_symlinks=False,
        )
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(
            f"Failed to download CosyVoice weights from {resolved_repo}. "
            "Set WAN_COSYVOICE_MODEL_REPO to a valid repository or place "
            "the weights manually."
        ) from exc

    return destination
