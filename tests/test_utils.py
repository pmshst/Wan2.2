import importlib.util
import os
from pathlib import Path
from unittest import mock

import pytest
import torch


def _load_module(name: str, relative_path: str):
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


memory_diag = _load_module("memory_diag", "wan/utils/memory_diag.py")
text_conditioning = _load_module("text_conditioning", "wan/utils/text_conditioning.py")
utils_mod = _load_module("wan_utils", "wan/utils/utils.py")


class _DummyT5Model(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise NotImplementedError


class _DummyTextEncoder:
    def __init__(self):
        super().__init__()
        self.model = _DummyT5Model()
        self.tokenizer = object()

    def __call__(self, prompts, device):
        return [
            torch.ones(1, 2, device=device) * (idx + 1) for idx in range(len(prompts))
        ]


class _DummyPipe:
    def __init__(self):
        self.text_encoder = _DummyTextEncoder()
        self.device = torch.device("cpu")


def test_prepare_text_conditioning_offloads_encoder():
    pipe = _DummyPipe()

    prompt_embeds, negative_embeds = text_conditioning.prepare_text_conditioning(
        pipe,
        prompts=["hello"],
        negative_prompts=["world"],
        precision="fp16",
        device=pipe.device,
        offload_strategy="set_none",
        print_memory=False,
    )

    assert pipe.text_encoder is None
    assert len(prompt_embeds) == 1
    assert prompt_embeds[0].dtype == torch.float16
    assert prompt_embeds[0].device == pipe.device
    assert negative_embeds is not None
    assert negative_embeds[0].dtype == torch.float16


def test_assert_text_encoder_off_gpu_accepts_cpu():
    pipe = _DummyPipe()
    pipe.text_encoder.model.to("cpu")
    memory_diag.assert_text_encoder_off_gpu(pipe)


def test_track_cuda_memory_no_cuda():
    with memory_diag.track_cuda_memory("unit-test"):
        pass


def test_download_cosyvoice_repo_skips_clone_when_exists(tmp_path):
    repo_dir = tmp_path / "CosyVoice"
    repo_dir.mkdir()

    with mock.patch.object(utils_mod.subprocess, "check_call") as check_call:
        path = utils_mod.download_cosyvoice_repo(str(repo_dir))

    assert path == repo_dir
    check_call.assert_not_called()


def test_download_cosyvoice_repo_clones_when_missing(tmp_path):
    repo_dir = tmp_path / "CosyVoice"

    def _fake_clone(cmd):
        repo_dir.mkdir()
        return 0

    with mock.patch.object(
        utils_mod.subprocess, "check_call", side_effect=_fake_clone
    ) as check_call:
        path = utils_mod.download_cosyvoice_repo(str(repo_dir))

    assert path == repo_dir
    assert repo_dir.exists()
    check_call.assert_called_once()


def test_download_cosyvoice_model_uses_snapshot(tmp_path):
    pytest.importorskip("huggingface_hub")
    target = tmp_path / "CosyVoiceModel"

    def _fake_snapshot(*_, **__):
        os.makedirs(target, exist_ok=True)
        (target / "config.json").write_text("{}", encoding="utf-8")
        return str(target)

    with mock.patch(
        "huggingface_hub.snapshot_download", side_effect=_fake_snapshot
    ) as snap:
        path = utils_mod.download_cosyvoice_model(
            "CosyVoice2-0.5B", str(target), repo_id="test/model"
        )

    assert path == target
    assert (target / "config.json").exists()
    snap.assert_called_once()
