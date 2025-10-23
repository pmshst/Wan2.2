# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
import time
import hashlib
from contextlib import contextmanager
from functools import partial
import gc

import numpy as np
import torch
from torch.cuda.amp import autocast
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae2_1 import Wan2_1_VAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

import psutil

dtype_c = torch.float16


class DynamicSwapInstaller:
    @staticmethod
    def _install_module(module: torch.nn.Module, **kwargs):
        original_class = module.__class__

        module.__dict__['forge_backup_original_class'] = original_class

        def hacked_get_attr(self, name: str):
            if '_parameters' in self.__dict__:
                _parameters = self.__dict__['_parameters']
                if name in _parameters:
                    p = _parameters[name]
                    if p is None:
                        return None
                    if p.__class__ == torch.nn.Parameter:
                        return torch.nn.Parameter(p.to(**kwargs), requires_grad=p.requires_grad)
                    else:
                        return p.to(**kwargs)
            if '_buffers' in self.__dict__:
                _buffers = self.__dict__['_buffers']
                if name in _buffers:
                    return _buffers[name].to(**kwargs)
            return super(original_class, self).__getattr__(name)

        module.__class__ = type('DynamicSwap_' + original_class.__name__, (original_class,), {
            '__getattr__': hacked_get_attr,
        })

        return

    @staticmethod
    def install_model(model: torch.nn.Module, **kwargs):
        count_modules = 0
        for m in model.modules():
            count_modules += 1
            DynamicSwapInstaller._install_module(m, **kwargs)
        # print("installed modules: " + str(count_modules) + "\n")
        return


def get_free_ram(human_readable=False):
    """
    Get available free RAM in bytes or human-readable format.

    Args:
        human_readable (bool): If True, returns formatted string (e.g., "8.5 GB").
                               If False, returns raw bytes (default).

    Returns:
        int or str: Available RAM in bytes or formatted string
    """
    mem = psutil.virtual_memory()
    available_bytes = mem.available

    if human_readable:
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if available_bytes < 1024:
                return f"{available_bytes:.2f} {unit}"
            available_bytes /= 1024
        return f"{available_bytes:.2f} PB"
    else:
        return available_bytes


class WanI2VLocal:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.boundary
        self.t5_fsdp = t5_fsdp

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        self.shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = None
        self.checkpoint_dir = checkpoint_dir

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = None

        self.low_noise_model = None
        self.use_sp = use_sp
        self.dit_fsdp = dit_fsdp

        self.high_noise_model = None

        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt

    def _prepare_model_for_timestep(self, t, boundary, offload_model):
        r"""
        Prepares and returns the required model for the current timestep.

        Args:
            t (torch.Tensor):
                current timestep.
            boundary (`int`):
                The timestep threshold. If `t` is at or above this value,
                the `high_noise_model` is considered as the required model.
            offload_model (`bool`):
                A flag intended to control the offloading behavior.

        Returns:
            torch.nn.Module:
                The active model on the target device for the current timestep.
        """
        if t.item() >= boundary:
            required_model_name = 'high_noise_model'
            offload_model_name = 'low_noise_model'
        else:
            required_model_name = 'low_noise_model'
            offload_model_name = 'high_noise_model'
        if offload_model or self.init_on_cpu:
            if next(getattr(
                    self,
                    offload_model_name).parameters()).device.type == 'cuda':
                getattr(self, offload_model_name).to('cpu')
            if next(getattr(
                    self,
                    required_model_name).parameters()).device.type == 'cpu':
                getattr(self, required_model_name).to(self.device)
        return getattr(self, required_model_name)

    def round_down_to_4x_plus_1(self, x):
        if x < 5:
            return 5
        k = math.floor((x - 1) / 4)
        return 4 * k + 1

    def get_frames_from_y(self, y):
        A = 12512899.2464896
        B = -0.836821147100733
        x = (y / A) ** (1 / B)
        x_rounded = self.round_down_to_4x_plus_1(x)
        return x_rounded

    def generate(self,
                 input_prompt,
                 img,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float` or tuple[`float`], *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
                If tuple, the first guide_scale will be used for low noise model and
                the second guide_scale will be used for high noise model.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        # preprocess
        guide_scale = (guide_scale, guide_scale) if isinstance(
            guide_scale, float) else guide_scale
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device, dtype=dtype_c)

        sampling_steps = 16  # 16+ best (high res 1280*720), 25+ (low res 640*360)
        # max_area = 720 * 1280 default

        # max_area = 1280*720

        # size 640*352
        # 81 frames             58.23 s/it 51.32 s/it (*FP8)
        # 33 frames             23.75 s/it              vae decode 4.5 sec

        # 704 * 396, sampling_steps 25+
        # frame_num = 49        24.72 s/it (FP16)
        # frame_num = 81        77.50 s/it (FP16)

        # size 720*405, sampling_steps 20+
        # frame_num = 17        21.23 s/it (FP16)        vae decode 5.4 sec
        # frame_num = 77        82.11 s/it (FP16)
        # frame_num = 81 (best) 70.74 s/it (*FP8)        vae decode 12.2 sec

        # size 832*464 / 848*448, sampling_steps 20+
        # frame_num = 17        23.68 s/it               vae decode 3.54 sec
        # frame_num = 53        74.34 s/it
        # 65                    79.73 s/it

        # size 960*540, sampling_steps 16+
        # 17 frames             34.30 s/it (FP16)
        # 41 frames             75.02 s/it (FP16)
        # 45 frames             72.35 s/it (*FP8)       vae decode 11.7 sec

        # size = 1120 * 630
        # 13 frames         29.24 s/it (*FP8)
        # 17 frames         37.90 s/it (*FP8)           vae decode 13.6 sec
        # 33 frames (max)   85.10 s/it (FP16)
        # 33 frames         76.49 s/it (*FP8)
        # 37                85.16 s/it (*FP8)

        ######################################################
        # for 8gb vram and sizes > 1120 * 630 vae use slow shared video memory

        # size 1280*720, sampling_steps 16+
        # 13 frames         48.70 s/it (FP16)
        # 13 frames         39.61 s/it (*FP8)          vae decode 17.4 sec
        # 17 frames         60.74 s/it (FP16)
        # 17 frames         54.02 s/it (*FP8)
        # 21 frames (max)   72.22 s/it (FP16)
        # 21 frames         66.18 s/it (*FP8)           vae decode 28 sec

        # size 1600*896 / 1568*896, sampling_steps 15+
        # 13 frames (max)   85.47 s/it (FP16)
        # 13 frames (max)   63.88 s/it (*FP8)           vae decode ~115 sec

        self.offload_large_tensors = False  # slower 20% inference but more frames per video
        # ################# large tensors offloading ##########################

        # 1280*720
        # 33 frames         118.83 s/it (*FP8)          vae decode 38 sec

        # 1568*896
        # 21 frames         127.01 s/it (*FP8)          vae decode: 182 sec

        self.infinity_loop = True  # for generating long videos (continued from the last frame)
        self.load_as_fp8 = True  # use 2x less ram, 10% faster

        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        free_memory, total_memory = torch.cuda.mem_get_info()
        if free_memory < 9*1024*1024*1024:
            # limit frames
            frame_num_limit = self.get_frames_from_y(w * h)
            if frame_num_limit < 81:
                frame_num = frame_num_limit

        #frame_num = 13

        F = frame_num

        logging.info(f"Generating {frame_num} frames {w}*{h}")

        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device, dtype=dtype_c)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        logging.info("Loading model T5EncoderModel")
        latent_file_path = "./final_latents.pt"
        if not os.path.exists(latent_file_path):
            logging.info("--- STAGE 1: Generating latents from text prompt ---")
            cache_dir = "./prompts_cache_video"
            os.makedirs(cache_dir, exist_ok=True)

            prompt_text = f"{input_prompt}-{n_prompt}"
            prompt_hash = hashlib.sha256(prompt_text.encode('utf-8')).hexdigest()
            cache_path = os.path.join(cache_dir, f"{prompt_hash}.pt")

            if os.path.exists(cache_path):
                logging.info(f"Loading cached prompt tensors from: {cache_path}")
                cached_tensors = torch.load(cache_path, map_location=self.device)
                context = [t.to(dtype_c) for t in cached_tensors['context']]
                context_null = [t.to(dtype_c) for t in cached_tensors['context_null']]
                del cached_tensors
            else:
                logging.info(f"No cache found. Encoding prompts and caching to: {cache_path}")
                logging.info(f"Loading model T5EncoderModel to device: {self.device}")
                self.text_encoder = T5EncoderModel(
                    text_len=self.config.text_len,
                    dtype=self.config.t5_dtype,
                    device=torch.device('cpu'),
                    checkpoint_path=os.path.join(self.checkpoint_dir, self.config.t5_checkpoint),
                    tokenizer_path=os.path.join(self.checkpoint_dir, self.config.t5_tokenizer),
                    shard_fn=self.shard_fn if self.t5_fsdp else None)

                DynamicSwapInstaller.install_model(self.text_encoder.model, device="cuda")
                context = self.text_encoder([input_prompt], torch.device('cuda'))
                context_null = self.text_encoder([n_prompt], torch.device('cuda'))
                context = [t.to(self.device) for t in context]
                context_null = [t.to(self.device) for t in context_null]
                self.text_encoder = None
                gc.collect()
                torch.cuda.empty_cache()

                torch.save({
                    'context': context,
                    'context_null': context_null
                }, cache_path)

            logging.info("Creating noise")
            start_inference = time.time_ns()
            noise = torch.randn(
                16,
                (F - 1) // self.vae_stride[0] + 1,
                lat_h,
                lat_w,
                dtype=dtype_c,
                generator=seed_g,
                device=self.device)

            last_frame_latent = "./last_frame_latents.pt"
            if os.path.exists(last_frame_latent):
                self.vae = Wan2_1_VAE(
                    vae_pth=os.path.join(self.checkpoint_dir, self.config.vae_checkpoint),
                    device=self.device,
                    dtype=dtype_c)
                logging.info(f"Found last frame continue video: {last_frame_latent}")
                lf = torch.load(last_frame_latent)
                last_frame_for_interp = lf.unsqueeze(0).to(self.device)
                resized_frame = torch.nn.functional.interpolate(
                    last_frame_for_interp, size=(h, w), mode='bicubic', align_corners=False
                )
                first_frame_prepared = resized_frame.squeeze(0).unsqueeze(1)
                zero_padding = torch.zeros(
                    3, F - 1, h, w,
                    device=self.device,
                    dtype=first_frame_prepared.dtype
                )
                prepared_video_tensor = torch.cat([first_frame_prepared, zero_padding], dim=1)

                new_y = self.vae.encode([prepared_video_tensor])[0]
                torch.save(new_y.to(dtype=dtype_c), "./y_latents.pt")
                os.remove("./last_frame_latents.pt")
                logging.info("y_latents prepared, run again to start inference")
                exit()

            y_latent = "./y_latents.pt"
            if os.path.exists(y_latent):
                logging.info("--- STAGE 1.1: Found pre-computed \"y\" latents. Loading. ---")
                logging.info(f"Loading latents from: {y_latent}")
                y = torch.load(y_latent)
            else:
                self.vae = Wan2_1_VAE(
                    vae_pth=os.path.join(self.checkpoint_dir, self.config.vae_checkpoint),
                    device=self.device,
                    dtype=dtype_c)
                y = self.vae.encode([
                    torch.concat([
                        torch.nn.functional.interpolate(
                            img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                            0, 1),
                        torch.zeros(3, F - 1, h, w)
                    ],
                        dim=1).to(self.device, dtype=dtype_c)
                ])[0]
                torch.save(y, "./y_latents.pt")
                logging.info("y_latents prepared, run again to start inference")
                exit()

            y = torch.concat([msk, y])

            self.vae = None
            gc.collect()
            torch.cuda.empty_cache()

            logging.info(f"Creating sample_scheduler")
            with (torch.no_grad()):
                boundary = self.boundary * self.num_train_timesteps

                if sample_solver == 'unipc':
                    sample_scheduler = FlowUniPCMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False)
                    sample_scheduler.set_timesteps(
                        sampling_steps, device=self.device, shift=shift)
                    timesteps = sample_scheduler.timesteps
                elif sample_solver == 'dpm++':
                    sample_scheduler = FlowDPMSolverMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False)
                    sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                    timesteps, _ = retrieve_timesteps(
                        sample_scheduler,
                        device=self.device,
                        sigmas=sampling_sigmas)
                else:
                    raise NotImplementedError("Unsupported solver.")

                latent = noise

                block_mem = 790000000
                if self.load_as_fp8:
                    block_mem = block_mem/2
                blocks_in_ram = int(get_free_ram() / block_mem)

                fk = (frame_num - 1) / 4 + 1
                x_sixe = 5120 * (((w/8) * (h/8)) / 4 * fk) * 2
                mem_need = (x_sixe * 23) / 1024 / 1024 / 1024 + (0.79 * 2) + 1
                blocks_in_vram = int(((total_memory - mem_need * 1024 * 1024 * 1024)) / block_mem)
                if blocks_in_vram < 0:
                    blocks_in_vram = 0

                torch.backends.cudnn.benchmark = True

                if blocks_in_ram > 40 - blocks_in_vram:
                    blocks_in_ram = 40 - blocks_in_vram

                blocks_in_ram = 40
                blocks_in_vram = 0

                it = 1
                model = None
                for _, t in enumerate(tqdm(timesteps)):
                    latent_model_input = [latent.to(self.device)]
                    timestep = [t]
                    timestep = torch.stack(timestep).to(self.device)

                    if t.item() >= boundary:
                        if not self.high_noise_model:
                            # print(f"\nLoading high_noise_checkpoint")
                            self.high_noise_model = WanModel.from_pretrained(
                                self.checkpoint_dir,
                                subfolder=self.config.high_noise_checkpoint,
                                model_type='i2v_h', blocks_in_ram=blocks_in_ram,
                                blocks_in_vram=blocks_in_vram, load_as_fp8=self.load_as_fp8,
                                offload_large_tensors=self.offload_large_tensors
                            )
                            model = self.high_noise_model
                    else:
                        if not self.low_noise_model:
                            model = None
                            self.high_noise_model.clear_mem()
                            del self.high_noise_model, model
                            self.high_noise_model = None
                            gc.collect()
                            torch.cuda.empty_cache()
                            # print(f"\nLoading low_noise_checkpoint")
                            self.low_noise_model = WanModel.from_pretrained(
                                self.checkpoint_dir,
                                subfolder=self.config.low_noise_checkpoint,
                                model_type='i2v_l', blocks_in_ram=blocks_in_ram,
                                blocks_in_vram=blocks_in_vram, load_as_fp8=self.load_as_fp8,
                                offload_large_tensors = self.offload_large_tensors
                            )
                            model = self.low_noise_model
                    it += 1
                    noise_pred_cond, noise_pred_uncond = model(
                        x=latent_model_input,
                        t=timestep,
                        context=[context[0]],
                        context_null=context_null,
                        seq_len=max_seq_len,
                        y=[y]
                    )
                    sample_guide_scale = guide_scale[1] \
                        if t.item() >= boundary else guide_scale[0]
                    noise_pred = noise_pred_uncond[0] + sample_guide_scale \
                                 * (noise_pred_cond[0] - noise_pred_uncond[0])

                    temp_x0 = sample_scheduler.step(
                        noise_pred.unsqueeze(0),
                        t,
                        latent.unsqueeze(0),
                        return_dict=False,
                        generator=seed_g)[0]
                    latent = temp_x0.squeeze(0)

                logging.info("Denoising complete. Offloading main models...")
                self.low_noise_model.clear_mem()
                del self.low_noise_model, model, sample_scheduler, noise
                self.low_noise_model = None
                gc.collect()
                torch.cuda.empty_cache()

                x0 = [latent]
                del latent_model_input, timestep
                end_inference = time.time_ns()
                logging.info(f"Inference time: {(end_inference - start_inference) / 1000000000} sec")

                logging.info(f"Saving final latents to: {latent_file_path}")
                torch.save(x0, latent_file_path)
                exit()
        else:
            if os.path.exists(latent_file_path):
                logging.info("--- STAGE 2: Found pre-computed latents. Decoding with VAE. ---")
                logging.info(f"Loading latents from: {latent_file_path}")
                x0 = torch.load(latent_file_path)
                x0 = [el.to(dtype=dtype_c) for el in x0]

        if self.rank == 0:
            logging.info("Loading VAE")
            self.vae = Wan2_1_VAE(
                vae_pth=os.path.join(self.checkpoint_dir, self.config.vae_checkpoint),
                device=self.device,
                dtype=dtype_c)

            start_decode = time.time_ns()
            videos = self.vae.decode(x0)
            end_decode = time.time_ns()
            logging.info(f"VAE decode: {(end_decode-start_decode)/1000000000} sec")

            self.vae = None
            gc.collect()
            torch.cuda.empty_cache()

            y = videos[0][:, -1, :, :]
            torch.save(y, "./last_frame_latents.pt")

            if self.infinity_loop:
                os.remove(latent_file_path)

            return videos[0] if self.rank == 0 else None

# kernprof -l -v generate_local.py --task i2v-A14B --size "1280*720" --image=./last_frame.png --ckpt_dir ./Wan2.2-I2V-A14B --prompt "Telephoto lens. A tiger walking through the jungle. The camera follows the tiger smoothly, keeping it in the center of the frame as the muscles ripple slowly beneath its striped fur. The tiger's eyes are focused intently ahead, and its steps are steady and powerful. The background is dense green vegetation and trees, with sunlight filtering through the leaves to cast mottled light and shadows."
# kernprof -l -v generate_local.py --task i2v-A14B --size "1280*720" --image=./last_frame.png --ckpt_dir ./Wan2.2-I2V-A14B --prompt "In close-up, a cheetah runs at full speed in a narrow canyon, its golden fur gleaming in the sun, and its black tear marks clearly visible. Shot from a low angle, the cheetah's body is close to the ground, its muscles flowing, and its limbs alternately and powerfully step over stones and soil, stirring up dust. The cheetah's eyes are sharp, staring at the target in front of it, showing unparalleled speed and strength. The camera follows the cheetah's running trajectory, capturing every moment of leaping and turning, showing its amazing agility. The whole scene unfolds in a tense chase rhythm, full of wild charm and competition for survival."
