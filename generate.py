# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import random

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import save_video, str2bool

EXAMPLE_PROMPT = {
    "t2v-A14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "i2v-A14B": {
        "prompt": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image": "examples/i2v_input.JPG",
    },
    "ti2v-5B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "s2v-14B": {
        "prompt": "A charismatic presenter greeting the audience with confident gestures.",
        "image": "examples/pose.png",
        "audio": "examples/talk.wav",
        "pose_video": "examples/pose.mp4",
        "tts_prompt_audio": "examples/zero_shot_prompt.wav",
        "tts_text": "你好，很高兴见到你。",
    },
    "animate-14B": {
        "prompt": "视频中的人在做动作",
        "animate_src_root": "examples/wan_animate/animate",
    },
}


def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    example = EXAMPLE_PROMPT[args.task]
    if args.prompt is None:
        args.prompt = example.get("prompt", args.prompt)

    # Populate optional inputs from examples when omitted
    for field in (
        "image",
        "audio",
        "pose_video",
        "tts_prompt_audio",
        "tts_prompt_text",
        "tts_text",
        "animate_src_root",
    ):
        if getattr(args, field, None) is None and field in example:
            setattr(args, field, example[field])

    task_lower = args.task.lower()

    # Task specific validations
    if "i2v" in task_lower or "ti2v" in task_lower or "s2v" in task_lower:
        assert (
            args.image is not None
        ), f"Task {args.task} requires --image. Provide a reference image path."

    if "s2v" in task_lower:
        if not args.enable_tts:
            assert (
                args.audio is not None
            ), "Speech-to-video requires --audio when TTS is disabled."
        else:
            assert (
                args.tts_text
            ), "Provide --tts_text when enabling text-to-speech synthesis."
            assert (
                args.tts_prompt_audio is not None
            ), "Provide --tts_prompt_audio with a reference speaker clip for TTS."
        if args.num_repeat is not None:
            assert args.num_repeat > 0, "--num_repeat should be a positive integer."

    if "animate" in task_lower:
        assert (
            args.animate_src_root is not None
        ), "Wan Animate requires --animate_src_root pointing to the preprocessed folder."
        if args.animate_refer_frames is not None:
            assert args.animate_refer_frames in (
                1,
                5,
            ), "--animate_refer_frames must be 1 or 5 for Wan Animate."
        if args.animate_clip_len is not None:
            assert args.animate_clip_len > 0, "--animate_clip_len must be positive."

    cfg = WAN_CONFIGS[args.task]

    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps
    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale
    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    args.base_seed = (
        args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
    )

    if getattr(args, "auto_safe_size_5b", False) and ("5b" in task_lower):
        safe_size = "832*480"
        if args.size != safe_size:
            logging.info(
                f"auto_safe_size_5b enabled: overriding size {args.size} -> {safe_size}"
            )
            args.size = safe_size
        if args.frame_num > 81:
            logging.info(
                f"auto_safe_size_5b enabled: capping frame_num {args.frame_num} -> 81"
            )
        args.frame_num = min(args.frame_num, 81)

    if "s2v" in task_lower and args.frame_num % 4 != 0:
        raise AssertionError(
            "Speech-to-video expects --frame_num to be a multiple of 4."
        )

    # Size check
    assert args.size in SUPPORTED_SIZES[args.task], (
        f"Unsupport size {args.size} for task {args.task}, supported sizes are: "
        f"{', '.join(SUPPORTED_SIZES[args.task])}"
    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-A14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image.",
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames of video are generated. Differs per task (e.g. 4n+1 for T2V/I2V/TI2V, multiples of 4 for S2V).",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.",
    )
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage.",
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.",
    )
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.",
    )
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.",
    )
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.",
    )
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated video to.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the video from.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="Optional negative prompt for classifier-free guidance.",
    )
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.",
    )
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.",
    )
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.",
    )
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the video.",
    )
    parser.add_argument(
        "--image", type=str, default=None, help="The image to generate the video from."
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Input audio track for speech-to-video tasks.",
    )
    parser.add_argument(
        "--pose_video",
        type=str,
        default=None,
        help="Optional pose driving video for speech-to-video motion guidance.",
    )
    parser.add_argument(
        "--sample_solver",
        type=str,
        default="unipc",
        choices=["unipc", "dpm++"],
        help="The solver used to sample.",
    )
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps."
    )
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.",
    )
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=None,
        help="Classifier free guidance scale.",
    )
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model paramerters dtype.",
    )
    parser.add_argument(
        "--num_repeat",
        type=int,
        default=None,
        help="Number of video clips to generate for speech-to-video (auto if omitted).",
    )
    parser.add_argument(
        "--enable_tts",
        action="store_true",
        default=False,
        help="Use CosyVoice TTS when no audio is provided for speech-to-video.",
    )
    parser.add_argument(
        "--tts_prompt_audio",
        type=str,
        default=None,
        help="Reference speaker audio for TTS when --enable_tts is set.",
    )
    parser.add_argument(
        "--tts_prompt_text",
        type=str,
        default=None,
        help="Optional reference transcript for the TTS prompt speaker.",
    )
    parser.add_argument(
        "--tts_text",
        type=str,
        default=None,
        help="Target text to synthesise when --enable_tts is set.",
    )
    parser.add_argument(
        "--init_first_frame",
        action="store_true",
        default=False,
        help="Initialise speech-to-video generation with the reference image as the first frame.",
    )
    parser.add_argument(
        "--animate_src_root",
        type=str,
        default=None,
        help="Directory containing the Wan Animate preprocessed assets (src_pose.mp4, src_ref.png, etc.).",
    )
    parser.add_argument(
        "--animate_replace",
        action="store_true",
        default=False,
        help="Enable background replacement for Wan Animate.",
    )
    parser.add_argument(
        "--animate_refer_frames",
        type=int,
        default=None,
        help="Reference frame count (1 or 5) used by Wan Animate for temporal guidance.",
    )
    parser.add_argument(
        "--animate_clip_len",
        type=int,
        default=None,
        help="Clip length processed per iteration in Wan Animate (defaults to model config).",
    )

    # Diagnostics and memory tracking flags
    parser.add_argument(
        "--diag_memory",
        action="store_true",
        default=False,
        help="Enable CUDA memory diagnostics and precompute text embeddings with encoder offload.",
    )
    parser.add_argument(
        "--diag_show_processes",
        action="store_true",
        default=False,
        help="Also print torch.cuda.list_gpu_processes() in diagnostics if available.",
    )
    parser.add_argument(
        "--diag_precision",
        type=str,
        choices=["fp16", "bf16"],
        default="fp16",
        help="Precision for text embeddings in diagnostics path (fp16 or bf16).",
    )
    parser.add_argument(
        "--auto_safe_size_5b",
        action="store_true",
        default=False,
        help="Auto-pick a safer size/frame count for 5B models (e.g., ti2v-5B).",
    )

    args = parser.parse_args()

    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl", init_method="env://", rank=rank, world_size=world_size
        )
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1
        ), f"sequence parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1:
        assert (
            args.ulysses_size == world_size
        ), f"The number of ulysses_size should be equal to the world size."
        init_distributed_group()

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=args.image is not None,
            )
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                task=args.task,
                is_vl=args.image is not None,
                device=rank,
            )
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {args.prompt_extend_method}"
            )

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert (
            cfg.num_heads % args.ulysses_size == 0
        ), f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    # Apply convenience safer-size for 5B models if requested
    if args.auto_safe_size_5b and "5B" in args.task:
        safe_size = "832*480"
        if args.size != safe_size:
            logging.info(
                f"auto_safe_size_5b enabled: overriding size {args.size} -> {safe_size}"
            )
            args.size = safe_size
        # Cap frames to 81 if higher, to reduce VRAM usage on 5B
        if args.frame_num is None:
            # _validate_args may have already set default; set conservatively to 81
            args.frame_num = 81
        elif args.frame_num > 81:
            logging.info(
                f"auto_safe_size_5b enabled: capping frame_num {args.frame_num} -> 81"
            )
            args.frame_num = 81

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if args.auto_safe_size_5b and "5b" in args.task.lower():
        logging.info(
            "auto_safe_size_5b active -> size=%s, frame_num=%s",
            args.size,
            args.frame_num,
        )

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    logging.info(f"Input prompt: {args.prompt}")
    img = None
    if args.image is not None:
        img = Image.open(args.image).convert("RGB")
        logging.info(f"Input image: {args.image}")

    # prompt extend
    if args.use_prompt_extend:
        logging.info("Extending prompt ...")
        if rank == 0:
            prompt_output = prompt_expander(
                args.prompt,
                image=img,
                tar_lang=args.prompt_extend_target_lang,
                seed=args.base_seed,
            )
            if prompt_output.status == False:
                logging.info(f"Extending prompt failed: {prompt_output.message}")
                logging.info("Falling back to original prompt.")
                input_prompt = args.prompt
            else:
                input_prompt = prompt_output.prompt
            input_prompt = [input_prompt]
        else:
            input_prompt = [None]
        if dist.is_initialized():
            dist.broadcast_object_list(input_prompt, src=0)
        args.prompt = input_prompt[0]
        logging.info(f"Extended prompt: {args.prompt}")

    neg_prompt = args.negative_prompt or ""

    if "t2v" in args.task:
        logging.info("Creating WanT2V pipeline.")
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )

        logging.info(f"Generating video ...")
        if args.diag_memory:
            from wan.utils.memory_diag import (
                assert_text_encoder_off_gpu,
                track_cuda_memory,
            )
            from wan.utils.text_conditioning import prepare_text_conditioning as _prep

            # Precompute embeddings once and offload encoder
            neg_text = (
                args.negative_prompt
                if args.negative_prompt is not None
                else wan_t2v.sample_neg_prompt
            )
            neg = [neg_text] if neg_text else None
            prompt_embeds, negative_prompt_embeds = _prep(
                pipe=wan_t2v,
                prompts=[args.prompt],
                negative_prompts=neg,
                precision=args.diag_precision,
                device=wan_t2v.device,
                offload_strategy="set_none",
                print_memory=True,
            )
            # Assert encoder is fully off GPU
            assert_text_encoder_off_gpu(wan_t2v)
            # Run generation under memory tracking
            from torch import inference_mode

            with track_cuda_memory(
                "generation", show_processes=args.diag_show_processes
            ):
                with inference_mode():
                    video = wan_t2v.generate(
                        args.prompt,
                        size=SIZE_CONFIGS[args.size],
                        frame_num=args.frame_num,
                        shift=args.sample_shift,
                        sample_solver=args.sample_solver,
                        sampling_steps=args.sample_steps,
                        guide_scale=args.sample_guide_scale,
                        n_prompt=neg_prompt,
                        seed=args.base_seed,
                        offload_model=args.offload_model,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        precision=args.diag_precision,
                    )
        else:
            video = wan_t2v.generate(
                args.prompt,
                size=SIZE_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                n_prompt=neg_prompt,
                seed=args.base_seed,
                offload_model=args.offload_model,
            )
    elif "ti2v" in args.task:
        logging.info("Creating WanTI2V pipeline.")
        wan_ti2v = wan.WanTI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )

        logging.info(f"Generating video ...")
        if args.diag_memory:
            from wan.utils.memory_diag import (
                assert_text_encoder_off_gpu,
                track_cuda_memory,
            )
            from wan.utils.text_conditioning import prepare_text_conditioning as _prep

            neg_text = (
                args.negative_prompt
                if args.negative_prompt is not None
                else wan_ti2v.sample_neg_prompt
            )
            neg = [neg_text] if neg_text else None
            prompt_embeds, negative_prompt_embeds = _prep(
                pipe=wan_ti2v,
                prompts=[args.prompt],
                negative_prompts=neg,
                precision=args.diag_precision,
                device=wan_ti2v.device,
                offload_strategy="set_none",
                print_memory=True,
            )
            assert_text_encoder_off_gpu(wan_ti2v)
            from torch import inference_mode

            with track_cuda_memory(
                "generation", show_processes=args.diag_show_processes
            ):
                with inference_mode():
                    video = wan_ti2v.generate(
                        args.prompt,
                        img=img,
                        size=SIZE_CONFIGS[args.size],
                        max_area=MAX_AREA_CONFIGS[args.size],
                        frame_num=args.frame_num,
                        shift=args.sample_shift,
                        sample_solver=args.sample_solver,
                        sampling_steps=args.sample_steps,
                        guide_scale=args.sample_guide_scale,
                        n_prompt=neg_prompt,
                        seed=args.base_seed,
                        offload_model=args.offload_model,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        precision=args.diag_precision,
                    )
        else:
            video = wan_ti2v.generate(
                args.prompt,
                img=img,
                size=SIZE_CONFIGS[args.size],
                max_area=MAX_AREA_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                n_prompt=neg_prompt,
                seed=args.base_seed,
                offload_model=args.offload_model,
            )
    elif "s2v" in args.task:
        logging.info("Creating WanS2V pipeline.")
        wan_s2v = wan.WanS2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )

        logging.info("Generating video ...")
        video = wan_s2v.generate(
            input_prompt=args.prompt,
            ref_image_path=args.image,
            audio_path=args.audio,
            enable_tts=args.enable_tts,
            tts_prompt_audio=args.tts_prompt_audio,
            tts_prompt_text=args.tts_prompt_text,
            tts_text=args.tts_text,
            num_repeat=args.num_repeat if args.num_repeat else 1,
            pose_video=args.pose_video,
            max_area=MAX_AREA_CONFIGS[args.size],
            infer_frames=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            n_prompt=neg_prompt,
            seed=args.base_seed,
            offload_model=args.offload_model,
            init_first_frame=args.init_first_frame,
        )
    elif "animate" in args.task:
        logging.info("Creating WanAnimate pipeline.")
        wan_animate = wan.WanAnimate(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )

        logging.info("Generating video ...")
        clip_len = args.animate_clip_len or args.frame_num
        refer_frames = args.animate_refer_frames or 1
        video = wan_animate.generate(
            src_root_path=args.animate_src_root,
            replace_flag=args.animate_replace,
            clip_len=clip_len,
            refert_num=refer_frames,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            input_prompt=args.prompt,
            n_prompt=neg_prompt,
            seed=args.base_seed,
            offload_model=args.offload_model,
        )
    else:
        logging.info("Creating WanI2V pipeline.")
        wan_i2v = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_sp=(args.ulysses_size > 1),
            t5_cpu=args.t5_cpu,
            convert_model_dtype=args.convert_model_dtype,
        )

        logging.info("Generating video ...")
        if args.diag_memory:
            from wan.utils.memory_diag import (
                assert_text_encoder_off_gpu,
                track_cuda_memory,
            )
            from wan.utils.text_conditioning import prepare_text_conditioning as _prep

            neg_text = (
                args.negative_prompt
                if args.negative_prompt is not None
                else wan_i2v.sample_neg_prompt
            )
            neg = [neg_text] if neg_text else None
            prompt_embeds, negative_prompt_embeds = _prep(
                pipe=wan_i2v,
                prompts=[args.prompt],
                negative_prompts=neg,
                precision=args.diag_precision,
                device=wan_i2v.device,
                offload_strategy="set_none",
                print_memory=True,
            )
            assert_text_encoder_off_gpu(wan_i2v)
            from torch import inference_mode

            with track_cuda_memory(
                "generation", show_processes=args.diag_show_processes
            ):
                with inference_mode():
                    video = wan_i2v.generate(
                        args.prompt,
                        img,
                        max_area=MAX_AREA_CONFIGS[args.size],
                        frame_num=args.frame_num,
                        shift=args.sample_shift,
                        sample_solver=args.sample_solver,
                        sampling_steps=args.sample_steps,
                        guide_scale=args.sample_guide_scale,
                        n_prompt=neg_prompt,
                        seed=args.base_seed,
                        offload_model=args.offload_model,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        precision=args.diag_precision,
                    )
        else:
            video = wan_i2v.generate(
                args.prompt,
                img,
                max_area=MAX_AREA_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                n_prompt=neg_prompt,
                seed=args.base_seed,
                offload_model=args.offload_model,
            )

    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/", "_")[:50]
            suffix = ".mp4"
            args.save_file = (
                f"{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.ulysses_size}_{formatted_prompt}_{formatted_time}"
                + suffix
            )

        logging.info(f"Saving generated video to {args.save_file}")
        save_video(
            tensor=video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
    del video

    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
