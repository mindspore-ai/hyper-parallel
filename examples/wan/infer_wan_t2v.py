# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Run local Wan2.1 T2V inference with a fine-tuned transformer, using the
stock `diffusers.WanPipeline`.

VAE / text encoder / tokenizer / scheduler are loaded straight from the
original base repo (they were frozen during training, i.e. untouched), and
only the transformer is swapped for the exported fine-tuned weights from
`export_wan_checkpoint.py`.

Requires:
    pip install -U diffusers transformers accelerate ftfy imageio imageio-ffmpeg

Basic usage (single GPU with >=48GB, e.g. A100-80GB/H100):
    python infer_wan_t2v.py \
        --transformer_dir ./wan2_1_t2v_14b_tomjerry_ft/transformer \
        --prompt "Tom and Jerry style cartoon: a small grey mouse tiptoes \
                   across a kitchen counter while a blue cat sleeps nearby" \
        --output_path ./sample.mp4

Memory-constrained single GPU (<48GB, e.g. 24GB consumer card):
    add --offload sequential   (or --offload model, a faster/less aggressive option)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import torch_npu
from torch_npu.contrib import transfer_to_npu

import torch


DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def build_pipeline(
    *,
    base_model: str,
    transformer_dir: Path,
    dtype: torch.dtype,
    offload: str,
):
    from diffusers import AutoencoderKLWan, WanPipeline, WanTransformer3DModel  # pylint: disable=import-outside-toplevel

    # Diffusers recommends keeping the Wan VAE in fp32 for decode quality
    # regardless of the transformer's compute dtype.
    vae = AutoencoderKLWan.from_pretrained(base_model, subfolder="vae", torch_dtype=torch.float32)
    transformer = WanTransformer3DModel.from_pretrained(str(transformer_dir), torch_dtype=dtype)

    pipe = WanPipeline.from_pretrained(
        base_model, vae=vae, transformer=transformer, torch_dtype=dtype,
    )

    if offload == "sequential":
        # Most memory-frugal: layers move to GPU one at a time. Slowest.
        pipe.enable_sequential_cpu_offload()
    elif offload == "model":
        # Whole submodules (text encoder / transformer / vae) offload as a
        # unit between steps. Good balance for ~24-32GB GPUs.
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    return pipe


def run_inference(
    *,
    pipe,
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    fps: int,
    seed: int | None,
    output_path: Path,
) -> None:
    from diffusers.utils import export_to_video  # pylint: disable=import-outside-toplevel

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    frames = output.frames[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(output_path), fps=fps)
    print(f"[infer] wrote {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", type=str, default="Wan-AI/Wan2.1-T2V-14B-Diffusers")
    parser.add_argument("--transformer_dir", type=Path, required=True,
                         help="Output of export_wan_checkpoint.py")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    # Defaults match the fine-tuning resolution/frame-count (see the training
    # yaml's video_max_size=480 / max_frames=81) so you're evaluating the
    # model at the same distribution it was actually trained on. Both height
    # and width must stay multiples of 16 (vae_scale_factor(8) * patch(2)).
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81,
                         help="Must satisfy (num_frames - 1) %% 4 == 0")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--offload", choices=["none", "model", "sequential"], default="none",
                         help="none: pipe.to('cuda') (needs ~48GB+ free); "
                              "model: enable_model_cpu_offload (~24-32GB); "
                              "sequential: enable_sequential_cpu_offload (slowest, lowest VRAM)")
    parser.add_argument("--output_path", type=Path, default=Path("./outputs/sample.mp4"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.num_frames - 1) % 4 != 0:
        raise ValueError("--num_frames must satisfy (num_frames - 1) % 4 == 0, e.g. 81")
    if args.height % 16 != 0 or args.width % 16 != 0:
        raise ValueError("--height/--width must be multiples of 16")

    dtype = getattr(torch, args.dtype)
    pipe = build_pipeline(
        base_model=args.base_model,
        transformer_dir=args.transformer_dir,
        dtype=dtype,
        offload=args.offload,
    )
    run_inference(
        pipe=pipe,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        fps=args.fps,
        seed=args.seed,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
