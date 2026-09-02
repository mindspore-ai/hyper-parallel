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
"""Frozen Wan 2.1 condition model used by the AutoModels DiT trainer."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.pipelines.wan.pipeline_wan import WanPipeline
from diffusers.video_processor import VideoProcessor
from PIL import Image
from torchvision.transforms import InterpolationMode, functional
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel


_DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
    "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, "
    "extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
    "fused fingers, still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)


@dataclass
class WanConditionConfig:
    """Runtime options for frozen Wan condition components."""

    base_model_path: str
    task: str = "t2v"
    tokenizer_subfolder: str = "tokenizer"
    text_encoder_subfolder: str = "text_encoder"
    vae_subfolder: str = "vae"
    scheduler_subfolder: str = "scheduler"
    image_processor_subfolder: str = "image_processor"
    image_encoder_subfolder: str = "image_encoder"
    max_sequence_length: int = 512
    num_train_timesteps: int = 1000
    cfg_negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT
    cfg_negative_prob: float = 0.1
    video_max_size: int = 480
    transformer_dtype: str = "bfloat16"
    local_files_only: bool = False
    seed: int = 42


def _torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {dtype!r}") from exc


def _retrieve_latents(
    encoder_output: torch.Tensor,
    generator: torch.Generator | None = None,
    sample_mode: str = "sample",
) -> torch.Tensor:
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    try:
        generator = torch.Generator(device=device)
    except (RuntimeError, TypeError):
        generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _normalize_sample_values(values: Any) -> list[Any]:
    if isinstance(values, (list, tuple)):
        return list(values)
    return [values]


def _first_sample_image(images: Any, fallback_video: torch.Tensor | None = None) -> Any:
    if images is None or (isinstance(images, dict) and not images):
        sample_images = []
    else:
        sample_images = _normalize_sample_values(images)
    if sample_images:
        first = sample_images[0]
        if isinstance(first, (list, tuple)):
            if not first:
                raise ValueError("I2V sample image list is empty")
            return first[0]
        return first
    if fallback_video is None:
        raise ValueError("Wan I2V training requires an image or a video first frame")
    return fallback_video[0]


def _tensor_frame_to_pil(frame: torch.Tensor) -> Image.Image:
    if frame.ndim != 3:
        raise ValueError(f"Expected frame tensor with shape (C,H,W) or (H,W,C), got {tuple(frame.shape)}")
    if frame.shape[0] in (1, 3):
        frame = frame.permute(1, 2, 0)
    if frame.dtype.is_floating_point:
        scale = 255.0 if frame.max() <= 1.0 else 1.0
        frame = (frame * scale).clamp(0, 255).to(torch.uint8)
    else:
        frame = frame.clamp(0, 255).to(torch.uint8)
    return Image.fromarray(frame.cpu().numpy()).convert("RGB")


def _image_for_processors(image: Any) -> Any:
    if torch.is_tensor(image):
        return _tensor_frame_to_pil(image)
    return image


class WanConditionModel(torch.nn.Module):
    """Frozen text/VAE/image encoder path that prepares Wan DiT training inputs."""

    def __init__(
        self,
        config: WanConditionConfig,
        *,
        device: torch.device,
        dp_rank: int = 0,
    ) -> None:
        super().__init__()
        if config.task not in {"t2v", "i2v"}:
            raise ValueError("Wan condition task must be either 't2v' or 'i2v'")
        self.config = config
        self.runtime_config = SimpleNamespace(expand_timesteps=False)
        self.target_dtype = _torch_dtype(config.transformer_dtype)
        self.device = device
        self.generator = _make_generator(device, int(config.seed) + int(dp_rank))
        self._timesteps_ready = False
        self._load_components()
        self.requires_grad_(False)
        self.eval()
        self.to(device)
        self._prepare_negative_prompt_embeds()

    @property
    def _execution_device(self) -> torch.device:
        return self.device

    @property
    def vae_scale_factor_temporal(self) -> int:
        return int(getattr(self.vae.config, "scale_factor_temporal", 4))

    @property
    def vae_scale_factor_spatial(self) -> int:
        return int(getattr(self.vae.config, "scale_factor_spatial", 8))

    @property
    def config_for_pipeline(self) -> SimpleNamespace:
        return self.runtime_config

    @property
    def boundary_ratio(self) -> None:
        return None

    def _load_components(self) -> None:
        base = self.config.base_model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            base,
            subfolder=self.config.tokenizer_subfolder,
            local_files_only=self.config.local_files_only,
        )
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            base,
            subfolder=self.config.text_encoder_subfolder,
            torch_dtype=torch.bfloat16,
            local_files_only=self.config.local_files_only,
        )
        self.vae = AutoencoderKLWan.from_pretrained(
            base,
            subfolder=self.config.vae_subfolder,
            torch_dtype=torch.float32,
            local_files_only=self.config.local_files_only,
        )
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base,
            subfolder=self.config.scheduler_subfolder,
            local_files_only=self.config.local_files_only,
        )
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        if self.config.task == "i2v":
            self.image_processor = CLIPImageProcessor.from_pretrained(
                base,
                subfolder=self.config.image_processor_subfolder,
                local_files_only=self.config.local_files_only,
            )
            self.image_encoder = CLIPVisionModel.from_pretrained(
                base,
                subfolder=self.config.image_encoder_subfolder,
                torch_dtype=torch.float32,
                local_files_only=self.config.local_files_only,
            )
        else:
            self.image_processor = None
            self.image_encoder = None

    @torch.no_grad()
    def _get_t5_prompt_embeds(self, **kwargs: Any) -> torch.Tensor:
        return WanPipeline._get_t5_prompt_embeds(self, **kwargs)

    @torch.no_grad()
    def _prepare_negative_prompt_embeds(self) -> None:
        prompt_embeds, _ = WanPipeline.encode_prompt(
            self,
            prompt=[self.config.cfg_negative_prompt],
            do_classifier_free_guidance=False,
            max_sequence_length=self.config.max_sequence_length,
        )
        self.negative_prompt_embeds = prompt_embeds[0].unsqueeze(0)

    def _resize_video(self, video: torch.Tensor) -> torch.Tensor:
        height, width = video.shape[-2:]
        size = min(self.config.video_max_size, min(width, height))
        return functional.resize(video, size, interpolation=InterpolationMode.BICUBIC).float().clamp(0, 255)

    def _encode_video_to_latents(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        resized_video = self._resize_video(video)
        processed_video = self.video_processor.preprocess_video(resized_video)
        processed_video = processed_video.to(device=self.vae.device, dtype=self.vae.dtype)
        posterior: DiagonalGaussianDistribution = self.vae.encode(processed_video).latent_dist
        return posterior.parameters, resized_video

    def _normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(
            1, self.vae.config.z_dim, 1, 1, 1
        )
        latents_std = torch.tensor(self.vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(
            1, self.vae.config.z_dim, 1, 1, 1
        )
        return (latents - latents_mean) / latents_std

    @torch.no_grad()
    def _encode_prompt(self, inputs: list[str]) -> list[torch.Tensor]:
        prompt_embeds, _ = WanPipeline.encode_prompt(
            self,
            prompt=inputs,
            do_classifier_free_guidance=False,
            max_sequence_length=self.config.max_sequence_length,
        )
        return [prompt_embed.unsqueeze(0).to(self.target_dtype) for prompt_embed in prompt_embeds]

    @torch.no_grad()
    def _encode_i2v_image_context(self, image: Any) -> torch.Tensor:
        if self.image_processor is None or self.image_encoder is None:
            raise ValueError("Wan I2V condition model was built without image encoder components")
        image = _image_for_processors(image)
        image_inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
        image_embeds = self.image_encoder(**image_inputs, output_hidden_states=True).hidden_states[-2]
        return image_embeds.to(self.target_dtype)

    def _prepare_i2v_latent_condition(
        self,
        image: Any,
        *,
        video: torch.Tensor,
        latents: torch.Tensor,
        height: int,
        width: int,
        num_frames: int,
    ) -> torch.Tensor:
        image = _image_for_processors(image)
        image = self.video_processor.preprocess(image, height=height, width=width).to(self.device, dtype=torch.float32)
        batch_size = latents.shape[0]
        latent_height = latents.shape[-2]
        latent_width = latents.shape[-1]

        image = image.unsqueeze(2)
        video_condition = torch.cat(
            [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)],
            dim=2,
        )
        video_condition = video_condition.to(device=self.vae.device, dtype=self.vae.dtype)
        latent_condition = _retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax")
        latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1).to(latents.device, dtype=latents.dtype)
        latent_condition = self._normalize_latents(latent_condition)

        mask_lat_size = torch.ones(
            batch_size,
            1,
            num_frames,
            latent_height,
            latent_width,
            dtype=latents.dtype,
            device=latents.device,
        )
        mask_lat_size[:, :, 1:] = 0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal)
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(
            batch_size,
            -1,
            self.vae_scale_factor_temporal,
            latent_height,
            latent_width,
        )
        mask_lat_size = mask_lat_size.transpose(1, 2)
        del video
        return torch.concat([mask_lat_size.to(latent_condition.device), latent_condition], dim=1)

    @torch.no_grad()
    def get_condition(self, inputs: list[str], videos: list[Any], images: list[Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        context_list = self._encode_prompt(inputs)
        latents_list = []
        resized_videos = []
        for sample_videos in videos:
            sample_videos = _normalize_sample_values(sample_videos)
            if len(sample_videos) != 1:
                raise ValueError("Wan DiT training currently supports one target video per sample")
            sample_latents, resized_video = self._encode_video_to_latents(sample_videos[0])
            latents_list.append(sample_latents)
            resized_videos.append(resized_video)

        if self.config.task == "t2v":
            return {"latents": latents_list, "context": context_list}

        image_context_list = []
        image_inputs = images if images is not None else [None] * len(videos)
        if len(image_inputs) != len(videos):
            raise ValueError(f"Wan I2V batch has {len(videos)} videos but {len(image_inputs)} image entries")
        latent_condition_inputs = []
        for sample_images, resized_video in zip(image_inputs, resized_videos):
            image = _first_sample_image(sample_images, resized_video)
            image_context_list.append(self._encode_i2v_image_context(image))
            latent_condition_inputs.append((image, resized_video))
        return {
            "latents": latents_list,
            "context": context_list,
            "image_context": image_context_list,
            "latent_condition_inputs": latent_condition_inputs,
        }

    @torch.no_grad()
    def process_condition(
        self,
        latents: list[torch.Tensor],
        context: list[torch.Tensor],
        image_context: list[torch.Tensor] | None = None,
        latent_condition_inputs: list[tuple[Any, torch.Tensor]] | None = None,
    ) -> dict[str, Any]:
        if not self._timesteps_ready:
            self.scheduler.set_timesteps(self.config.num_train_timesteps, device=self.device)
            self._timesteps_ready = True

        packed_conditions: dict[str, list[torch.Tensor]] = {
            "hidden_states": [],
            "timestep": [],
            "encoder_hidden_states": [],
            "training_target": [],
            "latents": [],
        }
        if self.config.task == "i2v":
            packed_conditions["encoder_hidden_states_image"] = []

        for sample_index, (sample_latents, sample_context) in enumerate(zip(latents, context)):
            sample_latents = DiagonalGaussianDistribution(sample_latents).mode()
            sample_latents = self._normalize_latents(sample_latents).to(self.device)
            noise = torch.randn(
                sample_latents.shape,
                dtype=sample_latents.dtype,
                device=self.device,
                generator=self.generator,
            )
            timestep_ids = torch.randint(
                0,
                len(self.scheduler.timesteps),
                (sample_latents.shape[0],),
                device=self.device,
                generator=self.generator,
            )
            timestep = self.scheduler.timesteps[timestep_ids].to(device=sample_latents.device, dtype=sample_latents.dtype)
            noisy_latents = self.scheduler.scale_noise(sample_latents, timestep, noise)
            training_target = noise - sample_latents

            if torch.rand((), device=self.device, generator=self.generator) < self.config.cfg_negative_prob:
                sample_context = self.negative_prompt_embeds.to(device=sample_latents.device, dtype=self.target_dtype)
            else:
                sample_context = sample_context.to(device=sample_latents.device, dtype=self.target_dtype)

            hidden_states = noisy_latents
            if self.config.task == "i2v":
                if image_context is None or latent_condition_inputs is None:
                    raise ValueError("Wan I2V process_condition requires image_context and latent_condition_inputs")
                image_input, resized_video = latent_condition_inputs[sample_index]
                latent_condition = self._prepare_i2v_latent_condition(
                    image_input,
                    video=resized_video,
                    latents=sample_latents,
                    height=int(resized_video.shape[-2]),
                    width=int(resized_video.shape[-1]),
                    num_frames=int(resized_video.shape[0]),
                )
                hidden_states = torch.cat([hidden_states, latent_condition], dim=1)
                packed_conditions["encoder_hidden_states_image"].append(
                    image_context[sample_index].to(device=sample_latents.device, dtype=self.target_dtype)
                )

            packed_conditions["hidden_states"].append(hidden_states.to(self.target_dtype))
            packed_conditions["timestep"].append(timestep)
            packed_conditions["encoder_hidden_states"].append(sample_context)
            packed_conditions["training_target"].append(training_target)
            packed_conditions["latents"].append(sample_latents)

        return packed_conditions


def build_wan_condition_model(
    *,
    base_model_path: str,
    task: str,
    device: torch.device,
    dp_rank: int = 0,
    seed: int = 42,
    **kwargs: Any,
) -> WanConditionModel:
    """Build the frozen Wan condition path for online DiT training."""
    config = WanConditionConfig(
        base_model_path=base_model_path,
        task=task,
        seed=seed,
        **kwargs,
    )
    return WanConditionModel(config, device=device, dp_rank=dp_rank)
