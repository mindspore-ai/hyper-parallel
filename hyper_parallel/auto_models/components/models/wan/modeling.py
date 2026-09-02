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
"""Wan 2.1 transformer wrapper wired into AutoModels infrastructure."""

from __future__ import annotations

import copy
import logging
import os
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import WanTransformer3DModel as _WanTransformer3DModel
from diffusers.models.transformers.transformer_wan import (
    WanAttention,
    WanAttnProcessor,
    _get_added_kv_projections,
    _get_qkv_projections,
)
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, ContextManagers

try:
    from transformers.modeling_utils import no_init_weights
except ImportError:  # pragma: no cover - transformers moved this helper.
    from transformers.initialization import no_init_weights

from hyper_parallel.auto_models._transformers.infrastructure import (
    apply_model_infrastructure,
    instantiate_infrastructure,
)
from hyper_parallel.auto_models.components.distributed.init_utils import get_world_size_safe
from hyper_parallel.auto_models.components.utils.device import get_device_type, get_torch_device
from hyper_parallel.auto_models.components.utils.model_utils import init_empty_weights

from .configuration import WanTransformer3DTrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class WanTrainingOutput(ModelOutput):
    """Training output matching the AutoModels ``ModelOutputLoss`` contract."""

    loss: dict[str, torch.FloatTensor] | None = None
    predictions: list[torch.FloatTensor] | None = None


class _WanTransformerInitShim(_WanTransformer3DModel):
    """Prevent ``PreTrainedModel`` from constructing a default Diffusers Wan."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        torch.nn.Module.__init__(self)


def _sample_count(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        return len(value)
    if torch.is_tensor(value):
        return int(value.shape[0])
    raise TypeError(f"Expected a tensor or list of tensors, got {type(value).__name__}")


def _as_sample_list(value: Any, *, count: int, keep_batch_dim: bool = True) -> list[Any]:
    if value is None:
        return [None] * count
    if isinstance(value, (list, tuple)):
        if len(value) != count:
            raise ValueError(f"Expected {count} samples, got {len(value)}")
        return list(value)
    if not torch.is_tensor(value):
        raise TypeError(f"Expected a tensor or list of tensors, got {type(value).__name__}")
    if value.shape[0] != count:
        raise ValueError(f"Expected tensor batch dimension {count}, got {value.shape[0]}")
    if keep_batch_dim:
        return [value[index : index + 1] for index in range(count)]
    return [value[index] for index in range(count)]


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _is_rank_zero() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def _format_tensor_stats(name: str, tensor: torch.Tensor | None) -> str:
    if tensor is None:
        return f"{name}=None"
    detached = tensor.detach().float()
    return (
        f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"mean={detached.mean().item():.6g}, std={detached.std().item():.6g}, "
        f"min={detached.min().item():.6g}, max={detached.max().item():.6g}, "
        f"norm={detached.norm().item():.6g}"
    )


def wan_eager_attention_forward(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    del module, kwargs
    attn_output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=False,
    )
    return attn_output.transpose(1, 2), None


def wan_flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    softcap: float | None = None,
    skip_ulysses: bool = False,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """VeOmni-compatible FA2 call path for DP-only Wan training."""
    del skip_ulysses
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        logger.warning("flash_attention_2 does not support output_attentions/head_mask; use eager for those.")

    if any(dim == 0 for dim in query.shape):
        raise ValueError("Wan flash-attention received a tensor with a zero dimension.")

    seq_len = query.shape[2]
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    target_dtype = None
    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(module.config, "_pre_quantization_dtype"):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype

    is_causal = kwargs.pop("is_causal", None)
    if is_causal is None:
        is_causal = module.is_causal

    try:
        from transformers.modeling_flash_attention_utils import (  # pylint: disable=import-outside-toplevel
            _flash_attention_forward,
        )
    except ImportError as exc:  # pragma: no cover - depends on transformers version.
        raise ImportError("Wan flash_attention_2 requires transformers.modeling_flash_attention_utils") from exc

    attn_output = _flash_attention_forward(
        query,
        key,
        value,
        attention_mask,
        query_length=seq_len,
        is_causal=is_causal,
        dropout=dropout,
        softmax_scale=scaling,
        sliding_window=sliding_window,
        softcap=softcap,
        use_top_left_mask=False,
        target_dtype=target_dtype,
        attn_implementation="flash_attention_2",
        layer_idx=getattr(module, "layer_idx", None),
        **kwargs,
    )
    return attn_output, None


class WanAttentionKernelModule:
    """Tiny adapter matching the object shape expected by HF attention kernels."""

    def __init__(self, config: SimpleNamespace, attn: WanAttention) -> None:
        target_dtype = attn.to_q.weight.dtype
        if target_dtype == torch.float32:
            target_dtype = torch.bfloat16
        self.config = SimpleNamespace(
            _attn_implementation=config._attn_implementation,
            _pre_quantization_dtype=target_dtype,
        )
        self.is_causal = False
        self.layer_idx = getattr(attn, "layer_idx", None)
        self._attn = attn

    def modules(self) -> Any:
        return self._attn.modules()


def _get_wan_full_sequence_varlen_kwargs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_length: int | None = None,
    key_length: int | None = None,
) -> dict[str, torch.Tensor | int]:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Wan flash-attention expects Q/K/V tensors with shape (batch, seq, heads, head_dim).")
    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        raise ValueError(
            "Wan flash-attention expects matching Q/K/V batch sizes; "
            f"got {query.shape[0]}/{key.shape[0]}/{value.shape[0]}."
        )
    batch_size = query.shape[0]
    query_length = query.shape[1] if query_length is None else query_length
    key_length = key.shape[1] if key_length is None else key_length
    cu_seq_lens_q = torch.arange(
        0,
        (batch_size + 1) * query_length,
        query_length,
        device=query.device,
        dtype=torch.int32,
    )
    cu_seq_lens_k = torch.arange(
        0,
        (batch_size + 1) * key_length,
        key_length,
        device=key.device,
        dtype=torch.int32,
    )
    return {
        "cu_seq_lens_q": cu_seq_lens_q,
        "cu_seq_lens_k": cu_seq_lens_k,
        "max_length_q": query_length,
        "max_length_k": key_length,
    }


def _assert_wan_flash_attention_bf16(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn: WanAttention,
) -> None:
    tensor_dtypes = {"query": query.dtype, "key": key.dtype, "value": value.dtype}
    weight_dtypes = {
        "to_q.weight": attn.to_q.weight.dtype,
        "to_k.weight": attn.to_k.weight.dtype,
        "to_v.weight": attn.to_v.weight.dtype,
    }
    assert all(dtype == torch.bfloat16 for dtype in tensor_dtypes.values()), (
        f"Wan flash-attention expects bf16 Q/K/V tensors, got {tensor_dtypes}."
    )
    assert all(dtype == torch.bfloat16 for dtype in weight_dtypes.values()), (
        f"Wan flash-attention expects bf16 projection weights, got {weight_dtypes}."
    )


class WanAutoModelsAttnProcessor(WanAttnProcessor):
    """VeOmni-compatible Wan attention processor for DP/FSDP AutoModels training."""

    def __init__(self, attn_implementation: str) -> None:
        self.attn_implementation = attn_implementation
        self.config = SimpleNamespace(_attn_implementation=attn_implementation)
        super().__init__()

    def __call__(
        self,
        attn: WanAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        is_cross_attention = encoder_hidden_states is not None

        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            if encoder_hidden_states is None:
                raise ValueError("Wan I2V cross-attention requires encoder_hidden_states")
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(
                states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ) -> torch.Tensor:
                x1, x2 = states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        use_flash_attention = self.attn_implementation in {"flash_attention_2", "veomni_flash_attention_2_with_sp"}
        if use_flash_attention:
            _assert_wan_flash_attention_bf16(query, key, value, attn)
            attention_interface: Callable = wan_flash_attention_forward
        elif self.attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.attn_implementation]
        else:
            attention_interface = wan_eager_attention_forward

        kernel_module = WanAttentionKernelModule(self.config, attn)

        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))
            if use_flash_attention:
                assert key_img.dtype == torch.bfloat16 and value_img.dtype == torch.bfloat16, (
                    "Wan image flash-attention expects bf16 added K/V tensors, "
                    f"got key={key_img.dtype}, value={value_img.dtype}."
                )
            image_attention_kwargs = (
                _get_wan_full_sequence_varlen_kwargs(query, key_img, value_img) if use_flash_attention else {}
            )
            hidden_states_img = attention_interface(
                kernel_module,
                query.transpose(1, 2),
                key_img.transpose(1, 2),
                value_img.transpose(1, 2),
                attention_mask=None,
                dropout=0.0,
                is_causal=False,
                skip_ulysses=True,
                **image_attention_kwargs,
            )[0]
            hidden_states_img = hidden_states_img.flatten(2, 3).type_as(query)

        attention_kwargs = _get_wan_full_sequence_varlen_kwargs(query, key, value) if use_flash_attention else {}
        hidden_states_out = attention_interface(
            kernel_module,
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attention_mask=attention_mask,
            dropout=0.0,
            is_causal=False,
            skip_ulysses=is_cross_attention,
            **attention_kwargs,
        )[0]
        hidden_states_out = hidden_states_out.flatten(2, 3)
        if hidden_states_img is not None:
            hidden_states_out = hidden_states_out + hidden_states_img

        hidden_states_out = hidden_states_out.type_as(attn.to_out[0].weight)
        hidden_states_out = attn.to_out[0](hidden_states_out)
        hidden_states_out = attn.to_out[1](hidden_states_out)
        return hidden_states_out


def _wan_transformer3d_forward(
    self: _WanTransformer3DModel,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: torch.Tensor | None = None,
) -> torch.Tensor:
    """VeOmni-compatible Wan forward without sequence-parallel slicing."""
    batch_size, _, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    rotary_emb = self.rope(hidden_states)
    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)

    if timestep.ndim == 2:
        timestep_seq_len = timestep.shape[1]
        timestep = timestep.flatten()
    else:
        timestep_seq_len = None

    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
        timestep,
        encoder_hidden_states,
        encoder_hidden_states_image,
        timestep_seq_len=timestep_seq_len,
    )
    if timestep_seq_len is not None:
        timestep_proj = timestep_proj.unflatten(2, (6, -1))
    else:
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    if torch.is_grad_enabled() and self.gradient_checkpointing:
        for block in self.blocks:
            hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
            )
    else:
        for block in self.blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

    if temb.ndim == 3:
        shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
        shift = shift.squeeze(2)
        scale = scale.squeeze(2)
    else:
        shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

    shift = shift.to(hidden_states.device)
    scale = scale.to(hidden_states.device)
    hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size,
        post_patch_num_frames,
        post_patch_height,
        post_patch_width,
        p_t,
        p_h,
        p_w,
        -1,
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)


class WanTransformer3DTrainingModel(PreTrainedModel, _WanTransformerInitShim):
    """Diffusers Wan transformer with an AutoModels full-finetune forward."""

    config_class = WanTransformer3DTrainingConfig
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_flash_attn_2 = True

    def __init__(self, config: WanTransformer3DTrainingConfig, **kwargs: Any) -> None:
        PreTrainedModel.__init__(self, config, **kwargs)
        del self._internal_dict
        _WanTransformer3DModel.__init__(self, **config.to_diffuser_dict())
        self.config: WanTransformer3DTrainingConfig = config
        self.config.tie_word_embeddings = False
        self._install_attention_processors()

    @property
    def config(self) -> WanTransformer3DTrainingConfig:
        return self._internal_dict

    @config.setter
    def config(self, value: WanTransformer3DTrainingConfig) -> None:
        self._internal_dict = value

    def _install_attention_processors(self) -> None:
        processor = WanAutoModelsAttnProcessor(attn_implementation=self.config._attn_implementation)
        for block in self.blocks:
            block.attn1.set_processor(processor)
            block.attn2.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.Tensor | list[torch.Tensor],
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        training_target: torch.Tensor | list[torch.Tensor],
        latents: torch.Tensor | list[torch.Tensor] | None = None,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> WanTrainingOutput:
        del kwargs
        count = _sample_count(hidden_states)
        hidden_states_list = _as_sample_list(hidden_states, count=count)
        timestep_list = _as_sample_list(timestep, count=count)
        text_context_list = _as_sample_list(encoder_hidden_states, count=count)
        image_context_list = _as_sample_list(encoder_hidden_states_image, count=count)
        target_list = _as_sample_list(training_target, count=count)
        latents_list = _as_sample_list(latents, count=count) if latents is not None else [None] * count

        per_sample_losses = []
        predictions = []
        samples = zip(
            hidden_states_list,
            timestep_list,
            text_context_list,
            image_context_list,
            target_list,
            latents_list,
        )
        for (
            sample_hidden,
            sample_timestep,
            sample_text_context,
            sample_image_context,
            sample_target,
            sample_latents,
        ) in samples:
            prediction = _wan_transformer3d_forward(
                self,
                hidden_states=sample_hidden,
                timestep=sample_timestep,
                encoder_hidden_states=sample_text_context,
                encoder_hidden_states_image=sample_image_context,
            )
            predictions.append(prediction)
            per_sample_loss = F.mse_loss(prediction.float(), sample_target.float(), reduction="none")
            per_sample_loss = per_sample_loss.view(per_sample_loss.shape[0], -1).mean(dim=1)
            per_sample_losses.append(per_sample_loss)

        loss = torch.stack(per_sample_losses).mean()
        return WanTrainingOutput(loss={"mse_loss": loss}, predictions=predictions)

    def save_pretrained(self, path: str | Path, **kwargs: Any) -> None:
        """Write a Diffusers-compatible Wan transformer directory."""
        original_config = copy.deepcopy(self.config)
        self.config = self.config.to_diffuser_dict()
        try:
            _WanTransformer3DModel.save_pretrained(self, path, **kwargs)
        finally:
            self.config = original_config


def _resolve_transformer_path(
    pretrained_model_name_or_path: str | None,
    transformer_subfolder: str | None,
    *,
    local_files_only: bool = False,
) -> str | None:
    if pretrained_model_name_or_path is None:
        return None
    path = Path(pretrained_model_name_or_path).expanduser()
    if path.is_dir() and not (path / "config.json").is_file() and transformer_subfolder:
        subfolder_path = path / transformer_subfolder
        if subfolder_path.exists():
            return str(subfolder_path)
    if path.is_dir() and (path / "config.json").is_file():
        return str(path)
    if transformer_subfolder:
        normalized = pretrained_model_name_or_path.rstrip("/").replace("\\", "/")
        if normalized.endswith(f"/{transformer_subfolder}"):
            repo_id = normalized[: -(len(transformer_subfolder) + 1)]
        else:
            repo_id = normalized
        if "/" in repo_id and not Path(repo_id).exists():
            from huggingface_hub import snapshot_download  # pylint: disable=import-outside-toplevel

            snapshot_path = Path(
                snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=[
                        f"{transformer_subfolder}/config.json",
                        f"{transformer_subfolder}/*.safetensors",
                        f"{transformer_subfolder}/*.safetensors.index.json",
                    ],
                    local_files_only=local_files_only,
                )
            )
            return str(snapshot_path / transformer_subfolder)
    return pretrained_model_name_or_path


def _resolve_config_source(
    config_path: str | None,
    transformer_path: str | None,
) -> str:
    if config_path is not None:
        return config_path
    if transformer_path is None:
        raise ValueError("Wan model requires config_path when pretrained_model_name_or_path is None")
    return transformer_path


def _validate_wan_parallel_axes(distributed_setup: Any) -> None:
    mesh = getattr(distributed_setup, "mesh_context", None)
    if mesh is None:
        return
    active = {
        name: value
        for name, value in {
            "tp_size": getattr(mesh, "tp_size", 1),
            "cp_size": getattr(mesh, "cp_size", 1),
            "pp_size": getattr(mesh, "pp_size", 1),
            "ep_size": getattr(mesh, "ep_size", 1),
        }.items()
        if int(value) != 1
    }
    for flag_name in ("sequence_parallel", "loss_parallel"):
        if bool(getattr(mesh, flag_name, False)):
            active[flag_name] = True
    if active:
        raise NotImplementedError(
            "Wan AutoModels migration currently supports DP/FSDP2 full finetuning only; "
            f"unsupported active axes/options: {active}."
        )


def _disable_fsdp_forward_input_cast_for_wan(distributed_setup: Any) -> None:
    """Keep Wan RoPE tensors in fp32 when entering FSDP-wrapped blocks."""
    fsdp_config = getattr(distributed_setup, "strategy_config", None)
    mix_precision = getattr(fsdp_config, "mix_precision", None)
    if mix_precision is None or not getattr(mix_precision, "cast_forward_inputs", False):
        return
    mix_precision.cast_forward_inputs = False
    logger.warning(
        "Disabled fsdp_config.mix_precision.cast_forward_inputs for Wan because "
        "FSDP block input casting would downcast rotary_emb from fp32 to bf16."
    )


def _current_device() -> torch.device:
    device_type = get_device_type()
    if device_type in ("cuda", "npu"):
        index = get_torch_device().current_device()
        return torch.device(f"{device_type}:{index}")
    return torch.device("cpu")


def build_wan_transformer_model(
    pretrained_model_name_or_path: str | None = None,
    *,
    transformer_subfolder: str | None = "transformer",
    config_path: str | None = None,
    task: str = "t2v",
    torch_dtype: str | torch.dtype = "bfloat16",
    attn_implementation: str = "sdpa",
    distributed_setup: Any = None,
    peft_config: Any = None,
    activation_checkpoint: str | None = None,
    swap_inputs: bool = False,
    activation_swap: str = "none",
    compile_config: Any = None,
    validate_placement: bool = False,
    **config_overrides: Any,
) -> WanTransformer3DTrainingModel:
    """Build Wan and apply AutoModels FSDP/checkpoint/loading infrastructure."""
    if peft_config is not None:
        raise ValueError("Wan full-finetune config must not set peft/lora options.")
    if task not in {"t2v", "i2v"}:
        raise ValueError("Wan task must be either 't2v' or 'i2v'")
    if distributed_setup is None:
        raise ValueError("Wan AutoModels builder requires distributed_setup from BaseTrainer")
    _validate_wan_parallel_axes(distributed_setup)
    _disable_fsdp_forward_input_cast_for_wan(distributed_setup)

    config_overrides.pop("condition_model_name_or_path", None)
    config_overrides.pop("condition_model", None)
    config_overrides.pop("trust_remote_code", None)
    local_files_only = bool(config_overrides.pop("local_files_only", False))

    transformer_path = _resolve_transformer_path(
        pretrained_model_name_or_path,
        transformer_subfolder,
        local_files_only=local_files_only,
    )
    config_source = _resolve_config_source(config_path, transformer_path)
    config = WanTransformer3DTrainingConfig.from_config_source(
        config_source,
        task=task,
        attn_implementation=attn_implementation,
        **config_overrides,
    )
    config.torch_dtype = str(torch_dtype)

    sharding_planner, fsdp2_manager, autopipeline = instantiate_infrastructure(
        distributed_setup=distributed_setup,
        device=_current_device(),
    )
    is_meta_device = transformer_path is not None or get_world_size_safe() > 1
    init_context = ContextManagers([no_init_weights(), init_empty_weights()]) if is_meta_device else nullcontext()
    with init_context:
        model = WanTransformer3DTrainingModel(config)

    model = apply_model_infrastructure(
        model,
        mesh=distributed_setup.mesh_context,
        sharding_planner=sharding_planner,
        fsdp2_manager=fsdp2_manager,
        autopipeline=autopipeline,
        peft_config=None,
        compile_config=compile_config,
        is_meta_device=is_meta_device,
        is_hf_model=True,
        device=_current_device(),
        load_base_model=transformer_path is not None,
        pretrained_path=transformer_path,
        validate_placement=validate_placement,
        distributed_setup=distributed_setup,
        activation_checkpoint=activation_checkpoint,
        swap_inputs=swap_inputs,
        activation_swap=activation_swap,
    )
    logger.info("Built Wan %s transformer for full finetuning from %s", task.upper(), transformer_path)
    model.train()
    return model
