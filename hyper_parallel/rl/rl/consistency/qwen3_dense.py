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
"""Qwen3 Dense training-rollout numerical consistency profile."""

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Any, Optional

from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from hyper_parallel import get_platform
from hyper_parallel.platform.platform import PlatformType

platform = get_platform()

CONSISTENCY_PROFILE_OFF = "off"
QWEN3_ASCEND_FA3_BATCH_INVARIANT_V1 = "qwen3_ascend_fa3_batch_invariant_v1"
_SUPPORTED_PROFILES = frozenset(
    (CONSISTENCY_PROFILE_OFF, QWEN3_ASCEND_FA3_BATCH_INVARIANT_V1)
)
_TRAINER_ATTENTION_IMPLEMENTATION = "hyper_qwen3_npu_consistent_v1"
_EXPECTED_PACKAGE_VERSIONS = {
    "batch-invariant-ops": "1.0.0",
    "flash-attn-npu": "0.2.0b1",
    "transformers": "5.5.4",
    "vllm": "0.22.1",
    "vllm-ascend": "0.22.1rc1",
}
_ROLLOUT_PROFILE_SETTINGS = {
    "attention_backend": "FLASH_ATTN",
    "batch_invariant": True,
    "block_size": 128,
    "dtype": "bfloat16",
    "enable_chunked_prefill": True,
    "enable_prefix_caching": True,
    "enforce_eager": True,
    "logprobs_mode": "raw_logprobs",
    "model_implementation": "hyper",
    "tensor_parallel_size": 1,
}
_TRAINER_MIXED_PRECISION_SETTINGS = {
    "enabled": True,
    "output_dtype": None,
    "param_dtype": "bfloat16",
    "reduce_dtype": "float32",
}


@dataclass
class _ConsistencyRuntime:
    """Process-global state for one irreversible numerical patch installation."""

    flash_attn_func: Optional[Callable[..., Any]] = None
    flash_attn_varlen_func: Optional[Callable[..., Any]] = None
    npu_rms_norm: Optional[Callable[..., Any]] = None
    installed_profile: str = CONSISTENCY_PROFILE_OFF
    installed_rollout_profile: str = CONSISTENCY_PROFILE_OFF


_runtime = _ConsistencyRuntime()


def _qwen3_npu_rms_norm_forward(module: Any, hidden_states: Any) -> Any:
    """Run the VERL-compatible fused Qwen3 RMSNorm primitive."""
    if _runtime.npu_rms_norm is None:
        raise ValueError("Qwen3 NPU RMSNorm was called before consistency profile installation")
    if hidden_states.dtype != module.weight.dtype:
        hidden_states = hidden_states.to(module.weight.dtype)
    return _runtime.npu_rms_norm(
        hidden_states,
        module.weight,
        epsilon=module.variance_epsilon,
    )[0]


def _install_qwen3_npu_rms_norm() -> None:
    """Install the shared Trainer and Hyper-vLLM Qwen3 RMSNorm path."""
    if _runtime.npu_rms_norm is not None:
        return
    try:
        # torch-npu is optional and must remain unloaded while the profile is off.
        import torch_npu  # pylint: disable=C0415
        from transformers.models.qwen3.modeling_qwen3 import (  # pylint: disable=C0415
            Qwen3RMSNorm,
        )
    except ImportError as error:
        raise ValueError(f"Qwen3 NPU RMSNorm is unavailable: {error}") from error
    if not callable(torch_npu.npu_rms_norm):
        raise ValueError("torch-npu does not expose the required npu_rms_norm interface")
    _runtime.npu_rms_norm = torch_npu.npu_rms_norm
    Qwen3RMSNorm.forward = _qwen3_npu_rms_norm_forward


def consistency_profile(config: Mapping[str, Any]) -> str:
    """Return and validate the selected consistency profile name."""
    consistency = config.get("consistency", {})
    if not isinstance(consistency, Mapping):
        raise ValueError("Configuration section 'consistency' must be a mapping")
    unknown = set(consistency) - {"profile"}
    if unknown:
        raise ValueError(f"Unsupported consistency configuration keys: {sorted(unknown)}")
    profile = consistency.get("profile", CONSISTENCY_PROFILE_OFF)
    if not isinstance(profile, str) or not profile:
        raise ValueError("consistency.profile must be a non-empty string")
    if profile not in _SUPPORTED_PROFILES:
        raise ValueError(
            f"Unsupported consistency.profile {profile!r}; available={sorted(_SUPPORTED_PROFILES)}"
        )
    return profile


def configure_consistency_profile(config: dict[str, Any]) -> str:
    """Atomically derive Trainer and rollout settings owned by one profile.

    Args:
        config: Mutable, fully merged Hyper-RL configuration.

    Returns:
        The normalized profile name.

    Raises:
        ValueError: If the profile does not support the selected model or rollout.
    """
    profile = consistency_profile(config)
    if profile == CONSISTENCY_PROFILE_OFF:
        return profile

    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("Consistency profiles require configuration section 'model' to be a mapping")
    if model.get("name") != "qwen3":
        raise ValueError(f"Consistency profile {profile!r} supports only model.name='qwen3'")

    rollout = config.get("rollout")
    if not isinstance(rollout, dict):
        raise ValueError("Consistency profiles require configuration section 'rollout' to be a mapping")
    if rollout.get("engine") != "vllm":
        raise ValueError(f"Consistency profile {profile!r} requires rollout.engine='vllm'")
    vllm = rollout.get("vllm")
    if not isinstance(vllm, dict):
        raise ValueError("Consistency profiles require configuration section 'rollout.vllm' to be a mapping")
    train = config.get("train")
    if not isinstance(train, dict):
        raise ValueError("Consistency profiles require configuration section 'train' to be a mapping")
    mixed_precision = train.get("mixed_precision")
    if not isinstance(mixed_precision, dict):
        raise ValueError("Consistency profiles require train.mixed_precision to be a mapping")

    model["attn_implementation"] = _TRAINER_ATTENTION_IMPLEMENTATION
    mixed_precision.update(_TRAINER_MIXED_PRECISION_SETTINGS)
    vllm.update(_ROLLOUT_PROFILE_SETTINGS)
    vllm["consistency_profile"] = profile
    rollout.update({"temperature": 1.0, "top_p": 1.0, "top_k": 0})
    return profile


def validate_consistency_model_identity(config: Mapping[str, Any], model_registration: Any) -> None:
    """Reject checkpoints outside the exact model family owned by the profile."""
    profile = consistency_profile(config)
    if profile == CONSISTENCY_PROFILE_OFF:
        return
    identity = (
        model_registration.hyper_model_name,
        model_registration.hf_architecture,
        model_registration.model_type,
        model_registration.text_model_type,
    )
    expected = ("qwen3", "Qwen3ForCausalLM", "qwen3", "qwen3")
    if identity != expected:
        raise ValueError(
            f"Consistency profile {profile!r} requires checkpoint identity {expected}, got {identity}"
        )


def _flash_attn_npu_attention_forward(
    _module: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    **kwargs: Any,
) -> tuple[Any, None]:
    """Run trainable FA v2 with the Transformers Qwen3 attention contract."""
    if _runtime.flash_attn_func is None:
        raise RuntimeError("The Qwen3 consistency attention profile was not installed")
    if dropout != 0.0:
        raise ValueError("The Qwen3 consistency profile requires attention dropout=0")
    if sliding_window is not None:
        raise ValueError("The Qwen3 consistency profile does not yet support sliding-window attention")
    if bool(kwargs.get("output_attentions", False)):
        raise ValueError("The Qwen3 consistency profile does not support output_attentions=True")

    packed_cu_seqlens = kwargs.get("packed_cu_seqlens")
    packed_max_seqlen = kwargs.get("packed_max_seqlen")
    if packed_cu_seqlens is not None:
        if _runtime.flash_attn_varlen_func is None:
            raise RuntimeError("The Qwen3 packed consistency attention profile was not installed")
        if attention_mask is not None:
            raise ValueError("Packed Qwen3 attention must not receive a padded attention mask")
        if query.shape[0] != 1 or not isinstance(packed_max_seqlen, int):
            raise ValueError("Packed Qwen3 attention requires one dummy batch and an integer max sequence length")
        output = _runtime.flash_attn_varlen_func(
            query.transpose(1, 2).squeeze(0).contiguous(),
            key.transpose(1, 2).squeeze(0).contiguous(),
            value.transpose(1, 2).squeeze(0).contiguous(),
            packed_cu_seqlens,
            packed_cu_seqlens,
            packed_max_seqlen,
            packed_max_seqlen,
            dropout_p=dropout,
            softmax_scale=scaling,
            causal=True,
        )
        return output.unsqueeze(0), None

    if attention_mask is not None:
        raise ValueError("Padded Qwen3 attention must be packed before the model forward")
    output = _runtime.flash_attn_func(
        query.transpose(1, 2).contiguous(),
        key.transpose(1, 2).contiguous(),
        value.transpose(1, 2).contiguous(),
        dropout_p=dropout,
        softmax_scale=scaling,
        causal=True,
    )
    return output, None


def trainer_sequence_log_probs(
    model: Any,
    sequences: Any,
    attention_mask: Any,
) -> Optional[Any]:
    """Compute packed selected-token logprobs when the Trainer profile is active.

    The public Actor contract remains padded ``[batch, sequence - 1]``. Internally,
    valid tokens are packed before every model layer, matching VERL's remove-padding
    path and preventing attention from observing artificial padding tokens.
    """
    if _runtime.installed_profile == CONSISTENCY_PROFILE_OFF:
        return None
    if _runtime.installed_profile != QWEN3_ASCEND_FA3_BATCH_INVARIANT_V1:
        raise RuntimeError(f"Unsupported installed Trainer profile {_runtime.installed_profile!r}")
    if sequences.ndim != 2 or tuple(attention_mask.shape) != tuple(sequences.shape):
        raise ValueError("Packed Trainer inputs require aligned two-dimensional sequences and attention_mask")

    valid_mask = attention_mask.bool()
    lengths = valid_mask.sum(dim=-1, dtype=platform.tensor_dtype.int32)
    lengths_cpu = lengths.tolist()
    if any(length < 2 for length in lengths_cpu):
        raise ValueError("Packed Trainer inputs require at least two valid tokens per sequence")
    expected_mask = platform.arange(sequences.shape[1], device=sequences.device).unsqueeze(0) < lengths.unsqueeze(1)
    if not bool((valid_mask == expected_mask).all().item()):
        raise ValueError("Packed Trainer inputs require contiguous right padding")

    flat_indices = valid_mask.reshape(-1).nonzero(as_tuple=False).flatten()
    packed_sequences = sequences.reshape(-1).index_select(0, flat_indices).unsqueeze(0)
    packed_position_ids = platform.cat(
        tuple(platform.arange(length, device=sequences.device) for length in lengths_cpu)
    ).unsqueeze(0)
    boundaries = [0]
    for length in lengths_cpu:
        boundaries.append(boundaries[-1] + length)
    cu_seqlens = platform.tensor(
        boundaries,
        dtype=platform.tensor_dtype.int32,
        device=sequences.device,
    )
    outputs = model(
        input_ids=packed_sequences,
        position_ids=packed_position_ids,
        use_cache=False,
        packed_cu_seqlens=cu_seqlens,
        packed_max_seqlen=max(lengths_cpu),
    )
    logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
    log_probs = logits.float().log_softmax(dim=-1)

    rows = []
    start = 0
    output_length = sequences.shape[1] - 1
    for row, length in enumerate(lengths_cpu):
        end = start + length
        labels = sequences[row, 1:length]
        selected = log_probs[0, start : end - 1].gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        rows.append(platform.cat((selected, selected.new_zeros((output_length - length + 1,)))).unsqueeze(0))
        start = end
    return platform.cat(rows, dim=0)


def _require_package_versions() -> None:
    """Fail closed unless every version-pinned profile dependency is installed."""
    for distribution, expected in _EXPECTED_PACKAGE_VERSIONS.items():
        try:
            installed = package_version(distribution).split("+", maxsplit=1)[0]
        except PackageNotFoundError as error:
            raise ValueError(
                f"Consistency profile requires {distribution}=={expected}, but it is not installed"
            ) from error
        if installed != expected:
            raise ValueError(
                f"Consistency profile requires {distribution}=={expected}, got {installed}"
            )


def validate_rollout_consistency_profile(profile: str) -> None:
    """Fail closed if an isolated rollout process cannot provide its profile."""
    if profile == CONSISTENCY_PROFILE_OFF:
        return
    if profile != QWEN3_ASCEND_FA3_BATCH_INVARIANT_V1:
        raise ValueError(f"Unsupported rollout consistency profile {profile!r}")
    _require_package_versions()
    try:
        # These optional packages are imported only by an explicitly profiled rollout.
        from flash_attn_npu_v3 import flash_attn_with_kvcache  # pylint: disable=C0415
        from vllm_ascend.batch_invariant import (  # pylint: disable=C0415
            HAS_ASCENDC_BATCH_INVARIANT,
        )
    except ImportError as error:
        raise ValueError(f"Rollout consistency profile {profile!r} is unavailable: {error}") from error
    if not callable(flash_attn_with_kvcache):
        raise ValueError("flash-attn-npu does not expose the required FA v3 KV-cache interface")
    if not HAS_ASCENDC_BATCH_INVARIANT:
        raise ValueError("AscendC batch-invariant operators are unavailable in the rollout process")


def install_rollout_consistency_profile(profile: str) -> None:
    """Validate and install model-level patches in an isolated rollout process."""
    validate_rollout_consistency_profile(profile)
    if profile == CONSISTENCY_PROFILE_OFF:
        if _runtime.installed_rollout_profile != CONSISTENCY_PROFILE_OFF:
            raise ValueError(
                "Cannot disable process-global rollout consistency profile "
                f"{_runtime.installed_rollout_profile!r} after installation"
            )
        return
    if _runtime.installed_rollout_profile == profile:
        return
    if _runtime.installed_rollout_profile != CONSISTENCY_PROFILE_OFF:
        raise ValueError(
            "Cannot replace process-global rollout consistency profile "
            f"{_runtime.installed_rollout_profile!r} with {profile!r}"
        )
    # Delay this import to avoid the rl.roles package cycle during facade initialization.
    from rl.roles.weight_sync.vllm_worker import (  # pylint: disable=C0415
        install_vllm_ascend_partial_prefill_rng_fix,
    )
    install_vllm_ascend_partial_prefill_rng_fix()
    _install_qwen3_npu_rms_norm()
    _runtime.installed_rollout_profile = profile


def install_trainer_consistency_profile(config: Mapping[str, Any]) -> None:
    """Install the selected process-global Trainer numerical profile."""
    profile = consistency_profile(config)
    if profile == CONSISTENCY_PROFILE_OFF:
        if _runtime.installed_profile != CONSISTENCY_PROFILE_OFF:
            raise ValueError(
                "Cannot disable process-global consistency profile "
                f"{_runtime.installed_profile!r} after installation"
            )
        return
    if _runtime.installed_profile == profile:
        return
    if _runtime.installed_profile != CONSISTENCY_PROFILE_OFF:
        raise ValueError(
            "Cannot replace process-global consistency profile "
            f"{_runtime.installed_profile!r} with {profile!r}"
        )
    if platform.platform_type != PlatformType.PYTORCH or platform.device_type() != "npu":
        raise ValueError(f"Consistency profile {profile!r} requires the Torch NPU platform")

    _require_package_versions()
    try:
        # These optional NPU packages must remain unloaded when the profile is off.
        from flash_attn_npu import (  # pylint: disable=C0415
            flash_attn_func,
            flash_attn_varlen_func,
        )
        from flash_attn_npu_v3 import flash_attn_with_kvcache  # pylint: disable=C0415
        from vllm_ascend.batch_invariant import (  # pylint: disable=C0415
            HAS_ASCENDC_BATCH_INVARIANT,
            enable_batch_invariant_mode,
        )
    except ImportError as error:
        raise ValueError(f"Consistency profile {profile!r} dependencies are unavailable: {error}") from error
    if not callable(flash_attn_func):
        raise ValueError("flash-attn-npu does not expose the required trainable FA v2 interface")
    if not callable(flash_attn_varlen_func):
        raise ValueError("flash-attn-npu does not expose the required trainable FA v2 varlen interface")
    if not callable(flash_attn_with_kvcache):
        raise ValueError("flash-attn-npu does not expose the required FA v3 KV-cache interface")
    if not HAS_ASCENDC_BATCH_INVARIANT:
        raise ValueError("AscendC batch-invariant operators are unavailable")

    _runtime.flash_attn_func = flash_attn_func
    _runtime.flash_attn_varlen_func = flash_attn_varlen_func
    ALL_ATTENTION_FUNCTIONS.register(
        _TRAINER_ATTENTION_IMPLEMENTATION,
        _flash_attn_npu_attention_forward,
    )
    ALL_MASK_ATTENTION_FUNCTIONS.register(
        _TRAINER_ATTENTION_IMPLEMENTATION,
        ALL_MASK_ATTENTION_FUNCTIONS["flash_attention_2"],
    )
    os.environ["HCCL_DETERMINISTIC"] = "strict"
    os.environ["LCCL_DETERMINISTIC"] = "1"
    enable_batch_invariant_mode()
    _install_qwen3_npu_rms_norm()
    _runtime.installed_profile = profile


__all__ = [
    "CONSISTENCY_PROFILE_OFF",
    "QWEN3_ASCEND_FA3_BATCH_INVARIANT_V1",
    "configure_consistency_profile",
    "consistency_profile",
    "install_rollout_consistency_profile",
    "install_trainer_consistency_profile",
    "trainer_sequence_log_probs",
    "validate_consistency_model_identity",
    "validate_rollout_consistency_profile",
]
