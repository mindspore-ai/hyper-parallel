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
"""HyperAutoModel classes — HF-compatible model interface.

Following design doc 01_hf_compatibility_layer.md §6.
Stub — provides from_pretrained/from_config as entry points.
"""

import logging
from typing import Any, Literal, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSequenceClassification,
    PretrainedConfig,
    PreTrainedModel,
)

# ``AutoModelForMultimodalLM`` was introduced in transformers 5.x. On older
# installs (e.g. 4.57.x, matching the Qwen3-Omni checkpoint's own
# ``transformers_version``), the omni architectures map instead under
# ``AutoModelForTextToWaveform`` (MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES).
# Pick whichever the installed transformers exposes so
# ``HyperAutoModelForMultimodalLM`` resolves a valid HF parent on both.
try:  # pylint: disable=import-outside-toplevel
    from transformers import AutoModelForMultimodalLM as _OMNI_AUTO_BASE
except ImportError:  # pragma: no cover - depends on installed transformers version
    from transformers import AutoModelForTextToWaveform as _OMNI_AUTO_BASE  # pylint: disable=import-outside-toplevel
AutoModelForMultimodalLM = _OMNI_AUTO_BASE

from hyper_parallel.components.checkpoint.huggingface_checkpointer import resolve_hf_loader
from hyper_parallel.components.checkpoint.weight_verifier import verify_hf_weights as check_hf_weights
from hyper_parallel.models._transformers.model_builder import (
    _init_model,
    apply_model_infrastructure,
    instantiate_infrastructure,
)
from hyper_parallel.models._transformers.config_resolver import get_hf_config, get_is_hf_model
from hyper_parallel.models.module_build import (
    attach_build_context, build_module_context, construct_family_model, resolve_effective_model_spec,
)
from hyper_parallel.distributed.mesh import DistributedSetup
from hyper_parallel.models.build_options import (  # pylint: disable=syntax-error
    CompileConfig,
    get_device_id,
    get_device_type,
)
from hyper_parallel.components.checkpoint.quantized_hf_storage import supports_hf_dequantization

logger = logging.getLogger(__name__)

def _current_device() -> torch.device:
    """Return the current accelerator device, or CPU when no accelerator exists."""
    device_type = get_device_type()
    if device_type == "cpu":
        return torch.device("cpu")
    return torch.device(device_type, get_device_id())


_CONFIG_ONLY_KWARGS = {
    "cache_dir",
    "config_path",
    "force_download",
    "ignore_mismatched_sizes",
    "local_files_only",
    "proxies",
    "resume_download",
    "revision",
    "subfolder",
    "token",
    "trust_remote_code",
    "use_auth_token",
}


def _model_init_kwargs(
    kwargs: dict[str, Any],
    *,
    load_base_model: bool,
) -> dict[str, Any]:
    """Drop config/download-only kwargs when constructing directly from config."""
    if load_base_model:
        return dict(kwargs)
    return {
        key: value
        for key, value in kwargs.items()
        if key not in _CONFIG_ONLY_KWARGS
    }


def _requires_quantized_initialization(
    hf_config: Any,
    explicit_quantization: Any,
    *,
    load_base_model: bool,
) -> bool:
    """Distinguish quantized model construction from DCP decoding into float parameters."""
    stored_quantization = getattr(hf_config, "quantization_config", None) if load_base_model else None
    quantized_init = explicit_quantization is not None or stored_quantization is not None
    if load_base_model and quantized_init and resolve_hf_loader() == "dcp":
        if explicit_quantization is not None or not supports_hf_dequantization(stored_quantization):
            raise ValueError(
                "DCP pretrained loading does not support quantized initialization; "
                "checkpoint decoding requires a supported format in the checkpoint's own config"
            )
        # Storage decoding produces floating-point parameters, not quantized model modules.
        return False
    return quantized_init


def _model_init_context(is_meta_device: bool) -> Any:
    """Resolve version-dependent initialization helpers when constructing a model."""
    # Preserve deferred imports for Transformers versions that moved no_init_weights.
    # pylint: disable=import-outside-toplevel
    from contextlib import nullcontext
    from transformers.modeling_utils import ContextManagers
    try:
        from transformers.modeling_utils import no_init_weights
    except ImportError:
        from transformers.initialization import no_init_weights
    from hyper_parallel import init_empty_weights
    # pylint: enable=import-outside-toplevel

    return ContextManagers([no_init_weights(), init_empty_weights()]) if is_meta_device else nullcontext()


def _load_pretrained_config(
        pretrained_model_name_or_path: str,
        attn_implementation: str,
        torch_dtype: Union[str, torch.dtype],
        config_overrides: Optional[dict[str, Any]],
        kwargs: dict[str, Any],
) -> PretrainedConfig:
    """Load the HF configuration with call-time overrides."""
    config_kwargs = {name: value for name, value in kwargs.items()
                     if name not in {"model_factory", "model_spec"}}
    config_kwargs.update(config_overrides or {})
    return get_hf_config(pretrained_model_name_or_path, attn_implementation, torch_dtype, **config_kwargs)


def _validate_weight_check_options(verify_hf_weights: Optional[str], load_base_model: bool) -> None:
    """Require pretrained loading when verification is requested."""
    if verify_hf_weights is not None and not load_base_model:
        raise ValueError("verify_hf_weights requires load_base_model=True")


def _prepare_pretrained_build(
        distributed_setup: DistributedSetup,
        compile_config: Optional[Union[CompileConfig, dict]],
        pretrained_model_name_or_path: str,
        attn_implementation: str,
        torch_dtype: Union[str, torch.dtype],
        config_overrides: Optional[dict[str, Any]],
        kwargs: dict[str, Any],
        force_hf: bool,
) -> tuple[Any, Any, PretrainedConfig, bool]:
    """Prepare infrastructure, HF configuration and model-path selection."""
    # ② Instantiate infrastructure
    sharding_planner, fsdp2_manager = instantiate_infrastructure(
        distributed_setup=distributed_setup,
        device=_current_device(),
        compile_config=compile_config,
    )

    # ③ Get HF config
    hf_config = _load_pretrained_config(
        pretrained_model_name_or_path, attn_implementation, torch_dtype, config_overrides, kwargs,
    )
    # ④ Determine model path
    is_hf_model = get_is_hf_model(hf_config, force_hf)

    return sharding_planner, fsdp2_manager, hf_config, is_hf_model


def _resolve_module_construction(kwargs, hf_config, distributed_setup, dtype):
    """Consume model construction arguments and create one shared build context."""
    factory = kwargs.pop("model_factory", None)
    spec = resolve_effective_model_spec(hf_config, kwargs.pop("model_spec", None))
    context = build_module_context(
        distributed_setup, device=_current_device(), dtype=dtype, model_spec=spec,
    )
    return factory, context


class _BaseHyperAutoModelClass:
    """Shared from_pretrained / from_config logic.

    Following design doc 01 §6.1-6.2.
    """

    @classmethod
    def _from_pretrained_parent_class(cls, *args, **kwargs):
        """Delegate to the parent HuggingFace AutoModel class.

        Used by the HF-native path in _init_model so that the model is loaded
        through the standard transformers checkpoint logic.
        """
        # Mixin: `super()` resolves to the HF AutoModel base at runtime (see
        # HyperAutoModelForCausalLM etc. below), which provides from_pretrained.
        return super().from_pretrained(*args, **kwargs)  # pylint: disable=E1101

    @classmethod
    def _from_config_parent_class(cls, *args, **kwargs):
        """Delegate to the parent Hugging Face AutoModel ``from_config`` path."""
        parent_classes = (
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoModelForSequenceClassification,
        )
        for parent_class in parent_classes:
            if issubclass(cls, parent_class):
                return parent_class.from_config(*args, **kwargs)
        raise TypeError(f"{cls.__name__} has no supported Hugging Face AutoModel parent")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        distributed_setup: Optional[DistributedSetup] = None,
        peft_config: Optional[Any] = None,
        torch_dtype: Union[str, torch.dtype] = "auto",
        attn_implementation: str = "sdpa",
        force_hf: bool = False,
        validate_placement: bool = False,
        qat_config: Optional[Any] = None,
        fp8_config: Optional[Any] = None,
        compile_config: Optional[Union[CompileConfig, dict]] = None,
        freeze_config: Optional[Any] = None,
        activation_checkpoint: Optional[str] = None,
        swap_inputs: bool = False,
        allow_cache_entry_mutation: bool = False,
        activation_swap: str = "none",
        model_init_dtype: Optional[Literal["float16", "bfloat16", "float32"]] = None,
        config_overrides: Optional[dict[str, Any]] = None,
        load_base_model: bool = True,
        verify_hf_weights: Optional[str] = None,
        **kwargs: Any,
    ) -> PreTrainedModel:
        """HF-compatible from_pretrained entry point.

        Unquantized models are constructed from configuration on meta, then
        the selected HF loader loads their finalized parameters once.

        Args:
            pretrained_model_name_or_path: HF checkpoint path or model identifier.
            *model_args: Positional arguments for model construction.
            verify_hf_weights: Optional report directory for checking all loaded shards
                against a local HF checkpoint before the first forward. All ranks must opt in.
            **kwargs: Additional model construction and HF configuration arguments.

        Returns:
            Constructed and parallelized pretrained model.
        Following design doc 01 §6.2:
        ① resolve distributed_setup → MeshContext
        ② instantiate_infrastructure → ShardingPlanner + FSDP2Manager
        ③ AutoConfig.from_pretrained → hf_config
        ④ get_is_hf_model → custom/HF path
        ⑤ _build_model → meta + shard + load
        """
        _validate_weight_check_options(verify_hf_weights, load_base_model)
        if distributed_setup is None:
            distributed_setup = DistributedSetup()
        mesh = distributed_setup.mesh_context
        prepared = _prepare_pretrained_build(
            distributed_setup,
            compile_config,
            pretrained_model_name_or_path,
            attn_implementation,
            torch_dtype,
            config_overrides,
            kwargs,
            force_hf,
        )
        sharding_planner, fsdp2_manager, hf_config, is_hf_model = prepared
        # ⑤ Build model
        model = cls._build_model(
            pretrained_model_name_or_path,
            *model_args,
            is_hf_model=is_hf_model,
            hf_config=hf_config,
            mesh=mesh,
            sharding_planner=sharding_planner,
            fsdp2_manager=fsdp2_manager,
            peft_config=peft_config,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            validate_placement=validate_placement,
            load_base_model=load_base_model,
            distributed_setup=distributed_setup,
            qat_config=qat_config,
            fp8_config=fp8_config,
            compile_config=compile_config,
            freeze_config=freeze_config,
            activation_checkpoint=activation_checkpoint,
            swap_inputs=swap_inputs,
            allow_cache_entry_mutation=allow_cache_entry_mutation,
            activation_swap=activation_swap,
            model_init_dtype=model_init_dtype,
            **kwargs,
        )
        if verify_hf_weights is not None:
            check_hf_weights(model, pretrained_model_name_or_path, verify_hf_weights,
                             phase="after_load_before_first_forward")
        return model

    @classmethod
    def from_config(  # pylint: disable=unused-argument
        cls,
        config: PretrainedConfig,
        *model_args: Any,
        distributed_setup: Optional[DistributedSetup] = None,
        device_mesh: Optional[Any] = None,
        peft_config: Optional[Any] = None,
        torch_dtype: Union[str, torch.dtype] = "auto",
        attn_implementation: str = "sdpa",
        validate_placement: bool = False,
        qat_config: Optional[Any] = None,
        fp8_config: Optional[Any] = None,
        compile_config: Optional[Union[CompileConfig, dict]] = None,
        freeze_config: Optional[Any] = None,
        activation_checkpoint: Optional[str] = None,
        swap_inputs: bool = False,
        allow_cache_entry_mutation: bool = False,
        activation_swap: str = "none",
        model_init_dtype: Optional[Literal["float16", "bfloat16", "float32"]] = None,
        **kwargs: Any,
    ) -> PreTrainedModel:
        """Build model from PretrainedConfig (no weight loading).

        Following design doc 01 §6.1.

        No checkpoint is loaded, so the from-config random initialization path
        is preserved.
        """
        if distributed_setup is None:
            distributed_setup = DistributedSetup()
        mesh = distributed_setup.mesh_context

        sharding_planner, fsdp2_manager = instantiate_infrastructure(
            distributed_setup=distributed_setup,
            device=_current_device(),
            compile_config=compile_config,
        )
        is_hf_model = get_is_hf_model(config, force_hf=False)

        return cls._build_model(
            None,
            *model_args,
            is_hf_model=is_hf_model,
            hf_config=config,
            mesh=mesh,
            sharding_planner=sharding_planner,
            fsdp2_manager=fsdp2_manager,
            peft_config=peft_config,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            validate_placement=validate_placement,
            load_base_model=False,
            distributed_setup=distributed_setup,
            qat_config=qat_config,
            fp8_config=fp8_config,
            compile_config=compile_config,
            freeze_config=freeze_config,
            activation_checkpoint=activation_checkpoint,
            swap_inputs=swap_inputs,
            allow_cache_entry_mutation=allow_cache_entry_mutation,
            activation_swap=activation_swap,
            model_init_dtype=model_init_dtype,
            **kwargs,
        )

    @classmethod
    def _build_model(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        is_hf_model,
        hf_config,
        mesh,
        sharding_planner,
        fsdp2_manager,
        peft_config,
        torch_dtype,
        attn_implementation,
        validate_placement,
        load_base_model,
        distributed_setup=None,
        qat_config=None,
        fp8_config=None,
        compile_config=None,
        freeze_config=None,
        activation_checkpoint: Optional[str] = None,
        activation_checkpoint_selection: Optional[Any] = None,
        swap_inputs: bool = False,
        allow_cache_entry_mutation: bool = False,
        activation_swap: str = "none",
        model_init_dtype: Optional[Literal["float16", "bfloat16", "float32"]] = None,
        **kwargs,
    ) -> PreTrainedModel:
        """Core model building orchestration.

        Following design doc 01 §6.3:
        Step 1: Determine meta device
        Step 2: Build model (meta or real device)
        Step 3-12: apply_model_infrastructure (PEFT, QAT, ShardingPlan,
        activation checkpoint, FSDP2, load, layer compile)
        """
        # Step 1: Determine meta device (inline world-size probe: auto_models
        # must not import the trainer runtime).
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        quantized_init = _requires_quantized_initialization(
            hf_config, kwargs.get("quantization_config"), load_base_model=load_base_model,
        )
        defer_hf_load = load_base_model and not quantized_init
        is_meta_device = (
            world_size > 1 or not is_hf_model or defer_hf_load
        ) and not quantized_init

        model_factory, module_context = _resolve_module_construction(
            kwargs, hf_config, distributed_setup, model_init_dtype or torch_dtype,
        )
        init_ctx = _model_init_context(is_meta_device)

        # Step 2: Config-first construction defers pretrained I/O until after sharding.
        init_pretrained_path = (
            pretrained_model_name_or_path if load_base_model and not defer_hf_load else None
        )
        init_kwargs = _model_init_kwargs(
            dict(kwargs),
            load_base_model=init_pretrained_path is not None,
        )
        with init_ctx:
            if model_factory is not None:
                if model_args:
                    raise ValueError("model_factory uses config and context, not additional positional model arguments")
                if quantized_init:
                    raise ValueError("model_factory requires config-first floating-point initialization")
                model = construct_family_model(model_factory, hf_config, context=module_context)
            else:
                _, model = _init_model(
                    cls, init_pretrained_path, hf_config, attn_implementation,
                    torch_dtype, is_hf_model, *model_args, **init_kwargs,
                )
                attach_build_context(model, module_context)

        # Step 3-12: Apply infrastructure
        model = apply_model_infrastructure(
            model,
            mesh=mesh,
            sharding_planner=sharding_planner,
            fsdp2_manager=fsdp2_manager,
            peft_config=peft_config,
            qat_config=qat_config,
            fp8_config=fp8_config,
            freeze_config=freeze_config,
            compile_config=compile_config,
            is_meta_device=is_meta_device,
            is_hf_model=is_hf_model,
            device=_current_device(),
            load_base_model=load_base_model,
            pretrained_path=pretrained_model_name_or_path,
            validate_placement=validate_placement,
            distributed_setup=distributed_setup,
            activation_checkpoint=activation_checkpoint,
            activation_checkpoint_selection=activation_checkpoint_selection,
            swap_inputs=swap_inputs,
            allow_cache_entry_mutation=allow_cache_entry_mutation,
            activation_swap=activation_swap,
            model_init_dtype=model_init_dtype,
        )

        model.train()
        return model


class HyperAutoModelForCausalLM(_BaseHyperAutoModelClass, AutoModelForCausalLM):
    """Hyper-Parallel CausalLM — equivalent to AutoModelForCausalLM.

    Following design doc 01 §6.1.
    """


class HyperAutoModelForImageTextToText(_BaseHyperAutoModelClass, AutoModelForImageTextToText):
    """Hyper-Parallel VLM — equivalent to AutoModelForImageTextToText."""


class HyperAutoModelForSequenceClassification(_BaseHyperAutoModelClass, AutoModelForSequenceClassification):
    """Hyper-Parallel SequenceClassification."""


class HyperAutoModelForMultimodalLM(_BaseHyperAutoModelClass, AutoModelForMultimodalLM):
    """Hyper-Parallel omni model — equivalent to AutoModelForMultimodalLM.

    Covers architectures registered in Transformers' multimodal-LM mapping that
    the image-text-to-text / causal-LM auto classes do not cover, such as
    ``Qwen3OmniMoeForConditionalGeneration`` (model_type ``qwen3_omni_moe``).
    The HF-native path (``force_hf=True``) loads the stock Transformers model.

    Some omni ``*ForConditionalGeneration`` wrappers (Qwen2.5-Omni, Qwen3-Omni)
    are generation-only: they expose ``generate`` but no training ``forward`` —
    the trainable module is the ``thinker`` submodule, which computes the LM loss
    from ``labels``. When ``enable_audio_output=false`` the talker/code2wav are
    not built, so the wrapper holds only the thinker. We add a thin ``forward``
    that delegates to ``self.thinker`` so the standard trainer call
    ``model(**batch)`` returns ``outputs.loss`` (thinker-only training).
    """

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> PreTrainedModel:
        """Patch omni thinker-only forward, then load via the shared path."""
        _ensure_omni_thinker_forward()
        # HF 4.57.1 stock-bug fixes (details in each _ensure_ docstring):
        # packed-audio dead branch / Conv3d-backward NPU segfault / deepstack
        # 3-D-mask joint-branch crash.
        _ensure_omni_packed_audio_input()
        _ensure_omni_patch_embed_linear()
        _ensure_omni_deepstack_2d_masks()
        # Optional: fused NPU MoE kernels, opt-in via HYPER_OMNI_MOE_FUSED_NPU=1;
        # stock forward otherwise.
        _ensure_omni_moe_fused_npu()
        return super().from_pretrained(*args, **kwargs)


def _ensure_omni_moe_fused_npu() -> None:
    """Patch the omni sparse-MoE to NPU fused kernels when opted in."""
    from hyper_parallel.models._transformers.omni_moe_fused_npu import (  # pylint: disable=import-outside-toplevel
        _ensure_omni_moe_fused_npu as _ensure_impl,
    )

    _ensure_impl()


def _ensure_omni_patch_embed_linear() -> None:
    """Route the omni vision ``patch_embed`` Conv3d through an equivalent Linear.

    CANN 8.5 (torch_npu 2.6) segfaults in ``aclnnConvolutionBackwardGetWorkspaceSize``
    for exactly the Qwen3-Omni patch-embed Conv3d configuration
    (in=3, out=1152, kernel/stride=(2,16,16), input (N,3,2,16,16), bf16) — a
    minimal Conv3d backward segfaults in libopapi_nn.so. Because the patch-embed
    kernel exactly covers the whole receptive field (kernel == input spatial/\
    temporal extent), the Conv3d is numerically identical to a Linear over the
    flattened feature map. Routing it through ``F.linear`` with a view of the
    conv weight: (a) produces bit-identical forward output, (b) backprops a
    gradient into the very same ``conv.weight`` storage via the contiguous view,
    and (c) avoids the broken conv backward path entirely. Applied only to the
    omni vision patch embed; idempotent.
    """
    try:
        from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as _M  # pylint: disable=import-outside-toplevel
    except ImportError:
        return
    name = "Qwen3OmniMoeVisionPatchEmbed"
    patch_embed_cls = getattr(_M, name, None)
    if patch_embed_cls is None:
        return
    if getattr(patch_embed_cls, "_hyper_parallel_linear_patch", False):
        return
    try:
        from torch.nn import functional as _F  # pylint: disable=import-outside-toplevel
    except ImportError:  # pragma: no cover
        return

    original_forward = patch_embed_cls.forward

    def linear_forward(self, hidden_states):
        # Same reshape the original forward performs before the Conv3d.
        x = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        kernel = self.proj.kernel_size
        stride = self.proj.stride
        if tuple(stride) == tuple(kernel) and tuple(x.shape[-3:]) == tuple(kernel):
            B = x.shape[0]
            w = self.proj.weight.view(self.proj.weight.shape[0], -1)
            return _F.linear(x.view(B, -1), w, self.proj.bias).view(B, -1)
        # Fall back to the stock Conv3d for any other receptive field.
        return original_forward(self, hidden_states)

    patch_embed_cls.forward = linear_forward
    patch_embed_cls._hyper_parallel_linear_patch = True
    logger.info("Patched %s.forward -> equivalent Linear (CANN conv3d-backward workaround).", name)


def _ensure_omni_packed_audio_input() -> None:
    """Make packed audio input reach the omni audio tower.

    The audio tower consumes packed ``(T_total, 128)`` mel features plus
    per-audio ``feature_lens`` — the layout the data pipeline emits. Stock
    4.57.1 ``get_audio_features`` discards the caller's
    ``audio_feature_lengths`` whenever ``feature_attention_mask`` is None,
    so the packed layout can never reach the tower (and both being None
    crashes with AttributeError). The replacement accepts both layouts:
    batched-padded ``(B, 128, T)`` (select valid frames, transpose) and
    packed ``(T_total, 128)`` (transpose only), and casts to the model dtype
    before the convs (fp32 mel vs bf16 conv weights segfaults on NPU).
    Guarded on the buggy stock pattern so an already-fixed modeling is left
    untouched; idempotent.
    """
    import inspect  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel

    # The buggy stock pattern: the else branch nulls the caller's
    # audio_feature_lengths whenever feature_attention_mask is None.
    buggy = (
        "        if feature_attention_mask is not None:\n"
        "            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)\n"
        "        else:\n"
        "            audio_feature_lengths = None\n"
    )
    targets = []
    for module_name, class_name in (
        ("transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe", "Qwen3OmniMoeThinkerForConditionalGeneration"),
        ("transformers.models.qwen2_5_omni.modeling_qwen2_5_omni", "Qwen2_5OmniThinkerForConditionalGeneration"),
    ):
        try:
            module = __import__(module_name, fromlist=[class_name])
            targets.append(getattr(module, class_name))
        except (ImportError, AttributeError):
            continue
    for thinker_cls in targets:
        if getattr(thinker_cls, "_hyper_parallel_packed_audio", False):
            continue
        # Only replace the broken stock body; a fixed upstream modeling that
        # already forwards audio_feature_lengths is left untouched.
        if buggy not in inspect.getsource(thinker_cls.get_audio_features):
            logger.warning(
                "%s.get_audio_features: packed-audio nulling pattern not found — skipping fix (non-fatal).",
                thinker_cls.__name__,
            )
            continue

        def _get_audio_features(
            self,
            input_features,
            feature_attention_mask=None,
            audio_feature_lengths=None,
        ):
            if feature_attention_mask is not None:
                # Batched padded layout (B, 128, T): select valid frames and
                # transpose to the tower's channel-first (128, T_total).
                audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
                input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
            elif input_features.dim() == 2 and input_features.shape[-1] == 128:
                # Packed layout (T_total, 128) from the VLM collator: transpose
                # to the tower's channel-first (128, T_total).
                input_features = input_features.t()
            if audio_feature_lengths is None:
                raise ValueError(
                    "packed audio requires audio_feature_lengths (or feature_attention_mask)"
                )
            # fp32 mel features vs bf16 conv weights segfault the kernel on
            # NPU, so cast first. qwen3_omni_moe's tower exposes ``conv2d1``;
            # qwen2_5_omni's uses ``conv1`` (1-D), so only the qwen3_omni_moe
            # target reaches this cast with audio flowing.
            input_features = input_features.to(self.audio_tower.conv2d1.weight.dtype)
            audio_outputs = self.audio_tower(input_features, feature_lens=audio_feature_lengths)
            return audio_outputs.last_hidden_state

        thinker_cls.get_audio_features = _get_audio_features
        thinker_cls._hyper_parallel_packed_audio = True
        logger.info(
            "Patched %s.get_audio_features -> packed (T_total,128)+feature_lens.",
            thinker_cls.__name__,
        )


def _ensure_omni_deepstack_2d_masks() -> None:
    """Fix the image+video deepstack joint-branch mask-shape crash.

    Before the deepstack scatter, ``forward`` expands the placeholder masks
    to 3-D via ``get_placeholder_mask``. The image+video joint branch then
    boolean-indexes with those 3-D masks, producing ``S*H`` rows where the
    per-modality feature tensors have ``S`` rows — ``RuntimeError: shape
    mismatch``. The convention downstream (``_deepstack_process`` squeezes
    with ``[..., 0]``) is 2-D token-dim masks. The fix squeezes the joint
    masks to the token dim inside the joint branch and makes the downstream
    squeeze conditional. Applied as a targeted source rewrite of the
    thinker ``forward`` (guarded on the buggy pattern being present and not
    already patched); idempotent.
    """
    import inspect  # pylint: disable=import-outside-toplevel
    import textwrap  # pylint: disable=import-outside-toplevel

    try:
        from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as _M  # pylint: disable=import-outside-toplevel
    except ImportError:
        return
    thinker_cls = getattr(_M, "Qwen3OmniMoeThinkerForConditionalGeneration", None)
    text_model_cls = getattr(_M, "Qwen3OmniMoeThinkerTextModel", None)
    if thinker_cls is None:
        return
    if getattr(thinker_cls, "_hyper_parallel_deepstack_2d_masks", False):
        return

    # --- fix 1: joint branch inside thinker.forward (source rewrite) ---
    source = inspect.getsource(thinker_cls.forward)
    # ``inspect.getsource`` includes BOTH decorators (``@can_return_tuple``
    # and ``@auto_docstring``); dedent then re-exec would re-run them on a
    # bare function and crash (auto_class_docstring expects a class) — strip
    # down to the ``def``. Training always runs with the default
    # ``return_dict=True``, so dropping ``can_return_tuple`` is safe here.
    source = source[source.index("def forward") :]
    buggy = (
        "visual_pos_masks = video_mask | image_mask\n"
        "                visual_embeds_multiscale_joint = ()\n"
        "                image_mask_joint = image_mask[visual_pos_masks]\n"
        "                video_mask_joint = video_mask[visual_pos_masks]"
    )
    fixed = (
        "visual_pos_masks = (video_mask | image_mask)[..., 0]\n"
        "                # masks are 3-D (expand_as) here; squeeze to token dim\n"
        "                visual_embeds_multiscale_joint = ()\n"
        "                image_mask_joint = image_mask[..., 0][visual_pos_masks]\n"
        "                video_mask_joint = video_mask[..., 0][visual_pos_masks]"
    )
    if buggy not in source:
        logger.warning(
            "deepstack joint-branch pattern not found — skipping fix (non-fatal)."
        )
        return
    source = source.replace(buggy, fixed)
    namespace = dict(_M.__dict__)
    exec(compile(textwrap.dedent(source), "<omni_deepstack_fix>", "exec"), namespace)  # noqa: S102
    thinker_cls.forward = namespace["forward"]
    thinker_cls._hyper_parallel_deepstack_2d_masks = True

    # --- fix 2: downstream squeeze must tolerate 2-D masks ---
    if text_model_cls is not None and not getattr(text_model_cls, "_hyper_parallel_deepstack_2d", False):
        def _deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds):
            if visual_pos_masks.ndim == 3:
                visual_pos_masks = visual_pos_masks[..., 0]
            visual_pos_masks = visual_pos_masks.to(hidden_states.device)
            local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
            hidden_states[visual_pos_masks, :] = local_this
            return hidden_states

        text_model_cls._deepstack_process = _deepstack_process
        text_model_cls._hyper_parallel_deepstack_2d = True

    logger.info(
        "Patched %s.forward joint deepstack branch + %s._deepstack_process "
        "(3-D mask index crash fix).",
        thinker_cls.__name__,
        text_model_cls.__name__ if text_model_cls else "-",
    )


def _omni_get_compile_layers(self):
    """Declare decoder-layer containers for ``distributed.compile``.

    Qwen3-Omni nests its MoE decoder under ``thinker.model.layers`` (and the
    talker under ``talker.model.layers`` when audio output is enabled), which
    the generic HF-convention paths (``layers``, ``model.layers``) cannot
    reach. This satisfies the model-owned ``get_compile_layers()`` contract.
    """
    layers = []
    for path in ("thinker.model.layers", "talker.model.layers"):
        container = self
        for part in path.split("."):
            container = getattr(container, part, None)
            if container is None:
                break
        if container is not None:
            layers.extend(container)
    return layers


def _ensure_omni_thinker_forward() -> None:
    """Give generation-only omni wrappers a training ``forward`` -> thinker.

    Idempotent and defensive: only classes that exist in the installed
    Transformers and that do not already define their own ``forward`` are
    patched. The delegating forward filters keyword arguments to those the
    thinker's ``forward`` accepts, which drops shared-VLM-pipeline fields the
    omni thinker does not consume (e.g. ``mm_token_type_ids`` — omni derives
    multimodal RoPE from ``input_ids`` directly).
    """
    import inspect  # pylint: disable=import-outside-toplevel

    def _make_thinker_forward(param_names, accepts_var_kw):
        def thinker_forward(self, **kwargs):
            if not accepts_var_kw:
                kwargs = {k: v for k, v in kwargs.items() if k in param_names}
            else:
                # Thinker takes **kwargs; still drop fields the omni stack never
                # consumes so they do not propagate into text/vision submodules.
                kwargs.pop("mm_token_type_ids", None)
            # The VLM collator produces packed (T_total, n_mels) input_features
            # + per-audio audio_feature_lengths, but the stock thinker forward
            # nulls audio_feature_lengths whenever
            # feature_attention_mask is None — get_rope_index then crashes on
            # audio_seqlens[...]. Rebuild the batched padded layout here so the
            # forward computes audio_feature_lengths itself; the patched
            # get_audio_features re-packs for the tower (same sample order).
            packed = kwargs.get("input_features")
            lens = kwargs.get("audio_feature_lengths")
            if (
                packed is not None
                and lens is not None
                and kwargs.get("feature_attention_mask") is None
                and packed.dim() == 2
            ):
                import torch  # pylint: disable=import-outside-toplevel

                lens = torch.as_tensor(lens)
                n_mels = packed.shape[-1]
                tmax = int(lens.max())
                batched = packed.new_zeros((lens.numel(), n_mels, tmax))
                mask = packed.new_zeros((lens.numel(), tmax), dtype=torch.long)
                ofs = 0
                for i, length in enumerate(lens.tolist()):
                    batched[i, :, :length] = packed[ofs : ofs + length].t()
                    mask[i, :length] = 1
                    ofs += length
                kwargs["input_features"] = batched
                kwargs["feature_attention_mask"] = mask
                kwargs.pop("audio_feature_lengths", None)
            return self.thinker(**kwargs)
        return thinker_forward

    candidates = []
    try:
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (  # pylint: disable=import-outside-toplevel
            Qwen3OmniMoeForConditionalGeneration,
        )
        candidates.append(Qwen3OmniMoeForConditionalGeneration)
    except ImportError:
        pass
    try:
        from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (  # pylint: disable=import-outside-toplevel
            Qwen2_5OmniForConditionalGeneration,
        )
        candidates.append(Qwen2_5OmniForConditionalGeneration)
    except ImportError:
        pass

    for model_cls in candidates:
        # Declare the nested decoder-layer containers for distributed.compile
        # regardless of whether the training-forward patch applies below.
        if "get_compile_layers" not in model_cls.__dict__:
            model_cls.get_compile_layers = _omni_get_compile_layers
        if "forward" in model_cls.__dict__:
            continue  # respects an upstream training forward if one is added later
        # The thinker attribute is instance-level; inspect the thinker *class*
        # forward to learn which kwargs to forward.
        thinker_cls_forward = None
        try:
            from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (  # pylint: disable=import-outside-toplevel
                Qwen3OmniMoeThinkerForConditionalGeneration,
            )
            if model_cls.__name__.startswith("Qwen3"):
                thinker_cls_forward = Qwen3OmniMoeThinkerForConditionalGeneration.forward
        except ImportError:
            pass
        if thinker_cls_forward is None:
            try:
                from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (  # pylint: disable=import-outside-toplevel
                    Qwen2_5OmniThinkerForConditionalGeneration,
                )
                thinker_cls_forward = Qwen2_5OmniThinkerForConditionalGeneration.forward
            except ImportError:
                thinker_cls_forward = None
        if thinker_cls_forward is None:
            continue
        params = inspect.signature(thinker_cls_forward).parameters
        accepts_var_kw = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        param_names = {name for name in params if name != "self"}
        model_cls.forward = _make_thinker_forward(param_names, accepts_var_kw)
        logger.info("Patched %s.forward to delegate to thinker (training path).", model_cls.__name__)


