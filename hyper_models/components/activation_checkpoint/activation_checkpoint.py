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
"""Activation checkpointing helpers for distributed model components."""

import logging
from collections.abc import Callable
from typing import Optional, Union

import torch
import torch.nn as nn

from hyper_parallel.core.activation_checkpoint.activation_checkpoint import (
    CheckpointPolicy,
    checkpoint_wrapper,
)
from hyper_parallel.platform import get_platform

logger = logging.getLogger(__name__)
platform = get_platform()


def _resolve_torch_op(dotted_path: str):
    """Resolve a torch operator to its default overload when available."""
    if dotted_path.startswith("ops."):
        dotted_path = dotted_path[4:]
    if dotted_path.count(".") == 1:
        dotted_path = f"{dotted_path}.default"
    obj = torch.ops
    try:
        for part in dotted_path.split("."):
            obj = getattr(obj, part)
    except AttributeError:
        return None
    return obj


def _resolve_op_attr(root: object, dotted_path: str):
    """Resolve an optional operator outside the regular ``torch.ops`` tree."""
    obj = root
    try:
        for part in dotted_path.split("."):
            obj = getattr(obj, part)
    except AttributeError:
        return None
    return obj


def _existing_ops(*ops):
    """Return the available operators, omitting optional torch operators."""
    return frozenset(op for op in ops if op is not None)


# Matmul operators alternate between saving and recomputing their outputs. The
# counter is scoped to one checkpoint region by ``make_selective_checkpoint_context_fn``.
_SELECTIVE_AC_MATMUL_OPS = _existing_ops(
    _resolve_torch_op("aten.matmul"),
    _resolve_torch_op("aten.mm"),
    _resolve_torch_op("aten.linear"),
    _resolve_torch_op("aten._grouped_mm"),
    _resolve_torch_op("aten._scaled_grouped_mm"),
)

# Some model implementations mutate these operator outputs in-place. Keeping
# the output by reference would either trip SAC's version check or replay the
# mutation on an already-mutated tensor during recomputation.
_SELECTIVE_AC_FORCE_RECOMPUTE_OPS = _existing_ops(
    _resolve_torch_op("aten.topk"),
)


def _default_compute_intensive_ops() -> tuple:
    """Get PyTorch's compute-intensive operator list when the private API exists."""
    try:
        # This private PyTorch API is unavailable on some supported versions.
        from torch._functorch.partitioners import get_default_op_list

        return tuple(op.default for op in get_default_op_list().compute_intensive_ops)
    except (ImportError, AttributeError, RuntimeError):
        return ()


def _ffpa_forward_ops() -> tuple:
    """Resolve optional FFPA forward operators after their extension registers."""
    try:
        # The optional extension registers its operators only when imported.
        import ffpa_attn.cute  # pylint: disable=C0415, W0611
    except (ImportError, OSError, RuntimeError):
        return ()
    return (
        _resolve_op_attr(torch.ops, "ffpa_attn._fwd_cute.default"),
        _resolve_op_attr(torch.ops, "ffpa_attn._varlen_fwd_cute.default"),
    )


def _build_selective_ac_must_save_ops():
    """Build the expensive/communication operator set for selective AC."""
    save_ops = set(_default_compute_intensive_ops())
    compute_ops = _existing_ops(
        *(
            _resolve_torch_op(name)
            for name in (
                "aten.mm",
                "aten.addmm",
                "aten.bmm",
                "aten.linear",
                "aten._scaled_mm",
                "aten._scaled_dot_product_cudnn_attention",
                "aten._scaled_dot_product_efficient_attention",
                "aten._scaled_dot_product_flash_attention",
                "aten._scaled_dot_product_flash_attention_for_cpu",
                "aten._scaled_dot_product_fused_attention_overrideable",
                "aten.scaled_dot_product_attention",
                "npu.npu_fusion_attention_v3",
                "aten._flex_attention",
                "aten.topk",
                "aten.max",
            )
        ),
        _resolve_op_attr(torch, "_higher_order_ops.flex_attention"),
        _resolve_op_attr(torch, "_higher_order_ops.inductor_compiled_code"),
        _resolve_op_attr(torch.ops, "torch_attn._varlen_attn.default"),
        *_ffpa_forward_ops(),
    )
    comm_ops = _existing_ops(
        *(
            _resolve_torch_op(name)
            for name in (
                "aten.all_to_all_single",
                "aten.reduce_scatter_tensor",
                "_c10d_functional.all_to_all_single",
                "_c10d_functional.reduce_scatter_tensor",
                "c10d.allreduce_",
            )
        ),
        _resolve_op_attr(torch.ops, "deepep.dispatch.default"),
        _resolve_op_attr(torch.ops, "deepep.combine.default"),
        _resolve_op_attr(torch.ops, "hybridep.dispatch.default"),
        _resolve_op_attr(torch.ops, "hybridep.combine.default"),
    )
    save_ops.update(compute_ops)
    save_ops.update(comm_ops)
    save_ops.difference_update(_SELECTIVE_AC_FORCE_RECOMPUTE_OPS)
    return frozenset(save_ops)


_SELECTIVE_AC_MUST_SAVE_OPS = _build_selective_ac_must_save_ops()

_LayerContainer = Union[nn.ModuleList, nn.ModuleDict]
_LayerContainerInfo = tuple[nn.Module, str, _LayerContainer, str]

# Each group contains alternative locations of one transformer layer container
# across transformers versions. The groups intentionally retain model roles so
# callers can choose HF-native checkpointing only for language-only models.
_GEMMA3_LAYER_CONTAINER_PATHS = (
    ("language", ("model.language_model.layers", "language_model.model.layers")),
    (
        "vision",
        (
            "model.vision_tower.vision_model.encoder.layers",
            "model.vision_tower.encoder.layers",
            "vision_tower.vision_model.encoder.layers",
        ),
    ),
)
_QWEN2_VL_LAYER_CONTAINER_PATHS = (
    ("language", ("model.language_model.layers", "model.layers")),
    ("vision", ("model.visual.blocks", "visual.blocks")),
)
_LLAVA_LAYER_CONTAINER_PATHS = (
    ("language", ("model.language_model.layers", "language_model.model.layers")),
    (
        "vision",
        (
            "model.vision_tower.vision_model.encoder.layers",
            "model.vision_tower.encoder.layers",
            "vision_tower.vision_model.encoder.layers",
        ),
    ),
)
_MODEL_LAYER_CONTAINER_PATHS = {
    "Gemma3ForConditionalGeneration": _GEMMA3_LAYER_CONTAINER_PATHS,
    "Qwen2_5_VLForConditionalGeneration": _QWEN2_VL_LAYER_CONTAINER_PATHS,
    "Qwen2VLForConditionalGeneration": _QWEN2_VL_LAYER_CONTAINER_PATHS,
    "SmolVLMForConditionalGeneration": (
        ("language", ("model.text_model.layers",)),
        ("vision", ("model.vision_model.encoder.layers",)),
    ),
    "LlavaForConditionalGeneration": _LLAVA_LAYER_CONTAINER_PATHS,
    "LlavaNextForConditionalGeneration": _LLAVA_LAYER_CONTAINER_PATHS,
    "LlavaNextVideoForConditionalGeneration": _LLAVA_LAYER_CONTAINER_PATHS,
    "LlavaOnevisionForConditionalGeneration": _LLAVA_LAYER_CONTAINER_PATHS,
    "Mistral3ForConditionalGeneration": (
        ("language", ("model.language_model.layers",)),
        (
            "vision",
            (
                "model.vision_tower.encoder.layers",
                "model.vision_tower.vision_model.encoder.layers",
                "model.vision_tower.transformer.layers",
            ),
        ),
    ),
    "Mistral3FP8VLMForConditionalGeneration": (
        ("language", ("model.language_model.layers",)),
        (
            "vision",
            (
                "model.vision_tower.encoder.layers",
                "model.vision_tower.vision_model.encoder.layers",
                "model.vision_tower.transformer.layers",
            ),
        ),
    ),
    "Ministral3BidirectionalModel": (("language", ("layers",)),),
    "LlamaNemotronVLModel": (
        ("language", ("language_model.layers",)),
        ("vision", ("vision_model.vision_model.encoder.layers", "vision_model.encoder.layers")),
    ),
    "Llama4ForConditionalGeneration": (
        ("language", ("language_model.model.layers",)),
        ("vision", ("vision_model.model.layers",)),
    ),
    "Qwen3_5ForConditionalGeneration": (
        ("language", ("model.language_model.layers",)),
        ("vision", ("model.visual.blocks",)),
    ),
    "Qwen3_5MoeForConditionalGeneration": (
        ("language", ("model.language_model.layers",)),
        ("vision", ("model.visual.blocks",)),
    ),
    "Qwen3VLMoeForConditionalGeneration": (
        ("language", ("model.language_model.layers",)),
        ("vision", ("model.visual.blocks",)),
    ),
    "Gemma4ForConditionalGeneration": (("language", ("model.language_model.layers",)),),
    "KimiVLForConditionalGeneration": (
        ("language", ("model.language_model.layers",)),
        ("vision", ("model.vision_tower.encoder.blocks",)),
    ),
    "KimiK25VLForConditionalGeneration": (
        ("language", ("model.language_model.layers",)),
        ("vision", ("model.vision_tower.encoder.blocks",)),
    ),
    "MiniMaxM3SparseForConditionalGeneration": (
        ("language", ("model.layers",)),
        ("vision", ("vision_tower.vision_model.encoder.layers",)),
    ),
    "Step3p7ForConditionalGeneration": (
        ("language", ("model.language_model.layers",)),
        ("vision", ("model.vision_model.transformer.resblocks",)),
    ),
    "BagelForUnifiedMultimodal": (
        ("language", ("model.language_model.model.layers",)),
        ("vision", ("model.vit_model.vision_model.encoder.layers",)),
    ),
    "NemotronHForCausalLM": (("language", ("backbone.layers", "model.layers")),),
    "GPT2LMHeadModel": (("language", ("transformer.h",)),),
}

_RETRIEVAL_WRAPPER_NAMES = frozenset(
    {"BiEncoderModel", "CrossEncoderModel", "FSDPBiEncoderModel"}
)


def ignore_sac_ops(ops: list[object | None]) -> None:
    """Exclude available runtime operators from selective-AC replay accounting.

    Args:
        ops: Backend operators to ignore. ``None`` entries represent optional
            operators that are unavailable in the installed PyTorch version.
    """
    platform.ignore_sac_ops(ops)


def ensure_profiler_ops_sac_ignored() -> None:
    """Keep profiler record-function operators out of selective-AC replay.

    FSDP hooks run under ``record_function`` and may execute a different number
    of profiler range operators in the original forward and recomputation. The
    range operators carry no activations, so excluding them only removes them
    from replay accounting while preserving their execution.
    """
    profiler_ops = getattr(torch.ops, "profiler", None)
    if profiler_ops is None:
        return

    ops_to_ignore = []
    for packet_name in (
        "_record_function_enter",
        "_record_function_enter_new",
        "_record_function_exit",
    ):
        packet = getattr(profiler_ops, packet_name, None)
        if packet is None:
            continue
        for overload_name in packet.overloads():
            ops_to_ignore.append(getattr(packet, overload_name))
    ignore_sac_ops(ops_to_ignore)


def ensure_fsdp_ops_sac_ignored() -> None:
    """Keep FSDP parameter-lifecycle operators out of selective-AC replay.

    Forward prefetch may unshard parameters before a checkpoint region, while
    recomputation may need to unshard them inside that region. These allocation,
    copy and collective operators manage parameters rather than model
    activations, and therefore must not be matched against the forward replay.
    """
    ignore_sac_ops(
        [
            _resolve_torch_op(op_name)
            for op_name in (
                "fsdp.all_gather_copy_in",
                "fsdp.split_with_sizes_copy",
                "fsdp.chunk_cat",
                "fsdp.copy_",
                "c10d._allgather_base_",
                "aten.empty.memory_format",
                "aten.empty_like",
                "aten.view",
            )
        ]
    )


def make_selective_checkpoint_context_fn() -> Callable[[], tuple[object, object]]:
    """Create a per-checkpoint-region selective activation policy context.

    Expensive operations are saved, ordinary operations are recomputed, and
    matmul operations alternate between the two decisions. A new counter is
    created every time the returned factory is invoked, matching the
    ``context_fn`` contract of non-reentrant checkpointing.

    Returns:
        A no-argument factory that creates the forward and recompute contexts.
    """
    ensure_profiler_ops_sac_ignored()
    ensure_fsdp_ops_sac_ignored()

    def selective_checkpoint_context_fn():
        matmul_counts = {False: 0, True: 0}

        def selective_checkpointing_policy(ctx, func, *args, **kwargs):
            del args, kwargs
            if func in _SELECTIVE_AC_FORCE_RECOMPUTE_OPS:
                return CheckpointPolicy.MUST_RECOMPUTE
            if func in _SELECTIVE_AC_MATMUL_OPS:
                matmul_counts[ctx.is_recompute] += 1
                if matmul_counts[ctx.is_recompute] % 2:
                    return CheckpointPolicy.MUST_SAVE
                return CheckpointPolicy.MUST_RECOMPUTE
            if func in _SELECTIVE_AC_MUST_SAVE_OPS:
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.MUST_RECOMPUTE

        return platform.create_selective_checkpoint_contexts(
            selective_checkpointing_policy
        )

    return selective_checkpoint_context_fn


def _resolve_layer_container(model: nn.Module, path: str) -> Optional[_LayerContainerInfo]:
    """Resolve one layer-container path and retain its parent for replacement."""
    path_parts = path.split(".")
    parent = model
    for part in path_parts[:-1]:
        parent = getattr(parent, part, None)
        if parent is None:
            return None

    if not isinstance(parent, nn.Module):
        return None
    child_name = path_parts[-1]
    container = getattr(parent, child_name, None)
    if not isinstance(container, (nn.ModuleList, nn.ModuleDict)):
        return None
    return parent, child_name, container, path


def _find_largest_layer_container(model: nn.Module) -> Optional[_LayerContainerInfo]:
    """Find the largest ModuleList or pipeline-split numeric ModuleDict."""
    largest = None
    largest_size = 0
    for module_path, module in model.named_modules():
        for child_name, child in module.named_children():
            is_layer_container = isinstance(child, nn.ModuleList) or (
                isinstance(child, nn.ModuleDict)
                and all(key.isdigit() for key in child.keys())
            )
            if not is_layer_container or len(child) <= largest_size:
                continue
            path = f"{module_path}.{child_name}" if module_path else child_name
            largest = module, child_name, child, path
            largest_size = len(child)
    return largest


def _find_transformer_layer_container_infos(model: nn.Module) -> dict[str, list[_LayerContainerInfo]]:
    """Find transformer layer containers grouped by model role.

    Known model classes use ordered alternative paths so different transformers
    module-tree versions resolve deterministically. Generic causal language
    models use their conventional ``model.layers`` or ``layers`` path. Unknown
    models fall back to the largest plausible layer container.

    Args:
        model: Model whose transformer layer containers should be located.

    Returns:
        Mapping from role names such as ``language``/``vision``/``audio`` to
        ``(parent, child_name, container, path)`` tuples. Each logical group
        contributes at most one container.
    """
    model_name = type(model).__name__
    if model_name in _RETRIEVAL_WRAPPER_NAMES:
        inner_model = getattr(model, "model", None)
        if isinstance(inner_model, nn.Module):
            inner_groups = _find_transformer_layer_container_infos(inner_model)
            return {
                group_name: [
                    (parent, child_name, container, f"model.{path}")
                    for parent, child_name, container, path in group_containers
                ]
                for group_name, group_containers in inner_groups.items()
            }

    # Known architectures use ordered, version-aware paths for each model role.
    # The first non-empty match wins so deprecated aliases cannot make the same
    # language or vision tower appear more than once.
    container_path_groups = _MODEL_LAYER_CONTAINER_PATHS.get(model_name)
    if container_path_groups is not None:
        layer_groups = {}
        for group_name, candidate_paths in container_path_groups:
            for path in candidate_paths:
                container_info = _resolve_layer_container(model, path)
                if container_info is None or not container_info[2]:
                    continue
                layer_groups[group_name] = [container_info]
                break
        if not layer_groups:
            logger.warning(
                "Layer-container spec for %s resolved no modules from expected paths %s",
                model_name,
                container_path_groups,
            )
        return layer_groups

    # Unregistered language models commonly expose their decoder blocks through
    # ``model.layers`` or ``layers``. A match is labeled as language-only so
    # downstream code can still evaluate HuggingFace-native checkpoint support.
    for path in ("model.layers", "layers"):
        container_info = _resolve_layer_container(model, path)
        if container_info is not None:
            return {"language": [container_info] if container_info[2] else []}

    # For unfamiliar module trees, use the largest plausible repeated-block
    # container as a conservative fallback. Numeric-only ModuleDict support in
    # the helper covers pipeline splits without mistaking named adapter sets for layers.
    logger.warning(
        "Unknown model type %s; using the largest layer-container heuristic.",
        model_name,
    )
    largest_container = _find_largest_layer_container(model)
    return {} if largest_container is None else {"unknown": [largest_container]}


def _flatten_layer_container_infos(
    layer_groups: dict[str, list[_LayerContainerInfo]],
) -> dict[str, list[nn.Module]]:
    """Expand grouped layer containers into grouped individual transformer layers."""
    flattened = {}
    for group_name, group_containers in layer_groups.items():
        layers = []
        for _, _, container, _ in group_containers:
            if isinstance(container, nn.ModuleDict):
                layers.extend(container.values())
            else:
                layers.extend(container)
        flattened[group_name] = layers
    return flattened


def _should_use_hf_native_gradient_checkpointing(
    model: nn.Module,
    layer_groups: dict[str, list[nn.Module]],
    *,
    enable_compile: bool = False,
) -> bool:
    """Return whether full checkpointing can use HuggingFace's native API."""
    if enable_compile or set(layer_groups) != {"language"}:
        return False

    language_layers = layer_groups.get("language", [])
    if not language_layers or any(
        not any(parameter.requires_grad for parameter in layer.parameters())
        for layer in language_layers
    ):
        return False

    try:
        from transformers.modeling_layers import GradientCheckpointingLayer
    except ImportError:
        return False

    return (
        all(isinstance(layer, GradientCheckpointingLayer) for layer in language_layers)
        and getattr(model, "supports_gradient_checkpointing", False)
        and hasattr(model, "gradient_checkpointing_enable")
    )


def _wrap_layer_containers(
    containers: list[_LayerContainerInfo],
    wrapper: Callable,
    *,
    context_fn: Optional[Callable[[], tuple[object, object]]] = None,
) -> int:
    """Wrap every layer in the discovered containers and return the count."""
    wrapped_count = 0
    for _, _, layers, _ in containers:
        layer_items = (
            list(layers.items())
            if isinstance(layers, nn.ModuleDict)
            else list(enumerate(layers))
        )
        for layer_key, layer in layer_items:
            checkpoint_kwargs = {} if context_fn is None else {"context_fn": context_fn}
            layers[layer_key] = wrapper(layer, **checkpoint_kwargs)
            wrapped_count += 1
    return wrapped_count


def _is_checkpoint_wrapped(module: nn.Module) -> bool:
    """Return whether a module is already wrapped by a supported checkpoint wrapper."""
    return hasattr(module, "_wrapped_module") or hasattr(
        module,
        "_checkpoint_wrapped_module",
    )


def _wrap_first_existing_attr(
    module: nn.Module,
    attr_names: tuple[str, ...],
    wrapper: Callable,
    *,
    skip: bool = False,
) -> int:
    """Checkpoint-wrap the first registered child matching ``attr_names``."""
    if skip:
        return 0

    for attr in attr_names:
        child = getattr(module, attr, None)
        if not isinstance(child, nn.Module):
            continue
        child_name = next(
            (
                name
                for name, registered_child in module._modules.items()
                if registered_child is child
            ),
            None,
        )
        if child_name is None:
            continue
        if _is_checkpoint_wrapped(child):
            return 0
        setattr(module, child_name, wrapper(child))
        return 1
    return 0


def apply_submodule_checkpointing(
    layers: list[nn.Module],
    has_kv_sharing: bool,
) -> int:
    """Apply full activation checkpointing to transformer-layer submodules."""
    wrapped_count = 0
    for layer in layers:
        wrapped_count += _wrap_first_existing_attr(
            layer,
            ("mlp", "feed_forward", "ffn"),
            checkpoint_wrapper,
        )
        wrapped_count += _wrap_first_existing_attr(
            layer,
            ("self_attn", "attention", "attn", "linear_attn"),
            checkpoint_wrapper,
            skip=has_kv_sharing,
        )
        wrapped_count += _wrap_first_existing_attr(
            layer,
            ("input_layernorm", "attention_norm", "layer_norm1", "norm1"),
            checkpoint_wrapper,
        )
        wrapped_count += _wrap_first_existing_attr(
            layer,
            ("post_attention_layernorm", "ffn_norm", "layer_norm2", "norm2"),
            checkpoint_wrapper,
        )
        for attr in (
            "mlp_moe_gen",
            "input_layernorm_moe_gen",
            "post_attention_layernorm_moe_gen",
        ):
            child = getattr(layer, attr, None)
            if isinstance(child, nn.Module) and not _is_checkpoint_wrapped(child):
                setattr(layer, attr, checkpoint_wrapper(child))
                wrapped_count += 1

    logger.info(
        "Applied submodule activation checkpointing to %d layer(s), wrapping %d submodule(s)",
        len(layers),
        wrapped_count,
    )
    return wrapped_count


def apply_compile_submodule_checkpointing(layers: list[nn.Module]) -> int:
    """Checkpoint-wrap attention and MLP submodules for compile mode."""
    wrapped_count = 0
    for layer in layers:
        for attr in (
            "self_attn",
            "attention",
            "attn",
            "linear_attn",
            "mlp",
            "feed_forward",
            "ffn",
        ):
            submodule = getattr(layer, attr, None)
            if submodule is not None:
                setattr(layer, attr, checkpoint_wrapper(submodule))
                wrapped_count += 1
    return wrapped_count


def _detect_kv_sharing_and_maybe_disable_cache(model: nn.Module) -> bool:
    """Detect cross-layer KV sharing and disable ordinary model caches."""
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None) or config
    has_kv_sharing = getattr(text_config, "num_kv_shared_layers", 0) > 0
    if has_kv_sharing or config is None:
        return has_kv_sharing

    sub_config_names = getattr(type(config), "sub_configs", None) or {
        "text_config": None,
    }
    sub_configs = (getattr(config, name, None) for name in sub_config_names)
    for sub_config in (config, *sub_configs):
        if sub_config is None or (
            sub_config is not config and not hasattr(sub_config, "use_cache")
        ):
            continue
        if getattr(sub_config, "use_cache", None) is not False:
            try:
                sub_config.use_cache = False
            except Exception:  # Configuration objects may reject assignment with custom errors.
                pass
    return False


def _apply_activation_checkpointing(
    model: nn.Module,
    activation_checkpoint: Optional[str],
    enable_compile: bool = False,
) -> nn.Module:
    """Apply full or selective recomputation to discovered transformer layers."""
    if activation_checkpoint not in ("full", "selective"):
        raise ValueError(
            "activation_checkpoint.mode must be 'full' or 'selective', but got "
            f"{activation_checkpoint!r}"
        )

    container_groups = _find_transformer_layer_container_infos(model)
    containers = [
        container_info
        for group_containers in container_groups.values()
        for container_info in group_containers
    ]
    if not containers:
        raise ValueError(
            f"{type(model).__name__} does not expose a supported transformer "
            "layer container"
        )

    layer_groups = _flatten_layer_container_infos(container_groups)
    ac_layers = [
        layer
        for group_layers in layer_groups.values()
        for layer in group_layers
    ]
    _has_kv_sharing = _detect_kv_sharing_and_maybe_disable_cache(model)

    # Selective recomputation normally wraps whole layers. KV-shared models
    # instead use submodule checkpointing so attention does not write the cache
    # again during backward recomputation.
    if activation_checkpoint == "selective":
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        if _has_kv_sharing:
            logger.warning(
                "Selective activation checkpointing is not supported for KV-shared models; "
                "falling back to submodule activation checkpointing."
            )
            apply_submodule_checkpointing(ac_layers, _has_kv_sharing)
        else:
            wrapped_count = _wrap_layer_containers(
                containers,
                checkpoint_wrapper,
                context_fn=make_selective_checkpoint_context_fn(),
            )
            logger.info(
                "Selective activation checkpointing applied to %d layer(s) in: %s",
                wrapped_count,
                ", ".join(path for _, _, _, path in containers),
            )

    elif activation_checkpoint == "full":
        # Prefer the HF-native implementation when all eligibility checks
        # pass. Otherwise use Hyper Parallel's submodule wrappers.
        if _should_use_hf_native_gradient_checkpointing(
            model,
            layer_groups,
            enable_compile=enable_compile,
        ):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
            logger.info("Using HuggingFace native gradient checkpointing for language layers.")
            return model

        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        if enable_compile:
            # Whole-layer/reentrant checkpointing can drop trainable parameter
            # gradients from the AOT-autograd graph. Wrap the compute-heavy
            # submodules before FSDP instead.
            wrapped_count = apply_compile_submodule_checkpointing(ac_layers)
        else:
            wrapped_count = apply_submodule_checkpointing(ac_layers, _has_kv_sharing)
        logger.info(
            "%s activation checkpointing wrapped %d submodule(s) in: %s",
            activation_checkpoint.capitalize(),
            wrapped_count,
            ", ".join(path for _, _, _, path in containers),
        )
    return model
