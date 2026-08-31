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
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn

from hyper_parallel.core.activation_checkpoint.activation_checkpoint import (
    CheckpointPolicy,
    checkpoint_wrapper,
)
from hyper_parallel.core.activation_checkpoint.swap import SwapManager
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
        from torch._functorch.partitioners import get_default_op_list  # pylint: disable=import-outside-toplevel

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


@dataclass(frozen=True)
class _TransformerBlockInfo:
    """A transformer block discovered below an HF checkpointing owner.

    ``parent`` is the registered repeated-block container (for example, a
    ``ModuleList``), while ``child_name`` is the actual registered key. Keeping
    both lets callers replace a block without assuming that the key is a
    contiguous integer or that the container is a specific PyTorch class.
    """

    fqn: str
    module: nn.Module
    parent: nn.Module
    child_name: str
    container_fqn: str


@dataclass(frozen=True)
class _LayerContainerInfo:
    """One repeated-block container and the blocks selected from it."""

    container: nn.Module
    path: str
    blocks: tuple[_TransformerBlockInfo, ...]


def _get_checkpoint_wrapped_module(module: nn.Module) -> Optional[nn.Module]:
    """Return the module directly held by a supported checkpoint wrapper."""
    for attr_name in ("_wrapped_module", "_checkpoint_wrapped_module"):
        wrapped_module = getattr(module, attr_name, None)
        if isinstance(wrapped_module, nn.Module):
            return wrapped_module
    return None


def _find_transformer_block_modules(
    model: nn.Module,
) -> tuple[list[_TransformerBlockInfo], set[int]]:
    """Find transformer blocks below modules with HF checkpointing support.

    HuggingFace model components expose ``gradient_checkpointing`` on modules
    that own a repeated block container.  The marker is the structural
    contract; no model class name or conventional ``layers`` path is needed.
    Each direct child of a marked repeated container is returned with its
    registered name so callers can safely replace it in-place.

    Args:
        model: Model whose transformer blocks should be located.

    Returns:
        A list of block metadata and the IDs of blocks already selected. The
        ID set is useful to callers that need to add other wrap targets while
        avoiding duplicate modules.
    """
    block_infos = []
    discovered_block_ids = set()
    for owner_fqn, owner in model.named_modules():
        # Match FSDP2's behavior for nested checkpointing owners: once an
        # owner has itself been selected as a block, do not scan inside it.
        if id(owner) in discovered_block_ids:
            continue
        if not hasattr(owner, "gradient_checkpointing"):
            continue

        for container_name, container in owner.named_children():
            children = list(container.named_children())
            if not children:
                continue
            container_fqn = (
                f"{owner_fqn}.{container_name}" if owner_fqn else container_name
            )
            for block_name, block in children:
                if id(block) in discovered_block_ids:
                    continue
                discovered_block_ids.add(id(block))
                wrapped_module = _get_checkpoint_wrapped_module(block)
                if wrapped_module is not None:
                    # A checkpoint wrapper may proxy attributes from its inner
                    # block, including ``gradient_checkpointing``. Treat the
                    # wrapper and inner module as one logical transformer block
                    # during the rest of the module-tree traversal.
                    discovered_block_ids.add(id(wrapped_module))
                block_fqn = f"{container_fqn}.{block_name}"
                block_infos.append(
                    _TransformerBlockInfo(
                        fqn=block_fqn,
                        module=block,
                        parent=container,
                        child_name=block_name,
                        container_fqn=container_fqn,
                    )
                )
    return block_infos, discovered_block_ids


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


def _make_selective_checkpoint_policy_fn() -> Callable:
    """Create an isolated selective activation checkpointing policy."""
    matmul_counts = {False: 0, True: 0}

    def selective_checkpointing_policy(
        ctx: Any,
        func: Any,
        *args: Any,
        **kwargs: Any,
    ) -> CheckpointPolicy:
        """Decide whether ``func``'s output is saved or recomputed.

        Follows the selective-activation-checkpointing policy contract: matmuls
        alternate between save and recompute, expensive/communication ops are
        always saved, and everything else is recomputed.

        Args:
            ctx: Checkpoint context carrying the ``is_recompute`` phase flag.
            func: The operator being traced.
            *args: Operator arguments (unused).
            **kwargs: Operator keyword arguments (unused).

        Returns:
            The checkpoint policy for ``func``.
        """
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

    return selective_checkpointing_policy


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

    def selective_checkpoint_context_fn() -> tuple[object, object]:
        """Create a fresh pair of forward/recompute selective-AC contexts.

        Returns:
            The ``(forward_context, recompute_context)`` pair expected by the
            non-reentrant checkpointing ``context_fn`` contract.
        """
        return platform.create_selective_checkpoint_contexts(
            _make_selective_checkpoint_policy_fn()
        )

    return selective_checkpoint_context_fn


def _find_transformer_layer_container_infos(
    model: nn.Module,
) -> list[_LayerContainerInfo]:
    """Find repeated block containers without model-specific path rules.

    Containers are derived from :func:`_find_transformer_block_modules`, so a
    model can freely name its towers and layer attributes.  A container is
    included once even when blocks are shared or the module tree exposes an
    alias to it.

    Args:
        model: Model whose transformer layer containers should be located.

    Returns:
        A list of discovered containers, ordered by the model's registration
        order. An empty list means no module advertises HF checkpointing
        support for a repeated block container.
    """
    block_infos, _ = _find_transformer_block_modules(model)
    containers = []
    blocks_by_container = {}
    seen_container_ids = set()
    for block_info in block_infos:
        container_id = id(block_info.parent)
        blocks_by_container.setdefault(container_id, []).append(block_info)

    for block_info in block_infos:
        container_id = id(block_info.parent)
        if container_id in seen_container_ids:
            continue
        seen_container_ids.add(container_id)
        containers.append(
            _LayerContainerInfo(
                container=block_info.parent,
                path=block_info.container_fqn,
                blocks=tuple(blocks_by_container[container_id]),
            )
        )
    return containers


def _flatten_layer_container_infos(
    containers: list[_LayerContainerInfo],
) -> list[nn.Module]:
    """Expand discovered containers into individual transformer blocks."""
    return [
        block_info.module
        for container in containers
        for block_info in container.blocks
    ]


def _should_use_hf_native_gradient_checkpointing(
    model: nn.Module,
    layers: list[nn.Module],
    *,
    enable_compile: bool = False,
) -> bool:
    """Return whether full checkpointing can use HuggingFace's native API."""
    if enable_compile:
        return False

    if not layers or any(
        not any(parameter.requires_grad for parameter in layer.parameters())
        for layer in layers
    ):
        return False

    try:
        # Guarded import: transformers is optional and older versions may not
        # expose GradientCheckpointingLayer.
        from transformers.modeling_layers import GradientCheckpointingLayer  # pylint: disable=import-outside-toplevel
    except ImportError:
        return False

    return (
        all(isinstance(layer, GradientCheckpointingLayer) for layer in layers)
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
    for container_info in containers:
        for block_info in container_info.blocks:
            checkpoint_kwargs = {} if context_fn is None else {"context_fn": context_fn}
            current_block = getattr(block_info.parent, block_info.child_name, None)
            if not isinstance(current_block, nn.Module) or _is_checkpoint_wrapped(current_block):
                continue
            setattr(
                block_info.parent,
                block_info.child_name,
                wrapper(current_block, **checkpoint_kwargs),
            )
            wrapped_count += 1
    return wrapped_count


def _is_checkpoint_wrapped(module: nn.Module) -> bool:
    """Return whether a module is already wrapped by a supported checkpoint wrapper."""
    return hasattr(module, "_wrapped_module") or hasattr(
        module,
        "_checkpoint_wrapped_module",
    )


def _find_checkpoint_wrappers(module: nn.Module, prefix: str = "") -> dict[str, nn.Module]:
    """Find outermost checkpoint wrappers by their relative module paths."""
    if _is_checkpoint_wrapped(module):
        return {prefix: module}

    wrappers = {}
    for child_name, child in module.named_children():
        child_path = f"{prefix}.{child_name}" if prefix else child_name
        wrappers.update(_find_checkpoint_wrappers(child, child_path))
    return wrappers


def _register_forward_prefetch_layers(containers: list[_LayerContainerInfo]) -> None:
    """Register swap prefetch chains within each repeated-block container."""
    swap_manager = SwapManager()
    for container_info in containers:
        wrapper_chains = {}
        for block_info in container_info.blocks:
            current_block = getattr(block_info.parent, block_info.child_name, None)
            if not isinstance(current_block, nn.Module):
                continue
            for relative_path, wrapper in _find_checkpoint_wrappers(current_block).items():
                wrapper_chains.setdefault(relative_path, []).append(wrapper)

        for wrappers in wrapper_chains.values():
            for current_wrapper, next_wrapper in zip(wrappers, wrappers[1:]):
                swap_manager.set_forward_prefetch_layer(current_wrapper, next_wrapper)


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
                for name, registered_child in module.named_children()
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
    enable_compile: bool = False,
    swap_inputs: bool = False,
) -> int:
    """Apply full activation checkpointing to transformer-layer submodules.

    Args:
        layers: Transformer layers whose selected submodules should be wrapped.
        has_kv_sharing: Whether attention submodules must remain outside the
            recomputation regions.
        enable_compile: Whether the wrapped regions will be compiled.
        swap_inputs: Whether checkpoint inputs should be offloaded in eager
            execution. This is ignored in compile mode.
    """
    checkpoint_kwargs = {"swap_inputs": swap_inputs} if not enable_compile else {}

    def submodule_checkpoint_wrapper(module: nn.Module) -> nn.Module:
        """Wrap one selected layer submodule with checkpointing."""
        return checkpoint_wrapper(module, **checkpoint_kwargs)

    wrapped_count = 0
    for layer in layers:
        wrapped_count += _wrap_first_existing_attr(
            layer,
            ("mlp", "feed_forward", "ffn"),
            submodule_checkpoint_wrapper,
        )
        wrapped_count += _wrap_first_existing_attr(
            layer,
            ("self_attn", "attention", "attn", "linear_attn"),
            submodule_checkpoint_wrapper,
            skip=has_kv_sharing,
        )
        wrapped_count += _wrap_first_existing_attr(
            layer,
            ("input_layernorm", "attention_norm", "layer_norm1", "norm1"),
            submodule_checkpoint_wrapper,
        )
        wrapped_count += _wrap_first_existing_attr(
            layer,
            ("post_attention_layernorm", "ffn_norm", "layer_norm2", "norm2"),
            submodule_checkpoint_wrapper,
        )
        for attr in (
            "mlp_moe_gen",
            "input_layernorm_moe_gen",
            "post_attention_layernorm_moe_gen",
        ):
            child = getattr(layer, attr, None)
            if isinstance(child, nn.Module) and not _is_checkpoint_wrapped(child):
                setattr(layer, attr, submodule_checkpoint_wrapper(child))
                wrapped_count += 1

    logger.info(
        "Applied submodule activation checkpointing to %d layer(s), wrapping %d submodule(s)",
        len(layers),
        wrapped_count,
    )
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
            except Exception:  # pylint: disable=broad-exception-caught
                # Configuration objects may reject assignment with custom errors.
                pass
    return False


def _apply_activation_checkpointing(
    model: nn.Module,
    activation_checkpoint: Optional[str],
    enable_compile: bool = False,
    swap_inputs: bool = False,
) -> nn.Module:
    """Apply full or selective recomputation to discovered transformer layers."""
    if activation_checkpoint not in ("full", "selective"):
        raise ValueError(
            "activation_checkpoint.mode must be 'full' or 'selective', but got "
            f"{activation_checkpoint!r}"
        )
    if not isinstance(swap_inputs, bool):
        raise ValueError(
            "activation_checkpoint.swap_inputs must be bool, but got "
            f"{type(swap_inputs).__name__}"
        )

    if swap_inputs and enable_compile:
        logger.warning(
            "activation_checkpoint.swap_inputs is not supported with torch.compile; "
            "input swapping will be disabled."
        )

    containers = _find_transformer_layer_container_infos(model)
    if not containers:
        raise ValueError(
            f"{type(model).__name__} has no module with a 'gradient_checkpointing' "
            "attribute and a non-empty repeated block container"
        )

    ac_layers = _flatten_layer_container_infos(containers)
    has_kv_sharing = _detect_kv_sharing_and_maybe_disable_cache(model)

    # Selective recomputation normally wraps whole layers. KV-shared models
    # instead use submodule checkpointing so attention does not write the cache
    # again during backward recomputation.
    if activation_checkpoint == "selective":
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        if has_kv_sharing:
            logger.warning(
                "Selective activation checkpointing is not supported for KV-shared models; "
                "falling back to submodule activation checkpointing."
            )
            apply_submodule_checkpointing(
                ac_layers,
                has_kv_sharing,
                enable_compile=enable_compile,
                swap_inputs=swap_inputs,
            )
        else:
            if enable_compile:
                ensure_profiler_ops_sac_ignored()
                ensure_fsdp_ops_sac_ignored()

                def compile_checkpoint_wrapper(layer: nn.Module) -> nn.Module:
                    """Wrap one layer with the compile-compatible SAC policy."""
                    return checkpoint_wrapper(
                        layer,
                        policy_fn=_make_selective_checkpoint_policy_fn(),
                    )

                wrapped_count = _wrap_layer_containers(
                    containers,
                    compile_checkpoint_wrapper,
                )
            else:
                def eager_checkpoint_wrapper(layer: nn.Module, **checkpoint_kwargs: Any) -> nn.Module:
                    """Wrap one layer with eager selective checkpointing."""
                    return checkpoint_wrapper(
                        layer,
                        swap_inputs=swap_inputs,
                        **checkpoint_kwargs,
                    )

                wrapped_count = _wrap_layer_containers(
                    containers,
                    eager_checkpoint_wrapper,
                    context_fn=make_selective_checkpoint_context_fn(),
                )
            logger.info(
                "Selective activation checkpointing applied to %d layer(s) in: %s",
                wrapped_count,
                ", ".join(container.path for container in containers),
            )

    elif activation_checkpoint == "full":
        # Prefer the HF-native implementation when all eligibility checks
        # pass. Otherwise use Hyper Parallel's submodule wrappers.
        if _should_use_hf_native_gradient_checkpointing(
            model,
            ac_layers,
            enable_compile=enable_compile,
        ):
            if swap_inputs:
                logger.warning(
                    "activation_checkpoint.swap_inputs is not supported by Hugging Face native "
                    "gradient checkpointing for now; input swapping will be disabled."
                )
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
            logger.info("Using HuggingFace native gradient checkpointing for discovered layers.")
            return model

        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        if has_kv_sharing:
            wrapped_count = apply_submodule_checkpointing(
                ac_layers,
                has_kv_sharing,
                enable_compile=enable_compile,
                swap_inputs=swap_inputs,
            )
        else:
            def full_checkpoint_wrapper(layer: nn.Module) -> nn.Module:
                """Wrap one complete transformer layer for full recomputation."""
                if enable_compile:
                    return checkpoint_wrapper(layer)
                return checkpoint_wrapper(layer, swap_inputs=swap_inputs)

            wrapped_count = _wrap_layer_containers(containers, full_checkpoint_wrapper)
        logger.info(
            "%s activation checkpointing wrapped %d submodule(s) in: %s",
            activation_checkpoint.capitalize(),
            wrapped_count,
            ", ".join(container.path for container in containers),
        )
    if swap_inputs and not enable_compile:
        _register_forward_prefetch_layers(containers)
    return model
