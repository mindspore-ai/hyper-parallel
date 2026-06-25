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
"""Activation optimization setup for LlamaFactory + HyperParallel."""

import logging
from typing import List, Tuple

import torch
from torch import nn
from torch.utils._pytree import tree_map

logger = logging.getLogger(__name__)


def find_transformer_blocks(
    model: nn.Module,
) -> List[Tuple[nn.Module, str, nn.ModuleList, str]]:
    """Discover transformer block containers via HF's ``gradient_checkpointing`` marker.

    HuggingFace ``PreTrainedModel`` containers that loop over a ``ModuleList``
    and apply gradient checkpointing set ``gradient_checkpointing`` on
    themselves.  This is exactly the set of containers whose blocks we should
    wrap, so we reuse the same marker (multi-tower models return all of them).

    Args:
        model: A HuggingFace model.

    Returns:
        A list of ``(parent, attr, blocks, block_path)`` tuples — one per
        ``ModuleList`` to wrap.  Empty when no container has the marker.
    """
    results = []
    seen_blocks = set()
    for container_name, container in model.named_modules():
        if not hasattr(container, "gradient_checkpointing"):
            continue
        for attr_name, child in container.named_children():
            if not isinstance(child, nn.ModuleList) or len(child) == 0:
                continue
            if id(child) in seen_blocks:
                continue
            seen_blocks.add(id(child))
            path = f"{container_name}.{attr_name}" if container_name else attr_name
            results.append((container, attr_name, child, path))
    return results


def _build_policy_fn(hp_args):
    """Build a per-op policy function based on activation configuration.

    Args:
        hp_args: HyperParallelArguments instance.

    Returns:
        A policy_fn callable or None (for plain recompute mode).
    """
    from hyper_parallel.core.activation_checkpoint import CheckpointPolicy  # pylint: disable=C0415

    mode = hp_args.activation_mode

    if mode == "swap":
        matmul_ops = {
            torch.ops.aten.matmul.default,
            torch.ops.aten.addmm.default,
            torch.ops.aten.bmm.default,
            torch.ops.aten.mm.default,
        }

        def policy_fn(ctx, op, *args, **kwargs):  # pylint: disable=W0613
            if op in matmul_ops:
                return CheckpointPolicy.MUST_SWAP
            return CheckpointPolicy.MUST_RECOMPUTE

        return policy_fn

    # recompute mode: no policy_fn needed (block-level full recompute)
    return None


def _clone_checkpoint_output(_module, _input, output):  # pylint: disable=C0103
    """Clone every tensor in checkpoint wrapper output to prevent in-place view errors.

    With ``use_reentrant=False``, checkpoint wraps forward in a custom autograd
    Function.  Some model architectures (e.g. Qwen3VL DeepStack) modify the
    decoder layer output in-place via indexed assignment.  Cloning breaks the
    view chain from the Function so that in-place ops are safe.  Use
    ``tree_map`` so dict / dataclass / nested-tuple outputs (e.g. HF
    ``BaseModelOutputWithPast``) are also handled.
    """
    return tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, output)


def _install_swap_tensor_backward_hooks(module, input, output):  # pylint: disable=W0622
    """Register tensor-level backward hooks in place of module-level ones.

    ``SwapManager``'s default ``register_full_backward_(pre_)hook`` wraps the
    module output through ``BackwardHookFunction``, producing view tensors
    that conflict with FSDP's ``PostBackwardFunction`` in-place updates.
    Equivalent timing is achieved with ``Tensor.register_hook`` on the
    first requires_grad output (= pre-backward) and input (= post-backward).
    """
    # Skip during recomputation (backward phase triggers forward again)
    if getattr(module, "_swap_state", None) == "pre_backward":
        return

    group_name = getattr(module, "_swap_group_name", None)
    if group_name is None:
        return

    from hyper_parallel.core.activation_checkpoint import SwapManager  # pylint: disable=C0415
    from torch.utils._pytree import tree_flatten  # pylint: disable=C0415

    # Pre-backward: fires at START of module backward
    def _pre_backward(grad):
        module._swap_state = "pre_backward"  # pylint: disable=W0212
        prev_name = module._swap_group_order.get("prev", None)  # pylint: disable=W0212
        if prev_name:
            SwapManager().launch_load(prev_name)
        next_name = module._swap_group_order.get("next", None)  # pylint: disable=W0212
        if next_name:
            SwapManager().wait_load(group_name)
        return grad

    # Post-backward: fires at END of module backward
    def _post_backward(grad):
        module._swap_state = "backward"  # pylint: disable=W0212
        SwapManager().release_group_storage(group_name)
        return grad

    # Register on first requires_grad output tensor
    flat_outputs, _ = tree_flatten(output)
    for output_tensor in flat_outputs:
        if isinstance(output_tensor, torch.Tensor) and output_tensor.requires_grad:
            output_tensor.register_hook(_pre_backward)
            break

    # Register on first requires_grad input tensor
    flat_inputs, _ = tree_flatten(input)
    for inp in flat_inputs:
        if isinstance(inp, torch.Tensor) and inp.requires_grad:
            inp.register_hook(_post_backward)
            break


def _replace_module_backward_hooks_with_tensor_hooks(wrapped_blocks):
    """Swap module-level backward hooks for tensor-level ones — see
    :func:`_install_swap_tensor_backward_hooks` for the rationale."""
    for wrapped in wrapped_blocks:
        for attr in ("_swap_backward_pre_hook_handle", "_swap_backward_hook_handle"):
            handle = getattr(wrapped, attr, None)
            if handle is not None:
                handle.remove()
                delattr(wrapped, attr)
        wrapped.register_forward_hook(_install_swap_tensor_backward_hooks)


def _build_wrap_kwargs(hp_args, mode: str) -> dict:
    """Return ``checkpoint_wrapper`` kwargs for the given ``activation_mode``."""
    if mode == "swap":
        return {
            "policy_fn": _build_policy_fn(hp_args),
            "swap_inputs": hp_args.activation_swap_inputs,
        }
    if mode == "recompute":
        return {}
    raise ValueError(f"Unknown activation_mode: {mode}")


def _wrap_one_container(
    parent, attr, blocks, block_path, mode, wrap_kwargs, hp_args
) -> int:
    """Wrap one transformer block container; install swap prefetch/hooks when in swap mode.

    Returns the number of newly wrapped (i.e. trainable) blocks.
    """
    from hyper_parallel.core.activation_checkpoint import (  # pylint: disable=C0415
        SwapManager,
        checkpoint_wrapper,
    )

    wrapped_blocks = []
    num_wrapped = 0
    for block in blocks:
        # Skip frozen blocks — matches LlamaFactory's runtime check semantics.
        if not any(p.requires_grad for p in block.parameters()):
            wrapped_blocks.append(block)
            continue
        wrapped = checkpoint_wrapper(block, **wrap_kwargs)
        wrapped.register_forward_hook(_clone_checkpoint_output)
        wrapped_blocks.append(wrapped)
        num_wrapped += 1

    num_skipped = len(blocks) - num_wrapped
    if num_skipped > 0:
        logger.info(
            "[Activation] [%s] Skipped %d frozen blocks (no trainable params)",
            block_path,
            num_skipped,
        )

    setattr(parent, attr, nn.ModuleList(wrapped_blocks))

    if mode == "swap":
        swap_blocks = [
            b for b in wrapped_blocks if hasattr(b, "_checkpoint_wrapped_module")
        ]
        for i in range(len(swap_blocks) - 1):
            SwapManager().set_forward_prefetch_layer(swap_blocks[i], swap_blocks[i + 1])
        _replace_module_backward_hooks_with_tensor_hooks(swap_blocks)
        logger.info(
            "[Activation] [%s] Applied swap checkpoint to %d/%d blocks, swap_inputs=%s",
            block_path,
            num_wrapped,
            len(blocks),
            hp_args.activation_swap_inputs,
        )
    else:
        logger.info(
            "[Activation] [%s] Applied checkpoint_wrapper to %d/%d blocks",
            block_path,
            num_wrapped,
            len(blocks),
        )

    return num_wrapped


def setup_activation_optimization(
    model: nn.Module, hp_args, block_info=None
) -> nn.Module:
    """Install activation optimization on transformer blocks.

    Can be called either before or after ``fully_shard``.  When called after
    ``fully_shard``, pass *block_info* obtained from an earlier call to
    :func:`find_transformer_blocks` (before the model tree was modified by FSDP).

    Args:
        model: The model to wrap.
        hp_args: HyperParallelArguments with activation_mode and related fields.
        block_info: Optional pre-detected list of ``(parent, attr, blocks,
            block_path)`` tuples from :func:`find_transformer_blocks`.  When
            ``None``, block detection runs internally.  Each tuple describes
            one container (e.g. visual tower + LLM tower in multimodal models).

    Returns:
        The model (modified in-place).
    """
    mode = hp_args.activation_mode
    if mode == "none":
        return model

    # Idempotency guard: avoid double-wrapping when setup is called more than once
    # on the same model (e.g. via callback or hot-reload paths).
    if getattr(model, "_hp_activation_setup_done", False):
        return model

    containers = (
        block_info if block_info is not None else find_transformer_blocks(model)
    )
    if not containers:
        raise ValueError(
            f"{type(model).__name__} has no module with a 'gradient_checkpointing' "
            f"attribute; this model does not support activation optimization via "
            f"HyperParallel. Set activation_mode='none' to disable."
        )

    paths = ", ".join(path for _, _, _, path in containers)
    total_blocks = sum(len(blocks) for _, _, blocks, _ in containers)
    logger.info(
        "[Activation] mode=%s, found %d blocks across %d container(s): %s",
        mode,
        total_blocks,
        len(containers),
        paths,
    )

    # Disable KV cache — required for activation checkpointing
    if hasattr(model, "config"):
        model.config.use_cache = False

    # Avoid double-checkpoint with HF native GC; the flag lives on inner
    # containers, so call the disable method unconditionally.
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
        logger.info(
            "[Activation] Disabled native gradient_checkpointing in favor of HP checkpoint_wrapper"
        )

    wrap_kwargs = _build_wrap_kwargs(hp_args, mode)

    total_wrapped = 0
    for parent, attr, blocks, block_path in containers:
        total_wrapped += _wrap_one_container(
            parent, attr, blocks, block_path, mode, wrap_kwargs, hp_args
        )

    if total_wrapped == 0:
        logger.warning(
            "[Activation] mode=%s requested but all %d blocks across %d container(s) "
            "are frozen; activation optimization has no effect. Consider setting "
            "activation_mode='none'.",
            mode,
            total_blocks,
            len(containers),
        )

    model._hp_activation_setup_done = True  # pylint: disable=W0212
    return model
