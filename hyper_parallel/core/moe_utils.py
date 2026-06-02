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
"""MoE utilities for distributed training."""

from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional, Any

from hyper_parallel.core.fully_shard.hsdp_utils import GroupInfo
from hyper_parallel.platform import get_platform

if TYPE_CHECKING:
    from hyper_parallel.platform.torch.common.moe import MoE

platform = get_platform()


def sync_and_update_expert_bias(
    moe: "MoE",
    lr: float = 1e-3,
    tp_group: Optional[Any] = None,
    cp_group: Optional[Any] = None,
    dp_group: Optional[Any] = None,
    num_recomputations: int = 1,
) -> None:
    """Synchronize tokens_per_expert across distributed ranks, then update expert bias.

    This function handles distributed synchronization for expert bias updates in
    MoE (Mixture of Experts) models. It should be called after optimizer.step() in
    the training loop.

    The synchronization is needed because:
    - **DP**: Different ranks process different samples, need to aggregate global batch stats
    - **TP+SP**: Standard TP does not need sync (all ranks see same input).
      Only needed when TP is combined with sequence parallelism (TP+SP),
      where activations are sequence-sharded across TP ranks.
    - **CP**: Sequence dimension is sharded across ranks

    For pure EP scenario (no DP/TP/CP), call moe.update_expert_bias() directly.

    Args:
        moe: The MoE module whose expert bias should be updated.
        lr: Step size for the bias update. Defaults to ``1e-3``.
        tp_group: Tensor parallel process group (ProcessGroup or GroupInfo).
            Only needed when TP is combined with sequence parallelism (TP+SP),
            where activations are sequence-sharded across TP ranks.
            Standard TP (weight-only sharding) does not need this.
        cp_group: Context parallel process group (ProcessGroup or GroupInfo).
            Sync across sequence shards dimension.
        dp_group: Data parallel process group (ProcessGroup or GroupInfo).
            Sync across different samples to get global batch statistics.
        num_recomputations: Number of forward executions per optimizer step.
            Default ``1`` (normal training). Set to ``2`` when activation
            checkpoint is enabled.

    Example:
        >>> # DP scenario
        >>> sync_and_update_expert_bias(moe_layer, lr=1e-3, dp_group=dp_group_info)
        >>>
        >>> # TP×CP×DP scenario (reference: Megatron-LM)
        >>> sync_and_update_expert_bias(
        ...     moe_layer,
        ...     lr=1e-3,
        ...     tp_group=tp_group_info,
        ...     cp_group=cp_group_info,
        ...     dp_group=dp_group_info,
        ... )

    Note:
        Reference implementation: Megatron-LM megatron/core/transformer/moe/moe_utils.py
        uses TP×CP×DP group synchronization for global batch statistics.
    """
    need_sync = tp_group is not None or cp_group is not None or dp_group is not None

    if need_sync:
        if tp_group is not None:
            group_info = _ensure_group_info(tp_group)
            platform.all_reduce(moe.tokens_per_expert, group_info)
        if cp_group is not None:
            group_info = _ensure_group_info(cp_group)
            platform.all_reduce(moe.tokens_per_expert, group_info)
        if dp_group is not None:
            group_info = _ensure_group_info(dp_group)
            platform.all_reduce(moe.tokens_per_expert, group_info)

    moe.update_expert_bias(lr=lr, num_recomputations=num_recomputations)


def _ensure_group_info(group: Any) -> GroupInfo:
    """Convert ProcessGroup to GroupInfo if needed.

    Args:
        group: Either a ProcessGroup or a GroupInfo object.

    Returns:
        GroupInfo object.
    """
    if isinstance(group, GroupInfo):
        return group

    if hasattr(group, "group"):
        return group

    return SimpleNamespace(group=group)


def _get_moe_layers(model: "nn.Module") -> list:
    """Collect all MoE sub-modules with ``enable_expert_bias=True``.

    Uses duck-typing (``getattr(m, 'enable_expert_bias', False)``) instead
    of ``isinstance`` to avoid importing platform-specific MoE classes in
    platform-agnostic ``core/`` code.

    Args:
        model: Root module to search.

    Returns:
        List of MoE module instances.
    """
    return [m for m in model.modules() if getattr(m, 'enable_expert_bias', False)]


class MoEMonitorCallback:
    """Callback that automates expert bias updates across all MoE layers.

    Provides a three-layer invocation architecture:

    - **Layer 0**: :func:`sync_and_update_expert_bias` — per-module sync + update
    - **Layer 1**: :meth:`on_step_end` — iterates all MoE layers, calling Layer 0
    - **Layer 2**: :meth:`register` — registers optimizer post-hook for auto-trigger

    Each layer can be used independently; Layer 0 is sufficient for basic usage,
    while Layers 1–2 add convenience for production training loops.

    Args:
        model: The model containing MoE layers to monitor.
        lr: Step size for expert bias updates. Defaults to ``1e-3``.
        tp_group: Tensor parallel process group. Only needed for TP+SP
            (sequence-sharded activations across TP ranks). Standard TP
            does not need this — pass ``None``.
        cp_group: Context parallel process group. Sync across sequence shards.
        dp_group: Data parallel process group. Sync across different samples.
        num_recomputations: Number of forward executions per optimizer step.
            Default ``1``. Set to ``2`` when activation checkpoint is enabled.

    Example:
        >>> # Layer 1: manual trigger in training loop
        >>> callback = MoEMonitorCallback(
        ...     model, lr=1e-3, dp_group=dp_group_info,
        ... )
        >>> for batch in dataloader:
        ...     loss = model(batch)
        ...     loss.backward()
        ...     optimizer.step()
        ...     callback.on_step_end()  # auto-updates all MoE layers
        ...     optimizer.zero_grad()
        >>>
        >>> # Layer 2: auto-trigger via optimizer post-hook
        >>> callback = MoEMonitorCallback(model, lr=1e-3, dp_group=dp_group_info)
        >>> callback.register(optimizer)
        >>> # optimizer.step() now auto-triggers callback.on_step_end()

    Note:
        Sync dimensions use ``dp_group``/``tp_group``/``cp_group`` (not ``ep_group``).
        EP does not need synchronization because all EP ranks have identical
        routing data (``tokens_per_expert`` shape = ``[total_experts]``).
    """

    def __init__(
        self,
        model: "nn.Module",
        lr: float = 1e-3,
        tp_group: Optional[Any] = None,
        cp_group: Optional[Any] = None,
        dp_group: Optional[Any] = None,
        num_recomputations: int = 1,
    ) -> None:
        """Initialize MoEMonitorCallback."""
        self.model = model
        self.lr = lr
        self.tp_group = tp_group
        self.cp_group = cp_group
        self.dp_group = dp_group
        self.num_recomputations = num_recomputations
        self._hook_handle = None
        self.last_mean_aux_loss: Optional[float] = None

    def on_step_end(self) -> None:
        """Update expert bias for all MoE layers in the model.

        Iterates over all :class:`MoE` sub-modules with
        ``enable_expert_bias=True`` and calls
        :func:`sync_and_update_expert_bias` for each one.

        When ``load_balance_coeff > 0`` on any MoE layer, collects
        ``last_aux_loss`` values and logs the mean.
        """
        moe_layers = _get_moe_layers(self.model)
        if not moe_layers:
            return

        for moe in moe_layers:
            sync_and_update_expert_bias(
                moe,
                lr=self.lr,
                tp_group=self.tp_group,
                cp_group=self.cp_group,
                dp_group=self.dp_group,
                num_recomputations=self.num_recomputations,
            )

        # Log aux_loss if any MoE layer has it enabled.
        aux_losses = [
            moe.last_aux_loss.item()
            for moe in moe_layers
            if moe.last_aux_loss is not None
        ]
        if aux_losses:
            mean_aux_loss = sum(aux_losses) / len(aux_losses)
            self.last_mean_aux_loss = mean_aux_loss
        else:
            self.last_mean_aux_loss = None

    def register(self, optimizer: Any) -> None:
        """Register an optimizer post-hook to auto-trigger :meth:`on_step_end`.

        On PyTorch 2.0+, uses ``optimizer.register_step_post_hook``.
        The hook fires after each ``optimizer.step()`` call, so that
        expert bias is updated once parameters have been updated.

        Args:
            optimizer: The optimizer instance to register the hook on.

        Raises:
            RuntimeError: If the optimizer does not support step post-hooks.
        """
        if not hasattr(optimizer, "register_step_post_hook"):
            raise RuntimeError(
                "Optimizer does not support register_step_post_hook. "
                "Requires PyTorch 2.0+. Call on_step_end() manually instead."
            )
        self._hook_handle = optimizer.register_step_post_hook(
            lambda *_: self.on_step_end(),
        )

    def remove(self) -> None:
        """Remove the optimizer post-hook if registered."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
