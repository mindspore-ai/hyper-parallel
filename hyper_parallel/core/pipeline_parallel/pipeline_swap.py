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
"""Pipeline-parallel activation swap scheduling helpers."""

from collections import defaultdict
from contextlib import nullcontext
from enum import IntEnum
import itertools
from typing import Any, ContextManager, FrozenSet, List

from hyper_parallel.core.activation_checkpoint.swap import SwapManager
from hyper_parallel.platform import get_platform

MIN_SWAP_GAP = 4
platform = get_platform()


class _BeforeActionPriority(IntEnum):
    WAIT_LOAD = 10
    LAUNCH_LOAD = 20


class _AfterActionPriority(IntEnum):
    WAIT_OFFLOAD = 10
    LAUNCH_OFFLOAD = 20


class PipelineSwapSession:
    """Own activation-swap groups for one pipeline schedule run."""

    _generation = itertools.count()

    def __init__(self, eligible_keys: FrozenSet[tuple[int, int]]) -> None:
        """Create one run-scoped session from build-time eligible chunk keys."""
        self._generation_id = next(self._generation)
        self._manager = SwapManager()
        self._eligible_keys = eligible_keys
        self._group_names = {
            key: f"pp_swap_run{self._generation_id}_s{key[0]}_m{key[1]}"
            for key in self._eligible_keys
        }
        self._forward_context_keys = set()

    @staticmethod
    def _key(step: Any) -> tuple[int, int]:
        """Return the logical chunk key carried by a swap or compute step."""
        return step.stage_index, step.micro_index

    def manages(self, step: Any) -> bool:
        """Return whether this session manages the step's pipeline chunk."""
        return self._key(step) in self._eligible_keys

    def group_context(self, step: Any) -> ContextManager[None]:
        """Enter the run-scoped swap group for one forward leaf."""
        if not self.manages(step):
            return nullcontext()
        key = self._key(step)
        group_name = self._group_names[key]
        self._manager.ensure_group(group_name)
        self._forward_context_keys.add(key)
        return self._manager.group_context(group_name)

    def require_forward_context(self, step: Any) -> None:
        """Require the managed forward leaf to have entered its swap context."""
        key = self._key(step)
        if key not in self._forward_context_keys:
            raise RuntimeError(
                "Pipeline swap did not observe the matching forward leaf. "
                "Custom overlap callbacks must call schedule.execute_fwd_leaf()."
            )

    def group_name(self, step: Any) -> str:
        """Return the physical group name for a managed step."""
        key = self._key(step)
        if key not in self._group_names:
            raise RuntimeError(f"Pipeline swap does not manage chunk {key}.")
        return self._group_names[key]

    def wait_load(self, step: Any) -> None:
        """Wait for H2D on the scheduler's current compute stream."""
        if self.manages(step):
            self._manager.wait_load(self.group_name(step))

    def protect_aliases(self, step: Any, tensors: Any) -> None:
        """Keep pipeline-owned aliases resident for a managed chunk."""
        if self.manages(step):
            self._manager.protect_alias_tensors(self.group_name(step), tensors)

    def close(self) -> None:
        """Release every group still owned by this run."""
        for group_name in self._group_names.values():
            self._manager.abort_group(group_name)


def _is_compute_step(step) -> bool:
    from hyper_parallel.core.pipeline_parallel.scheduler import MetaStepType  # pylint: disable=C0415

    return step is not None and step.type in (
        MetaStepType.FWD,
        MetaStepType.BWD,
        MetaStepType.BWD_INPUT,
        MetaStepType.BWD_WEIGHT,
    )


def _is_comm_step(step) -> bool:
    from hyper_parallel.core.pipeline_parallel.scheduler import MetaStepType  # pylint: disable=C0415

    return step is not None and step.type in (
        MetaStepType.FWD_RECV,
        MetaStepType.FWD_SEND,
        MetaStepType.BWD_RECV,
        MetaStepType.BWD_SEND,
        MetaStepType.BATCH_SEND_RECV,
    )


def _is_composite_compute_step(step) -> bool:
    from hyper_parallel.core.pipeline_parallel.scheduler import MetaStepType  # pylint: disable=C0415

    return (
        step is not None
        and step.type in (MetaStepType.OVERLAP_F_B, MetaStepType.OVERLAP_B_F)
        and step.sub_steps
    )


class _ComputeLeaf:
    """A real FWD/BWD leaf and the top-level container that owns it."""

    __slots__ = ("step", "container_index", "compute_index")

    def __init__(self, step: Any, container_index: int, compute_index: int) -> None:
        """Record a compute leaf and its physical container positions."""
        self.step = step
        self.container_index = container_index
        self.compute_index = compute_index


def _iter_compute_leaf_steps(step):
    """Yield real FWD/BWD steps, expanding composite containers."""
    if _is_compute_step(step):
        yield step
        return
    if _is_composite_compute_step(step):
        for sub_step in step.sub_steps:
            if _is_compute_step(sub_step):
                yield sub_step


def _collect_compute_leaves(order):
    """Collect compute leaves while counting each composite as one slot."""
    leaves = []
    container_by_compute_index = {}
    compute_index = 0
    for container_index, step in enumerate(order):
        leaf_steps = list(_iter_compute_leaf_steps(step))
        if not leaf_steps:
            continue
        container_by_compute_index[compute_index] = container_index
        for leaf_step in leaf_steps:
            leaves.append(_ComputeLeaf(leaf_step, container_index, compute_index))
        compute_index += 1
    return leaves, container_by_compute_index


def _append_after(after_steps, index, priority, step):
    after_steps[index].append((priority, step))


def _append_before(before_steps, index, priority, step):
    before_steps[index].append((priority, step))


def _iter_steps_by_priority(priority_steps):
    """Yield steps from high priority to low priority."""
    for _, step in sorted(priority_steps, key=lambda item: item[0], reverse=True):
        yield step


def _comm_block_anchor(order, index):
    """Return the last immediately following communication step."""
    anchor = index
    for next_index in range(index + 1, len(order)):
        next_step = order[next_index]
        if _is_comm_step(next_step):
            anchor = next_index
            continue
        break
    return anchor


def _post_compute_anchor(order, index, leaf_step=None):
    """Return the safe index after which post-compute swap steps may run."""
    from hyper_parallel.core.pipeline_parallel.scheduler import MetaStepType  # pylint: disable=C0415

    step = leaf_step if leaf_step is not None else order[index]
    fallback_anchor = _comm_block_anchor(order, index)
    if step.type == MetaStepType.FWD:
        send_type = MetaStepType.FWD_SEND
    elif step.type in (MetaStepType.BWD, MetaStepType.BWD_INPUT):
        send_type = MetaStepType.BWD_SEND
    else:
        return fallback_anchor

    for next_index in range(index + 1, fallback_anchor + 1):
        next_step = order[next_index]
        next_valid = (next_step is not None and next_step.type == send_type
                   and next_step.stage_index == step.stage_index and next_step.micro_index == step.micro_index)
        if next_valid:
            return next_index
        if next_step is not None and next_step.type == MetaStepType.BATCH_SEND_RECV:
            for sub_step in next_step.sub_steps:
                if (
                        sub_step.type == send_type
                        and sub_step.stage_index == step.stage_index
                        and sub_step.micro_index == step.micro_index):
                    return next_index
    return fallback_anchor


def _post_compute_launch_anchor(leaf):
    """Return the fallback point after compute where D2H may be launched."""
    return leaf.container_index


def _load_launch_anchor(
        order: List[Any], fwd_leaf: _ComputeLeaf, bwd_leaf: _ComputeLeaf,
        compute_between: List[int]) -> int:
    """Choose the latest safe H2D launch point for plain or FSDP execution."""
    from hyper_parallel.core.pipeline_parallel.scheduler import MetaStepType  # pylint: disable=C0415

    has_fsdp_steps = any(
        step is not None and step.type in (
            MetaStepType.FSDP_UNSHARD,
            MetaStepType.FSDP_RESHARD,
            MetaStepType.FSDP_REDUCE_GRAD,
        )
        for step in order
    )
    if not has_fsdp_steps:
        return compute_between[-1]

    for index in range(bwd_leaf.container_index - 1, fwd_leaf.container_index, -1):
        step = order[index]
        if (
                step is not None
                and step.type == MetaStepType.FSDP_UNSHARD
                and step.stage_index == bwd_leaf.step.stage_index):
            return index
    return bwd_leaf.container_index


def inject_pipeline_swap_steps(order: List[Any]) -> List[Any]:
    """Inject asynchronous transfer steps into one rank's pipeline order.

    Forward collection is executed directly by the forward leaf executor.
    Transfer launch/wait actions, including the H2D wait before the backward
    consumer container, appear in the top-level order.
    """
    from hyper_parallel.core.pipeline_parallel.scheduler import MetaStep, MetaStepType  # pylint: disable=C0415

    fwd_index = {}
    bwd_index = {}
    compute_leaves, container_by_compute_index = _collect_compute_leaves(order)
    for leaf in compute_leaves:
        step = leaf.step
        key = (step.stage_index, step.micro_index)
        if step.type == MetaStepType.FWD:
            fwd_index[key] = leaf
        elif step.type in (MetaStepType.BWD, MetaStepType.BWD_INPUT):
            # BWD_INPUT (dxdw split) needs activations → place WAIT_LOAD before it.
            # BWD_WEIGHT is intentionally excluded: it does not consume the
            # original forward activations restored by swap.
            bwd_index[key] = leaf

    before_steps = defaultdict(list)
    after_steps = defaultdict(list)
    chunk_gaps = {
        key: bwd_index[key].compute_index - fwd_leaf.compute_index
        for key, fwd_leaf in fwd_index.items()
        if key in bwd_index
    }
    for key, fwd_leaf in fwd_index.items():
        bwd_leaf = bwd_index.get(key)
        if bwd_leaf is None:
            continue
        if chunk_gaps[key] < MIN_SWAP_GAP:
            continue
        compute_between = [
            container_by_compute_index[index]
            for index in range(fwd_leaf.compute_index + 1, bwd_leaf.compute_index)
        ]
        if not compute_between:
            continue
        stage_index, micro_index = key

        first_between_anchor = _post_compute_anchor(order, compute_between[0])

        # Always launch offload immediately after the FWD container so that
        # the async D2H starts before any FSDP_RESHARD or FWD_SEND that may
        # sit between the FWD and the next compute step.
        fwd_anchor = _post_compute_launch_anchor(fwd_leaf)
        _append_after(
            after_steps, fwd_anchor, _AfterActionPriority.LAUNCH_OFFLOAD,
            MetaStep(micro_index, MetaStepType.SWAP_LAUNCH_OFFLOAD, stage_index),
        )

        _append_after(
            after_steps, first_between_anchor, _AfterActionPriority.WAIT_OFFLOAD,
            MetaStep(micro_index, MetaStepType.SWAP_WAIT_OFFLOAD, stage_index),
        )

        load_launch_anchor = _load_launch_anchor(order, fwd_leaf, bwd_leaf, compute_between)
        _append_before(
            before_steps, load_launch_anchor, _BeforeActionPriority.LAUNCH_LOAD,
            MetaStep(micro_index, MetaStepType.SWAP_LAUNCH_LOAD, stage_index),
        )
        _append_before(
            before_steps, bwd_leaf.container_index, _BeforeActionPriority.WAIT_LOAD,
            MetaStep(micro_index, MetaStepType.SWAP_WAIT_LOAD, stage_index),
        )
    injected = []
    for index, step in enumerate(order):
        injected.extend(_iter_steps_by_priority(before_steps[index]))
        injected.append(step)
        injected.extend(_iter_steps_by_priority(after_steps[index]))
    return injected


def _protect_pipeline_owned_tensors(step, schedule, arg_mbs, kwarg_mbs, group_name: str) -> None:
    """Keep long-lived module and pipeline-owned tensors alive on device.

    Swap offload clears the device storage of saved tensors after D2H copy.
    If a saved tensor aliases a parameter, registered buffer, or pipeline
    boundary tensor, clearing it would invalidate the long-lived owner. The
    alias protection below marks those saved tensors as keep-on-device.
    """
    stage = schedule._stage_dict[step.stage_index]  # pylint: disable=protected-access
    manager = SwapManager()

    # Saved-tensor hooks may receive a plain Tensor view of a Parameter. Some
    # backends do not preserve parameter metadata on that view, so protect by
    # storage ownership before any group member can be resized.
    parameters = tuple(param for _, param in platform.parameters_dict(stage.submodule))
    buffers = tuple(buffer for _, buffer in platform.buffers_dict(stage.submodule))
    if stage.is_first_stage:
        # First-stage inputs come from split_microbatches(), outside the
        # wrapped stage.  They are not stage outputs, but they can alias
        # tensors saved by the first layer and must not have their storage
        # resized by the swap group.
        boundary_inputs = (arg_mbs[step.micro_index], kwarg_mbs[step.micro_index])
    else:
        recv_infos = stage.args_recv_info.get(step.micro_index, ())
        boundary_inputs = tuple(info.buffer for info in recv_infos if info.buffer is not None)

    # Protect parameters, registered buffers, and pipeline-owned inputs in one
    # pass over the collected saved tensors. Forward outputs are protected
    # earlier by execute_fwd_leaf(), while the direct return value is still
    # available.
    manager.protect_alias_tensors(group_name, (parameters, buffers, boundary_inputs))


def swap_launch_offload(
        step: Any,
        schedule: Any,
        arg_mbs: List[Any],
        kwarg_mbs: List[Any],
        session: PipelineSwapSession) -> None:
    """Launch D2H for a pipeline swap group."""
    session.require_forward_context(step)
    group_name = session.group_name(step)
    manager = SwapManager()
    _protect_pipeline_owned_tensors(step, schedule, arg_mbs, kwarg_mbs, group_name)
    manager.launch_offload(group_name)


def swap_wait_offload(step: Any, session: PipelineSwapSession) -> None:
    """Wait for a pipeline swap group's D2H and release device storage."""
    SwapManager().wait_offload(session.group_name(step))


def swap_launch_load(step: Any, session: PipelineSwapSession) -> None:
    """Launch H2D for a pipeline swap group."""
    SwapManager().launch_load(session.group_name(step))


def swap_wait_load(step: Any, session: PipelineSwapSession) -> None:
    """Wait for a pipeline swap group's H2D before its backward container."""
    session.wait_load(step)
