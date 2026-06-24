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
from enum import IntEnum

from hyper_parallel.core.activation_checkpoint.swap import SwapManager

MIN_SWAP_GAP = 4


class _BeforeActionPriority(IntEnum):
    SET_GROUP = 10
    WAIT_LOAD = 20
    LAUNCH_LOAD = 30


class _AfterActionPriority(IntEnum):
    WAIT_OFFLOAD = 10
    LAUNCH_OFFLOAD = 20


def pp_swap_group_name(stage_index: int, micro_index: int) -> str:
    """Return the swap group name for a pipeline chunk."""
    return f"pp_swap_s{stage_index}_m{micro_index}"


def _is_compute_step(step) -> bool:
    from hyper_parallel.core.pipeline_parallel.scheduler import MetaStepType  # pylint: disable=C0415

    return step is not None and step.type in (MetaStepType.FWD, MetaStepType.BWD)


def _is_comm_step(step) -> bool:
    from hyper_parallel.core.pipeline_parallel.scheduler import MetaStepType  # pylint: disable=C0415

    return step is not None and step.type in (
        MetaStepType.FWD_RECV,
        MetaStepType.FWD_SEND,
        MetaStepType.BWD_RECV,
        MetaStepType.BWD_SEND,
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

    def __init__(self, step, container_index, compute_index):
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
    elif step.type == MetaStepType.BWD:
        send_type = MetaStepType.BWD_SEND
    else:
        return fallback_anchor

    for next_index in range(index + 1, fallback_anchor + 1):
        next_step = order[next_index]
        next_valid = (next_step is not None and next_step.type == send_type
                   and next_step.stage_index == step.stage_index and next_step.micro_index == step.micro_index)
        if next_valid:
            return next_index
    return fallback_anchor


def inject_pipeline_swap_steps(order):
    """Inject SWAP_* steps into one rank's pipeline order.

    The injected order preserves the required lifecycle:
    SET_GROUP -> FWD -> LAUNCH_OFFLOAD -> WAIT_OFFLOAD ->
    LAUNCH_LOAD -> WAIT_LOAD -> BWD.
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
        elif step.type == MetaStepType.BWD:
            bwd_index[key] = leaf

    before_steps = defaultdict(list)
    after_steps = defaultdict(list)
    for key, fwd_leaf in fwd_index.items():
        bwd_leaf = bwd_index.get(key)
        if bwd_leaf is None:
            continue
        if bwd_leaf.compute_index - fwd_leaf.compute_index < MIN_SWAP_GAP:
            continue
        compute_between = [
            container_by_compute_index[index]
            for index in range(fwd_leaf.compute_index + 1, bwd_leaf.compute_index)
        ]
        if not compute_between:
            continue
        stage_index, micro_index = key

        _append_before(
            before_steps, fwd_leaf.container_index, _BeforeActionPriority.SET_GROUP,
            MetaStep(micro_index, MetaStepType.SWAP_SET_GROUP, stage_index),
        )
        fwd_anchor = _post_compute_anchor(order, fwd_leaf.container_index, fwd_leaf.step)
        first_between_anchor = _post_compute_anchor(order, compute_between[0])

        _append_after(
            after_steps, fwd_anchor, _AfterActionPriority.LAUNCH_OFFLOAD,
            MetaStep(micro_index, MetaStepType.SWAP_LAUNCH_OFFLOAD, stage_index),
        )

        _append_after(
            after_steps, first_between_anchor, _AfterActionPriority.WAIT_OFFLOAD,
            MetaStep(micro_index, MetaStepType.SWAP_WAIT_OFFLOAD, stage_index),
        )

        _append_before(
            before_steps, compute_between[-1], _BeforeActionPriority.LAUNCH_LOAD,
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


def _protect_pipeline_owned_tensors(step, schedule, arg_mbs, kwarg_mbs) -> None:
    """Keep pipeline-owned boundary tensors alive on device.

    Swap offload clears the device storage of saved tensors after D2H copy.
    If a saved tensor aliases a pipeline boundary tensor, clearing it would
    also invalidate the object still held by the pipeline runtime.  The alias
    protection below marks those saved tensors as keep-on-device.
    """
    stage = schedule._stage_dict[step.stage_index]  # pylint: disable=protected-access
    group_name = pp_swap_group_name(step.stage_index, step.micro_index)
    manager = SwapManager()

    if stage.is_first_stage:
        # First-stage inputs come from split_microbatches(), outside the
        # wrapped stage.  They are not stage outputs, but they can alias
        # tensors saved by the first layer and must not have their storage
        # resized by the swap group.
        manager.protect_alias_tensors(
            group_name,
            (arg_mbs[step.micro_index], kwarg_mbs[step.micro_index]),
        )

    if stage.is_last_stage:
        # Last-stage outputs are consumed by the schedule as losses / sens
        # roots, so they must stay device-valid until backward has used them.
        outputs = stage.fwd_outputs_cache.get(step.micro_index)
        if outputs is None:
            outputs = stage.last_stage_outputs
        if outputs is not None:
            manager.protect_alias_tensors(group_name, outputs)


def swap_set_group(step) -> None:
    """Set the active SwapManager group for the next pipeline forward chunk."""
    group_name = pp_swap_group_name(step.stage_index, step.micro_index)
    manager = SwapManager()
    manager.ensure_group(group_name)
    manager.set_current_group_name(group_name)


def swap_launch_offload(step, schedule, arg_mbs, kwarg_mbs) -> None:
    """Launch D2H for a pipeline swap group."""
    group_name = pp_swap_group_name(step.stage_index, step.micro_index)
    manager = SwapManager()
    _protect_pipeline_owned_tensors(step, schedule, arg_mbs, kwarg_mbs)
    manager.launch_offload(group_name)
    # Only the immediately preceding forward belongs to this swap group.
    manager.set_current_group_name("")


def swap_wait_offload(step) -> None:
    """Wait for a pipeline swap group's D2H and release device storage."""
    SwapManager().wait_offload(pp_swap_group_name(step.stage_index, step.micro_index))


def swap_launch_load(step) -> None:
    """Launch H2D for a pipeline swap group."""
    SwapManager().launch_load(pp_swap_group_name(step.stage_index, step.micro_index))


def swap_wait_load(step) -> None:
    """Wait for a pipeline swap group's H2D before backward uses activations."""
    SwapManager().wait_load(pp_swap_group_name(step.stage_index, step.micro_index))
