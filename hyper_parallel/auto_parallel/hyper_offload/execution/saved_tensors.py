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
"""Replay immutable autograd saves without intercepting unrelated operators."""

from __future__ import annotations

import weakref
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from hyper_parallel.auto_parallel.hyper_offload.execution.native_saved_tensor import get_native_saved_tensor
from hyper_parallel.auto_parallel.hyper_offload.execution.tensor import ShadowTensor
from hyper_parallel.auto_parallel.hyper_offload.ir.schedule import ResidencyActionType as Action
from hyper_parallel.auto_parallel.hyper_offload.ir.trace import AccessKind


@dataclass(frozen=True, slots=True)
class SavedSlot:
    """Metadata for a saved value; never owns the recorded activation."""

    storage_id: int | None
    metadata: tuple
    op_id: int


def tensor_metadata(tensor: torch.Tensor) -> tuple:
    """Describe the saved view without reading device data.

    Args:
        tensor: Saved activation or parameter view.
    """
    return (tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset(), tensor.dtype, tensor.device)


class SavedTensorRecorder:
    """Associate autograd save slots with the warmup storage identities."""

    def __init__(self, executor: Any) -> None:
        """Bind the executor for the duration of a complete warmup step.

        Args:
            executor: Warmup executor recording the same step.
        """
        self.executor = executor
        self.slots: list[SavedSlot] = []
        self.unpacks: list[tuple[int, int]] = []
        self.save_unpack_counts: list[int] = []

    def pack(self, tensor: torch.Tensor) -> tuple:
        """Record a save without retaining a tensor in the persistent guide.

        Args:
            tensor: Tensor being saved by autograd.
        """
        sid = (tensor.storage_id if isinstance(tensor, ShadowTensor)
               else self.executor._tracker.get_activation_sid(tensor))
        index = len(self.slots)
        self.slots.append(SavedSlot(sid, tensor_metadata(tensor), self.executor.op_idx))
        self.save_unpack_counts.append(len(self.unpacks))
        return index, tensor.detach(), tensor._version

    def unpack(self, packed: tuple) -> torch.Tensor:
        """Record the backward access anchor and restore the saved tensor.

        Args:
            packed: Save slot and detached tensor from pack.
        """
        index, tensor, version = packed
        if tensor._version != version:
            raise RuntimeError("a tensor saved for backward was modified in place")
        self.unpacks.append((index, self.executor.op_idx))
        return tensor


class NativeSavedValue(Protocol):
    """Native and Python saved values expose the same replay access contract."""

    slot_index: int | None

    def unpack(self) -> torch.Tensor:
        """Return a saved view after checking its shared version counter."""


@dataclass(slots=True)
class SavedNative:
    """Keep native autograd values detached and preserve mutation checks."""

    tensor: torch.Tensor
    version: int
    slot_index: int | None = None

    def unpack(self) -> torch.Tensor:
        """Check mutations before returning a Python-managed saved view."""
        if self.tensor._version != self.version:
            raise RuntimeError("a tensor saved for backward was modified in place")
        return self.tensor


@dataclass(frozen=True, slots=True)
class SavedActivation:
    """Reference moved storage without retaining its original device bytes."""

    storage_id: int
    metadata: tuple
    slot_index: int | None = None


class SavedTensorReplay:
    """Run a validated forward-save/backward-use plan through autograd hooks.

    This path requires immutable, non-retained forward activations with one
    offload interval each. Other schedules use the operator replay executor.
    Like operator replay, the saved-slot sequence is a static trace contract.
    """

    def __init__(self, manager: Any, slots: list[SavedSlot], selected: set[int]) -> None:
        """Create a replay guide containing metadata only.

        Args:
            manager: Existing physical residency controller.
            slots: Recorded autograd saves.
            selected: Storage IDs selected by the unchanged planner.
        """
        self.manager = manager
        self._native_saved_tensor = get_native_saved_tensor()
        self.slot_count = len(slots)
        self.slots = {index: slot for index, slot in enumerate(slots) if slot.storage_id in selected}
        self.selected = selected
        self._next_slot = 0
        self._bound: dict[int, tuple[int, list[tuple[Any, int]]]] = {}
        self._unpacked = 0
        self._prefetched = False
        self._pending_release: set[int] = set()
        self.prefetch_at: dict[int, list[int]] = {0: list(selected)}
        self.release_at: dict[int, list[int]] = {}
        self.unpack_count: int | None = None
        self._next_unpack = 0
        self._restored: set[int] = set()
        self._remaining_uses: dict[int, int] = dict(Counter(slot.storage_id for slot in self.slots.values()))
        self._use_counts = dict(self._remaining_uses)
        self.save_unpack_counts: list[int] | None = None
        self.unpack_slots: list[int] | None = None
        self._forward_slot_count = len(slots)

    @staticmethod
    def add_planning_hints(recorder: SavedTensorRecorder, trace: Any) -> None:
        """Prefer immutable forward-save/backward-use gaps without relaxing budgets.

        Args:
            recorder: Completed autograd save and use recording.
            trace: Planner input receiving optional candidate preferences.
        """
        if not recorder.unpacks:
            return
        backward = recorder.unpacks[0][1]
        used = {recorder.slots[index].storage_id for index, _ in recorder.unpacks}
        first_saves = {}
        for slot in recorder.slots:
            if slot.storage_id is not None and slot.storage_id in used and slot.op_id < backward:
                first_saves.setdefault(slot.storage_id, slot.op_id)
        positions, writes = {}, {}
        for op_id, op in enumerate(trace.ops):
            for access in op.accesses:
                sid = access.storage_id
                if sid not in first_saves:
                    continue
                rows = positions.setdefault(sid, [])
                if not rows or rows[-1] != op_id:
                    rows.append(op_id)
                if access.kind == AccessKind.WRITE:
                    writes.setdefault(sid, []).append(op_id)
        for sid, accesses in positions.items():
            boundary = bisect_left(accesses, backward)
            if not 0 < boundary < len(accesses) or sid in trace.retained_sids:
                continue
            if not SavedTensorReplay._writes_are_immutable(first_saves[sid], writes.get(sid, []), trace):
                continue
            start, end = accesses[boundary - 1], accesses[boundary]
            if end > start + 1:
                trace.preferred_eviction_gaps.append((sid, start, end))

    @staticmethod
    def _immutable_save_slots(sid: int, recorder: SavedTensorRecorder, trace: Any) -> bool:
        slots = [index for index, slot in enumerate(recorder.slots) if slot.storage_id == sid]
        used = [index for index, _ in recorder.unpacks if index in slots]
        if not used:
            return False
        return SavedTensorReplay._immutable_writes(sid, recorder.slots[slots[0]].op_id, trace)

    @staticmethod
    def _immutable_writes(sid: int, first_save: int, trace: Any) -> bool:
        writes = [op_id for op_id, op in enumerate(trace.ops)
                  if any(access.storage_id == sid and access.kind == AccessKind.WRITE for access in op.accesses)]
        return SavedTensorReplay._writes_are_immutable(first_save, writes, trace)

    @staticmethod
    def _writes_are_immutable(first_save: int, writes: list[int], trace: Any) -> bool:
        aliases = {"detach.default", "view.default", "_unsafe_view.default", "alias.default", "t.default",
                   "transpose.int", "permute.default", "slice.Tensor", "select.int", "as_strided.default",
                   "squeeze.default", "squeeze.dim", "unsqueeze.default", "expand.default",
                   "split.Tensor", "split.sizes", "split_with_sizes.default", "unbind.int"}
        if not writes or writes[0] > first_save:
            return False
        # Initialization can update storage before its first autograd save.
        # Later writes must only create aliases of the saved bytes.
        return all(op_id < first_save or trace.ops[op_id].name in aliases for op_id in writes[1:])

    @staticmethod
    def _copy_actions(schedule: Any) -> list:
        copies = []
        for phase in (schedule.pre, schedule.post):
            for actions in phase.values():
                copies.extend(action for action in actions if action.kind in (Action.COPY_D2H, Action.COPY_H2D))
        return copies

    @classmethod
    def from_trace(
        cls, manager: Any, recorder: SavedTensorRecorder, trace: Any, schedule: Any,
    ) -> SavedTensorReplay | None:
        """Compile complete immutable intervals onto autograd save/use anchors.

        Args:
            manager: Physical residency controller shared with operator replay.
            recorder: Completed save and unpack recording.
            trace: Warmup trace passed to the planner.
            schedule: Schedule returned by the planner.
        """
        copies = cls._copy_actions(schedule)
        counts = Counter((action.storage_id, action.kind) for action in copies)
        selected = {action.storage_id for action in copies}
        if not selected or not recorder.unpacks or selected & trace.retained_sids:
            return None
        if any(counts[sid, kind] != 1 for sid in selected for kind in (Action.COPY_D2H, Action.COPY_H2D)):
            return None
        if not cls._forward_intervals(copies, recorder):
            return None
        if not all(cls._immutable_save_slots(sid, recorder, trace) for sid in selected):
            return None
        replay = cls(manager, recorder.slots, selected)
        replay.save_unpack_counts = getattr(recorder, "save_unpack_counts", None)
        replay.unpack_slots = [index for index, _ in recorder.unpacks]
        replay._forward_slot_count = (
            replay.save_unpack_counts.count(0) if replay.save_unpack_counts is not None
            else sum(slot.op_id < recorder.unpacks[0][1] for slot in recorder.slots)
        )
        replay._use_counts = dict(Counter(
            recorder.slots[index].storage_id for index, _ in recorder.unpacks if index in replay.slots
        ))
        replay._remaining_uses = dict(replay._use_counts)
        replay._compile_prefetch(copies, recorder)
        replay._compile_releases(schedule, recorder)
        return replay

    @staticmethod
    def _forward_intervals(copies: list, recorder: SavedTensorRecorder) -> bool:
        first_backward = recorder.unpacks[0][1]
        # Checkpoint recomputation can save additional aliases after backward
        # starts. Each moved storage must still be saved in the forward.
        for sid in {action.storage_id for action in copies}:
            saves = [slot.op_id for slot in recorder.slots if slot.storage_id == sid]
            if not saves or min(saves) >= first_backward:
                return False
        for action in copies:
            if action.kind == Action.COPY_D2H and action.op_id >= first_backward:
                return False
            if action.kind == Action.COPY_H2D and action.op_id < first_backward:
                return False
        return True

    def _compile_prefetch(self, copies: list, recorder: SavedTensorRecorder) -> None:
        self.prefetch_at.clear()
        unpack_ops = [op_id for _, op_id in recorder.unpacks]
        # Round prefetch later to the next autograd access. Moving all restores
        # to backward entry would violate the planner's device-memory budget.
        for action in copies:
            if action.kind == Action.COPY_H2D:
                anchor = bisect_left(unpack_ops, action.op_id)
                first_use = next(index for index, (slot, _) in enumerate(recorder.unpacks)
                                 if recorder.slots[slot].storage_id == action.storage_id)
                self.prefetch_at.setdefault(min(anchor, first_use), []).append(action.storage_id)
        self.unpack_count = len(recorder.unpacks)

    def _compile_releases(self, schedule: Any, recorder: SavedTensorRecorder) -> None:
        save_ops = [slot.op_id for slot in recorder.slots]
        actions = [action for phase in (schedule.pre, schedule.post) for group in phase.values() for action in group]
        for sid in self.selected:
            restore_op = next(action.op_id for action in actions
                              if action.storage_id == sid and action.kind == Action.COPY_H2D)
            deadlines = [action.op_id for action in actions if action.storage_id == sid
                         and action.kind == Action.RELEASE_DEVICE and action.op_id < restore_op]
            if deadlines:
                anchor = bisect_left(save_ops, min(deadlines))
                self.release_at.setdefault(anchor, []).append(sid)

    def _release_due_sources(self, index: int) -> None:
        for sid in self.release_at.get(index, ()):
            if sid in self._pending_release:
                self.manager.release_device(sid)
                self._pending_release.remove(sid)

    def reset(self) -> None:
        """Reset metadata after the preceding step has released its graph."""
        self._next_slot = 0
        self._bound.clear()
        self._unpacked = 0
        self._prefetched = False
        self._pending_release.clear()
        self._next_unpack = 0
        self._restored.clear()
        self._remaining_uses = dict(self._use_counts)

    def _retire_sources(self) -> None:
        for sid in tuple(self._pending_release):
            if self.manager.try_release_device(sid):
                self._pending_release.remove(sid)

    def pack(self, tensor: torch.Tensor) -> NativeSavedValue | SavedActivation:
        """Save unrelated values natively and move only planned immutable storage.

        Args:
            tensor: Value saved for backward by native autograd.
        """
        index = self._next_slot
        self._next_slot += 1
        if index >= self.slot_count:
            raise RuntimeError("saved-tensor replay save sequence differs from warmup")
        if self.save_unpack_counts is not None and self.save_unpack_counts[index] != self._next_unpack:
            raise RuntimeError("saved-tensor replay save/use order differs from warmup")
        if self._pending_release:
            self._release_due_sources(index)
        slot = self.slots.get(index)
        if slot is None:
            if self._pending_release and index % 32 == 0:
                self._retire_sources()
            if self._native_saved_tensor is not None and type(tensor) is torch.Tensor:
                return self._native_saved_tensor(tensor, index)
            return SavedNative(tensor.detach(), tensor._version, index)
        if tensor_metadata(tensor) != slot.metadata:
            raise RuntimeError(f"saved-tensor replay metadata differs at slot={index}")
        sid = slot.storage_id
        storage_id = tensor.untyped_storage()._cdata
        version = tensor._version
        if sid not in self._bound:
            self._bound[sid] = (storage_id, [])
            self.manager.bind(sid, tensor)
            self.manager.copy_d2h(sid)
            self._pending_release.add(sid)
        if sid in self._restored:
            storage_id_expected = self.manager.device_storage_id(sid)
        else:
            storage_id_expected = self._bound[sid][0]
        if storage_id_expected != storage_id:
            raise RuntimeError(f"saved-tensor replay alias differs at slot={index}")
        self._bound[sid][1].append((weakref.ref(tensor), version))
        return SavedActivation(sid, slot.metadata, index)

    def _validate_forward(self) -> None:
        if self._next_slot != self._forward_slot_count or set(self._bound) != self.selected:
            raise RuntimeError("saved-tensor replay forward save count differs from warmup")
        for sid in self._bound:
            self._validate_versions(sid)
        self._release_due_sources(self._next_slot)
        self._prefetched = True

    def _validate_versions(self, sid: int) -> None:
        for reference, version in self._bound[sid][1]:
            tensor = reference()
            if tensor is not None and tensor._version != version:
                raise RuntimeError("an offloaded saved tensor was modified in place")

    def _prefetch(self, sid: int) -> None:
        self.manager.release_device(sid)
        self.manager.copy_h2d(sid)
        self.manager.release_host(sid)
        self._pending_release.discard(sid)
        self._restored.add(sid)

    def unpack(self, packed: NativeSavedValue | SavedActivation) -> torch.Tensor:
        """Prefetch at compiled backward anchors and resolve moved saved views.

        Args:
            packed: Native saved value or activation metadata.
        """
        if not self._prefetched:
            self._validate_forward()
        if self.unpack_slots is not None:
            if (self._next_unpack >= len(self.unpack_slots)
                    or packed.slot_index != self.unpack_slots[self._next_unpack]):
                raise RuntimeError("saved-tensor replay unpack sequence differs from warmup")
        for sid in self.prefetch_at.get(self._next_unpack, ()):
            self._prefetch(sid)
        self._next_unpack += 1
        if not isinstance(packed, SavedActivation):
            return packed.unpack()
        if packed.storage_id not in self._restored:
            raise RuntimeError("saved-tensor replay unpack sequence differs from warmup")
        self._validate_versions(packed.storage_id)
        self._unpacked += 1
        shape, stride, offset, dtype, device = packed.metadata
        storage = self.manager.device_storage(packed.storage_id)
        result = torch.empty(0, dtype=dtype, device=device)
        result.set_(storage, offset, shape, stride)
        self._remaining_uses[packed.storage_id] -= 1
        if self._remaining_uses[packed.storage_id] == 0:
            # The returned native view owns its storage until its backward
            # consumer finishes; the manager must not retain it afterwards.
            self.manager.release_device(packed.storage_id)
        return result

    def finish(self) -> None:
        """Require the complete recorded step and release restored device storage."""
        if self._next_slot != self.slot_count or self._unpacked != sum(self._use_counts.values()):
            raise RuntimeError("saved-tensor replay step differs from warmup")
        if self.unpack_count is not None and self._next_unpack != self.unpack_count:
            raise RuntimeError("saved-tensor replay backward use count differs from warmup")
        for sid in self.selected:
            self.manager.release_device(sid)
