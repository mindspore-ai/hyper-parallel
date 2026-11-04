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
# Adapted from torchtitan-npu's graph_trainer ACLGraph implementation.
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# (c) Meta Platforms, Inc. and affiliates.
# The original BSD-3-Clause license is retained in
# Third_Party_Open_Source_Software_Notice.
"""ACLGraph capture/replay, input guards and buffer ownership."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import timedelta
import importlib
import logging
import os
from operator import itemgetter
from typing import Any

import torch
from torch._C._dynamo.guards import RootGuardManager
from torch._dynamo.guards import GuardManagerType
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._pytree import keystr, tree_flatten, tree_flatten_with_path, tree_leaves, tree_map, tree_unflatten

logger = logging.getLogger(__name__)


def _get_torch_npu() -> Any:
    # Keep this optional dependency out of CPU imports and disabled pipelines.
    return importlib.import_module("torch_npu")


def prepare_aclgraph_runtime() -> None:
    """Select a task-queue mode supported by NPU graph recording.

    Raises:
        ValueError: If task queue mode 2 cannot be switched dynamically.
    """
    npu = _get_torch_npu().npu
    get_mode = getattr(npu, "get_task_queue_enable", None)
    mode = get_mode() if get_mode is not None else int(os.environ.get("TASK_QUEUE_ENABLE", "1"))
    if mode == 2:
        set_mode = getattr(npu, "set_task_queue_enable", None)
        if set_mode is None:
            raise ValueError("ACLGraph requires TASK_QUEUE_ENABLE=0/1 or the dynamic task-queue API")
        set_mode(1)
        logger.info("ACLGraph switched task queue mode from 2 to 1 before recording")


class _RecaptureCoordinator:
    """Agree across ranks before capture/replay, without capturing a collective."""

    def __init__(self, *, coordinate_recapture: bool = True) -> None:
        """Track the common capture order and recording-pool generation."""
        self.wrappers = []
        self.group: torch.distributed.ProcessGroup | None = None
        self.initialized = False
        self.graph_pool = None
        self.coordinate_recapture = coordinate_recapture

    def _begin_step(self):
        pools = {value.pool_id for value in self.wrappers if value.graph is not None}
        if len(pools) > 1:
            torch_npu = _get_torch_npu()

            # Reclaim old generations before another step records a third
            # pool. Escaping gradients are independent copies by contract.
            torch_npu.npu.synchronize()
            for value in self.wrappers:
                value.teardown()
            logger.info("ACLGraph: recycling mixed capture generations at step boundary")
        # A new recording generation must not allocate over live buffers
        # from graphs replayed this step: those belong to older pools.
        self.graph_pool = None

    def _initialize(self):
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            group = torch.distributed.new_group(backend="gloo", timeout=timedelta(seconds=600))
            self.group = group
            names = [value.label for value in self.wrappers]
            gathered: list[list[str] | None] = [None] * torch.distributed.get_world_size(group)
            torch.distributed.all_gather_object(gathered, names, group=group)
            if any(value != names for value in gathered):
                self.teardown()
                raise ValueError("ACLGraph requires identical region order across ranks")
        self.initialized = True

    def before_call(self, wrapper: "ACLGraphWrapper", args: list[Any]) -> None:
        """Agree on capture state before executing any captured collective."""
        if wrapper is self.wrappers[0]:
            self._begin_step()
        if not self.initialized:
            self._initialize()
        reason = wrapper._input_change_reason(args) if wrapper.graph is not None else None
        if not self.coordinate_recapture and wrapper.graph is not None:
            if reason is not None:
                raise ValueError(
                    f"ACLGraph static replay inputs changed at {wrapper.label}: {reason}. "
                    "Use aclgraph_mode='dynamic' for coordinated recapture."
                )
            return
        self._coordinate(wrapper, reason)

    def _coordinate(self, wrapper, reason):
        flags = torch.tensor(
            [int(reason is not None), int(wrapper.graph is None), int(wrapper.graph is not None)],
            device="cpu",
            dtype=torch.int32,
        )
        if self.group is not None:
            with torch.profiler.record_function("ACLGraph/rank_coordination"):
                torch.distributed.all_reduce(flags, op=torch.distributed.ReduceOp.MAX, group=self.group)
        if flags[0].item() or (flags[1].item() and flags[2].item()):
            torch_npu = _get_torch_npu()

            torch_npu.npu.synchronize()
            # New captures use this step's fresh pool, never an older replay's
            # allocator state. There is no checkpoint rollback across HCCL.
            wrapper.teardown()
            logger.info(
                "ACLGraph %s: input signature changed; recapturing region; reason=%s",
                wrapper.label,
                reason or "another rank changed its inputs or capture state",
            )

    def teardown(self) -> None:
        """Release captured buffers or the coordination process group."""
        if self.group is not None:
            torch.distributed.destroy_process_group(self.group)
            self.group = None
        self.graph_pool = None
        self.initialized = False


class ACLGraphWrapper:
    """Warm up once, then capture/replay using independent input buffers.

    Lifted state and mutated inputs retain their addresses. Escaping gradients
    are copied: GraphTrainer may assign a returned gradient to ``param.grad``.
    Intermediates use non-owning replay handles; owning capture-time results
    follow normal FX liveness. New recordings never allocate in an older pool.
    """

    def __init__(
        self,
        runnable: Callable[..., Any],
        *,
        static_input_indices: Iterable[int],
        capture_runnable: Callable[..., Any] | None = None,
        label: str = "joint step",
        clone_outputs: bool = True,
        recapture_coordinator: _RecaptureCoordinator | None = None,
        borrowed_input_indices: Iterable[int] = (),
        capture_boxed: bool = False,
    ) -> None:
        """Bind a region and declare its lifted-state and borrowed inputs."""
        self.runnable = runnable
        self.capture_runnable = capture_runnable or runnable
        self.label = label
        self.static_input_indices = set(static_input_indices)
        self.clone_outputs = clone_outputs
        self.recapture_coordinator = recapture_coordinator
        self.borrowed_input_indices = set(borrowed_input_indices)
        self.capture_boxed = capture_boxed
        self.warmed_up = False
        self.graph = None
        self.pool_id = None
        self.args = None
        self.input_specs = None
        self.output = None
        self._flat_outputs = None
        self._output_spec = None
        self._clone_output_indices: tuple[int, ...] = ()
        self._input_guard = None
        self._static_pointers = ()
        self._scalar_inputs = ()
        self._container_inputs = {}

    def _prepare_input_guard(self, args):
        guard = RootGuardManager()
        guard.add_length_check_guard(len(args), [], None)
        pointers, scalars = [], []
        self._container_inputs = {}
        for i, arg in enumerate(args):
            if not isinstance(arg, torch.Tensor):
                leaves, structure = tree_flatten_with_path(arg)
                if structure.is_leaf():
                    scalars.append((i, arg))
                else:
                    # Containers are borrowed as-is during capture. Their tensor
                    # leaves need address guards, never Python value equality.
                    self._container_inputs[i] = (
                        structure,
                        [
                            (
                                keystr(path),
                                None if isinstance(value, torch.Tensor) else value,
                                (value.shape, value.stride(), value.dtype, value.device, value.data_ptr())
                                if isinstance(value, torch.Tensor)
                                else None,
                            )
                            for path, value in leaves
                        ],
                    )
                continue
            manager = guard.list_getitem_manager(i, f"input[{i}]", arg, GuardManagerType.GUARD_MANAGER)
            manager.add_tensor_match_guard(
                arg,
                list(arg.shape),
                list(arg.stride()),
                f"input[{i}]",
                [],
                None,
                type(arg),
                torch._C._dispatch_keys(arg),
            )
            if i in self.static_input_indices:
                pointers.append((i, arg.data_ptr()))
        self._input_guard = guard
        self._static_pointers = tuple(pointers)
        self._scalar_inputs = tuple(scalars)

    @staticmethod
    def _tensor_change_reason(arg, spec, *, static):
        if not isinstance(arg, torch.Tensor):
            return f"Tensor -> {type(arg).__name__}"
        for field, old, new in zip(
            ("shape", "stride", "dtype", "device"),
            spec[:4],
            (arg.shape, arg.stride(), arg.dtype, arg.device),
            strict=True,
        ):
            if old != new:
                return f"{field} {old} -> {new}"
        if static and arg.data_ptr() != spec[4]:
            return f"static address {spec[4]:#x} -> {arg.data_ptr():#x}"
        return None

    def _container_change_reason(self, args):
        for i, (structure, saved_leaves) in self._container_inputs.items():
            leaves, current_structure = tree_flatten_with_path(args[i])
            if current_structure != structure:
                return f"input[{i}] container structure changed"
            for (_, value), (path, saved, spec) in zip(leaves, saved_leaves, strict=True):
                if spec is not None:
                    reason = self._tensor_change_reason(value, spec, static=True)
                    if reason is not None:
                        return f"input[{i}]{path} {reason}"
                elif value is not saved and (type(value) is not type(saved) or value != saved):
                    return f"input[{i}]{path} scalar {saved!r} -> {value!r}"
        return None

    def _fast_change_reason(self, args):
        for i, address in self._static_pointers:
            if args[i].data_ptr() != address:
                return f"input[{i}] static address {address:#x} -> {args[i].data_ptr():#x}"
        for i, saved in self._scalar_inputs:
            arg = args[i]
            if arg is not saved and (type(arg) is not type(saved) or arg != saved):
                return f"input[{i}] scalar {saved!r} -> {arg!r}"
        return None

    def _input_change_reason(self, args):
        assert self.args is not None and self.input_specs is not None
        if len(args) != len(self.args):
            return f"argument count {len(self.args)} -> {len(args)}"
        reason = self._container_change_reason(args)
        if reason is not None:
            return reason
        # C++ guards check all tensor layouts without a Python loop per input.
        # Address checks remain explicit: tensor identity does not guard .data.
        if self._input_guard is not None and self._input_guard.check(args):
            return self._fast_change_reason(args)
        for i, (arg, saved, spec) in enumerate(zip(args, self.args, self.input_specs, strict=True)):
            if isinstance(saved, torch.Tensor):
                assert spec is not None
                reason = self._tensor_change_reason(arg, spec, static=i in self.static_input_indices)
                if reason is not None:
                    return f"input[{i}] {reason}"
            elif (
                i not in self._container_inputs and arg is not saved and (type(arg) is not type(saved) or arg != saved)
            ):
                return f"input[{i}] scalar {saved!r} -> {arg!r}"
        return None

    def __call__(self, *args: Any) -> Any:
        """Execute a region using positional graph inputs."""
        return self.call_boxed(list(args))

    def _prepare_capture_inputs(self, args):
        self.args = tuple(
            arg if i in self.static_input_indices or not isinstance(arg, torch.Tensor) else arg.detach().clone()
            for i, arg in enumerate(args)
        )
        self.input_specs = [
            (arg.shape, arg.stride(), arg.dtype, arg.device, arg.data_ptr())
            if isinstance(arg, torch.Tensor)
            else None
            for arg in args
        ]
        self._prepare_input_guard(args)
        for i, (arg, saved) in enumerate(zip(args, self.args, strict=True)):
            if isinstance(arg, torch.Tensor) and arg.stride() != saved.stride():
                raise ValueError(
                    f"ACLGraph {self.label}: dynamic input[{i}] copy must preserve strides; "
                    f"shape={tuple(arg.shape)}, source={arg.stride()}, copied={saved.stride()}"
                )

    def _replay_handle_factory(self, torch_npu):
        protected_storages = {
            StorageWeakRef(value.untyped_storage())
            for i, arg in enumerate(self.args)
            if i not in self.borrowed_input_indices
            for value in tree_leaves(arg)
            if isinstance(value, torch.Tensor)
        }

        def _replay_handle(value):
            if (
                self.recapture_coordinator is not None
                and isinstance(value, torch.Tensor)
                and value.device.type == "npu"
                and StorageWeakRef(value.untyped_storage()) not in protected_storages
            ):
                return torch_npu._C._weak_ref_tensor(value)
            return value

        return _replay_handle

    def _capture(self, args, torch_npu):
        self._prepare_capture_inputs(args)
        graph = torch_npu.npu.NPUGraph()
        coordinator = self.recapture_coordinator
        if coordinator is not None:
            if coordinator.graph_pool is None:
                coordinator.graph_pool = torch_npu.npu.graph_pool_handle()
            pool = coordinator.graph_pool
        else:
            pool = torch_npu.npu.graph_pool_handle()
        capture_inputs = list(self.args)
        _replay_handle = self._replay_handle_factory(torch_npu)
        self.args = tuple(
            tree_map(_replay_handle, arg) if i in self.borrowed_input_indices else arg
            for i, arg in enumerate(self.args)
        )
        args.clear()
        # All regions use the same default capture stream and execute once,
        # in joint-FX order. Capture-time outputs follow the caller's normal
        # liveness; replay only needs non-owning handles to their addresses.
        with torch_npu.npu.graph(graph, pool=pool):
            self.output = (
                self.capture_runnable(capture_inputs)
                if self.capture_boxed
                else self.capture_runnable(*capture_inputs)
            )
            # Join pending HCCL streams before ending this region's capture.
            for value in tree_leaves(self.output):
                if isinstance(value, torch.Tensor) and value.device.type == "npu":
                    torch.ops._c10d_functional.wait_tensor.default(value)
        self.graph = graph
        self.pool_id = pool
        captured_output = self.output
        if coordinator is not None:
            self.output = tree_map(_replay_handle, captured_output)
            owner = getattr(self.capture_runnable, "__self__", None)
            if isinstance(owner, torch.fx.Interpreter):
                owner.env.clear()
        if self.clone_outputs:
            self._prepare_output_clones()
        logger.info("ACLGraph %s: capture SUCCEEDED", self.label)
        return captured_output

    def _prepare_output_clones(self):
        static_storages = {
            StorageWeakRef(value.untyped_storage())
            for i, arg in enumerate(self.args)
            if i in self.static_input_indices and i not in self.borrowed_input_indices
            for value in tree_leaves(arg)
            if isinstance(value, torch.Tensor)
        }
        self._flat_outputs, self._output_spec = tree_flatten(self.output)
        self._clone_output_indices = tuple(
            i
            for i, value in enumerate(self._flat_outputs)
            if isinstance(value, torch.Tensor)
            and StorageWeakRef(value.untyped_storage()) not in static_storages
        )

    def _replay_outputs(self, captured_output):
        if not self.clone_outputs:
            # During recording, owning tensors keep inputs live for subsequent
            # captures until their last FX consumer. Replay allocates no tensors
            # in this pool, so non-owning handles suffice on subsequent steps.
            return captured_output if captured_output is not None else self.output
        assert self._flat_outputs is not None and self._output_spec is not None
        outputs = list(self._flat_outputs)
        for i in self._clone_output_indices:
            outputs[i] = outputs[i].clone()
        return tree_unflatten(outputs, self._output_spec)

    @torch.no_grad()
    def call_boxed(self, args: list[Any]) -> Any:
        """Warm up, record, or replay a region, consuming its argument list."""
        torch_npu = _get_torch_npu()
        if self.recapture_coordinator is not None:
            self.recapture_coordinator.before_call(self, args)
        elif self.graph is not None and self._input_change_reason(args) is not None:
            raise ValueError("ACLGraph inputs changed without a recapture coordinator")
        if not self.warmed_up:
            output = self.runnable(*args)
            self.warmed_up = True
            logger.info("ACLGraph %s: warmup completed", self.label)
            return output
        captured_output = None
        if self.graph is None:
            captured_output = self._capture(args, torch_npu)
        else:
            for i, (arg, saved) in enumerate(zip(args, self.args, strict=True)):
                if isinstance(saved, torch.Tensor) and i not in self.static_input_indices:
                    saved.copy_(arg)
        self.graph.replay()
        return self._replay_outputs(captured_output)

    def teardown(self) -> None:
        """Release captured buffers or the coordination process group."""
        self.graph = None
        self.pool_id = None
        self.args = None
        self.input_specs = None
        self.output = None
        self._flat_outputs = None
        self._output_spec = None
        self._clone_output_indices = ()
        self._input_guard = None
        self._static_pointers = ()
        self._scalar_inputs = ()
        self._container_inputs = {}
        owner = getattr(self.capture_runnable, "__self__", None)
        if isinstance(owner, torch.fx.Interpreter):
            owner.env.clear()


class _PartitionInterpreter(torch.fx.Interpreter):
    """Release the caller's last references before entering a boxed region."""

    def __init__(
        self, module: torch.fx.GraphModule, *, staged_input_indices: Iterable[int] = (),
        input_aliases: dict[int, int] | None = None,
    ) -> None:
        """Keep normal FX liveness while staging changing user input storage."""
        super().__init__(module)
        self._staged_input_indices = tuple(sorted(staged_input_indices))
        self._producer_input_indices = None
        self._input_buffers = {}
        self._input_aliases = input_aliases or {}
        # Keep the last-use map, but perform collection before boxed calls.
        self.garbage_collect_values = False
        self._inputs = [node for node in self.graph.nodes if node.op == "placeholder"]
        self._body = [node for node in self.graph.nodes if node.op != "placeholder"]
        self._boxed_args = {
            node: itemgetter(*node.args)
            for node in self._body
            if node.op == "call_module"
            and node.args
            and not node.kwargs
            and all(isinstance(arg, torch.fx.Node) for arg in node.args)
        }

    def run(
        self, *args: Any, initial_env: dict[torch.fx.Node, Any] | None = None,
        enable_io_processing: bool = True,
    ) -> Any:
        """Execute partitions and release values before entering captured regions."""
        if initial_env is not None or len(args) != len(self._inputs):
            return super().run(*args, initial_env=initial_env, enable_io_processing=enable_io_processing)
        if enable_io_processing:
            args = self.graph.process_inputs(*args)
        if self._staged_input_indices:
            args = self._stage_inputs(args)
        # Lifted parameters need bindings, not a Python interpreter call each.
        self.env = dict(zip(self._inputs, args, strict=True))
        for node in self._body:
            result = self.env[node] = self.run_node(node)
            if node.op == "output":
                return self.graph.process_outputs(result) if enable_io_processing else result

        raise RuntimeError("ACLGraph partition graph has no output node")

    def _stage_inputs(self, args):
        # Share one persistent buffer across every consumer, including aliases
        # returned by captured regions. Update it once before this step's work.
        if self._producer_input_indices is None:
            # Producer-owned buffers pass straight through. Region guards check
            # their addresses/layouts; no root staging or duplicate guard loop.
            self._producer_input_indices = {
                i
                for i in self._staged_input_indices
                if getattr(args[i], "_dynamo_static_input_type", None) == "guarded"
            }
            self._staged_input_indices = tuple(
                i for i in self._staged_input_indices if i not in self._producer_input_indices
            )
        values = list(args)
        destinations, sources = [], []
        for i in self._staged_input_indices:
            arg = args[i]
            if i in self._input_aliases:
                source = self._input_aliases[i]
                other = args[source]
                if arg.data_ptr() != other.data_ptr() or (arg.shape, arg.stride(), arg.dtype) != (
                    other.shape,
                    other.stride(),
                    other.dtype,
                ):
                    raise ValueError(f"ACLGraph shared input alias changed: input[{i}] and input[{source}]")
                values[i] = values[source]
                continue
            saved = self._input_buffers.get(i)
            if saved is None or (saved.shape, saved.stride(), saved.dtype, saved.device) != (
                arg.shape,
                arg.stride(),
                arg.dtype,
                arg.device,
            ):
                saved = self._input_buffers[i] = arg.detach().clone()
            else:
                destinations.append(saved)
                sources.append(arg)
            values[i] = saved
        if sources:
            torch._foreach_copy_(destinations, sources)
        return values

    def run_node(self, n: torch.fx.Node) -> Any:
        """Drop last-use arguments before the wrapper records their consumer."""
        region = self.fetch_attr(n.target) if n.op == "call_module" else None
        wrapper = getattr(region, "_aclgraph_wrapper", None)
        if wrapper is not None:
            fetch = self._boxed_args.get(n)
            if fetch is not None:
                values = fetch(self.env)
                boxed = [values] if len(n.args) == 1 else list(values)
                del values
            else:
                args, kwargs = self.fetch_args_kwargs_from_env(n)
                if kwargs:
                    raise ValueError("ACLGraph partition calls must use positional inputs")
                boxed = list(args)
                del args
            for value in self.user_to_last_uses.get(n, []):
                del self.env[value]
            return wrapper.call_boxed(boxed)
        result = super().run_node(n)
        for value in self.user_to_last_uses.get(n, []):
            del self.env[value]
        return result


def teardown_aclgraph(graph_module: torch.fx.GraphModule) -> None:
    """Release graphs before the caller destroys HCCL process groups."""
    wrappers = getattr(graph_module, "_aclgraph_wrappers", ())
    if any(wrapper.graph is not None for wrapper in wrappers):
        _get_torch_npu().npu.synchronize()
    for wrapper in wrappers:
        wrapper.teardown()
    coordinator = getattr(graph_module, "_aclgraph_recapture_coordinator", None)
    if coordinator is not None:
        coordinator.teardown()
    interpreter = getattr(graph_module.forward, "__self__", None)
    if isinstance(interpreter, _PartitionInterpreter):
        interpreter.env.clear()
        interpreter._input_buffers.clear()
