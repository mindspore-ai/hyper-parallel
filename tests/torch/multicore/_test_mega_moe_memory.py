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
"""Device regression for local-capacity buffers with outstanding forwards."""

import os
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.checkpoint import checkpoint

from hyper_parallel.core.multicore import MegaMoeExperts, profiler as multicore_profiler, shmem
from hyper_parallel.core.multicore.torch import ops
from tests.torch.multicore import _test_mega_moe as baseline
from tests.torch.multicore._mega_moe_utils import start_shmem_lifetime, write_evidence


def _route(shape: baseline.MoeShape, pattern: str) -> tuple:
    """Produce exact counts with odd tails or a completely empty destination."""
    positions = torch.arange(shape.local_num_tokens * shape.top_k).reshape(-1, shape.top_k)
    ids = (positions + baseline.RANK * shape.top_k).remainder(shape.num_experts)
    if pattern == "tail" and baseline.RANK == 0:
        ids[0, 0] = shape.local_experts
    elif pattern.startswith("destination"):
        destination = int(pattern[-1])
        ids = positions.remainder(shape.local_experts) + destination * shape.local_experts
    counts = torch.bincount(ids.reshape(-1), minlength=shape.num_experts).to(torch.int32)
    weights = torch.full(ids.shape, 1.0 / shape.top_k, dtype=torch.float32, device=baseline.DEVICE)
    return ids.to(device=baseline.DEVICE, dtype=torch.int32), weights, counts.to(baseline.DEVICE)


def _deferred_forwards(layers: list, hidden: list, routes: list) -> list:
    """Submit both forwards, optionally with checkpoint replay on alternating streams."""
    outputs = []
    replay = os.getenv("HP_MEGA_MOE_GROWTH_CASE") == "checkpoint" and isinstance(layers[0], MegaMoeExperts)
    streams = [torch.npu.Stream() for _ in range(2)] if replay else [torch.npu.current_stream()] * 2
    default_stream = torch.npu.current_stream()
    for member, tensor, (ids, weights, counts), stream in zip((layers * 2)[:2], hidden, routes, streams):
        with torch.npu.stream(stream):
            stream.wait_stream(default_stream)
            if replay:
                output = checkpoint(baseline.forward_layer, member, tensor, ids, weights,
                                    tokens_per_expert=counts, use_reentrant=False)
            else:
                output = baseline.forward_layer(member, tensor, ids, weights, tokens_per_expert=counts)
        default_stream.wait_stream(stream)
        outputs.append(output)
    return outputs


def _deferred_pair(layer: torch.nn.Module, shape: baseline.MoeShape, patterns: tuple) -> list:
    """Run two routes before reverse backward without changing the weights."""
    layers = layer if isinstance(layer, list) else [layer]
    for member in layers:
        member.zero_grad(set_to_none=True)
    source, upstream = baseline.make_data(shape)
    hidden = [(source * (1.0 + index / 8.0)).detach().requires_grad_(True) for index in range(2)]
    routes = [_route(shape, pattern) for pattern in patterns]
    route_weights = [weights.requires_grad_(True) for _, weights, _ in routes]
    outputs = _deferred_forwards(layers, hidden, routes)
    for output in reversed(outputs):
        output.backward(upstream, retain_graph=True)
        output.backward(upstream)
    torch.npu.synchronize()
    compared = outputs + [tensor.grad for tensor in hidden + route_weights]
    for member in layers:
        compared.extend(baseline.expert_weight_gradients(member))
    for index, tensor in enumerate(compared):
        assert tensor is not None, f"rank={baseline.RANK}: missing memory regression tensor {index}."
        baseline.assert_finite(f"memory regression tensor {index}", tensor)
    return [tensor.detach().cpu() for tensor in compared]


def test_mega_moe_local_capacity_lifetime() -> None:
    """Feature: Transport storage lifetime and conservative receive-buffer reuse.

    Description: Compare retained backward, hotspot/tail routes and profiling with the common backend.
    Expectation: All gradients match with reuse on/off, and pull fits a source-sized heap.
    """
    shape = baseline.MoeShape(local_num_tokens=4096, hidden_size=1024)
    mode = os.getenv("HP_MEGA_MOE_DISPATCH_MODE", "push")
    results = []
    for factor in (None, 1.5):
        start_shmem_lifetime()
        mega, common = baseline.new_layers(shape, expert_capacity_factor=factor)
        resources = mega._get_execution_resources(baseline.make_data(shape)[0])  # pylint: disable=protected-access
        assert resources.plan.reuse_backward_dispatch
        original = ops.mega_moe_grad_with_profile_buffer
        pairs = [("balanced", "tail"), ("tail", "balanced")]
        if factor is None:
            pairs.extend([("destination0", "destination1"), ("destination1", "tail")])
        try:
            for reuse in (False, True):
                resources.plan = replace(resources.plan, reuse_backward_dispatch=reuse)

                def _backward(*args):
                    """Check receive-buffer ownership at the native boundary."""
                    assert (args[0].data_ptr() == args[12].data_ptr()) == reuse
                    assert args[17].data_ptr() != args[0].data_ptr()
                    return original(*args)

                with patch.object(ops, "mega_moe_grad_with_profile_buffer", side_effect=_backward):
                    for patterns in pairs:
                        expected = _deferred_pair(common, shape, patterns)
                        with multicore_profiler.mega_kernel_profile(
                            schedule=multicore_profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
                        ) as profiler:
                            actual = _deferred_pair(mega, shape, patterns)
                            profiler.step()
                        trace = profiler.export_chrome_trace(Path(os.environ["HP_MEGA_MOE_LEVEL1_RESULT"]).with_name(
                            f"{mode}_rank{baseline.RANK}_trace.json"))
                        assert trace["megaKernelCycleTrace"]["invocationCount"] == 6
                        assert trace["megaKernelCycleTrace"]["droppedRecordCount"] == 0
                        for index, (reference, observed) in enumerate(zip(expected, actual)):
                            baseline.assert_close(f"capacity patterns={patterns} tensor={index}", observed, reference)
                        results.append({"factor": factor, "reuse": reuse, "patterns": patterns, "match": True})
            if mode == "pull":
                data_bytes = shape.local_num_tokens * shape.top_k * shape.hidden_size * 2
                assert 2 * data_bytes < shmem.debug_state()["config"]["heap_size_bytes"] < 3 * data_bytes
                assert resources.workspace.expert_buffer is None
        finally:
            mega.close()
    write_evidence({"mode": mode, "local_capacity_lifetime": results})


def _check_unknown_owner(mega: MegaMoeExperts, shape: baseline.MoeShape) -> None:
    """Converge one rank's unmanaged reference and preserve every old buffer."""
    resources = mega._get_execution_resources(baseline.make_data(shape)[0])
    resources.workspace.ensure(resources.spec, torch.bfloat16, baseline.DEVICE)
    before = resources.workspace.expert_buffer
    if baseline.RANK == 0:
        shmem.acquire()
    try:
        resources.heap_manager.ensure_capacity(resources, shape.local_num_tokens * shape.top_k + 1)
    except RuntimeError as error:
        assert "unmanaged SHMEM owner" in str(error), error
    else:
        raise AssertionError("Unknown owner did not prevent growth")
    finally:
        if baseline.RANK == 0:
            shmem.release()
    assert resources.workspace.expert_buffer is before
    assert before.untyped_storage().nbytes() > 0


def _check_companions(pending: list, lazy: MegaMoeExperts, lazy_reference: torch.nn.Module,
                      shape: baseline.MoeShape) -> None:
    """Finish a pull graph and bind a previously unused push owner after growth."""
    snapshots = []
    for member, output, source, weights, upstream in pending:
        output.backward(upstream)
        snapshots.append([output, source.grad, weights.grad, *baseline.expert_weight_gradients(member)])
    for reference, actual in zip(*snapshots):
        baseline.assert_close("mixed owner across rebuild", actual, reference)
    expected = _deferred_pair(lazy_reference, shape, ("balanced", "tail"))
    actual = _deferred_pair(lazy, shape, ("balanced", "tail"))
    for reference, actual_tensor in zip(expected, actual):
        baseline.assert_close("lazy owner after rebuild", actual_tensor, reference)


def test_mega_moe_heap_growth() -> None:
    """Feature: Online push heap growth.

    Description: Grow between two forwards, then reverse and repeat both backwards.
    Expectation: All outputs and gradients match common EP; stale storage is invalidated.
    """
    os.environ.pop("HYPER_PARALLEL_SHMEM_HEAP_SIZE", None)
    start_shmem_lifetime()
    shape = baseline.MoeShape(local_num_tokens=4096, hidden_size=1024)
    mega, common = baseline.new_layers(shape, expert_capacity_factor=1.0, capacity_policy="grow")
    variant = os.getenv("HP_MEGA_MOE_GROWTH_CASE", "single")
    managed, references, handles = [mega], [common], []
    companions = []
    pending = []
    lazy = lazy_reference = small = None
    if variant == "shared":
        second, reference = baseline.new_layers(shape, expert_capacity_factor=1.0, capacity_policy="grow")
        managed.append(second)
        references.append(reference)
        MegaMoeExperts.share_execution_resources(managed)
    elif variant == "mixed":
        small = replace(shape, hidden_size=512)
        sidecar, side_reference = baseline.new_layers(small, dispatch_mode="pull")
        lazy, lazy_reference = baseline.new_layers(small, dispatch_mode="push")
        companions.extend((sidecar, lazy))
        for member in (side_reference, sidecar):
            source, upstream = baseline.make_data(small)
            ids, weights, counts = _route(small, "balanced")
            source.requires_grad_(True)
            weights.requires_grad_(True)
            output = baseline.forward_layer(member, source, ids, weights, tokens_per_expert=counts)
            pending.append((member, output, source, weights, upstream))
    observations = []
    old_buffers = []

    def observe(_module: torch.nn.Module, _args: tuple, _output: torch.Tensor) -> None:
        """Record the physical heap and retain one stale tensor for invalidation checks.

        Args:
            _module: Layer invoking the hook.
            _args: Forward positional inputs.
            _output: Completed forward output.
        """
        resources = mega._resource_group.resources  # pylint: disable=protected-access
        observations.append((resources.heap_manager.epoch, resources.workspace.expert_capacity,
                             shmem.debug_state()["config"]["heap_size_bytes"]))
        old_buffers.append(resources.workspace.expert_buffer)

    handles = [member.register_forward_hook(observe) for member in managed]
    try:
        if variant == "single":
            _check_unknown_owner(mega, shape)
        for patterns in (("balanced", "destination0"), ("destination1", "balanced")):
            expected = _deferred_pair(references, shape, patterns)
            actual = _deferred_pair(managed, shape, patterns)
            for index, (reference, observed) in enumerate(zip(expected, actual)):
                baseline.assert_close(f"grow patterns={patterns} tensor={index}", observed, reference)
        epochs = [item[0] for item in observations]
        assert epochs[:2] == [0, 1] and all(epoch == 1 for epoch in epochs[2:]), observations
        assert observations[1][2] > observations[0][2], observations
        assert old_buffers[0].untyped_storage().nbytes() == 0
        if pending:
            _check_companions(pending, lazy, lazy_reference, small)
        assert shmem.debug_state()["reference_count"] == 1 + len(companions)
        resources = mega._resource_group.resources  # pylint: disable=protected-access
        write_evidence({"variant": variant, "observations": observations,
                        "growth": resources.heap_manager.growth_records})
    finally:
        for handle in handles:
            handle.remove()
        for member in managed + companions:
            member.close()
    assert shmem.debug_state()["state"] == "Uninitialized"
