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
"""Shared-resource training, ready generations and checkpoint lifecycle acceptance."""

import os
import time
from dataclasses import asdict
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from hyper_parallel.core.multicore import MegaMoeExperts, shmem
from hyper_parallel.core.multicore.modules.mega_moe import function as function_module
from hyper_parallel.core.multicore.scheduler.config import (
    READY_CACHE_LINE_BYTES,
)
from tests.torch.multicore import _test_mega_moe as baseline
from tests.torch.multicore._mega_moe_utils import (
    assert_stable_memory,
    collect_memory,
    start_shmem_lifetime,
    write_evidence,
)


def _new_stack(shape: baseline.MoeShape, layers: int = 3) -> list[MegaMoeExperts]:
    """Build distinct expert layers without initializing any native resource."""
    return [
        MegaMoeExperts(
            hidden_size=shape.hidden_size,
            intermediate_size=shape.intermediate_size,
            num_experts=shape.num_experts,
            top_k=shape.top_k,
            ep_size=shape.ep_size,
            ep_group=dist.group.WORLD,
            dispatch_mode=os.getenv("HP_MEGA_MOE_DISPATCH_MODE", "push"),
        ).to(device=baseline.DEVICE, dtype=torch.bfloat16)
        for _ in range(layers)
    ]


def _stack_step(layers: list, shape: baseline.MoeShape, streams: list, replay: bool, step: int) -> list:
    """Queue two stack forwards and reverse backward without per-layer synchronization."""
    source, upstream = baseline.make_data(shape)
    ids, weights, counts = baseline.make_balanced_route(shape)
    inputs = [(source * (1.0 + index / 8.0)).detach().requires_grad_(True) for index in range(2)]
    selected = [weights.clone().requires_grad_(True) for _ in range(6)]
    for layer in layers:
        layer.zero_grad(set_to_none=True)
    default_stream = torch.npu.current_stream()
    outputs = []
    for microbatch, value in enumerate(inputs):
        previous_stream = default_stream
        for index, layer in enumerate(layers):
            if (baseline.RANK + index + microbatch + step) % shape.ep_size == 0:
                time.sleep(0.002)
            stream = streams[(microbatch + index) % len(streams)]
            with torch.npu.stream(stream):
                stream.wait_stream(previous_stream)
                arguments = (layer, value, ids, selected[microbatch * 3 + index])
                if replay:
                    value = checkpoint(*arguments, tokens_per_expert=counts, use_reentrant=False)
                else:
                    value = layer(*arguments[1:], tokens_per_expert=counts)
            previous_stream = stream
        outputs.append(value)
    for index, output in enumerate(reversed(outputs)):
        if (baseline.RANK + index + step) % shape.ep_size == 0:
            time.sleep(0.003)
        output.backward(upstream / shape.local_num_tokens**0.5)
    torch.npu.synchronize()
    return _snapshot_and_update(layers, outputs, inputs + selected, step)


def _snapshot_and_update(layers: list, outputs: list, leaves: list, step: int) -> list:
    """Validate every result/gradient before retaining snapshots and applying the same SGD step."""
    tensors = outputs + [tensor.grad for tensor in leaves]
    tensors += [parameter.grad for layer in layers for parameter in layer.parameters()]
    for index, tensor in enumerate(tensors):
        assert tensor is not None, f"rank={baseline.RANK}: missing tensor {index}, expected a result or gradient."
        baseline.assert_finite(f"ready step={step} tensor={index}", tensor)
        assert torch.count_nonzero(tensor).item() > 0, (
            f"rank={baseline.RANK}: lifecycle step={step} tensor={index} is entirely zero"
        )
    snapshots = [tensor.detach().cpu() for tensor in tensors]
    with torch.no_grad():
        for layer in layers:
            for parameter in layer.parameters():
                parameter.add_(parameter.grad, alpha=-1e-3)
            layer.zero_grad(set_to_none=True)
    return snapshots


def _ready_generations(workspace, ep_size: int) -> tuple[int, int]:
    """Read persistent counters only after a complete validation step."""
    offset = workspace.event_counter_bytes + ep_size * READY_CACHE_LINE_BYTES
    return tuple(
        int(events[offset:offset + 4].view(torch.int32).cpu().item())
        for events in (workspace.forward_event_counters, workspace.backward_event_counters)
    )


def _check_resources_and_failed_launch(independent: list, shared: list, shape: baseline.MoeShape,
                                      samples: list) -> None:
    """Check stable allocation, sharing identity and lease release after an injected launch failure."""
    if samples:
        assert_stable_memory(samples)
        allocations = [sample["shmem_allocations"] for sample in samples]
        assert len(set(allocations)) == 1, f"rank={baseline.RANK}: unstable SHMEM allocations={allocations}"
    # Sharing identity and numerical parity are separate contracts.
    resources = shared[0]._resource_group.resources  # pylint: disable=protected-access
    workspace = resources.workspace
    shared_resources = [layer._resource_group.resources for layer in shared]  # pylint: disable=protected-access
    assert all(resource is resources for resource in shared_resources), (
        f"rank={baseline.RANK}: shared layers did not share the same resources"
    )
    # Each independent layer must retain a distinct physical allocation group.
    independent_ids = {
        id(layer._resource_group.resources) for layer in independent  # pylint: disable=protected-access
    }
    assert len(independent_ids) == 3, f"rank={baseline.RANK}: independent resource count={len(independent_ids)}"
    source, _ = baseline.make_data(shape)
    ids, weights, counts = baseline.make_balanced_route(shape)
    with patch.object(
        function_module.multicore_ops, "mega_moe",
        side_effect=RuntimeError("injected launch error"),
    ):
        try:
            shared[0](source, ids, weights, tokens_per_expert=counts)
        except RuntimeError as error:
            assert str(error) == "injected launch error", f"unexpected launch failure: {error}"
        else:
            raise AssertionError("Injected launch error was not propagated.")
    assert not workspace.in_use, f"rank={baseline.RANK}: failed launch left lease in_use={workspace.in_use}."


def _assert_step_parity(reference: list, actual: list, independent: list, shared: list,
                        replay: bool, step: int) -> None:
    """Compare the complete result/gradient snapshots and updated parameters for one step."""
    for index, (expected, observed) in enumerate(zip(reference, actual)):
        torch.testing.assert_close(
            observed, expected, rtol=2e-2, atol=1e-5,
            msg=f"rank={baseline.RANK}: replay={replay} step={step} tensor={index}",
        )
    for source, target in zip(independent, shared):
        for expected, observed in zip(source.parameters(), target.parameters()):
            baseline.assert_close("lifecycle updated parameter", observed, expected)


def _run_lifecycle_case(alternate_streams: bool, replay: bool) -> dict:
    """Compare shared checkpoint execution against independent ordinary layers."""
    start_shmem_lifetime()
    shape = baseline.MoeShape(local_num_tokens=128 if replay else 1024)
    independent = _new_stack(shape)
    shared = _new_stack(shape)
    for source, target in zip(independent, shared):
        target.load_state_dict(source.state_dict())
    MegaMoeExperts.share_execution_resources(shared)
    streams = [torch.npu.Stream() for _ in range(2 if alternate_streams else 1)]
    generations, samples = [], []
    steps = 4 if replay else 6
    try:
        for step in range(steps):
            reference = _stack_step(independent, shape, streams, False, step)
            if step == 0:
                actual = _stack_step(shared, shape, streams, replay, step)
            else:
                with patch.object(
                    shmem, "host_barrier", wraps=shmem.host_barrier,
                ) as fence:
                    actual = _stack_step(shared, shape, streams, replay, step)
                assert fence.call_count == 0, (
                    f"rank={baseline.RANK}: steady-state Host barriers={fence.call_count}, expected 0."
                )
            _assert_step_parity(reference, actual, independent, shared, replay, step)
            if not replay and step >= 3:
                sample = collect_memory()
                sample["shmem_allocations"] = shmem.debug_state()["allocated_count"]
                samples.append(sample)
            workspace = shared[0]._resource_group.resources.workspace  # pylint: disable=protected-access
            current = _ready_generations(workspace, shape.ep_size)
            if generations:
                assert all(new > old for new, old in zip(current, generations[-1])), (
                    f"rank={baseline.RANK}: ready generations={current}, expected greater than {generations[-1]}."
                )
            generations.append(current)

        _check_resources_and_failed_launch(independent, shared, shape, samples)
        reference = _stack_step(independent, shape, streams, False, steps)
        actual = _stack_step(shared, shape, streams, replay, steps)
        for index, (expected, observed) in enumerate(zip(reference, actual)):
            torch.testing.assert_close(
                observed, expected, rtol=2e-2, atol=1e-5,
                msg=f"rank={baseline.RANK}: recovery tensor={index}",
            )
        return {
            "shape": asdict(shape), "checkpoint_replay": replay, "layers": 3, "microbatches": 2,
            "steps": steps, "alternating_streams": len(streams), "rank_staggered": True,
            "generations": generations, "stable_memory": samples, "parameter_updates_match": True,
            "steady_host_barriers": 0, "launch_exception_recovered": True,
            "outputs_and_all_gradients_match": True,
        }
    finally:
        for layer in independent + shared:
            layer.close()


def test_mega_moe_resource_lifecycle() -> None:
    """Validate shared training and ready recovery in serial, alternating and replay schedules."""
    results = [_run_lifecycle_case(alternate, replay) for alternate, replay in
               ((False, False), (True, False), (True, True))]
    write_evidence({"dispatch_mode": os.getenv("HP_MEGA_MOE_DISPATCH_MODE", "push"),
                    "resource_lifecycle": results})
