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
"""Exercise peer-ready generations with delayed ranks and checkpoint replay."""

import time
from unittest.mock import patch

import torch
from torch.utils.checkpoint import checkpoint

from hyper_parallel.core.multicore import MegaMoeExperts, shmem
from hyper_parallel.core.multicore.modules.mega_moe import function as function_module
from hyper_parallel.core.multicore.scheduler.config import (
    READY_CACHE_LINE_BYTES,
)
from tests.torch.multicore import _test_mega_moe as baseline
from tests.torch.multicore import _test_mega_moe_resources as resources_test
from tests.torch.multicore._mega_moe_utils import start_shmem_lifetime, write_evidence


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
    tensors = outputs + [tensor.grad for tensor in inputs + selected]
    tensors += [parameter.grad for layer in layers for parameter in layer.parameters()]
    for index, tensor in enumerate(tensors):
        assert tensor is not None, f"rank={baseline.RANK}: missing tensor {index}, expected a result or gradient."
        baseline.assert_finite(f"ready step={step} tensor={index}", tensor)
    return [tensor.detach().cpu() for tensor in tensors]


def _ready_generations(workspace, ep_size: int) -> tuple[int, int]:
    """Read persistent counters only after a complete validation step."""
    offset = workspace.event_counter_bytes + ep_size * READY_CACHE_LINE_BYTES
    return tuple(
        int(events[offset:offset + 4].view(torch.int32).cpu().item())
        for events in (workspace.forward_event_counters, workspace.backward_event_counters)
    )


def _run_ready_case(replay: bool) -> dict:
    """Compare shared checkpoint execution against independent ordinary layers."""
    start_shmem_lifetime()
    shape = baseline.MoeShape(local_num_tokens=128)
    independent = resources_test._new_stack(shape)  # pylint: disable=protected-access
    shared = resources_test._new_stack(shape)  # pylint: disable=protected-access
    for source, target in zip(independent, shared):
        target.load_state_dict(source.state_dict())
    MegaMoeExperts.share_execution_resources(shared)
    streams = [torch.npu.Stream() for _ in range(2)]
    generations = []
    try:
        for step in range(4):
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
            for index, (expected, observed) in enumerate(zip(reference, actual)):
                torch.testing.assert_close(
                    observed, expected, rtol=2e-2, atol=1e-5,
                    msg=f"rank={baseline.RANK}: replay={replay} step={step} tensor={index}",
                )
            workspace = shared[0]._resource_group.resources.workspace  # pylint: disable=protected-access
            current = _ready_generations(workspace, shape.ep_size)
            if generations:
                assert all(new > old for new, old in zip(current, generations[-1])), (
                    f"rank={baseline.RANK}: ready generations={current}, expected greater than {generations[-1]}."
                )
            generations.append(current)

        source, _ = baseline.make_data(shape)
        ids, weights, counts = baseline.make_balanced_route(shape)
        with patch.object(
            function_module.multicore_ops, "mega_moe_with_profile_buffer",
            side_effect=RuntimeError("injected launch error"),
        ):
            try:
                shared[0](source, ids, weights, tokens_per_expert=counts)
            except RuntimeError as error:
                assert str(error) == "injected launch error", f"unexpected launch failure: {error}"
            else:
                raise AssertionError("Injected launch error was not propagated.")
        assert not workspace.in_use, f"rank={baseline.RANK}: failed launch left lease in_use={workspace.in_use}."
        reference = _stack_step(independent, shape, streams, False, 4)
        actual = _stack_step(shared, shape, streams, replay, 4)
        for index, (expected, observed) in enumerate(zip(reference, actual)):
            torch.testing.assert_close(
                observed, expected, rtol=2e-2, atol=1e-5,
                msg=f"rank={baseline.RANK}: recovery tensor={index}",
            )
        return {
            "checkpoint_replay": replay, "layers": 3, "microbatches": 2,
            "alternating_streams": 2, "rank_staggered": True, "generations": generations,
            "steady_host_barriers": 0, "launch_exception_recovered": True,
            "outputs_and_all_gradients_match": True,
        }
    finally:
        for layer in independent + shared:
            layer.close()


def test_mega_moe_device_ready_lifecycle() -> None:
    """Recreate resources across plain and checkpointed staggered schedules."""
    results = [_run_ready_case(replay) for replay in (False, True)]
    write_evidence({"device_ready_lifecycle": results})
