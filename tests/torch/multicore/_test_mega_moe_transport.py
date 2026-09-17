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

"""Acceptance of switchable transport, isolated heaps and pull completion."""

import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from hyper_parallel.core.multicore import profiler as multicore_profiler
from tests.torch.multicore import _test_mega_moe as baseline
from tests.torch.multicore._test_mega_moe_memory import _deferred_pair, _route
from tests.torch.multicore._mega_moe_utils import start_shmem_lifetime, write_evidence


def _repeated_backward(layer, shape, replay):
    """Preserve Router and expert state under replay and retained backward."""
    hidden, upstream = baseline.make_data(shape)
    hidden = hidden.detach().requires_grad_(True)
    ids, weights, counts = _route(shape, "tail")
    weights.requires_grad_(True)
    layer.zero_grad(set_to_none=True)
    if replay:
        output = checkpoint(layer, hidden, ids, weights, tokens_per_expert=counts, use_reentrant=False)
    else:
        output = baseline.forward_layer(layer, hidden, ids, weights, tokens_per_expert=counts)
    output.backward(upstream, retain_graph=True)
    output.backward(upstream)
    torch.npu.synchronize()
    return [tensor.detach().cpu() for tensor in
            [output, hidden.grad, weights.grad, *baseline.expert_weight_gradients(layer)]]


def _assert_tensors(expected, actual, label):
    """Compare every independently produced output and gradient."""
    for index, (reference, observed) in enumerate(zip(expected, actual)):
        baseline.assert_close(f"{label} tensor={index}", observed, reference)


def _profile_transport(layer, common, shape, mode):
    """Exercise GET/PUT completion in the instrumented forward and backward path."""
    hidden, upstream = baseline.make_data(shape)
    ids, weights, counts = _route(shape, "balanced")
    reference = baseline.run_layer(common, hidden, ids, weights, counts, upstream)
    with multicore_profiler.mega_kernel_profile(
        schedule=multicore_profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        detailed_task_names=True,
    ) as profiler:
        actual = baseline.run_layer(layer, hidden, ids, weights, counts, upstream)
        profiler.step()
    baseline.assert_results_close(reference, actual)
    path = Path(os.environ["HP_MEGA_MOE_LEVEL1_RESULT"]).with_name(f"{mode}_rank{baseline.RANK}_trace.json")
    trace = profiler.export_chrome_trace(path)
    metadata = trace["megaKernelCycleTrace"]
    events = [event for event in trace["traceEvents"] if event.get("cat") == "MegaKernelInternal"]
    assert metadata["invocationCount"] == 2 and metadata["droppedRecordCount"] == 0, metadata
    assert {event["args"]["direction"] for event in events} == {"forward", "backward"}
    dispatch_events = [event for event in events if event["args"].get("task_stage") == "Dispatch"]
    assert dispatch_events and all(event["args"]["core_type"] == "AIV" for event in dispatch_events), mode
    if mode == "pull":
        local_base = baseline.RANK * shape.local_experts
        assert all(local_base <= event["args"]["owner_id"] < local_base + shape.local_experts
                   for event in dispatch_events), mode
    return metadata


def test_push_pull_coexistence() -> None:
    """Alternate modes through hot ranks, retained graphs, checkpoint and profiling."""
    start_shmem_lifetime()
    shape = baseline.MoeShape(local_num_tokens=128, ep_size=dist.get_world_size(), num_experts=8)
    pairs = {mode: baseline.new_layers(shape, dispatch_mode=mode) for mode in ("push", "pull")}
    samples = []
    try:
        for mode, (layer, common) in pairs.items():
            for patterns in (("balanced", "tail"), ("destination0", "destination3"), ("tail", "balanced")):
                _assert_tensors(_deferred_pair(common, shape, patterns), _deferred_pair(layer, shape, patterns), mode)
            for replay in (False, True):
                _assert_tensors(_repeated_backward(common, shape, False),
                                _repeated_backward(layer, shape, replay), f"{mode} replay={replay}")
            samples.append({"mode": mode, "profile": _profile_transport(layer, common, shape, mode)})
        streams = [torch.npu.Stream(), torch.npu.Stream()]
        for step in range(4):
            for index, (mode, (layer, common)) in enumerate(pairs.items()):
                reference = _deferred_pair(common, shape, ("tail", "balanced"))
                stream = streams[(step + index) % 2]
                stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(stream):
                    actual = _deferred_pair(layer, shape, ("tail", "balanced"))
                _assert_tensors(reference, actual, f"{mode} stream step={step}")
        push = pairs["push"][0]._resource_group.resources.workspace  # pylint: disable=protected-access
        pull = pairs["pull"][0]._resource_group.resources.workspace  # pylint: disable=protected-access
        assert push.expert_buffer is not None and push.source_buffer is None
        assert pull.source_buffer is not None and pull.expert_buffer is None
        assert push.expert_buffer.shape[0] == shape.ep_size * shape.local_num_tokens * shape.top_k
        assert pull.source_buffer.shape[0] == shape.local_num_tokens * shape.top_k
        write_evidence({"coexistence": samples, "retained_backward": True, "checkpoint": True,
                        "alternating_streams": True, "hotspot_and_empty_ranks": True})
    finally:
        for layer, _ in pairs.values():
            layer.close()


def test_pull_hotspot_small_heap() -> None:
    """Move all receive traffic across four ranks within a source-sized heap."""
    start_shmem_lifetime()
    shape = baseline.MoeShape(local_num_tokens=4096, hidden_size=1024, intermediate_size=128,
                              num_experts=8, top_k=2, ep_size=dist.get_world_size())
    layer, common = baseline.new_layers(shape, dispatch_mode="pull")
    try:
        for patterns in (("balanced", "tail"), ("destination0", "destination3"), ("tail", "balanced")):
            _assert_tensors(_deferred_pair(common, shape, patterns), _deferred_pair(layer, shape, patterns), "pull")
        workspace = layer._resource_group.resources.workspace  # pylint: disable=protected-access
        heap = int(os.environ["HYPER_PARALLEL_SHMEM_HEAP_SIZE"])
        data_bytes = shape.local_num_tokens * shape.top_k * shape.hidden_size * 2
        assert 2 * data_bytes < heap < (shape.ep_size + 1) * data_bytes
        assert workspace.source_buffer.numel() * 2 == data_bytes
        assert workspace.expert_buffer is None
        write_evidence({"shape": vars(shape), "heap_bytes": heap, "symmetric_data_bytes": 2 * data_bytes,
                        "push_data_bytes": (shape.ep_size + 1) * data_bytes, "all_gradients_match": True})
    finally:
        layer.close()


def test_push_pull_qwen_shape() -> None:
    """Validate real Qwen dimensions across every automatic pull scheduling tier."""
    start_shmem_lifetime()
    shape = baseline.MoeShape(local_num_tokens=4096, hidden_size=5120, intermediate_size=1792,
                              num_experts=48, top_k=8, ep_size=dist.get_world_size())
    pairs = {mode: baseline.new_layers(shape, dispatch_mode=mode) for mode in ("push", "pull")}
    samples = []
    hidden, upstream = baseline.make_data(shape)
    try:
        for load in (1.0, 1.25, 2.5):
            slots = shape.local_num_tokens * shape.top_k
            counts_by_destination = [int(slots * load)]
            remaining = slots * shape.ep_size - counts_by_destination[0]
            counts_by_destination += [remaining // (shape.ep_size - 1)] * (shape.ep_size - 1)
            counts_by_destination[-1] += slots * shape.ep_size - sum(counts_by_destination)
            destinations = torch.repeat_interleave(torch.arange(shape.ep_size), torch.tensor(counts_by_destination))
            destinations = destinations[baseline.RANK::shape.ep_size]
            ids = destinations * shape.local_experts + torch.arange(slots).remainder(shape.local_experts)
            ids = ids.reshape(shape.local_num_tokens, shape.top_k).to(device=baseline.DEVICE, dtype=torch.int32)
            weights = torch.full(ids.shape, 1 / shape.top_k, device=baseline.DEVICE, dtype=torch.float32)
            counts = torch.bincount(ids.reshape(-1).long(), minlength=shape.num_experts).to(torch.int32)
            for mode, (layer, common) in pairs.items():
                reference = baseline.run_layer(common, hidden, ids, weights, counts, upstream)
                actual = baseline.run_layer(layer, hidden, ids, weights, counts, upstream)
                baseline.assert_results_close(reference, actual)
                samples.append({"mode": mode, "max_over_mean_load": load, "all_gradients_match": True})
        write_evidence({"shape": vars(shape), "real_shape": samples})
    finally:
        for layer, _ in pairs.values():
            layer.close()
