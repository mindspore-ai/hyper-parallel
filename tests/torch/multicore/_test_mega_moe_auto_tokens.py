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

"""Validate automatic token capacity for push and pull, including deferred backward."""

import json
import os

import torch
import torch.distributed as dist
import torch_npu

from hyper_parallel.core.multicore import MegaMoeExperts
from tests.torch.multicore._mega_moe_utils import start_shmem_lifetime, write_evidence
from tests.torch.multicore._test_mega_moe_dynamic_tokens import (
    _HIDDEN,
    _INTERMEDIATE,
    _check_gradients,
    _deferred,
    _forward,
    _inputs,
    _lifecycle,
    _rounded_oracle,
    _weights,
)


def _layer(mode: str, weights: tuple, rank: int, world: int, device: torch.device) -> MegaMoeExperts:
    """Construct an unbounded module without predicting token counts."""
    layer = MegaMoeExperts(hidden_size=_HIDDEN, intermediate_size=_INTERMEDIATE,
                           num_experts=world * 6, top_k=6, ep_size=world,
                           ep_group=dist.group.WORLD, dispatch_mode=mode).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        layer.gate_up_weight.copy_(weights[0][rank * 6:(rank + 1) * 6])
        layer.down_weight.copy_(weights[1][rank * 6:(rank + 1) * 6])
    return layer


def _state(layer: MegaMoeExperts) -> dict:
    """Read physical allocation sizes without synchronizing tensor contents."""
    resources = layer._resource_group.resources  # pylint: disable=protected-access
    workspace = resources.workspace
    return {"routed_rows": workspace.routed_slots, "receive_rows": workspace.expert_capacity,
            "heap_bytes": resources.heap_manager.heap_bytes, "epoch": resources.heap_manager.epoch}


def _verify(layer: MegaMoeExperts, lengths: list[int], weights: tuple, rank: int,
            device: torch.device, hot: bool = False) -> dict:
    """Check the complete local output and gradient contract against the independent oracle."""
    layer.zero_grad(set_to_none=True)
    inputs = _inputs(lengths, layer.num_experts, layer.top_k, hot=hot)
    expected = _rounded_oracle(weights, inputs)
    errors = _check_gradients(layer, _forward(layer, inputs, rank, device), expected, lengths, rank)
    return {"lengths": lengths, "hot": hot, "max_abs_error": errors, "capacity": _state(layer)}


def _pending_growth(layer: MegaMoeExperts, weights: tuple, rank: int, world: int,
                    device: torch.device) -> dict:
    """Grow source storage while an earlier forward still needs backward."""
    lengths = [1] * world
    inputs = _inputs(lengths, layer.num_experts, layer.top_k)
    pending = _forward(layer, inputs, rank, device)
    before = _state(layer)
    result = _verify(layer, [257] * world, weights, rank, device, hot=True)
    after = _state(layer)
    if after["epoch"] <= before["epoch"] or after["routed_rows"] <= before["routed_rows"]:
        raise AssertionError("larger input did not grow the source allocation")
    layer.zero_grad(set_to_none=True)
    _check_gradients(layer, pending, _rounded_oracle(weights, inputs), lengths, rank)
    return result


def _pending_bucket_growth(layer: MegaMoeExperts, weights: tuple, rank: int, world: int,
                           device: torch.device) -> dict:
    """Replace a partial bucket while backward still retains its previous plan."""
    lengths = [257] * world
    inputs = _inputs(lengths, layer.num_experts, layer.top_k)
    pending = _forward(layer, inputs, rank, device)
    resources = layer._resource_group.resources  # pylint: disable=protected-access
    retained = resources._plans[512]  # pylint: disable=protected-access
    coverage = retained.spec.plan_tokens
    result = _verify(layer, [385] * world, weights, rank, device)
    expanded = resources._plans[512]  # pylint: disable=protected-access
    if expanded is retained or retained.spec.plan_tokens != coverage or expanded.spec.plan_tokens < 512:
        raise AssertionError(f"partial bucket was not replaced immutably: old={coverage}, "
                             f"new={expanded.spec.plan_tokens}")
    layer.zero_grad(set_to_none=True)
    _check_gradients(layer, pending, _rounded_oracle(weights, inputs), lengths, rank)
    return result


def _bucket_reuse(layer: MegaMoeExperts, weights: tuple, rank: int, world: int,
                  device: torch.device) -> dict:
    """Cycle through more than four buckets without rebuilding stable plans or storage."""
    lengths = (0, 1, 129, 257, 513, 1025, 2049, 4097)
    for tokens in lengths:
        _verify(layer, [tokens] * world, weights, rank, device)
    resources = layer._resource_group.resources  # pylint: disable=protected-access
    plans = dict(resources._plans)  # pylint: disable=protected-access
    epoch = _state(layer)["epoch"]
    if len(plans) < 7:
        raise AssertionError(f"wide token cycle did not retain seven buckets: cached={list(plans)}")
    errors = []
    for tokens in lengths:
        errors.append(_verify(layer, [tokens] * world, weights, rank, device)["max_abs_error"])
        if _state(layer)["epoch"] != epoch:
            raise AssertionError(f"warm token cycle restarted the heap: expected_epoch={epoch}")
        if any(resources._plans.get(bucket) is not plan for bucket, plan in plans.items()):  # pylint: disable=protected-access
            raise AssertionError(f"warm token cycle replaced cached plans: buckets={list(plans)}")
    return {"buckets": sorted(plans), "epoch": epoch, "max_abs_errors": errors}


def _other_workspace_growth(layer: MegaMoeExperts, weights: tuple, rank: int, world: int,
                            device: torch.device) -> None:
    """Rebuild a root containing both transports without invalidating pending graphs."""
    lengths = [17] * world
    inputs = _inputs(lengths, layer.num_experts, layer.top_k)
    pending = _forward(layer, inputs, rank, device)
    other = _layer("pull" if layer.dispatch_mode == "push" else "push", weights, rank, world, device)
    _verify(other, [769] * world, weights, rank, device)
    layer.zero_grad(set_to_none=True)
    _check_gradients(layer, pending, _rounded_oracle(weights, inputs), lengths, rank)
    other.close()


def test_auto_tokens() -> None:
    """Exercise empty bootstrap, growth, reuse, shared layers, checkpoint and old plans."""
    torch.set_num_threads(2)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("npu", local_rank)
    mode = os.environ.get("HP_DYNAMIC_DISPATCH", "push")
    start_shmem_lifetime()
    weights = _weights(world * 6)
    layer = _layer(mode, weights, rank, world, device)
    peer = _layer(mode, weights, rank, world, device)
    MegaMoeExperts.share_execution_resources([layer, peer])
    results = [_verify(layer, [0] * world, weights, rank, device)]
    if _state(layer)["routed_rows"] != 128 * layer.top_k:
        raise AssertionError("empty first input reserved more than the bootstrap capacity")
    results.append(_pending_growth(layer, weights, rank, world, device))
    results.append(_pending_bucket_growth(layer, weights, rank, world, device))
    _other_workspace_growth(layer, weights, rank, world, device)
    for lengths, hot in (([129 + index for index in range(world)], False),
                         ([769] + [0] * (world - 1), True),
                         ([1] * world, False), ([0] * world, False),
                         ([4097] + [1] * (world - 1), False)):
        previous = _state(layer)
        result = _verify(layer, lengths, weights, rank, device, hot)
        current = result["capacity"]
        required = max(lengths) * layer.top_k
        if current["routed_rows"] < required or current["routed_rows"] < previous["routed_rows"]:
            raise AssertionError("source capacity is insufficient or shrank")
        if (required <= previous["routed_rows"] and not hot and max(lengths) <= 1
                and current["epoch"] != previous["epoch"]):
            raise AssertionError("small input unnecessarily restarted the heap")
        results.append(result)
    results.append(_verify(peer, [127] * world, weights, rank, device))
    if layer._resource_group.resources is not peer._resource_group.resources:  # pylint: disable=protected-access
        raise AssertionError("shared second layer acquired independent execution resources")
    _lifecycle([layer, peer], rank, device)
    with torch.no_grad():
        layer.gate_up_weight.copy_(weights[0][rank * 6:(rank + 1) * 6])
        layer.down_weight.copy_(weights[1][rank * 6:(rank + 1) * 6])
    peer.close()
    _deferred(layer, weights, rank, device)
    bucket_reuse = _bucket_reuse(layer, weights, rank, world, device)
    final_state = _state(layer)
    layer.close()
    result = {"status": "passed", "dispatch_mode": mode, "results": results,
              "source_growth_pending_backward": "passed", "mixed_transport_restart": "passed",
              "same_bucket_growth_pending_backward": "passed", "geometric_plan_reuse": bucket_reuse,
              "checkpoint_shared_stream_sgd": "passed", "deferred_plan_reuse": "passed",
              "final_capacity": final_state, "torch": torch.__version__, "torch_npu": torch_npu.__version__}
    print(json.dumps({"rank": rank, **result}), flush=True)
    write_evidence(result)
    dist.destroy_process_group()
