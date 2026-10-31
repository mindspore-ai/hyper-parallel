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
"""Interleaved original HP / MoonEP measurements over the identical base payload."""
from __future__ import annotations

import gc
import json
import os
import statistics
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist

from hyper_parallel.core.multicore import shmem
from hyper_parallel.core.expert_parallel.moonep.mega_moe import MoonEPMegaMoeExperts
from tests.torch.multicore import _test_mega_moe as reference


def _rank_balanced_expert_hot_route(shape):
    """Choose distinct TopK entries from a small active subset of each EP rank."""
    positions = torch.arange(shape.local_num_tokens * shape.top_k, device=reference.DEVICE).view(
        shape.local_num_tokens, shape.top_k)
    active_per_rank = (shape.top_k + shape.ep_size - 1) // shape.ep_size
    selected = (positions.remainder(shape.ep_size) * shape.local_experts
                + positions.div(shape.ep_size, rounding_mode="floor").remainder(active_per_rank)).long()
    scores = torch.zeros((shape.local_num_tokens, shape.num_experts), device=reference.DEVICE)
    preference = torch.arange(shape.top_k, 0, -1, device=reference.DEVICE, dtype=torch.float32)
    scores.scatter_(1, selected, preference.expand_as(selected))
    # Match the real HP Router's scores.topk selection semantics.
    ids = scores.topk(shape.top_k, dim=-1).indices.int()
    probabilities = torch.full(ids.shape, 1.0 / shape.top_k, device=reference.DEVICE)
    counts = torch.bincount(ids.flatten().long(), minlength=shape.num_experts).int()
    return ids, probabilities, counts


def _measure(shape, pattern, mode):
    baseline, _ = reference.new_layers(shape, dispatch_mode=mode)
    moon = MoonEPMegaMoeExperts(
        **{key: value for key, value in asdict(shape).items() if key != "local_num_tokens"},
        ep_group=dist.group.WORLD, dispatch_mode=mode).to(
        device=reference.DEVICE, dtype=torch.bfloat16)
    moon.load_state_dict(baseline.state_dict())
    hidden, upstream = reference.make_data(shape)
    ids, probabilities, counts = (_rank_balanced_expert_hot_route(shape)
                                  if pattern == "zero_token_experts"
                                  else reference._make_fixed_route(shape, pattern))
    sorted_ids = ids.sort(dim=1).values
    assert bool((sorted_ids[:, 1:] != sorted_ids[:, :-1]).all().item()), (
        "Performance routes must use distinct TopK IDs, as the production Router does")
    warmup = int(os.getenv("HP_MOONEP_AB_WARMUP", "30"))
    samples = {"baseline": [], "moonep": []}
    try:
        expected = reference.run_layer(baseline, hidden, ids, probabilities, counts, upstream)
        actual = reference.run_layer(moon, hidden, ids, probabilities, counts, upstream)
        reference.assert_results_close(expected, actual)
        del expected, actual
        for _ in range(warmup):
            for layer in (baseline, moon):
                reference._timed_fwd_bwd(layer, hidden, ids, probabilities, counts, upstream)
        # ABBA cancels drift without changing either operator's normal Python path.
        order = ("baseline", "moonep", "moonep", "baseline") * 20
        layers = {"baseline": baseline, "moonep": moon}
        for tag in order:
            samples[tag].append(reference._timed_fwd_bwd(
                layers[tag], hidden, ids, probabilities, counts, upstream))
        medians = {tag: statistics.median(values) for tag, values in samples.items()}
        ratio = medians["moonep"] / medians["baseline"]
        result = {"shape": asdict(shape), "pattern": pattern, "dispatch_mode": mode,
                  "topk_ids_distinct": True, "warmup_per_backend": warmup, "order": list(order), "samples_ms": samples,
                  "median_ms": medians, "moonep_over_baseline": ratio,
                  "speedup": 1.0 / ratio, "correctness": "output/dX/probs/dW/SGD original HP tolerances",
                  "statistics": dict(moon._transfer_pool.statistics) if moon._transfer_pool is not None else {},
                  "placement_statistics": dict(moon.placement_statistics),
                  "bank_bytes": (sum(x.numel()*x.element_size() for x in moon._transfer_pool.weights)
                                 if moon._transfer_pool is not None else 0),
                  "native_payload": str(Path(__file__).resolve().parents[3] / "build/native/payload"),
                  "shmem_heap_bytes": shmem.debug_state().get("config", {}).get("heap_size_bytes")}
        saved_calls = result["placement_statistics"]
        if saved_calls["moonep_calls"]:
            assert result["statistics"]["remote_gradient_returns"] > 0
        else:
            assert saved_calls["native_balanced_calls"] + saved_calls["native_cost_skip_calls"] > 0
            assert moon._transfer_pool is None and moon._virtual_executor is None, (
                "Ordinary fast path allocated placement resources")
        print("MOONEP_AB_RESULT", json.dumps(result), flush=True)
        return result
    finally:
        moon.close()
        baseline.close()
        gc.collect()
        torch.npu.empty_cache()
        dist.barrier()


def test_moonep_interleaved_performance():
    """Compare small uniform and representative large uniform/hotspot cases."""
    mode = os.environ["HP_MEGA_MOE_DISPATCH_MODE"]
    small = reference.MoeShape(num_experts=8, ep_size=4)
    large = reference.performance_shape()
    results = [_measure(shape, route, mode) for shape, route in (
        (small, "balanced"), (small, "zero_token_experts"), (small, "single_destination"),
        (large, "balanced"), (large, "zero_token_experts"), (large, "single_destination"))]
    if dist.get_rank() == 0:
        Path(os.environ["HP_MOONEP_AB_RESULT"]).write_text(json.dumps(results, indent=2), encoding="utf-8")
