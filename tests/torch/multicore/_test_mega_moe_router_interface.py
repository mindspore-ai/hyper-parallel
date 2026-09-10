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
"""MegaMoe Router interface acceptance without caller-provided expert counts."""

from __future__ import annotations

import os
from dataclasses import asdict

import torch

from hyper_parallel.core.multicore import MegaMoeExperts
from tests.torch.multicore import _test_mega_moe as baseline
from tests.torch.multicore._mega_moe_utils import (
    start_shmem_lifetime,
    write_evidence,
)


def _router_weight(shape: baseline.MoeShape) -> torch.nn.Parameter:
    """Use identical trainable Router weights for the two expert backends."""
    generator = torch.Generator().manual_seed(706)
    weight = torch.randn(shape.num_experts, shape.hidden_size, generator=generator)
    return torch.nn.Parameter((weight / shape.hidden_size**0.5).to(baseline.DEVICE))


def _route(hidden: torch.Tensor, weight: torch.Tensor, top_k: int) -> tuple:
    """Compute Router projection, softmax, Top-K and selected-weight normalization."""
    probabilities = (hidden.float() @ weight.T).softmax(dim=-1)
    values, indices = probabilities.topk(top_k, dim=-1)
    return indices.to(torch.int32), values / values.sum(dim=-1, keepdim=True)


def _run_router_layer(layer: torch.nn.Module, shape: baseline.MoeShape) -> dict[str, torch.Tensor]:
    """Validate a connected Router/expert graph, including Router dW and SGD."""
    hidden, upstream = baseline.make_data(shape)
    hidden.requires_grad_(True)
    router = _router_weight(shape)
    ids, weights = _route(hidden, router, shape.top_k)
    weights.retain_grad()
    if isinstance(layer, MegaMoeExperts):
        output = layer(hidden, ids, weights)
    else:
        output = baseline.forward_layer(layer, hidden, ids, weights)
    output.backward(upstream)
    optimizer = torch.optim.SGD([*layer.parameters(), router], lr=1e-3)
    optimizer.step()
    torch.npu.synchronize()
    gate_up_grad, down_grad = baseline.expert_weight_gradients(layer)
    gate_up_weight, down_weight = baseline.expert_weights(layer)
    compared = {
        "output": output,
        "input_grad": hidden.grad,
        "route_weight_grad": weights.grad,
        "router_grad": router.grad,
        "router_weight": router,
        "gate_up_weight_grad": gate_up_grad,
        "down_weight_grad": down_grad,
        "gate_up_weight": gate_up_weight,
        "down_weight": down_weight,
    }
    for name, tensor in compared.items():
        assert tensor is not None, f"rank={baseline.RANK}: missing Router graph tensor {name}."
        baseline.assert_finite(f"Router graph {name}", tensor)
    return {name: tensor.detach().clone() for name, tensor in compared.items()}


def test_mega_moe_router_interface() -> None:
    """Validate omitted counts and Router gradients with one step per backend."""
    shape = baseline.performance_shape()
    start_shmem_lifetime()
    os.environ.pop("SYMMETRIC_MEMORY_HEAP_SIZE", None)
    mega, common = baseline.new_layers(shape)
    try:
        reference = _run_router_layer(common, shape)
        actual = _run_router_layer(mega, shape)
        for name, expected in reference.items():
            baseline.assert_close(f"Router graph {name}", actual[name], expected)
    finally:
        mega.close()
    write_evidence({
        "shape": asdict(shape), "dtype": "bfloat16", "ep_is_whole_world": True,
        "capacity_factor": None, "counts_supplied": False, "router_projection_dtype": "float32",
        "backend_steps": {"common": 1, "mega_moe": 1},
        "validated_tensors": list(reference), "output_gradient_update_validation": True,
    })
