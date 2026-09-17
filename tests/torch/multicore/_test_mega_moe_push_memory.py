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
"""Exercise push receive-buffer reuse through both supported autograd entry points."""

from dataclasses import replace
from typing import Any
from unittest.mock import patch

import torch

from hyper_parallel.core.multicore import MegaMoeExperts
from hyper_parallel.core.multicore.modules.mega_moe.function import execute_mega_moe
from hyper_parallel.core.multicore.modules.mega_moe.route import prepare_topk_route, restore_topk_output
from hyper_parallel.core.multicore.torch import ops
from tests.torch.multicore import _test_mega_moe as baseline
from tests.torch.multicore._test_mega_moe_memory import _route
from tests.torch.multicore._mega_moe_utils import start_shmem_lifetime, write_evidence


def _forward(layer, hidden, ids, weights, counts, legacy):
    """Use the public expert-major bridge with ordinary differentiable permutation."""
    if not legacy or not isinstance(layer, MegaMoeExperts):
        return baseline.forward_layer(layer, hidden, ids, weights, tokens_per_expert=counts)
    resources = layer._get_execution_resources(hidden)  # pylint: disable=protected-access
    route = prepare_topk_route(hidden, ids, weights, resources.spec, counts, resources.workspace)
    output = execute_mega_moe(route.routed_tokens, layer.gate_up_weight, layer.down_weight,
                             route.metadata, resources.plan, resources.workspace)
    return restore_topk_output(output, route.unpermute_mapping, weights)


def _backward_results(layer, output, hidden, weights, upstream):
    """Compare both retained and final backward without accumulating gradients."""
    results = []
    for retained in (True, False):
        layer.zero_grad(set_to_none=True)
        hidden.grad = None
        weights.grad = None
        output.backward(upstream, retain_graph=retained)
        torch.npu.synchronize()
        tensors = [output, weights.grad, *baseline.expert_weight_gradients(layer)]
        if hidden.requires_grad:
            tensors.append(hidden.grad)
        for index, tensor in enumerate(tensors):
            assert tensor is not None, f"missing push-memory gradient {index}"
            baseline.assert_finite(f"push-memory gradient {index}", tensor)
        results.extend(tensor.detach().cpu().clone() for tensor in tensors)
    return results


def _deferred_results(layer, shape, patterns, legacy, input_grad):
    """Hold two routes while shared receive storage is overwritten between calls."""
    source, upstream = baseline.make_data(shape)
    saved = []
    for index, pattern in enumerate(patterns):
        hidden = (source * (1 + index / 8)).detach().requires_grad_(input_grad)
        ids, weights, counts = _route(shape, pattern)
        weights.requires_grad_(True)
        output = _forward(layer, hidden, ids, weights, counts, legacy)
        saved.append((output, hidden, weights))
    results = []
    for output, hidden, weights in reversed(saved):
        results.extend(_backward_results(layer, output, hidden, weights, upstream))
    return results


def test_mega_moe_push_memory_reuse() -> None:
    """Check alias/fallback, tails, empty ranks, retained graphs and trainable/frozen managed input."""
    shape = baseline.MoeShape()
    start_shmem_lifetime()
    mega, common = baseline.new_layers(shape, expert_capacity_factor=None)
    reference_input, _ = baseline.make_data(shape)
    resources = mega._get_execution_resources(reference_input)  # pylint: disable=protected-access
    assert resources.plan.reuse_backward_dispatch, "managed push plan must enable checked storage reuse"
    original = ops.mega_moe_grad_with_profile_buffer
    results = []
    try:
        for reuse in (False, True):
            resources.plan = replace(resources.plan, reuse_backward_dispatch=reuse)
            calls = []

            def checked_backward(*args: Any) -> None:
                """Check the actual native pointers, including the local receive prefix."""
                assert (args[0].data_ptr() == args[12].data_ptr()) == reuse
                assert args[17].data_ptr() != args[0].data_ptr()
                assert args[12].shape == args[17].shape
                calls.append(int(args[12].shape[0]))
                return original(*args)

            with patch.object(ops, "mega_moe_grad_with_profile_buffer", new=checked_backward):
                for legacy, input_grad in ((True, True), (False, False), (False, True)):
                    for patterns in (("balanced", "tail"), ("destination0", "destination1")):
                        expected = _deferred_results(common, shape, patterns, legacy, input_grad)
                        actual = _deferred_results(mega, shape, patterns, legacy, input_grad)
                        for index, (left, right) in enumerate(zip(expected, actual)):
                            baseline.assert_close(f"push reuse={reuse} legacy={legacy} tensor={index}", right, left)
                        results.append({"reuse": reuse, "legacy": legacy, "patterns": patterns,
                                        "input_grad": input_grad, "all_gradients_match": True})
            assert len(calls) == 24, f"expected twenty-four real backward calls, got {len(calls)}"
    finally:
        mega.close()
    write_evidence({"push_memory_reuse": results})
