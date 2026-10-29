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
"""Same-input KDA AG/hybrid gradients and invocation-local cache checks."""
import copy
from datetime import timedelta
import os

import torch
import torch.distributed as dist
import torch_npu
from torch.utils.checkpoint import checkpoint

import hyper_parallel as hp
from hyper_parallel.components.functional.kimi_delta_attention_fla_adapter import get_fla_kda_staged_ops
from hyper_parallel.components.functional.kimi_delta_attention_state_summary import (
    kda_state_gradient_summary_from_prepared,
    kda_state_summary_forward_from_prepared,
)
from hyper_parallel.components.modules import KimiDeltaAttention
from hyper_parallel.distributed._builder.forward_rewriter import _commit_forward_rewrite
from hyper_parallel.models.kimi_k3.adapter.distributed.context_parallel import kimi_delta_attention_cp_wrapper


def _model():
    """Build the shared BF16 layer with deterministic weights and sensitive gates."""
    torch.manual_seed(91237)
    model = KimiDeltaAttention(hidden_size=512, num_heads=96)
    with torch.no_grad():
        for name, value in model.named_parameters():
            if name == "A_log":
                value.zero_()
            elif name == "dt_bias":
                value.fill_(-10)
            elif name == "o_norm.weight":
                value.fill_(1)
            elif value.ndim == 1:
                value.zero_()
            else:
                value.normal_(0, .02)
    return model.bfloat16().train()


def _saved_transfers(outputs):
    """Verify the actual autograd storage, not merely a view's logical size."""
    stack = [output.grad_fn for output in outputs]
    seen, transfers = set(), []
    while stack:
        node = stack.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        if type(node).__name__ == "_KDAStateP2PFunctionBackward" and node.state_cp is not None:
            matrix = node.saved_tensors[12]
            size = matrix.numel() * matrix.element_size()
            if matrix.ndim != 4 or matrix.untyped_storage().nbytes() != size:
                raise AssertionError("State transfer cache retains more than one local M")
            transfers.append(size)
        stack.extend(value[0] for value in node.next_functions)
    return transfers


def _execute(model, inputs, grads, group, *, gather=False, checkpointed=False):
    """Run live calls in reverse backward order and collect local and reduced gradients."""
    values = [value.npu().requires_grad_() for value in inputs]
    outputs = [checkpoint(model, value, use_reentrant=False) if checkpointed else model(value) for value in values]
    if not checkpointed and len(_saved_transfers(outputs)) != (len(outputs) if gather else 0):
        raise AssertionError("Expected one compact local M per gather invocation")
    for index in reversed(range(len(values))):
        outputs[index].backward(grads[index].npu())
    local_params, params = {}, {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise AssertionError(f"Missing parameter gradient: {name}")
        local_params[name] = parameter.grad.detach().cpu().clone()
        gradient = parameter.grad.float()
        dist.all_reduce(gradient, group=group)
        params[name] = gradient.cpu()
    return {"output": [value.detach().cpu() for value in outputs],
            "hidden": [value.grad.cpu() for value in values], "params": params, "local_params": local_params}


def _assert_gradient(actual, expected):
    delta = actual.double() - expected.double()
    error = float(delta.norm() / expected.double().norm().clamp_min(1e-30))
    if not bool(torch.isfinite(actual).all()) or error > .003:
        raise AssertionError(f"KDA same-input gradient relative L2 {error} exceeds 0.003")


def _assert_same(actual, expected, *, lifetime=False):
    for kind in ("output", "hidden"):
        torch.testing.assert_close(actual[kind][0], expected[kind][0], atol=0, rtol=0)
    kind = "local_params" if lifetime else "params"
    for name, gradient in actual[kind].items():
        _assert_gradient(gradient, expected[kind][name])
    if lifetime and torch.count_nonzero(actual["hidden"][1]):
        raise AssertionError("The zero-loss invocation has a nonzero input gradient")



def _assert_native_summary_contract(length: int, heads: int) -> None:
    """Catch feedback dtype and scale/cast regressions against native FLA consumers."""
    generator = torch.Generator().manual_seed(713)
    shape = (1, length, heads, 128)
    key, weight, value, query, grad_output, grad_value = (
        (torch.randn(shape, generator=generator, dtype=torch.bfloat16) * .02).npu()
        for _ in range(6)
    )
    gate = torch.full(shape, -.02, device=key.device, dtype=torch.float32)
    state, transition = kda_state_summary_forward_from_prepared(key, weight, value, gate)
    ops = get_fla_kda_staged_ops()
    expected_state = ops.chunk_gated_delta_rule_fwd_h(
        k=key, w=weight, u=value, gk=gate, initial_state=None,
        output_final_state=True, chunk_size=64)[2]
    scale = 128 ** -.5
    gradient = kda_state_gradient_summary_from_prepared(
        query, key, weight, gate, grad_output, grad_value, scale)
    expected_gradient = ops.chunk_gated_delta_rule_bwd_dhu(
        q=query, k=key, w=weight, gk=gate, do=grad_output, dv=grad_value,
        h0=torch.zeros_like(state), dht=None, scale=scale, chunk_size=64)[1]
    # Independent FP64 recurrence checks M without duplicating the device kernel.
    reference = torch.eye(128, dtype=torch.float64)
    key_ref, weight_ref, gate_ref = (tensor[0, :, 0].cpu().double() for tensor in (key, weight, gate))
    for start in range(0, length, 64):
        keys = key_ref[start:start + 64]
        weights = weight_ref[start:start + 64]
        reference = gate_ref[start + 63].exp2()[:, None] * reference - keys.T @ (weights @ reference)
    torch.testing.assert_close(transition[0, 0].cpu().double(), reference, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(state, expected_state, atol=0, rtol=0)
    torch.testing.assert_close(gradient, expected_gradient, atol=0, rtol=0)


def test_kda_ag_layer() -> None:
    """Check supported combinations with identical weights, inputs and upstream gradients."""
    torch.set_num_threads(1)
    torch_npu.npu.set_device(int(os.environ["LOCAL_RANK"]))
    torch.npu.matmul.allow_hf32 = False
    torch.npu.config.allow_internal_format = False
    dist.init_process_group("hccl", timeout=timedelta(seconds=240))
    size = dist.get_world_size()
    root = hp.init_device_mesh("npu", (size,), mesh_dim_names=("cp",))
    length = int(os.environ.get("KDA_TEST_LENGTH", "4096"))
    if length < 4096 or length % 64:
        raise ValueError("Qualification requires local length >= 4096 and full chunks")
    _assert_native_summary_contract(length, (96, 48, 24)[root.get_local_rank() % 3])
    base = _model()
    generator = torch.Generator().manual_seed(1097 + sum(root.rank_list))
    full = torch.randn(1, length * size, 512, generator=generator, dtype=torch.bfloat16)
    grad_output = torch.randn(full.shape, generator=generator, dtype=torch.bfloat16) * .01
    local = full.chunk(size, 1)[root.get_local_rank()].contiguous()
    local_do = grad_output.chunk(size, 1)[root.get_local_rank()].contiguous()
    cases = [("p2p", 1, 1), ("allgather", 1, 1), ("grouped_allgather_p2p", 1, 2),
             ("p2p", 2, 1), ("allgather", 2, 1)]
    if size >= 8:
        cases.append(("grouped_allgather_p2p", 2, 2))
    controls = {}
    for protocol, ulysses, width in cases:
        model = copy.deepcopy(base).npu()
        request = kimi_delta_attention_cp_wrapper(
            model, None, None, root, None, state_cp_method=protocol,
            ulysses_degree=ulysses, allgather_group_size=width, backend="triton")
        _commit_forward_rewrite(request)
        gather = protocol != "p2p" and size // ulysses > 1
        result = _execute(model, [local], [local_do], root.get_group(), gather=gather)
        if protocol == "p2p":
            controls[ulysses] = result
        _assert_same(result, controls[ulysses])
        if gather and ulysses == 1:
            model.zero_grad(set_to_none=True)
            repeated = _execute(
                model, [local, -local], [local_do, torch.zeros_like(local_do)], root.get_group(),
                gather=True, checkpointed=protocol == "grouped_allgather_p2p")
            _assert_same(repeated, result, lifetime=True)
            del repeated
        print(f"PASS KDA {protocol}/U{ulysses}/g{width} gradients and local-M cache", flush=True)
        del result, model, request
    dist.destroy_process_group()
