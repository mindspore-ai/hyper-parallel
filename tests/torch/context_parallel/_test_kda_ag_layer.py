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
from hyper_parallel.components.modules import KimiDeltaAttention
from hyper_parallel.distributed._builder.forward_rewriter import _commit_forward_rewrite
from hyper_parallel.models.kimi_k3.adapter.distributed.context_parallel import kimi_delta_attention_cp_wrapper


def _model():
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
        if type(node).__name__ == "_KDAStateP2PFunctionBackward" and node.boundary is not None:
            matrix = node.saved_tensors[11]
            size = matrix.numel() * matrix.element_size()
            if matrix.ndim != 4 or matrix.untyped_storage().nbytes() != size:
                raise AssertionError("Boundary transfer cache retains more than one local M")
            transfers.append(size)
        stack.extend(value[0] for value in node.next_functions)
    return transfers


def _execute(model, inputs, grads, group, *, gather=False, checkpointed=False):
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
    base = _model()
    generator = torch.Generator().manual_seed(1097 + sum(root.rank_list))
    full = torch.randn(1, length * size, 512, generator=generator, dtype=torch.bfloat16)
    dout = torch.randn(full.shape, generator=generator, dtype=torch.bfloat16) * .01
    local = full.chunk(size, 1)[root.get_local_rank()].contiguous()
    local_do = dout.chunk(size, 1)[root.get_local_rank()].contiguous()
    cases = [("p2p", 1, 1), ("allgather", 1, 1), ("grouped_allgather_p2p", 1, 2),
             ("p2p", 2, 1), ("allgather", 2, 1)]
    if size >= 8:
        cases.append(("grouped_allgather_p2p", 2, 2))
    controls = {}
    for protocol, ulysses, width in cases:
        model = copy.deepcopy(base).npu()
        request = kimi_delta_attention_cp_wrapper(
            model, None, None, root, None, boundary_protocol=protocol,
            ulysses_degree=ulysses, group_size=width, backend="triton")
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


if __name__ == "__main__":
    test_kda_ag_layer()
