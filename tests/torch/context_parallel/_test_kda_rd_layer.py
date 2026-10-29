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
"""K3-shaped layer compatibility and autograd-cache lifecycle checks."""
import copy
from datetime import timedelta
import os

import torch
import torch.distributed as dist
import torch_npu
from torch.utils.checkpoint import checkpoint

import hyper_parallel as hp
from hyper_parallel.distributed._builder.forward_rewriter import _commit_forward_rewrite
from hyper_parallel.models.kimi_k3.adapter.distributed.context_parallel import kimi_delta_attention_cp_wrapper
from tests.torch.context_parallel._test_kda_ag_layer import _model


def _setup():
    """Initialize CP and identical global inputs for the requested K3 shape."""
    torch.set_num_threads(1)
    torch_npu.npu.set_device(int(os.environ["LOCAL_RANK"]))
    torch.npu.matmul.allow_hf32 = False
    torch.npu.config.allow_internal_format = False
    dist.init_process_group("hccl", timeout=timedelta(seconds=240))
    size = dist.get_world_size()
    dp_size = int(os.environ.get("KDA_TEST_DP_SIZE", "1"))
    if dp_size < 1 or size % dp_size:
        raise ValueError("DP size must divide the available devices")
    root = hp.init_device_mesh("npu", (size // dp_size, dp_size), mesh_dim_names=("cp", "dp"))
    mesh = root["cp"]
    length = int(os.environ.get("KDA_TEST_LENGTH", "4096"))
    if length < 4096 or length % 64:
        raise ValueError("Qualification requires local length >= 4096 and full chunks")
    generator = torch.Generator().manual_seed(1103)
    full = torch.randn(1, length * mesh.size(), 512, generator=generator, dtype=torch.bfloat16)
    full_do = torch.randn(full.shape, generator=generator, dtype=torch.bfloat16) * .01
    local, dout = [value.chunk(mesh.size(), 1)[mesh.get_local_rank()].contiguous().npu() for value in (full, full_do)]
    return mesh, _model(), local, dout


def _wrap(base, mesh, protocol, ulysses):
    """Install the explicit state CP method on a fresh copy of the same layer."""
    model = copy.deepcopy(base).npu()
    request = kimi_delta_attention_cp_wrapper(
        model, None, None, mesh, None, backend="triton", state_cp_method=protocol, ulysses_degree=ulysses,
    )
    _commit_forward_rewrite(request)
    return model


def _assert_invocation_cache(node):
    """Check the invocation tensor layout and uniquely owned compact transfers."""
    cache = node.saved_tensors[12:12 + node.state_cache_count]
    if node.sequence_info is None:
        expected = sum(node.cp_rank >= distance for distance in node.state_cp.distances)
    else:
        expected = node.state_cp.plan.cache_count(node.cp_rank)
    metadata_count = 0 if node.sequence_info is None else 3
    if len(node.saved_tensors) != 12 + expected + metadata_count:
        raise AssertionError("RD saved-tensor layout does not match its invocation plan")
    if len(cache) != expected:
        raise AssertionError("RD saved a duplicate local M or omitted a round transfer")
    for matrix in cache:
        if matrix.untyped_storage().nbytes() != matrix.numel() * matrix.element_size():
            raise AssertionError("RD autograd cache retains oversized storage")
    storages = {matrix.untyped_storage().data_ptr() for matrix in cache}
    if len(storages) != len(cache):
        raise AssertionError("RD round transfers unexpectedly share storage")


def _assert_compact_cache(outputs):
    """Check every live RD autograd context without retaining producer workspaces."""
    stack, seen = [item.grad_fn for item in outputs], set()
    while stack:
        node = stack.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        if type(node).__name__ == "_KDAStateP2PFunctionBackward" and node.state_cp is not None:
            _assert_invocation_cache(node)
        stack.extend(item[0] for item in node.next_functions)


def _capture(model, value, dout, *, lifetime=False, checkpointed=False, group=None):
    """Capture outputs and gradients while checking invocation-owned compact caches."""
    model.zero_grad(set_to_none=True)
    inputs = [value.detach().clone().requires_grad_() for _ in range(2 if lifetime else 1)]
    outputs = [checkpoint(model, item, use_reentrant=False) if checkpointed else model(item) for item in inputs]
    if not checkpointed:
        _assert_compact_cache(outputs)
    for index in reversed(range(len(inputs))):
        outputs[index].backward(dout if index == 0 else torch.zeros_like(dout))
    result = {"output": outputs[0].detach().cpu(), "hidden": inputs[0].grad.cpu()}
    if lifetime and torch.count_nonzero(inputs[1].grad):
        raise AssertionError("Zero-loss invocation has a nonzero input gradient")
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise AssertionError(f"Missing gradient for {name}")
        gradient = parameter.grad.float()
        if group is not None:
            dist.all_reduce(gradient, group=group)
        result["parameter/" + name] = gradient.cpu()
    return result


def _assert_compatible(actual, expected):
    """Keep the established relative L2 budget and report every failing interface."""
    failures = []
    for name, value in actual.items():
        delta = value.double() - expected[name].double()
        relative = float(delta.norm() / expected[name].double().norm().clamp_min(1e-30))
        if not torch.isfinite(value).all() or relative > .003:
            failures.append(f"{name}: relative L2={relative:.9g}, limit=0.003")
    if failures:
        raise AssertionError("KDA compatibility: " + "; ".join(failures))


def test_kda_rd_layer():
    """Compare pure/hybrid RD with identical same-U controls and live caches."""
    mesh, base, local, dout = _setup()
    for ulysses in (1, 2):
        model = _wrap(base, mesh, "recursive_doubling", ulysses)
        local_reference = _capture(model, local, dout)
        for checkpointed in (False, True):
            result = _capture(model, local, dout, lifetime=True, checkpointed=checkpointed)
            _assert_compatible(result, local_reference)
        print(f"PASS RD/U{ulysses} rank={mesh.get_local_rank()} reverse-live/cache/checkpoint", flush=True)
        actual = _capture(model, local, dout, group=mesh.get_group())
        del model, local_reference, result
        control = _wrap(base, mesh, "p2p", ulysses)
        expected = _capture(control, local, dout, group=mesh.get_group())
        _assert_compatible(actual, expected)
        print(f"PASS RD/U{ulysses} rank={mesh.get_local_rank()} same-input P2P comparison", flush=True)
        del control, actual, expected
    dist.destroy_process_group()


if __name__ == "__main__":
    test_kda_rd_layer()
