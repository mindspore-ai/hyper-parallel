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
"""Packed linear-attention semantics, sample isolation and saved-call ownership."""
import copy
from datetime import timedelta
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.checkpoint import checkpoint

import hyper_parallel as hp
from hyper_parallel.components.modules.kimi_delta_attention import KimiDeltaAttention, KimiRMSNormGated
from hyper_parallel.distributed.context_parallel.gated_delta_net import GatedDeltaNetP2PCP, GatedDeltaNetUlyssesCP
from hyper_parallel.distributed.context_parallel.kimi_delta_attention import KimiDeltaAttentionLayerCP


class _GDN(nn.Module):
    """Small projection shell with the same GDN layer contract as Qwen3.5."""

    def __init__(self, heads: int, dim: int) -> None:
        """Build a projection shell without importing model-specific optional packages."""
        super().__init__()
        self.num_k_heads = self.num_v_heads = heads
        self.head_k_dim = self.head_v_dim = dim
        self.key_dim = self.value_dim = heads * dim
        self.conv_dim = 3 * heads * dim
        self.activation = "silu"
        for name, channels in (("in_proj_qkv", self.conv_dim), ("in_proj_z", self.value_dim),
                               ("in_proj_b", heads), ("in_proj_a", heads)):
            setattr(self, name, nn.Linear(32, channels, bias=False))
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, 4, groups=self.conv_dim, padding=3)
        self.A_log = nn.Parameter(torch.zeros(heads))
        self.dt_bias = nn.Parameter(torch.full((heads,), -8.))
        self.norm = KimiRMSNormGated(dim)
        self.out_proj = nn.Linear(self.value_dim, 32, bias=False)


def _model(kind, npu):
    heads, dim = (96, 128) if npu else (4, 4)
    torch.manual_seed(451)
    model = (KimiDeltaAttention(hidden_size=32, num_heads=heads, head_k_dim=dim, head_v_dim=dim,
                               conv_kernel_size=4, chunk_size=64 if npu else 4)
             if kind == "kda" else _GDN(heads, dim))
    with torch.no_grad():
        model.A_log.zero_()
        model.dt_bias.fill_(-8.)
    return model


def _wrapper(model, kind, mesh, npu, degree=1, method="p2p"):
    options = {"backend": "triton" if npu else "eager", "chunk_size": 64 if npu else 4}
    if kind == "kda":
        return KimiDeltaAttentionLayerCP(
            model, mesh, ulysses_degree=degree, state_cp_method=method,
            allgather_group_size=2 if method == "grouped_allgather_p2p" else 1, **options)
    cls = GatedDeltaNetUlyssesCP if degree > 1 else GatedDeltaNetP2PCP
    return cls(model, mesh, **options)


def _execute(model, wrapper, inputs, gradients, boundaries, *, live=False):
    """Use FP32 master parameters and identical backward order for all schedules."""
    model.zero_grad(set_to_none=True)
    values = [tensor.detach().clone().requires_grad_() for tensor in inputs]

    def call(value: torch.Tensor, offsets: tuple[int, ...]) -> torch.Tensor:
        """Execute with differentiable BF16 views of FP32 master weights."""
        parameters = {name: param if name.endswith(("A_log", "dt_bias")) else param.to(value.dtype)
                      for name, param in wrapper.named_parameters()}
        return torch.func.functional_call(wrapper, parameters, (value,), {"cu_seq_lens": offsets})

    outputs = [None] * len(values)
    if live:
        outputs = [checkpoint(call, value, offsets, use_reentrant=False)
                   for value, offsets in zip(values, boundaries)]
    for index in reversed(range(len(values))):
        if not live:
            outputs[index] = call(values[index], boundaries[index])
        outputs[index].backward(gradients[index])
    result = {f"output_{index}": tensor.detach().cpu() for index, tensor in enumerate(outputs)}
    result.update({f"hidden_{index}": tensor.grad.cpu() for index, tensor in enumerate(values)})
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and parameter.grad.dtype == torch.float32, name
        grad = parameter.grad.detach().clone()
        dist.all_reduce(grad)
        result[name] = grad.cpu()
    assert all(torch.isfinite(tensor).all() for tensor in result.values())
    return result


def _assert_same(actual, expected, *, exact=True):
    for name, value in actual.items():
        torch.testing.assert_close(value, expected[name], rtol=0 if exact else 2e-4,
                                   atol=0 if exact else 2e-6, msg=lambda message: f"{name}: {message}")


def _isolation(wrapper, inputs, grad, offsets, rank, length):
    """Loss in later documents must have exactly zero derivative on the first."""
    value = inputs.detach().clone().requires_grad_()
    parameters = {name: param if name.endswith(("A_log", "dt_bias")) else param.to(value.dtype)
                  for name, param in wrapper.named_parameters()}
    output = torch.func.functional_call(wrapper, parameters, (value,), {"cu_seq_lens": offsets})
    positions = torch.arange(rank * length, (rank + 1) * length, device=value.device)
    output.backward(grad * (positions >= offsets[1])[:, None])
    assert torch.count_nonzero(value.grad[positions < offsets[1]]) == 0


def _run(npu):
    device = "npu" if npu else "cpu"
    if npu:
        # Keep the CPU worker usable when the optional NPU package is absent.
        import torch_npu  # pylint: disable=import-outside-toplevel
        torch_npu.npu.set_device(int(os.environ["LOCAL_RANK"]))
        torch_npu.npu.matmul.allow_hf32 = False
        torch_npu.npu.config.allow_internal_format = False
        torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    dist.init_process_group("hccl" if npu else "gloo", timeout=timedelta(seconds=300))
    rank, size = dist.get_rank(), dist.get_world_size()
    root = hp.init_device_mesh(device, (size, 1), mesh_dim_names=("cp", "local"))
    mesh = root["cp"]
    length = int(os.environ.get("LINEAR_TND_LOCAL_TOKENS", "4096")) if npu else 96
    offsets = (0, length + length // 2 + 1, 3 * length, size * length - 1, size * length)
    aligned = tuple(range(0, size * length + 1, length))
    dtype = torch.bfloat16 if npu else torch.float32
    for kind in ("kda", "gdn"):
        base = _model(kind, npu)
        full = (torch.randn(size * length, 32) * .05).to(dtype)
        grad = (torch.randn_like(full.float()) * .01).to(dtype)
        local = full[rank * length:(rank + 1) * length].to(device)
        do = grad[rank * length:(rank + 1) * length].to(device)
        options = [(1, "p2p"), (size, "p2p")]
        if kind == "kda":
            options.insert(1, (2, "p2p"))
            if npu:
                options += [(1, "allgather"), (1, "grouped_allgather_p2p"), (2, "allgather")]
                if size >= 8:
                    options.append((2, "grouped_allgather_p2p"))
        references = {}
        for degree, method in options:
            model = copy.deepcopy(base).to(device)
            wrapper = _wrapper(model, kind, mesh, npu, degree, method)
            # Different metadata and nonzero losses in two simultaneously live calls.
            inputs, gradients = [local, local * .75], [do, do * .5]
            expected = _execute(model, wrapper, inputs, gradients, [offsets, aligned])
            actual = _execute(model, wrapper, inputs, gradients, [offsets, aligned], live=True)
            _assert_same(actual, expected)
            if method == "p2p":
                references[degree] = expected
            else:
                _assert_same(expected, references[degree])
            _isolation(wrapper, local, do, offsets, rank, length)
            if not npu:
                # Independent per-document CP1 evaluation: no cross-rank summary or halo.
                serial_model = copy.deepcopy(base)
                serial_wrapper = _wrapper(serial_model, kind, root["local"], False)
                # Every rank runs the same serial graph; avoid reducing its gradients size times.
                serial = _execute(serial_model, serial_wrapper, [full, full * .75],
                                  [grad, grad * .5], [offsets, aligned])
                for name in serial:
                    if name.startswith(("output_", "hidden_")):
                        serial[name] = serial[name][rank * length:(rank + 1) * length]
                    else:
                        serial[name] /= size
                _assert_same(expected, serial, exact=False)
            if rank == 0:
                print(f"PASS TND {kind} CP{size} U{degree} {method} T{length}", flush=True)
            del model, wrapper
        del references
    dist.barrier()
    dist.destroy_process_group()


def test_linear_tnd_gloo():
    """Check KDA/GDN packed semantics against serial per-document references."""
    _run(False)


def test_linear_tnd_npu():
    """Check true resets, existing protocols, nonzero checkpoint and live-call ownership."""
    _run(True)
