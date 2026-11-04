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
"""Fixed real-layer core inputs: packet arithmetic and autograd ownership."""

import gc
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from hyper_parallel.components.modules import KimiDeltaAttention
from hyper_parallel.components.functional.kimi_delta_attention_state_summary import (
    apply_kda_state_summary, apply_kda_state_gradient_summary,
)
from hyper_parallel.distributed.context_parallel.kimi_delta_attention import (
    KimiDeltaAttentionLayerCP, KimiDeltaAttentionCoreCP,
)
from hyper_parallel.distributed.context_parallel.kda_onesided import KDAOneSidedP2P
from tests.torch.context_parallel._test_kda_rd_layer import _setup


_GRAD_NAMES = ("query", "key", "value", "gate", "beta", "a_log", "dt_bias")


def _assert_fp32_state_apply():
    """An enclosing model autocast must not reduce state/adjoint apply precision."""
    generator = torch.Generator().manual_seed(8147)
    matrix, state, local = [torch.randn((1, 3, 128, 128), generator=generator).npu() * .02 for _ in range(3)]
    expected = (matrix @ state + local, matrix.transpose(-2, -1) @ state + local)
    with torch.autocast("npu", dtype=torch.bfloat16):
        actual = (apply_kda_state_summary(local, matrix, state),
                  apply_kda_state_gradient_summary(local, matrix, state))
    for value, reference in zip(actual, expected):
        assert value.dtype == torch.float32 and torch.equal(value, reference), "State apply was autocast"


def _model_with_heads(heads, seed=91237):
    """Keep the real layer contract while varying the projected head count."""
    torch.manual_seed(seed)
    model = KimiDeltaAttention(hidden_size=512, num_heads=heads)
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


def _bootstrap_for_group(group):
    """Use distinct endpoints for disjoint CP groups within one torchrun job."""
    offset = 4 * dist.get_process_group_ranks(group)[0]
    def _endpoint(value):
        host, port = value.rsplit(":", 1)
        return f"{host}:{int(port) + offset}"
    os.environ["SHMEM_UID_SESSION_ID"] = _endpoint(os.environ["SHMEM_UID_SESSION_ID"])
    return _endpoint(os.environ["STATE_PULL_BOOTSTRAP"])


def _fixtures(layer, length, size):
    """Capture projected/ShortConv inputs with long, mixed and aligned resets."""
    values = torch.randn((length, 512), generator=torch.Generator().manual_seed(1103 + dist.get_rank()),
                         dtype=torch.bfloat16).npu()
    total = length * size
    mixed = {point for block in range(0, size, 2) for point in (
        block * length, block * length + length // 2 + 1, (block + 1) * length + 67,
        (block + 2) * length - 1, (block + 2) * length)}
    offsets = {
        "long": (0, total),
        "mixed": tuple(sorted({0, total} | {point for point in mixed if 0 < point < total})),
        "reset": tuple(range(0, total + 1, length)),
    }
    result = {}
    current = None

    def _capture_inputs(_module, inputs, kwargs):
        """Retain identical projected tensors for both state transports."""
        result[current] = (tuple(x.detach() for x in (*inputs, kwargs["a_log"], kwargs["dt_bias"])),
                           kwargs.get("scale"), kwargs.get("sequence_info"))

    handle = layer.core.register_forward_pre_hook(_capture_inputs, with_kwargs=True)
    with torch.no_grad():
        for document, ends in offsets.items():
            current = document
            layer(values, cu_seqlens=ends)
    handle.remove()
    # Same projected values, but exercise the metadata-free dense kernel path.
    result["dense"] = (*result["long"][:2], None)
    return result


def _invoke(model, tensors, fixture):
    """Run the same raw-input core contract and immutable packed metadata."""
    _, scale, sequence = fixture
    return model(*tensors[:5], a_log=tensors[5], dt_bias=tensors[6], scale=scale, sequence_info=sequence)


def _capture(model, fixtures, documents, grad_output, checkpointed=False):
    """Reverse backward order across three live graphs and check zero-loss isolation."""
    inputs = [[x.detach().clone().requires_grad_() for x in fixtures[doc][0]] for doc in documents]
    outputs = []
    for tensors, doc in zip(inputs, documents):
        def _call(*values, doc=doc):
            """Bind document metadata before optional checkpoint replay."""
            return _invoke(model, values, fixtures[doc])
        outputs.append(checkpoint(_call, *tensors, use_reentrant=False) if checkpointed else _call(*tensors))
    for index in reversed(range(len(inputs))):
        outputs[index].backward(grad_output if index == 0 else torch.zeros_like(grad_output))
    torch.npu.synchronize()
    for tensors in inputs[1:]:
        assert all(torch.count_nonzero(x.grad).item() == 0 for x in tensors), "Cross-invocation gradient contamination"
    return {"output": outputs[0].detach().cpu(), **{
        name: tensor.grad.detach().cpu() for name, tensor in zip(_GRAD_NAMES, inputs[0])}}


def _compare(label, actual, expected):
    """Require finite and bitwise-identical outputs and all seven input gradients."""
    passed = all(torch.equal(value, expected[name]) and bool(torch.isfinite(value).all())
                 for name, value in actual.items())
    vote = torch.tensor(int(passed), device="npu", dtype=torch.int32)
    dist.all_reduce(vote, op=dist.ReduceOp.MIN)
    assert vote.item(), f"One-sided KDA arithmetic/lifetime regression: {label}"


def _check_close(model, transport, fixture, grad_output):
    """Reject live graphs; completed, discarded and no-grad graphs must not leak."""
    inputs = [value.detach().clone().requires_grad_() for value in fixture[0]]
    output = _invoke(model, inputs, fixture)
    try:
        transport.close()
    except RuntimeError as error:
        assert "pending backward" in str(error), str(error)
    else:
        raise AssertionError("Closed an arena while its backward was still pending")
    output.backward(grad_output)
    del output
    with torch.no_grad():
        output = _invoke(model, inputs, fixture)
    del output
    output = _invoke(model, inputs, fixture)
    del output, inputs
    gc.collect()
    transport.close()


def test_kda_onesided_core():
    """Cover complete core and mandatory lifetimes without a performance sweep."""
    os.environ.setdefault("KDA_TEST_LENGTH", "8192")
    mesh, base, unused, unused_grad = _setup()
    del unused, unused_grad
    _assert_fp32_state_apply()
    rank, size = mesh.get_local_rank(), mesh.size()
    heads = int(os.environ.get("KDA_ONESIDED_HEADS", "96"))
    if heads != 96:
        base = _model_with_heads(heads)
    length = int(os.environ["KDA_TEST_LENGTH"])
    layer = KimiDeltaAttentionLayerCP(base.npu(), mesh, backend="triton")
    fixtures = _fixtures(layer, length, size)
    control = layer.core
    del layer, base
    uri = _bootstrap_for_group(control.state.cp_group)
    configurations = [("groups1", 1, 0)]
    if heads > 1:
        configurations.extend((("groups2", 2, 0), ("groups2sg16", 2, 16)))
    transports = {
        name: KDAOneSidedP2P(control.state.cp_group, bootstrap_uri=uri, num_heads=heads,
                             producer_groups=groups, reserved_workers=workers)
        for name, groups, workers in configurations
    }
    assert len({id(transport.arena.runtime) for transport in transports.values()}) == 1
    models = {name: KimiDeltaAttentionCoreCP(mesh, backend="triton", p2p_transport=transport)
              for name, transport in transports.items()}
    grad_output = (torch.randn(fixtures["long"][0][2].shape,
                               generator=torch.Generator().manual_seed(99103 + dist.get_rank()),
                               dtype=torch.bfloat16) * .01).npu()
    checks = []
    for doc in fixtures:
        expected = _capture(control, fixtures, [doc], grad_output)
        for name, model in models.items():
            label = f"{doc}/{name}"
            _compare(label, _capture(model, fixtures, [doc], grad_output), expected)
            checks.append(label)
            if doc == "mixed":
                for checkpointed in (False, True):
                    label = f"{name}/three_flights/checkpoint={checkpointed}"
                    actual = _capture(model, fixtures, ["mixed", "long", "mixed"], grad_output, checkpointed)
                    _compare(label, actual, expected)
                    checks.append(label)
        del expected
    for name, transport in transports.items():
        _check_close(models[name], transport, fixtures["long"], grad_output)
    if os.environ.get("KDA_ONESIDED_RESULTS"):
        output = Path(os.environ["KDA_ONESIDED_RESULTS"])
        output.mkdir(parents=True, exist_ok=True)
        (output / f"rank{dist.get_rank()}.json").write_text(json.dumps({
            "rank": rank, "global_rank": dist.get_rank(), "size": size, "heads": heads, "length": length,
            "group_ranks": dist.get_process_group_ranks(mesh.get_group()), "completed": True,
            "bitwise_checks": len(checks) * 8, "cases": checks, "lifetime_pass": True,
        }, indent=2), encoding="utf-8")
    print(f"PASS one-sided KDA core CP{size} rank={rank}: {len(checks) * 8} bitwise checks and close lifetimes")
    dist.destroy_process_group()


if __name__ == "__main__":
    test_kda_onesided_core()
