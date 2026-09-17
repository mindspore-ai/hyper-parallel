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
"""Long-sequence KDA AG/hybrid qualification with a common serial FLA control."""
import copy
from datetime import timedelta
import gc
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch_npu
from torch.utils.checkpoint import checkpoint

import hyper_parallel as hp
from hyper_parallel.components.modules import KimiDeltaAttention
from hyper_parallel.distributed._builder.forward_rewriter import _commit_forward_rewrite
from hyper_parallel.distributed.context_parallel.kimi_delta_attention import KimiDeltaAttentionLayerP2PCP
from hyper_parallel.distributed.context_parallel.kimi_delta_attention_mesh import split_kda_mesh
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


def _metrics(actual, expected):
    delta = actual.double() - expected.double()
    return {"rel_l2": float(delta.norm() / expected.double().norm().clamp_min(1e-30)),
            "max_abs": float(delta.abs().max()), "reference_rms": float(expected.double().square().mean().sqrt()),
            "finite": bool(torch.isfinite(actual).all())}


def _execute(model, inputs, grads, *, use_checkpoint=False, reduce_group=None):
    values = [value.npu().requires_grad_() for value in inputs]
    torch.npu.synchronize()
    before = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    begin = time.perf_counter()
    outputs = [checkpoint(model, value, use_reentrant=False) if use_checkpoint else model(value) for value in values]
    transfers = [] if use_checkpoint else _saved_transfers(outputs)
    for index in reversed(range(len(values))):
        outputs[index].backward(grads[index].npu())
    torch.npu.synchronize()
    elapsed = (time.perf_counter() - begin) * 1000
    result = {"output": [value.detach().cpu() for value in outputs],
              "hidden": [value.grad.cpu() for value in values], "params": {}, "local_params": {}}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise AssertionError(f"Missing parameter gradient: {name}")
        result["local_params"][name] = parameter.grad.detach().cpu().clone()
        gradient = parameter.grad.float()
        if reduce_group is not None:
            dist.all_reduce(gradient, group=reduce_group)
        result["params"][name] = gradient.cpu()
    result["memory"] = {"baseline": before, "peak_allocated": torch.npu.max_memory_allocated(),
                        "ms": elapsed, "saved_transfers": transfers}
    return result


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
            transfers.append({"shape": list(matrix.shape), "storage_bytes": size})
        stack.extend(value[0] for value in node.next_functions)
    return transfers


def _check_lifetime(model, local, local_do, control, group, use_checkpoint):
    """Keep two distinct forwards alive and backpropagate in reverse order."""
    model.zero_grad(set_to_none=True)
    result = _execute(model, [local, -local], [local_do, torch.zeros_like(local_do)],
                      use_checkpoint=use_checkpoint, reduce_group=group)
    for kind in ("output", "hidden"):
        torch.testing.assert_close(result[kind][0], control[kind][0], atol=0, rtol=0)
    if torch.count_nonzero(result["hidden"][1]):
        raise AssertionError("The zero-loss invocation has a nonzero input gradient")
    # Check local gradients before the independent HCCL reduction. Native
    # parameter reductions need the same norm contract as single-call tests.
    parameter_metrics = {}
    for name, gradient in result["local_params"].items():
        metric = _metrics(gradient, control["local_params"][name])
        parameter_metrics[name] = metric
        if not metric["finite"] or metric["rel_l2"] > .003:
            raise AssertionError(f"Local parameter lifetime check failed for {name}: {metric}")
    return {**result["memory"], "parameter_metrics": parameter_metrics}


def _comparison(result, control):
    """Compare the complete returned layer contract using a specified control."""
    return {
        "output": _metrics(result["output"][0], control["output"][0]),
        "hidden": _metrics(result["hidden"][0], control["hidden"][0]),
        "params": {name: _metrics(value, control["params"][name]) for name, value in result["params"].items()},
        "bitwise": all(torch.equal(result[kind][0], control[kind][0]) for kind in ("output", "hidden"))
                   and all(torch.equal(value, control["params"][name]) for name, value in result["params"].items()),
        "local_bitwise": all(torch.equal(result[kind][0], control[kind][0]) for kind in ("output", "hidden"))
                         and all(torch.equal(value, control["local_params"][name])
                                 for name, value in result["local_params"].items()),
    }


def _assert_compatibility(records):
    """Gate the migrated contract; keep optional native-FLA diagnostics separate."""
    for record in records:
        for name, metric in record["p2p_comparison"].items():
            if name in ("bitwise", "local_bitwise"):
                continue
            values = metric.values() if name == "params" else [metric]
            for value in values:
                if not value["finite"] or value["rel_l2"] > .003:
                    raise AssertionError(f"KDA P2P-compatibility failed: {record['protocol']}/{name}: {value}")


def _record_lifetime_checks(record, model, local, local_do, result, group):
    """Keep lifecycle diagnostics separate from the protocol comparison loop."""
    if os.environ.get("KDA_TEST_LIFETIME") != "1" or record["ulysses"] != 1:
        return
    if record["protocol"] == "p2p":
        model.zero_grad(set_to_none=True)
        repeated = _execute(model, [local], [local_do], reduce_group=group)
        record["repeat_control"] = _comparison(repeated, result)
        record["repeat_control"]["local_params"] = {
            name: _metrics(value, result["local_params"][name])
            for name, value in repeated["local_params"].items()}
    else:
        record["lifetime"] = _check_lifetime(
            model, local, local_do, result, group, record["protocol"] == "grouped_allgather_p2p")


def test_kda_ag_layer() -> None:
    """Compare all combinations on identical weights/data/do, including live-call lifetimes."""
    torch.set_num_threads(1)
    torch_npu.npu.set_device(int(os.environ["LOCAL_RANK"]))
    torch.npu.matmul.allow_hf32 = False
    torch.npu.config.allow_internal_format = False
    dist.init_process_group("hccl", timeout=timedelta(seconds=240))
    size = dist.get_world_size()
    root = hp.init_device_mesh("npu", (size,), mesh_dim_names=("cp",))
    if os.environ.get("KDA_TEST_DP") == "2":
        root = hp.init_device_mesh("npu", (size // 2, 2), mesh_dim_names=("cp", "dp"))["cp"]
    rank, size = root.get_local_rank(), root.size()
    length = int(os.environ.get("KDA_TEST_LENGTH", "4096"))
    if length < 4096 or length % 64:
        raise ValueError("Qualification requires local length >= 4096 and full chunks")
    base = _model()
    generator = torch.Generator().manual_seed(1097 + sum(root.rank_list))
    full = torch.randn(1, length * size, 512, generator=generator, dtype=torch.bfloat16)
    dout = torch.randn(full.shape, generator=generator, dtype=torch.bfloat16) * .01
    reference = None
    if os.environ.get("KDA_TEST_SERIAL") == "1":
        singleton = split_kda_mesh(root, 1, ("kda_replica", "kda_single"))["kda_single"]
        serial = KimiDeltaAttentionLayerP2PCP(copy.deepcopy(base).npu(), singleton, backend="triton")
        reference = _execute(serial, [full], [dout])
        reference["output"] = [reference["output"][0].chunk(size, 1)[rank].clone()]
        reference["hidden"] = [reference["hidden"][0].chunk(size, 1)[rank].clone()]
        reference["params"] = {name.removeprefix("module."): value for name, value in reference["params"].items()}
        reference["local_params"] = {name.removeprefix("module."): value
                                     for name, value in reference["local_params"].items()}
        del serial
        gc.collect()
        torch.npu.empty_cache()
    cases = [("p2p", 1, 1), ("allgather", 1, 1), ("grouped_allgather_p2p", 1, 2),
             ("p2p", 2, 1), ("allgather", 2, 1)]
    if size >= 8:
        cases.append(("grouped_allgather_p2p", 2, 2))
    records = []
    controls = {}
    for protocol, ulysses, group_size in cases:
        model = copy.deepcopy(base).npu()
        request = kimi_delta_attention_cp_wrapper(model, None, None, root, None,
                                                  boundary_protocol=protocol, ulysses_degree=ulysses,
                                                  group_size=group_size, backend="triton")
        _commit_forward_rewrite(request)
        local = full.chunk(size, 1)[rank].contiguous()
        local_do = dout.chunk(size, 1)[rank].contiguous()
        result = _execute(model, [local], [local_do], reduce_group=root.get_group())
        if protocol != "p2p" and size // ulysses > 1 and len(result["memory"]["saved_transfers"]) != 1:
            raise AssertionError("Expected exactly one saved local transfer per state invocation")
        metrics = _comparison(result, reference) if reference is not None else None
        record = {"protocol": protocol, "ulysses": ulysses, "group_size": group_size,
                  "rank": rank, "length": length, "metrics": metrics, **result["memory"]}
        if protocol == "p2p":
            controls[ulysses] = result
        control = controls[ulysses]
        record["p2p_comparison"] = _comparison(result, control)
        _record_lifetime_checks(record, model, local, local_do, result, root.get_group())
        records.append(record)
        print(json.dumps(record), flush=True)
        del result, model, request
        gc.collect()
        torch.npu.empty_cache()
    out = os.environ.get("KDA_TEST_OUTPUT")
    if out:
        destination = Path(out)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / f"layer_rank{dist.get_rank()}.json").write_text(json.dumps(records, indent=2))
    _assert_compatibility(records)
    dist.destroy_process_group()


if __name__ == "__main__":
    test_kda_ag_layer()
