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
"""Real Qwen3-MoE async K/V AG consumer: CP1, old AR, new RS on CP8."""
from contextlib import nullcontext
from copy import deepcopy
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention
from hyper_parallel.distributed.context_parallel import collectives
from hyper_parallel.models.qwen3_moe.adapter.distributed.context_parallel_async import (
    qwen3_moe_async_colossal_cp_wrapper, _qwen3_moe_async_colossal_forward,
)
from tests.torch.context_parallel._test_sequence_halo import WorldMesh, init_group


def _legacy_sum_chunk(gradient, dim, size, group):
    """Prior SUM all-reduce/chunk implementation, confined to the comparison."""
    gradient = gradient.contiguous().clone()
    dist.all_reduce(gradient, group=group)
    return gradient.chunk(size, dim=dim)[dist.get_rank(group)].contiguous()


class SingleMesh:
    """Single-rank context for the full-sequence numerical oracle."""

    def size(self) -> int:
        """Return the unsharded context size."""
        return 1

    def get_local_rank(self) -> int:
        """Return the sole local rank."""
        return 0

    def get_group(self) -> None:
        """No collective is needed for the single-rank reference."""
        return None


def _compare_reduction_results(mode, values, saved):
    """Check the legacy suite tolerance and retain per-tensor error metrics."""
    comparisons = {}
    for ref in (["cp1"] if mode == "old_ar" else ["cp1", "old_ar"] if mode == "new_rs" else []):
        rows = []
        for name, value_cpu in values.items():
            target = saved[ref][name]
            actual, expected = value_cpu.float(), target.float()
            assert torch.isfinite(actual).all(), (mode, name)
            # Existing NPU async-CP suite's elementwise tolerances.
            torch.testing.assert_close(
                actual, expected, atol=1e-2, rtol=1e-2, msg=f"{mode}/{ref}/{name}",
            )
            error = actual.double()-expected.double()
            rows.append({"name": name, "max_abs": float(error.abs().max()),
                         "relative_L2": float(error.norm()/expected.double().norm().clamp_min(1e-30))})
        comparisons[ref] = rows
    return comparisons


def test_qwen_async_rs_cp8():
    """Compare both reduction implementations with the unsharded GQA oracle."""
    torch.use_deterministic_algorithms(True)
    device = init_group("hccl")
    rank = dist.get_rank()
    assert dist.get_world_size() == 8
    records = []
    try:
        for dtype in (torch.float16, torch.bfloat16):
            for length in (1024, 2048):
                local = length // 8
                start = rank * local
                torch.manual_seed(818)
                config = Qwen3MoeConfig()
                config.hidden_size = 512
                config.num_attention_heads = 8
                config.num_key_value_heads = 2
                config.head_dim = 64
                config.attention_dropout = 0.
                template = Qwen3MoeAttention(config, 0)
                full = torch.randn(1, length, 512).to(device=device, dtype=dtype)
                upstream = torch.randn(1, length, 512).to(device=device, dtype=dtype) / (length*512)**.5
                angles = torch.arange(length).float()[:, None] * torch.pow(10000., -torch.arange(0, 64, 2).float()/64)
                angles = torch.cat([angles, angles], dim=-1)[None]
                cos, sin = angles.cos().to(device=device, dtype=dtype), angles.sin().to(device=device, dtype=dtype)
                saved = {}
                for mode in ("cp1", "old_ar", "new_rs"):
                    model = deepcopy(template).to(device=device, dtype=dtype)
                    value = (full if mode == "cp1" else full[:, start:start+local]).clone().requires_grad_()
                    context = (patch.object(collectives, "_sum_scatter_sequence_gradient", _legacy_sum_chunk)
                               if mode == "old_ar" else nullcontext())
                    with context:
                        if mode == "cp1":
                            output = _qwen3_moe_async_colossal_forward(
                                model, value, (cos, sin), None, cp_mesh=SingleMesh(),
                            )[0]
                            (output*upstream).sum().backward()
                        else:
                            request = qwen3_moe_async_colossal_cp_wrapper(model, None, None, WorldMesh(), None)
                            model.forward = request.forward
                            output = model(value, (cos[:, start:start+local], sin[:, start:start+local]), None)[0]
                            (output*upstream[:,start:start+local]).sum().backward()
                            for parameter in model.parameters():
                                dist.all_reduce(parameter.grad)
                    values = {"output": output.detach().cpu(), "input_grad": value.grad.detach().cpu()}
                    values.update({"grad."+name: p.grad.detach().cpu() for name, p in model.named_parameters()})
                    if mode == "cp1":
                        values["output"] = values["output"][:, start:start+local]
                        values["input_grad"] = values["input_grad"][:, start:start+local]
                    saved[mode] = values
                    comparisons = _compare_reduction_results(mode, values, saved)
                    records.append({"rank": rank, "dtype": str(dtype), "length": length,
                                    "mode": mode, "comparisons": comparisons})
                    if rank == 0:
                        print("CASE "+json.dumps(records[-1]), flush=True)
                    del model, value, output, values
                dist.barrier()
        if os.environ.get("V41_TEST_REPORT_DIR"):
            path = Path(os.environ["V41_TEST_REPORT_DIR"]) / "qwen"
            path.mkdir(parents=True, exist_ok=True)
            (path/f"rank{rank}.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        dist.barrier()
        if rank == 0:
            print("PASS Qwen3-MoE async CP8 K/V, FP16/BF16, GQA8:2, S1024/2048, CP1/oldAR/newRS", flush=True)
    finally:
        dist.destroy_process_group()
