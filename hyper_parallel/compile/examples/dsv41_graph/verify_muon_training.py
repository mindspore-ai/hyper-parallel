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
"""Check cropped DeepSeek V4.1 Muon updates against reconstructed NS matrices.

This reference adds work and allocations; use benchmark_training.py separately
for performance measurements. The sampled recipe has evenly sized local shards.
"""

import json
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.compile.examples.dsv41_graph.benchmark_training import BenchmarkTrainer
from hyper_parallel.core.optimizer.dtensor_compat import to_local_if_dtensor
from hyper_parallel.core.optimizer.muon import compute_muon_slice_scale, zeropower_via_newtonschulz5
import hyper_parallel.core.optimizer.muon as muon_module
from hyper_parallel.trainer.config import TrainerConfig
from hyper_parallel.trainer.config.parser import parse_training_args


_NAMES = (
    "model.layers.0.self_attn.q_a_proj.weight",
    "model.layers.0.self_attn.q_b_proj.weight",
    "model.layers.0.self_attn.o_b_proj.weight",
    "model.layers.0.mlp.gate.weight",
    "model.layers.0.mlp.experts.down_proj",
)


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(((actual.float() - expected.float()).norm() /
                  expected.float().norm().clamp_min(1e-30)).item())


def _gather_matrix(local: torch.Tensor, layout: Any) -> tuple[torch.Tensor, list]:
    value, stages = local, []
    # Expert/batch shards already contain complete independent NS matrices.
    if value.ndim == 3 and all(not p.is_shard() or p.dim == 0 for p in layout.placements):
        return value, stages
    for axis, placement in enumerate(layout.placements):
        if not placement.is_shard():
            continue
        group = layout.mesh.get_group(axis)
        size, rank = dist.get_world_size(group), dist.get_rank(group)
        if size <= 1:
            continue
        stages.append((placement.dim, rank, value.shape[placement.dim]))
        gathered = torch.empty((size, *value.shape), dtype=value.dtype, device=value.device)
        dist.all_gather_into_tensor(gathered.view(-1), value.contiguous().view(-1), group=group)
        value = torch.cat(list(gathered.unbind(0)), dim=placement.dim)
    return value, stages


def _ns_reference(tensor: torch.Tensor, group: dict) -> torch.Tensor:
    update = zeropower_via_newtonschulz5(
        tensor, group["ns_steps"], ns_variant=group["ns_variant"],
        epsilon=group["ns_epsilon"], ns_coefficients=group["ns_coefficients"],
    )
    return update.mul_(compute_muon_slice_scale(update, group["matched_adamw_rms"]))


def _batch_errors(actual: torch.Tensor, local_ns: torch.Tensor, shapes: list, group: dict) -> dict:
    errors = {}
    for shape in set(shapes):
        if len(shape) != 3 or shape[-2:] != tuple(local_ns.shape[-2:]) or shape[0] < local_ns.shape[0]:
            continue
        count = shape[0]
        repeats = (count + local_ns.shape[0] - 1) // local_ns.shape[0]
        padded = torch.cat([local_ns] * repeats, dim=0)[:count].contiguous()
        update = _ns_reference(padded, group)
        errors[str(count)] = _relative_error(actual, update[:local_ns.shape[0]])
    return errors


class VerificationTrainer(BenchmarkTrainer):
    """Compare real updates and momentum without changing the native algorithm."""

    def __init__(self, config: TrainerConfig) -> None:
        """Retain independent local momentum across several optimizer steps."""
        super().__init__(config)
        self.reference_momentum: dict[str, torch.Tensor] = {}
        self.parameter_ids = None
        self.verification_path = self.metrics_path.with_name(f"verification_rank{self.base.global_rank}.jsonl")
        self.verification_path.write_text("", encoding="utf-8")

    def _capture_reference(self, name: str, muon: Any, named: dict) -> dict:
        model_param = named[name]
        param = self.base.optimizer.optimizer_param_by_model_param[model_param]
        group = next(group for group in muon.param_groups if any(p is param for p in group["params"]))
        grad = model_param.grad.detach().to(dtype=param.dtype)
        momentum = self.reference_momentum.setdefault(name, torch.zeros_like(grad))
        direction = torch._foreach_add([grad], [momentum], alpha=group["momentum"][0])
        torch._foreach_mul_([momentum], group["momentum"][1])
        torch._foreach_add_([momentum], [grad])
        if group["nesterov"]:
            torch._foreach_mul_(direction, group["momentum"][0])
            torch._foreach_add_(direction, [grad])
        full, stages = _gather_matrix(direction[0].to(torch.bfloat16), param.layout)
        update = _ns_reference(full, group)
        for dim, rank, size in reversed(stages):
            update = update.narrow(dim, rank * size, size)
        return {"name": name, "model_param": model_param, "param": param, "group": group,
                "input": full, "expected": update.contiguous(),
                "before": to_local_if_dtensor(param).detach().clone()}

    def _check_parameter(self, item: dict, muon: Any, shapes: list) -> dict:
        name, param, group = item["name"], item["param"], item["group"]
        actual = (item["before"] - to_local_if_dtensor(param)) / group["lr"]
        error = _relative_error(actual, item["expected"])
        state = muon.state.get(param, {}).get("momentum_buffer")
        momentum_error = (_relative_error(to_local_if_dtensor(state), self.reference_momentum[name])
                          if state is not None else None)
        model_param = item["model_param"]
        copy_error = float((model_param.float() -
                            to_local_if_dtensor(param).to(model_param.dtype).float()).abs().max().item())
        row = {"name": name, "update_relative_l2": error, "momentum_relative_l2": momentum_error,
               "model_copy_max_abs": copy_error, "global_shape": list(param.shape),
               "local_shape": list(to_local_if_dtensor(param).shape),
               "placements": [str(p) for p in param.placements]}
        threshold = 0.001 if "q_a_proj" in name else 0.02
        if "experts.down_proj" in name:
            batch_errors = _batch_errors(actual, item["input"], shapes, group)
            row["reference_batch_errors"] = batch_errors
            if batch_errors and min(batch_errors.values()) >= 0.001:
                raise AssertionError(f"{name}: matched-batch errors={batch_errors}, expected < 0.001")
        if error >= threshold:
            raise AssertionError(f"{name}: update relative L2={error}, expected < {threshold}")
        if momentum_error is not None and momentum_error >= 1e-6:
            raise AssertionError(f"{name}: momentum relative L2={momentum_error}, expected < 1e-6")
        if copy_error != 0.0:
            raise AssertionError(f"{name}: model/main difference={copy_error}, expected=0")
        return row

    @torch.no_grad()
    def timed_optimizer_step(self) -> None:
        """Check references before and after the unchanged native update."""
        muon = self.base.optimizer.optimizers_dict["muon"]
        named = dict(self.base.model.named_parameters())
        identities = tuple(id(p) for group in muon.param_groups for p in group["params"])
        if self.parameter_ids is not None and identities != self.parameter_ids:
            raise AssertionError(f"Optimizer identities changed: expected={self.parameter_ids}, got={identities}")
        self.parameter_ids = identities
        captured = [self._capture_reference(name, muon, named) for name in _NAMES]
        shapes = []
        native_ns = muon_module.zeropower_via_newtonschulz5

        def record_ns(tensor, *args, **kwargs):
            shapes.append(tuple(tensor.shape))
            return native_ns(tensor, *args, **kwargs)

        muon_module.zeropower_via_newtonschulz5 = record_ns
        try:
            super().timed_optimizer_step()
        finally:
            muon_module.zeropower_via_newtonschulz5 = native_ns
        rows = [self._check_parameter(item, muon, shapes) for item in captured]
        uncleared = [name for name, param in named.items() if param.grad is not None]
        if uncleared:
            raise AssertionError(f"Uncleared graph gradients={uncleared}, expected=[]")
        result = {"step": self.base.state.global_step + 1, "rank": self.base.global_rank,
                  "groups": [{"no_comm": len(local), "communication_groups": len(groups)}
                             for local, groups in muon._hsdp_grouping.values()],
                  "parameters": rows, "ns_batch_shapes": shapes}
        with self.verification_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result) + "\n")
        if self.base.global_rank == 0:
            print("MUON_VERIFIED", json.dumps(result), flush=True)


if __name__ == "__main__":
    VerificationTrainer(parse_training_args()).train()
