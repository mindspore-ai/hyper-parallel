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
"""Compare a real DSV4.1 MoE block with MegaMoe using torchrun on Ascend NPUs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from types import MethodType
from unittest.mock import patch
from typing import Any

import torch
import torch.distributed as dist
import torch_npu
import transformers
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4SparseMoeBlock

from hyper_parallel.core.multicore._loader import get_multicore_paths
from hyper_parallel.models.deepseek_v41.adapter.expert_parallel import deepseek_v41_ep_compute_fn
from hyper_parallel.core.multicore.examples.mega_moe.deepseek_v41_accuracy import fp32_accuracy
from hyper_parallel.core.multicore.examples.mega_moe.deepseek_v41_oracle import evaluate_fp32_moe
from hyper_parallel.models.deepseek_v41.adapter.activation import configure_deepseek_v41_swiglu
from hyper_parallel.models.deepseek_v41.adapter.megamoe import DeepseekV41MegaMoe
from hyper_parallel.models.deepseek_v41.modeling_deepseek_v41 import (
    DeepseekV41TopKRouter,
    _v41_sparse_moe_forward,
)


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--intermediate-size", type=int, default=128)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--ep-size", type=int, default=None,
                        help="EP group size; smaller than WORLD creates strided disjoint subgroups")
    parser.add_argument("--dispatch-mode", choices=("push", "pull"), default="push")
    parser.add_argument("--reference", choices=("owner_ep", "hf_replicated"), default="owner_ep")
    parser.add_argument("--route", choices=("learned", "hotspot"), default="learned")
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--acceptance", choices=("fp32", "hf_bf16"), default="fp32",
                        help="FP32 acceptance synchronizes step weights; hf_bf16 retains the legacy gate")
    parser.add_argument("--fp32-oracle", action="store_true",
                        help="Record independent CPU FP32 comparisons, including with the legacy HF BF16 gate")
    parser.add_argument(
        "--synchronize-step-weights", action="store_true",
        help="Copy the HF path's current weights before each step, after validating the previous update",
    )
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.acceptance == "fp32":
        args.fp32_oracle = True
        args.synchronize_step_weights = True
    for name in ("tokens", "hidden_size", "intermediate_size", "num_experts", "top_k", "steps"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.tokens % 128 or args.top_k > args.num_experts:
        parser.error("tokens must be divisible by 128 and top-k cannot exceed num-experts")
    if any(not math.isfinite(value) or value < 0 for value in (args.rtol, args.atol)):
        parser.error("rtol and atol must be finite and non-negative")
    if args.ep_size is not None and args.ep_size <= 0:
        parser.error("ep-size must be positive")
    return args


def _source_block(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    config = DeepseekV4Config(  # pylint: disable=unexpected-keyword-arg
        hidden_size=args.hidden_size, moe_intermediate_size=args.intermediate_size,
        n_routed_experts=args.num_experts, n_shared_experts=1, num_experts_per_tok=args.top_k,
        num_hidden_layers=1, mlp_layer_types=["moe"], swiglu_limit=0.0,
        scoring_func="sqrtsoftplus", routed_scaling_factor=1.7,
    )
    config.v41_vision_enabled = args.vision
    source = DeepseekV4SparseMoeBlock(config, 0)
    source.gate = DeepseekV41TopKRouter(config)
    source.forward = MethodType(_v41_sparse_moe_forward, source)
    generator = torch.Generator().manual_seed(2026)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.normal_(std=0.02, generator=generator)
    configure_deepseek_v41_swiglu(source)
    return source.to(device=device, dtype=torch.bfloat16)


class _GroupMesh:
    """Expose the existing group through the EP factory's mesh contract."""

    def __init__(self, group: Any) -> None:
        """Keep an already-created EP process group."""
        self.group = group

    def __getitem__(self, _axis: str) -> _GroupMesh:
        """Expose the single EP axis."""
        return self

    def size(self) -> int:
        """Return the group-local EP degree."""
        return dist.get_world_size(self.group)

    def get_group(self, _axis: str) -> Any:
        """Return the explicit EP collective group.

        Args:
            _axis: EP axis name supplied by the factory.
        """
        return self.group


def _configure_owner_reference(source, ep_group):
    compute = deepseek_v41_ep_compute_fn(
        module=source,
        mesh=None,
        tp_mesh=None,
        cp_mesh=None,
        ep_mesh=_GroupMesh(ep_group),
        use_grouped_gemm=True,
    )
    count = source.experts.num_experts // dist.get_world_size(ep_group)
    start = dist.get_rank(ep_group) * count
    for name, parameter in tuple(source.experts.named_parameters()):
        setattr(source.experts, name, torch.nn.Parameter(parameter[start:start + count].detach().clone(),
                                                       requires_grad=parameter.requires_grad))
    source.forward = MethodType(compute, source)
    source._dsv41_owner_ep = True


def _mapped_reference(source, name, parameter, ep_rank, local_experts):
    if getattr(source, "_dsv41_owner_ep", False) and name.startswith("experts."):
        return None if parameter is None else parameter.transpose(1, 2)
    return _reference_parameter(name, parameter, ep_rank, local_experts)


def _full_source_parameters(source, ep_group):
    if not getattr(source, "_dsv41_owner_ep", False):
        return dict(source.named_parameters())
    parameters = {}
    for name, parameter in source.named_parameters():
        if name.startswith("experts."):
            shards = [torch.empty_like(parameter) for _ in range(dist.get_world_size(ep_group))]
            dist.all_gather(shards, parameter, group=ep_group)
            parameter = torch.cat(shards)
        parameters[name] = parameter
    return parameters


def _compare(name: str, actual: torch.Tensor | None, expected: torch.Tensor | None,
             args: argparse.Namespace) -> dict[str, Any]:
    if actual is None or expected is None:
        return {"name": name, "passed": actual is None and expected is None, "gradient_absent": True}
    if actual.shape != expected.shape:
        return {"name": name, "passed": False, "shape_matches": False}
    actual = actual.detach().float()
    expected = expected.detach().float()
    difference = (actual - expected).abs()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    tolerance = args.atol + args.rtol * expected.abs()
    violation = difference > tolerance
    worst = int((difference / tolerance.clamp_min(1e-12)).reshape(-1).argmax())
    return {
        "name": name, "finite": finite, "shape_matches": True,
        "passed": finite and torch.allclose(actual, expected, rtol=args.rtol, atol=args.atol),
        "max_abs_error": float(difference.max()), "reference_max_abs": float(expected.abs().max()),
        "relative_l2": float(difference.norm() / expected.norm().clamp_min(1e-12)),
        "elements_outside_tolerance": int(violation.sum()), "elements": actual.numel(),
        "worst_actual": float(actual.reshape(-1)[worst]),
        "worst_expected": float(expected.reshape(-1)[worst]),
        "worst_tolerance_ratio": float((difference / tolerance.clamp_min(1e-12)).max()),
    }


def _compare_fp32(name, actual, expected, args):
    comparison = _compare(name, actual, expected, args)
    comparison["elementwise_passed"] = comparison["passed"]
    comparison.update(fp32_accuracy(actual, expected))
    return comparison


def _reference_parameter(name: str, parameter: torch.Tensor | None, rank: int, local_experts: int):
    if parameter is None or not name.startswith("experts."):
        return parameter
    return parameter[rank * local_experts:(rank + 1) * local_experts].transpose(1, 2)


def _parameter_comparisons(source, candidate, args, ep_rank, *, gradients: bool):
    reference_parameters = dict(source.named_parameters())
    comparisons = []
    for name, parameter in candidate.named_parameters():
        source_name = name.replace("gate_up_weight", "gate_up_proj").replace("down_weight", "down_proj")
        expected = reference_parameters[source_name]
        expected = expected.grad if gradients else expected
        expected = _mapped_reference(source, name, expected, ep_rank, candidate.experts.local_experts)
        actual = parameter.grad if gradients else parameter
        comparisons.append(_compare(f"{'grad' if gradients else 'weight'}/{name}", actual, expected, args))
    return comparisons


def _capture_route(storage):
    def _capture(_module, _inputs, output):
        _, weights, indices = output
        weights.retain_grad()
        storage.update(weights=weights, indices=indices)
    return _capture


def _forward_pair(source, candidate, inputs, other_inputs, args, image_mask):
    if args.route == "hotspot":
        indices = torch.arange(args.top_k, device=inputs.device).expand(args.tokens, -1).contiguous()
        weights = torch.full((args.tokens, args.top_k), 1.7 / args.top_k,
                             device=inputs.device, dtype=torch.float32, requires_grad=True)
        other_weights = weights.detach().clone().requires_grad_()
        if getattr(source, "_dsv41_owner_ep", False):
            with patch.object(source.gate, "forward", return_value=(None, weights, indices)):
                expected = source(inputs)
        else:
            expected = source.experts(inputs.reshape(-1, args.hidden_size), indices, weights).reshape_as(inputs)
            expected = expected + source.shared_experts(inputs)
        actual = candidate.experts(other_inputs, indices, other_weights) + candidate.shared_experts(other_inputs)
        return (expected, actual, {"weights": weights, "indices": indices},
                {"weights": other_weights, "indices": indices})
    expected_route, actual_route = {}, {}
    source_hook = source.gate.register_forward_hook(_capture_route(expected_route))
    candidate_hook = candidate.gate.register_forward_hook(_capture_route(actual_route))
    try:
        expected = source(inputs, image_mask=image_mask)
        actual = candidate(other_inputs, image_mask=image_mask)
    finally:
        source_hook.remove()
        candidate_hook.remove()
    return expected, actual, expected_route, actual_route


def _full_native_parameters(candidate, ep_group):
    parameters = {}
    with torch.no_grad():
        for name, parameter in candidate.named_parameters():
            source_name = name.replace("gate_up_weight", "gate_up_proj").replace("down_weight", "down_proj")
            if name.startswith("experts."):
                shards = [torch.empty_like(parameter) for _ in range(dist.get_world_size(ep_group))]
                dist.all_gather(shards, parameter, group=ep_group)
                parameters[source_name] = torch.cat(shards).transpose(1, 2)
            else:
                parameters[source_name] = parameter
    return parameters


def _fp32_evidence(module, parameters, inputs, output, dy, route, args, ep_group):
    oracle = evaluate_fp32_moe(
        inputs, dy, parameters, route["indices"], route["weights"],
        learned_routing=args.route == "learned", scaling_factor=1.7,
    )
    # Sum the FP32 per-rank oracle before comparing local expert dW. Reducing
    # already rounded BF16 gradients would add a second reference error.
    for name in ("experts.gate_up_proj", "experts.down_proj"):
        gradient = oracle["gradients"][name].to(device=inputs.device)
        dist.all_reduce(gradient, group=ep_group)
        oracle["gradients"][name] = gradient.cpu()
    checks = [
        _compare_fp32("output", output.cpu(), oracle["output"], args),
        _compare_fp32("input_grad", inputs.grad.cpu(), oracle["input_grad"], args),
        _compare_fp32("route_weights", route["weights"].cpu(), oracle["route_weights"], args),
        _compare_fp32("route_weight_grad", route["weights"].grad.cpu(), oracle["route_weight_grad"], args),
    ]
    checks.extend(_fp32_parameter_checks(module, oracle, args, ep_group, gradients=True))
    return oracle, checks


def _fp32_parameter_checks(module, oracle, args, ep_group, *, gradients):
    checks = []
    for name, parameter in module.named_parameters():
        source_name = name.replace("gate_up_weight", "gate_up_proj").replace("down_weight", "down_proj")
        expected = oracle["gradients"][source_name]
        if not gradients:
            before = oracle["parameters"][source_name]
            expected = before if expected is None else before - 0.01 * expected
        if getattr(module, "_dsv41_owner_ep", False) and name.startswith("experts."):
            count = module.experts.local_expert_count
            start = dist.get_rank(ep_group) * count
            expected = None if expected is None else expected[start:start + count]
        if source_name != name:
            expected = _reference_parameter(name, expected, dist.get_rank(ep_group), module.experts.local_experts)
        actual = parameter.grad if gradients else parameter
        checks.append(_compare_fp32(f"{'grad' if gradients else 'weight'}/{name}",
                               None if actual is None else actual.cpu(), expected, args))
    return checks


def _synchronize_parameters(source, candidate, ep_rank):
    reference = dict(source.named_parameters())
    with torch.no_grad():
        for name, parameter in candidate.named_parameters():
            source_name = name.replace("gate_up_weight", "gate_up_proj").replace("down_weight", "down_proj")
            expected = _mapped_reference(source, name, reference[source_name], ep_rank, candidate.experts.local_experts)
            parameter.copy_(expected)


def _initial_parameter_checks(source, candidate, ep_rank):
    reference = dict(source.named_parameters())
    checks = []
    for name, parameter in candidate.named_parameters():
        source_name = name.replace("gate_up_weight", "gate_up_proj").replace("down_weight", "down_proj")
        expected = _mapped_reference(source, name, reference[source_name], ep_rank, candidate.experts.local_experts)
        checks.append({"name": f"initial_weight/{name}", "passed": torch.equal(parameter, expected)})
    return checks


def _run_steps(source, candidate, args, device, ep_group):
    ep_rank = dist.get_rank(ep_group)
    source_optimizer = torch.optim.SGD(source.parameters(), lr=0.01)
    candidate_optimizer = torch.optim.SGD(candidate.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(4000 + dist.get_rank())
    image_mask = torch.arange(args.tokens, device=device).reshape(1, -1).remainder(3) == 0 if args.vision else None
    results = []
    for step in range(args.steps):
        if step > 0 and args.synchronize_step_weights:
            _synchronize_parameters(source, candidate, ep_rank)
        initial_checks = (_initial_parameter_checks(source, candidate, ep_rank)
                          if step == 0 or args.synchronize_step_weights else [])
        inputs = torch.randn(1, args.tokens, args.hidden_size, generator=generator).to(device, torch.bfloat16)
        inputs.requires_grad_()
        other_inputs = inputs.detach().clone().requires_grad_()
        output_grad = torch.randn(inputs.shape, generator=generator).to(device) / math.sqrt(args.tokens)
        source_optimizer.zero_grad(set_to_none=True)
        candidate_optimizer.zero_grad(set_to_none=True)
        expected, actual, expected_route, actual_route = _forward_pair(
            source, candidate, inputs, other_inputs, args, image_mask
        )
        comparisons = [*initial_checks, _compare("output", actual, expected, args)]
        comparisons.append({"name": "route/indices", "passed": torch.equal(
            actual_route["indices"], expected_route["indices"])})
        (expected.float() * output_grad).sum().backward()
        (actual.float() * output_grad).sum().backward()
        # The full-expert oracle saw only this rank's tokens. Native local dW
        # includes tokens dispatched from every rank, so sum before slicing.
        if not getattr(source, "_dsv41_owner_ep", False):
            for parameter in source.experts.parameters():
                dist.all_reduce(parameter.grad, group=ep_group)
        comparisons.extend((
            _compare("input_grad", other_inputs.grad, inputs.grad, args),
            _compare("route_weight_grad", actual_route["weights"].grad, expected_route["weights"].grad, args),
        ))
        comparisons.extend(_parameter_comparisons(source, candidate, args, ep_rank, gradients=True))
        fp32_checks = {}
        if args.fp32_oracle:
            source_oracle, fp32_checks["hf_bf16"] = _fp32_evidence(
                source, _full_source_parameters(source, ep_group), inputs, expected, output_grad,
                expected_route, args, ep_group,
            )
            native_oracle, fp32_checks["megamoe"] = _fp32_evidence(
                candidate, _full_native_parameters(candidate, ep_group), other_inputs, actual, output_grad,
                actual_route, args, ep_group,
            )
        source_optimizer.step()
        candidate_optimizer.step()
        comparisons.extend(_parameter_comparisons(source, candidate, args, ep_rank, gradients=False))
        if args.fp32_oracle:
            fp32_checks["hf_bf16"].extend(_fp32_parameter_checks(
                source, source_oracle, args, ep_group, gradients=False))
            fp32_checks["megamoe"].extend(_fp32_parameter_checks(
                candidate, native_oracle, args, ep_group, gradients=False))
        results.append({"step": step, "checks": comparisons, "fp32_checks": fp32_checks})
    return _rank_result(results, args, ep_group)


def _requires_exact_check(check):
    return (check["name"].startswith("initial_weight/") or check["name"] == "route/indices"
            or check.get("gradient_absent") or check.get("shape_matches") is False)


def _rank_result(results, args, ep_group):
    legacy_passed = all(check["passed"] for result in results for check in result["checks"])
    structural_passed = all(check["passed"] for result in results for check in result["checks"]
                            if _requires_exact_check(check))
    fp32_passed = args.fp32_oracle and all(
        check["passed"] for result in results for checks in result["fp32_checks"].values() for check in checks
    )
    return {"rank": dist.get_rank(), "ep_rank": dist.get_rank(ep_group),
            "ep_ranks": dist.get_process_group_ranks(ep_group),
            "steps": results,
            "hf_bf16_passed": legacy_passed, "structural_passed": structural_passed,
            "passed": structural_passed and fp32_passed if args.acceptance == "fp32" else legacy_passed}


def _ep_group(ep_size):
    world_size = dist.get_world_size()
    if world_size % ep_size:
        raise ValueError("WORLD size must be divisible by ep-size")
    if ep_size == world_size:
        return dist.group.WORLD
    stride = world_size // ep_size
    selected = None
    for offset in range(stride):
        ranks = list(range(offset, world_size, stride))
        group = dist.new_group(ranks)
        if dist.get_rank() in ranks:
            selected = group
    return selected


def _environment() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[5]
    vendor, adapter = get_multicore_paths()
    paths = [adapter, *vendor.rglob("*.so"), *vendor.rglob("*.o")]
    source_paths = [*Path(__file__).parent.glob("deepseek_v41*.py"),
                    *root.glob("hyper_parallel/models/deepseek_v41/**/*.py")]
    return {
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=root)).hexdigest(),
        "source_files_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in source_paths},
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "transformers": transformers.__version__, "transformers_path": transformers.__file__,
        "cann": os.environ.get("ASCEND_HOME_PATH"), "opp": os.environ.get("ASCEND_OPP_PATH"),
        "chip": torch.npu.get_device_name(),
        "native_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    }


def _fp32_passes(results):
    return {backend: all(check["passed"] for rank in results for step in rank["steps"]
                         for check in step["fp32_checks"][backend])
            for backend in ("hf_bf16", "megamoe")}


def main(argv: list[str] | None = None) -> None:
    """Compare per-rank evidence and fail the torchrun job on any mismatch.

    Args:
        argv: Explicit worker arguments, or None to parse the process command line.
    """
    args = _arguments(argv)
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("hccl")
    args.ep_size = args.ep_size or dist.get_world_size()
    if args.num_experts % args.ep_size:
        raise ValueError("num-experts must be divisible by ep-size")
    ep_group = _ep_group(args.ep_size)
    device = torch.device("npu", torch.npu.current_device())
    source = _source_block(args, device)
    candidate = DeepseekV41MegaMoe(
        copy.deepcopy(source), local_num_tokens=args.tokens,
        ep_group=ep_group, dispatch_mode=args.dispatch_mode,
    )
    if args.reference == "owner_ep":
        _configure_owner_reference(source, ep_group)
    try:
        result = _run_steps(source, candidate, args, device, ep_group)
        results = [None] * dist.get_world_size()
        dist.all_gather_object(results, result)
        passed = all(item["passed"] for item in results)
        if dist.get_rank() == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            report = {"passed": passed, "swiglu_limit": 0.0, "ep_size": args.ep_size,
                      "owner_use_grouped_gemm": args.reference == "owner_ep",
                      "world_size": dist.get_world_size(),
                      "config": {key: str(value) if isinstance(value, Path) else value
                                 for key, value in vars(args).items()},
                      "environment": _environment(), "ranks": results}
            report["hf_bf16_passed"] = all(item["hf_bf16_passed"] for item in results)
            report["step_state"] = ("identical weights before each step; previous optimizer updates checked first"
                                    if args.synchronize_step_weights else "independent optimizer trajectories")
            if args.fp32_oracle:
                report["fp32_passed"] = _fp32_passes(results)
                report["fp32_contract"] = (
                    "CPU FP32 math with fixed selected IDs and each path's current weights; "
                    "expert gradients reduced in FP32 within the actual EP group; "
                    "per-step SGD update checked from that same state; "
                    "each tensor must be finite with matching shape/presence, "
                    "relative L2 <= 1% and peak-normalized error <= 2%; zero reference requires exact zero"
                )
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"DeepSeek-V4.1 MegaMoe precision passed={passed}: {args.output}", flush=True)
        if not passed:
            raise AssertionError(f"DeepSeek-V4.1 MegaMoe precision mismatch; inspect {args.output}")
    finally:
        candidate.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
