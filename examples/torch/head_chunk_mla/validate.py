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
"""Attention-only MLA parity on the current checkout.

The native mathematical reference is MLAAttention from master. Only its
attention-interface boundary adds Ulysses; no Trainer or model recipe is changed.
Random initialization is explicit and identical by parameter FQN and rank.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import traceback
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
import torch_npu
from torch import nn
from transformers import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Attention

import hyper_parallel
from hyper_parallel import SkipDTensorDispatch, fully_shard, init_device_mesh
from hyper_parallel.components.functional import npu_fusion_attention_forward
from hyper_parallel.components.modules.mla_attention import MLAAttention
from hyper_parallel.core.activation_memory import checkpoint_exclude_wrapper, checkpoint_wrapper
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.distributed.context_parallel.collectives import ulysses_head_to_seq, ulysses_seq_to_head
from hyper_parallel.distributed._builder.forward_rewriter import _commit_forward_rewrite
from hyper_parallel.models.deepseek_v3.adapter.distributed.context_parallel import mla_cp_wrapper

from hyper_parallel.components.modules import HeadChunkMLA


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
SHAPES = {
    "structure": {"hidden_size": 512, "num_attention_heads": 16, "q_lora_rank": 128, "kv_lora_rank": 64},
    "deepseek": {"hidden_size": 7168, "num_attention_heads": 128, "q_lora_rank": 1536, "kv_lora_rank": 512},
}
# Frozen before device parity. These are module-level BF16 tolerances, not a
# claim that a full pretrained model converges or that every element is equal.
BUDGET = {"output_rel_l2": 0.01, "dinput_rel_l2": 0.03, "parameter_grad_rel_l2": 0.05,
          "loss_relative": 0.01, "norm_relative": 0.03, "final_parameter_rel_l2": 0.01}


def tensor_hash(value: torch.Tensor) -> str:
    """Hash actual tensor bytes outside timed execution."""
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def local_tensor(value: torch.Tensor) -> torch.Tensor:
    """Expose a local FSDP shard for diagnostics and the explicit main optimizer."""
    return value.to_local() if hasattr(value, "to_local") else value


def metric(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    """Compute full-value errors rather than substituting summary fingerprints."""
    reference, candidate = reference.double(), candidate.double()
    delta = candidate - reference
    return {"finite": bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all()),
            "relative_l2": float(delta.norm() / reference.norm().clamp_min(1e-30)),
            "max_abs": float(delta.abs().max()), "elements": reference.numel()}


def native_core(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float) -> torch.Tensor:
    """Use native MLA FA at the selectively retained operator boundary.

    Args:
        query: Query in BNSD layout.
        key: Key in BNSD layout.
        value: Value in BNSD layout.
        scale: Native attention score scale.

    Returns:
        Causal attention output in BNSD layout.
    """
    output, _ = npu_fusion_attention_forward(SimpleNamespace(is_causal=True), query, key, value,
                                            None, scaling=scale)
    return output.transpose(1, 2)


SELECTIVE_CORE = checkpoint_exclude_wrapper(native_core)


def native_interface(mesh: Any, backend: str) -> Any:
    """Apply Ulysses after complete MLA Q/K/V composition, preserving RoPE heads."""
    core = SELECTIVE_CORE if backend == "selective" else native_core
    def attention(module: nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                  attention_mask: torch.Tensor | None, **kwargs: Any) -> tuple[torch.Tensor, None]:
        """Exchange complete QKV heads, evaluate FA and restore token ownership."""
        query, key, value = [ulysses_seq_to_head(x, 2, 1, mesh) for x in (query, key, value)]
        if backend == "native":
            output = npu_fusion_attention_forward(module, query, key, value, attention_mask, **kwargs)[0]
            output = output.transpose(1, 2)
        else:
            output = core(query, key, value, module.scaling)
        return ulysses_head_to_seq(output, 2, 1, mesh).transpose(1, 2).contiguous(), None
    return attention


def master_interface(module: nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                     attention_mask: torch.Tensor | None, **kwargs: Any) -> tuple[torch.Tensor, None]:
    """Keep CP/projections in current master; match only the native FA-v3 ABI."""
    if attention_mask is not None or kwargs.get("actual_seq_qlen") is not None:
        raise ValueError("This isolated experiment does not support packed inputs")
    output = SELECTIVE_CORE(query, key, value, module.scaling)
    return output.transpose(1, 2), None


def make_attention(args: argparse.Namespace, mesh: Any, index: int, variant: str) -> nn.Module:
    """Select the public candidate or the native MLA CP adapter, with the same FA ABI."""
    shape = SHAPES[args.shape]
    config = DeepseekV3Config(
        **shape, num_key_value_heads=shape["num_attention_heads"], qk_nope_head_dim=128,
        qk_rope_head_dim=64, v_head_dim=128, rope_interleave=False, attention_bias=False,
        attention_dropout=0.0, num_hidden_layers=args.layers,
    )
    source = DeepseekV3Attention(config, layer_idx=index)
    native = MLAAttention(module=source, attention_interface=native_interface(mesh, "plain"))
    if variant == "master":
        for request in mla_cp_wrapper(
            native, None, None, mesh, None, strategy="expanded_ulysses", overlap=True,
            attention_interface=master_interface,
        ):
            _commit_forward_rewrite(request)
        return native
    return HeadChunkMLA(module=native, head_chunk_size=args.chunk, cp_mesh=mesh,
                        offload_saved_stats=args.offload, prefetch_slots=args.slots,
                        overlap_backward=args.overlap)


class Block(nn.Module):
    """Attention-only stack; no MLP, token embedding or language-model objective."""

    def __init__(self, args: argparse.Namespace, mesh: Any, variant: str) -> None:
        """Compose the requested Attention modules without introducing MLP parameters."""
        super().__init__()
        self.layers = nn.ModuleList([
            checkpoint_exclude_wrapper(make_attention(args, mesh, index, variant))
            if variant != "master" else make_attention(args, mesh, index, variant)
            for index in range(args.layers)
        ])

    @property
    def attention(self) -> nn.Module:
        """Expose aggregate diagnostics without registering another module alias."""
        return self

    def saved_stats_offload_statistics(self) -> dict[str, int]:
        """Sum counters only; never retain invocation-local tensors."""
        result = {}
        for module in self.layers:
            function = getattr(module, "saved_stats_offload_statistics", None)
            for key, value in ({} if function is None else function()).items():
                result[key] = result.get(key, 0) + value
        return result

    def compressed_kv_statistics(self) -> dict[str, int]:
        """Sum communication counts across the Attention modules."""
        result = dict.fromkeys(("completed_backward", "completed_forward", "forward_allgather_calls",
                               "backward_allgather_calls", "backward_reduce_scatter_calls"), 0)
        for module in self.layers:
            function = getattr(module, "compressed_kv_statistics", None)
            for key, value in ({} if function is None else function()).items():
                result[key] = result.get(key, 0) + value
        return result

    def forward(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Compose two Attention residual branches when more than one is selected."""
        for module in self.layers:
            attention = module(hidden, position_embeddings=(cos, sin))[0]
            hidden = attention if len(self.layers) == 1 else hidden + 0.125 * attention
        return hidden


def build(args: argparse.Namespace, mesh: Any, rank: int, device: torch.device,
          variant: str) -> tuple[nn.Module, Block, dict[str, str]]:
    """Initialize identical FP32 local shards after applying the same FSDP policy."""
    with torch.device("meta"):
        block = Block(args, mesh, variant)
        model = checkpoint_wrapper(block, use_reentrant=False) if args.checkpoint else block
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                  output_dtype=None, cast_forward_inputs=False)
    fully_shard(model, mesh=mesh, reshard_after_forward=True, mp_policy=policy)
    model.to_empty(device=device)
    hashes = {}
    with torch.no_grad(), SkipDTensorDispatch():
        for name, parameter in model.named_parameters():
            local = local_tensor(parameter)
            if local.dtype != torch.float32:
                raise RuntimeError("Expected FP32 main storage")
            seed = int.from_bytes(hashlib.sha256(f"mla:20261029:{rank}:{name}".encode()).digest()[:4], "little")
            torch.manual_seed(seed)
            if "layernorm" in name:
                local.fill_(1)
            else:
                local.normal_(0, 0.02)
            hashes[name] = tensor_hash(local)
    return model, block, hashes


def inputs(args: argparse.Namespace, rank: int, world: int, device: torch.device,
           step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use aligned positions and deterministic actual local BF16 input bytes."""
    local_length = args.sequence // world
    generator = torch.Generator().manual_seed(2026102900 + rank + step * 17)
    hidden = torch.randn(1, local_length, SHAPES[args.shape]["hidden_size"],
                         generator=generator, dtype=torch.bfloat16).mul_(0.2)
    positions = torch.arange(rank * local_length, (rank + 1) * local_length, dtype=torch.float32)
    angles = positions[:, None] / (1e6 ** (torch.arange(0, 64, 2) / 64))[None, :]
    angles = torch.cat((angles, angles), dim=-1)[None]
    return hidden.to(device), angles.cos().bfloat16().to(device), angles.sin().bfloat16().to(device)


def source_manifest() -> dict[str, str]:
    """Hash the example, candidate and native dependencies before and after execution."""
    files = [Path(__file__)]
    files += list((REPO / "hyper_parallel/components/modules").glob("*head_chunk_mla*.py"))
    files += [Path(inspect.getfile(value)) for value in (MLAAttention, npu_fusion_attention_forward,
                                                       ulysses_seq_to_head, checkpoint_wrapper, mla_cp_wrapper)]
    files += [REPO / "hyper_parallel/distributed/context_parallel" / name
              for name in ("mla_context_parallel.py", "mla_backward_overlap.py")]
    files += [REPO / "hyper_parallel/core/activation_memory" / name
              for name in ("wrapper.py", "checkpoint_exclude.py", "pinned_memory_pool.py")]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def resource_statistics(block: Block) -> dict[str, Any]:
    """Require invocation-local offload storage to be closed after backward."""
    function = getattr(block.attention, "saved_stats_offload_statistics", None)
    statistics_ = {} if function is None else function()
    for name in ("active_stores", "live_host_bytes", "device_slots", "device_slot_bytes",
                 "live_d2h_source_tickets", "live_d2h_source_bytes"):
        if statistics_.get(name, 0):
            raise RuntimeError(f"MLA offload resource not released: {name}={statistics_[name]}")
    latent_stats = getattr(block.attention, "compressed_kv_statistics", None)
    if latent_stats is not None:
        counters = latent_stats()
        completed = counters["completed_backward"]
        if (counters["completed_forward"] != completed
                or counters["forward_allgather_calls"] != completed
                or counters["backward_allgather_calls"] != completed
                or counters["backward_reduce_scatter_calls"] != 2 * completed):
            raise RuntimeError("Latent publication/gradient reduction counts differ from the declared dataflow")
        statistics_["compressed_kv"] = counters
    return statistics_


def parity(args: argparse.Namespace, mesh: Any, rank: int, world: int,
           device: torch.device, result: dict[str, Any], path: Path) -> None:
    """Compare every output/input/parameter gradient and continuous AdamW updates."""
    reference_args = argparse.Namespace(**vars(args))
    reference_args.chunk = args.reference_chunk or args.chunk
    built = [build(reference_args, mesh, rank, device, args.reference),
             build(args, mesh, rank, device, args.variant)]
    if built[0][2] != built[1][2]:
        raise RuntimeError("Initial parameters/FQNs differ")
    result["initial_parameter_hashes"] = built[0][2]
    mains, optimizers = [], []
    with SkipDTensorDispatch():
        for model, _, _ in built:
            aliases = {name: nn.Parameter(local_tensor(value).detach()) for name, value in model.named_parameters()}
            mains.append(aliases)
            optimizers.append(torch.optim.AdamW(aliases.values(), lr=1e-4, weight_decay=0.01))
    all_pass = True
    for step in range(args.steps):
        batches = [inputs(args, rank, world, device, step * args.micro_batches + micro)
                   for micro in range(args.micro_batches)]
        values = batches[0]
        states = []
        for (model, block, _), main_parameters, optimizer in zip(built, mains, optimizers):
            with SkipDTensorDispatch():
                model.zero_grad()
            outputs, input_gradients, losses = [], [], []
            for values in batches:
                hidden = values[0].detach().clone().requires_grad_()
                output = model(hidden, values[1], values[2])
                loss = output.float().square().mean() / args.micro_batches
                loss.backward()
                outputs.append(output.detach().float().cpu())
                input_gradients.append(hidden.grad.detach().float().cpu())
                losses.append(float(loss.detach()))
                resource_statistics(block)
            with SkipDTensorDispatch():
                norm_squared = torch.zeros((), dtype=torch.float32, device=device)
                gradients = {}
                for name, parameter in model.named_parameters():
                    local_grad = local_tensor(parameter.grad)
                    if (local_grad.dtype != torch.float32
                            or local_tensor(parameter).data_ptr() != main_parameters[name].data_ptr()):
                        raise RuntimeError("FSDP main-parameter/gradient contract changed")
                    main_parameters[name].grad = local_grad
                    norm_squared.add_(local_grad.square().sum())
                    gradients[name] = local_grad.detach().float().cpu()
            dist.all_reduce(norm_squared)
            states.append({"output": torch.cat(outputs), "dinput": torch.cat(input_gradients),
                           "gradients": gradients, "loss": sum(losses), "norm": float(norm_squared.sqrt()),
                           "offload_statistics": resource_statistics(block)})
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            del output, loss, hidden
        left, right = states[0], states[1]
        row = {"step": step + 1, "input_hashes": [tensor_hash(batch[0]) for batch in batches], "learning_rate": 1e-4,
               "output": metric(left["output"], right["output"]),
               "dinput": metric(left["dinput"], right["dinput"]),
               "gradients": {name: metric(value, right["gradients"][name])
                             for name, value in left["gradients"].items()},
               "reference_loss": left["loss"], "candidate_loss": right["loss"],
               "reference_norm": left["norm"], "candidate_norm": right["norm"],
               "reference_offload": left["offload_statistics"], "candidate_offload": right["offload_statistics"]}
        for name in ("loss", "norm"):
            row[name + "_relative"] = abs(left[name] - right[name]) / max(abs(left[name]), 1e-30)
        row["pass"] = (row["output"]["finite"] and row["dinput"]["finite"]
                       and row["output"]["relative_l2"] <= BUDGET["output_rel_l2"]
                       and row["dinput"]["relative_l2"] <= BUDGET["dinput_rel_l2"]
                       and row["loss_relative"] <= BUDGET["loss_relative"]
                       and row["norm_relative"] <= BUDGET["norm_relative"]
                       and all(v["finite"] and v["relative_l2"] <= BUDGET["parameter_grad_rel_l2"]
                               for v in row["gradients"].values()))
        if args.exact:
            row["exact"] = all(item["max_abs"] == 0 for item in
                               (row["output"], row["dinput"], *row["gradients"].values()))
            row["pass"] &= row["exact"] and row["loss_relative"] == 0 and row["norm_relative"] == 0
        all_pass &= row["pass"]
        result["rows"].append(row)
        path.write_text(json.dumps(result, indent=2))
    result["final_parameters"] = {name: metric(value.detach().float().cpu(), mains[1][name].detach().float().cpu())
                                  for name, value in mains[0].items()}
    all_pass &= all(v["finite"] and v["relative_l2"] <= BUDGET["final_parameter_rel_l2"]
                    for v in result["final_parameters"].values())
    if args.exact:
        all_pass &= all(v["max_abs"] == 0 for v in result["final_parameters"].values())
    flag = torch.tensor(int(all_pass), device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    result["status"] = "PASS_MODULE_10STEP" if bool(flag) and args.steps >= 10 else (
        "PASS_MODULE_SMOKE" if bool(flag) else "PRECISION_FAIL")


def main() -> None:
    """Launch a declared topology without implicitly downloading model assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", choices=tuple(SHAPES), default="structure")
    parser.add_argument("--sequence", type=int, default=1024)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--reference-chunk", type=int)
    parser.add_argument("--reference", choices=("master", "chunk"), default="master")
    parser.add_argument("--variant", choices=("chunk",), default="chunk")
    parser.add_argument("--layers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--micro-batches", type=int, choices=(1, 2), default=1)
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--overlap", action="store_true")
    parser.add_argument("--slots", type=int, choices=(1, 2), default=2)
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if args.sequence % world or not 0 < args.sequence // world <= 16384:
        raise ValueError("Require integral local sequence <=16K")
    if args.chunk % world or not 0 < args.chunk <= SHAPES[args.shape]["num_attention_heads"]:
        raise ValueError("Require a positive CP-divisible chunk no larger than the head count")
    if args.reference_chunk is not None and (
            args.reference_chunk % world or SHAPES[args.shape]["num_attention_heads"] % args.reference_chunk):
        raise ValueError("Invalid reference chunk")
    if Path(hyper_parallel.__file__).resolve().parent.parent != REPO:
        raise RuntimeError(f"Wrong editable checkout: {hyper_parallel.__file__}")
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"rank{rank}.json"
    if path.exists():
        raise FileExistsError(path)
    result = {"status": "STARTING", "scope": "random_attention_only", "rank": rank, "cp_fsdp": world,
              "environment": {"torch": torch.__version__, "torch_npu": torch_npu.__version__,
                              "checkout": str(REPO)},
              "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "precision_budget": BUDGET, "source_hashes": source_manifest(), "rows": []}
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
        device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
        if os.environ.get("HCCL_DETERMINISTIC", "").lower() != "true":
            raise RuntimeError("Set HCCL_DETERMINISTIC=true before launching precision validation")
        dist.init_process_group("hccl")
        mesh = init_device_mesh("npu", (world,), mesh_dim_names=("cp_fsdp",))
        parity(args, mesh, rank, world, device, result, path)
        result["source_hashes_at_finish"] = source_manifest()
        if result["source_hashes"] != result["source_hashes_at_finish"]:
            raise RuntimeError("Sources changed during execution")
    except BaseException:
        result["status"], result["error"] = "FAILED", traceback.format_exc()
        raise
    finally:
        path.write_text(json.dumps(result, indent=2))
        if dist.is_initialized():
            dist.destroy_process_group()
    print(json.dumps({"rank": rank, "status": result["status"]}), flush=True)
    if result["status"] == "PRECISION_FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
