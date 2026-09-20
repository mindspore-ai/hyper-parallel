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
"""Hugging Face checkpoints converted and loaded by HFLoadPlanner, checked against the legacy loader."""
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch import nn

from hyper_parallel import DTensor
from hyper_parallel.components.checkpoint.conversion_ops import AddScalar, InterleaveQKV
from hyper_parallel.components.checkpoint.weight_conversion import ConversionOps, WeightConverter
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.components.checkpoint.huggingface_checkpointer import (
    HuggingFaceCheckpointer,
)
from hyper_parallel.components.checkpoint.huggingface_load_planner import (
    HFLoadPlanner,
    load_hf_checkpoint,
)
from tests.torch.utils import _DEVICE_TYPE, init_backend, to_device

# Model tensor -> (placements over the (dp, tp) = (2, 2) mesh, local shape on every rank).
_TARGETS = {
    "embed.weight": ((Shard(0), Replicate()), (4, 4)),
    "layers.0.linear_qkv.weight": ((Replicate(), Shard(0)), (8, 4)),
    "layers.0.norm.weight": ((Replicate(), Replicate()), (4,)),
    "layers.0.experts.gate_up_proj": ((Shard(0), Replicate()), (1, 6, 4)),
    "layers.0.scaled.weight": ((Replicate(), Shard(0)), (2,)),
}


class _StackThenConcatenate(ConversionOps):
    """Stacks the experts of every source pattern and concatenates the stacks, as MergeModulelist and Concatenate do."""

    def convert(self, input_dict: dict[str, Any], source_patterns: list[str], target_patterns: list[str],
                **kwargs: Any) -> dict[str, torch.Tensor]:
        """Stack each pattern's experts along a new first dimension, then concatenate along dimension 1."""
        del kwargs
        stacks = [torch.stack(input_dict[pattern], dim=0) for pattern in source_patterns]
        return {target_patterns[0]: torch.cat(stacks, dim=1)}


class _Product(ConversionOps):
    """Multiplies two checkpoint tensors elementwise, which the planner can only run on real tensors."""

    def convert(self, input_dict: dict[str, Any], source_patterns: list[str], target_patterns: list[str],
                **kwargs: Any) -> dict[str, torch.Tensor]:
        """Multiply the tensors of the two source patterns."""
        del kwargs
        first, second = (input_dict[pattern][0] for pattern in source_patterns)
        return {target_patterns[0]: first * second}


def _mapping() -> list[Any]:
    """Rules grouping Q, K and V, shifting a norm, merging experts, and multiplying two tensors."""
    return [
        WeightConverter(source_patterns=["q_proj.weight", "k_proj.weight", "v_proj.weight"],
                        target_patterns="linear_qkv.weight",
                        operations=[InterleaveQKV(2, 2, 2, 2, source_is_fused=False)]),
        WeightConverter(source_patterns="norm.weight", target_patterns="norm.weight", operations=[AddScalar(1.0)]),
        WeightConverter(source_patterns=["experts.*.gate_proj.weight", "experts.*.up_proj.weight"],
                        target_patterns="experts.gate_up_proj", operations=[_StackThenConcatenate()]),
        WeightConverter(source_patterns=["scaled.weight", "scale.weight"], target_patterns="scaled.weight",
                        operations=[_Product()]),
    ]


def _checkpoint_tensors() -> dict[str, torch.Tensor]:
    """The tensors of the checkpoint, the same on every rank."""
    generator = torch.Generator().manual_seed(0)

    def randn(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.randn(*shape, generator=generator).to(dtype)

    tensors = {
        "embed.weight": randn(8, 4, dtype=torch.bfloat16),
        "layers.0.q_proj.weight": randn(8, 4),
        "layers.0.k_proj.weight": randn(4, 4),
        "layers.0.v_proj.weight": randn(4, 4),
        "layers.0.norm.weight": randn(4, dtype=torch.bfloat16),
        "layers.0.scaled.weight": randn(4),
        "layers.0.scale.weight": randn(4),
    }
    for expert in range(2):
        tensors[f"layers.0.experts.{expert}.gate_proj.weight"] = randn(3, 4)
        tensors[f"layers.0.experts.{expert}.up_proj.weight"] = randn(3, 4)
    return tensors


def _save_pretrained_style(checkpoint_dir: Path, tensors: dict[str, torch.Tensor]) -> None:
    """Write ``tensors`` into two safetensors files and an index, the way ``save_pretrained`` shards a model."""
    names = sorted(tensors)
    weight_map = {}
    for part, keys in enumerate((names[::2], names[1::2]), start=1):
        file_name = f"model-0000{part}-of-00002.safetensors"
        save_file({key: tensors[key] for key in keys}, str(checkpoint_dir / file_name), metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(keys, file_name))
    index = {"metadata": {"total_size": 0}, "weight_map": weight_map}
    (checkpoint_dir / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")


def _make_shared_temp_dir(prefix: str) -> Path:
    """Create a temporary directory on rank 0 and tell every rank where it is."""
    path_holder = [tempfile.mkdtemp(prefix=prefix) if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(path_holder, src=0)
    return Path(path_holder[0])


def _build_model(mesh: Any) -> nn.Module:
    """A model holding every target as a zero DTensor buffer laid out over ``mesh``."""
    model = nn.Module()
    for name, (placements, local_shape) in _TARGETS.items():
        *path, leaf = name.split(".")
        module = model
        for part in path:
            if part not in module._modules:  # pylint: disable=protected-access
                module.add_module(part, nn.Module())
            module = module._modules[part]  # pylint: disable=protected-access
        local = to_device(torch.zeros(*local_shape), _DEVICE_TYPE)
        module.register_buffer(leaf, DTensor.from_local(local, mesh, placements))
    return model


def _full_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
    """Every buffer of ``model`` gathered whole onto the host, collectively on every rank."""
    return {name: buffer.full_tensor().cpu() for name, buffer in model.named_buffers()}


def test_dcp_hf_planner_matches_legacy_loader() -> None:
    """
    Feature: Test HFLoadPlanner loads of a converted Hugging Face checkpoint against the legacy loader.
    Description:
        1. Rank 0 writes a checkpoint into two safetensors files and an index: Q, K and V projections,
           a bfloat16 norm, per-expert gate and up projections, and two tensors whose product is a
           model tensor.
        2. Every rank loads it with the legacy loader into DTensors on a (dp, tp) = (2, 2) mesh: the
           grouped QKV sharded over tp across its head boundaries, experts sharded over dp, the norm
           replicated everywhere.
        3. Every rank loads it through HFLoadPlanner into a second model, once with the default
           collectives and once with use_collectives=False.
    Expectation: Both DCP loads rebuild every tensor exactly as the legacy loader does, with the same
        report, and only the product is loaded through a whole-tensor read.
    """
    init_backend(_DEVICE_TYPE)
    rank = dist.get_rank()
    checkpoint_dir = _make_shared_temp_dir("test_dcp_hf_planner_load_")

    try:
        if rank == 0:
            _save_pretrained_style(checkpoint_dir, _checkpoint_tensors())
        dist.barrier()
        mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
        legacy = _build_model(mesh)
        legacy_state = {"model": legacy}
        HuggingFaceCheckpointer(loader="legacy", weights_mapping=_mapping()).load(
            str(checkpoint_dir), legacy_state
        )
        legacy_report = legacy_state["load_report"]
        expected = _full_tensors(legacy)

        for load_kwargs in ({}, {"use_collectives": False}):
            model = _build_model(mesh)
            planner = HFLoadPlanner(model, weights_mapping=_mapping())
            report = load_hf_checkpoint(model, str(checkpoint_dir), planner=planner, **load_kwargs)

            assert report == legacy_report, f"reports differ with {load_kwargs}: legacy={legacy_report}, dcp={report}"
            assert "layers.0.scaled.weight" not in planner.table and len(planner.deferred) == 1, \
                f"expected only the product as a whole-tensor read, got table={list(planner.table)}"
            for name, loaded in _full_tensors(model).items():
                assert torch.equal(loaded, expected[name]), \
                    f"{name} mismatch with {load_kwargs}: legacy={expected[name]}, dcp={loaded}"
    finally:
        dist.barrier()
        if rank == 0:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
        dist.barrier()
