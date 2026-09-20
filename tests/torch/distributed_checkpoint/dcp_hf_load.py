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
"""DCP loads of safetensors checkpoints written outside DCP, resharded onto a device mesh."""
import inspect
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as torch_dcp
from torch.distributed.device_mesh import init_device_mesh as torch_init_device_mesh
from torch.distributed.tensor import DTensor as TorchDTensor
from torch.distributed.tensor import Replicate as TorchReplicate
from torch.distributed.tensor import Shard as TorchShard
from safetensors.torch import save_file

from hyper_parallel import DTensor
from hyper_parallel.core.distributed_checkpoint import (
    HuggingFaceStorageReader,
    TorchShardedSafetensorsReader,
    load,
)
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from tests.torch.utils import _DEVICE_TYPE, init_backend, to_device

# torch's Hugging Face writer, on a torch that has one.
_TORCH_HF_WRITER = getattr(torch_dcp, "HuggingFaceStorageWriter", None)

_SHARD_FILES = {
    "model-00001-of-00002.safetensors": ("model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight"),
    "model-00002-of-00002.safetensors": ("model.layers.0.mlp.down_proj.weight", "model.norm.weight"),
}


def _tensor_specs() -> dict[str, tuple]:
    """
    The tensors of the checkpoints, and how each is laid out when saved and when loaded.

    Loaded, a tensor is sharded over dp, over tp, over both or over neither of a (dp, tp) = (2, 2)
    mesh, so that a shard is held by two ranks, by one, and by all four. Saved by torch, a tensor is
    sharded over all four ranks along one dimension or replicated, so that the shards a rank loads
    are put together out of more than one file.

    Returns:
        dict[str, tuple]: Name -> (global shape, dtype in the files, placements when loaded, local
        shape when loaded, dimension torch shards it along when saving, or None to replicate it).
    """
    return {
        "model.embed_tokens.weight": ((16, 8), torch.bfloat16, (Shard(0), Replicate()), (8, 8), 0),
        "model.layers.0.self_attn.q_proj.weight": ((8, 8), torch.float32, (Replicate(), Shard(1)), (8, 4), 1),
        "model.layers.0.mlp.down_proj.weight": ((8, 12), torch.float32, (Shard(0), Shard(1)), (4, 6), 0),
        "model.norm.weight": ((8,), torch.float32, (Replicate(), Replicate()), (8,), None),
    }


def _reference_tensors() -> dict[str, torch.Tensor]:
    """The whole tensors of the checkpoints, the same on every rank."""
    torch.manual_seed(0)
    return {name: torch.randn(*spec[0]).to(spec[1]) for name, spec in _tensor_specs().items()}


def _make_shared_temp_dir(prefix: str) -> Path:
    """Create a temporary directory on rank 0 and tell every rank where it is."""
    path_holder = [tempfile.mkdtemp(prefix=prefix) if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(path_holder, src=0)
    return Path(path_holder[0])


def _save_pretrained_style(checkpoint_dir: Path, tensors: dict[str, torch.Tensor]) -> None:
    """
    Write ``tensors`` the way ``save_pretrained`` shards a model: two safetensors files and an index.

    A ``consolidated.safetensors`` holding one of the tensors again goes beside them, as some
    repositories ship a second copy of their weights. The index leaves it out, and so must the load.

    Args:
        checkpoint_dir (Path): Directory to write into.
        tensors (dict[str, torch.Tensor]): The tensors of the checkpoint, by name.
    """
    weight_map = {}
    for file_name, keys in _SHARD_FILES.items():
        save_file({key: tensors[key] for key in keys}, str(checkpoint_dir / file_name), metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(keys, file_name))
    save_file({"model.norm.weight": torch.ones(8)}, str(checkpoint_dir / "consolidated.safetensors"))
    total_size = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (checkpoint_dir / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")


def _save_with_torch(checkpoint_dir: Path, tensors: dict[str, torch.Tensor]) -> None:
    """
    Save ``tensors`` as torch's own DTensors through HuggingFaceStorageWriter(save_distributed=True).

    Every rank writes its shards to a file of its own, recording where in the whole tensor each one
    sits, and no index is written.

    Args:
        checkpoint_dir (Path): Directory every rank writes into.
        tensors (dict[str, torch.Tensor]): The whole tensors, by name, the same on every rank.
    """
    world_size, rank = dist.get_world_size(), dist.get_rank()
    mesh = torch_init_device_mesh(_DEVICE_TYPE, (world_size,))
    state_dict = {}
    for name, (_, _, _, _, dim) in _tensor_specs().items():
        whole = to_device(tensors[name], _DEVICE_TYPE)
        if dim is None:
            state_dict[name] = TorchDTensor.from_local(whole, mesh, [TorchReplicate()])
        else:
            local = whole.chunk(world_size, dim=dim)[rank].contiguous()
            state_dict[name] = TorchDTensor.from_local(local, mesh, [TorchShard(dim)])
    torch_dcp.save(state_dict, storage_writer=_TORCH_HF_WRITER(path=str(checkpoint_dir), save_distributed=True))


def _load_and_check(reader_type: Any, checkpoint_dir: Path, reference: dict[str, torch.Tensor]) -> None:
    """
    Load ``checkpoint_dir`` twice into float32 DTensors on a (dp, tp) = (2, 2) mesh, and compare.

    The first load keeps the default collectives, where a shard several ranks hold on the device is
    read by one of them and broadcast; the second passes use_collectives=False, where every rank
    reads its own.

    Args:
        reader_type (Any): The storage reader class to load through.
        checkpoint_dir (Path): The checkpoint directory.
        reference (dict[str, torch.Tensor]): What every tensor has to come back as, before it is
            converted to float32.
    """
    mesh = init_device_mesh(device_type=_DEVICE_TYPE, mesh_shape=(2, 2), mesh_dim_names=("dp", "tp"))
    for load_kwargs in ({}, {"use_collectives": False}):
        state_dict = {
            name: DTensor.from_local(to_device(torch.zeros(*local_shape), _DEVICE_TYPE), mesh, placements)
            for name, (_, _, placements, local_shape, _) in _tensor_specs().items()
        }
        load(state_dict, storage_reader=reader_type(checkpoint_dir), **load_kwargs)

        for name, dtensor in state_dict.items():
            loaded = dtensor.full_tensor().cpu()
            expected = reference[name].to(torch.float32)
            assert torch.equal(loaded, expected), \
                (f"{name} mismatch after loading through {reader_type.__name__} with {load_kwargs}: "
                 f"expected={expected}, got={loaded}")


def test_dcp_hf_load_resharded() -> None:
    """
    Feature: Test DCP load of a Hugging Face checkpoint through HuggingFaceStorageReader.
    Description:
        1. Rank 0 writes four tensors, one of them bfloat16, into two safetensors files and a
           model.safetensors.index.json, beside a stray safetensors file the index does not name.
        2. Every rank loads them into float32 DTensors on a (dp, tp) = (2, 2) mesh, sharded over dp,
           over tp, over both and over neither: once with the default collectives, and once with
           use_collectives=False.
    Expectation: Both loads rebuild every tensor exactly, converted to float32, and the stray file
        is never read.
    """
    init_backend(_DEVICE_TYPE)
    rank = dist.get_rank()
    reference = _reference_tensors()
    checkpoint_dir = _make_shared_temp_dir("test_dcp_hf_load_")

    try:
        if rank == 0:
            _save_pretrained_style(checkpoint_dir, reference)
        dist.barrier()
        _load_and_check(HuggingFaceStorageReader, checkpoint_dir, reference)
    finally:
        dist.barrier()
        if rank == 0:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
        dist.barrier()


def test_dcp_torch_sharded_load_resharded() -> None:
    """
    Feature: Test DCP load of the shards torch's HuggingFaceStorageWriter writes, through
        TorchShardedSafetensorsReader.
    Description:
        1. Every rank saves four tensors, one of them bfloat16, as torch's own DTensors through
           HuggingFaceStorageWriter(save_distributed=True): three sharded over the four ranks along
           different dimensions and one replicated. Every rank writes a file of its own, and no
           index is written.
        2. Every rank loads them into float32 DTensors of hyper on a (dp, tp) = (2, 2) mesh, so that
           the shards a rank loads are put together out of more than one file: once with the default
           collectives, and once with use_collectives=False.
    Expectation: Both loads rebuild every tensor exactly, converted to float32. The case is skipped
        on a torch whose HuggingFaceStorageWriter cannot save_distributed.
    """
    if _TORCH_HF_WRITER is None or "save_distributed" not in inspect.signature(_TORCH_HF_WRITER).parameters:
        pytest.skip("this torch's HuggingFaceStorageWriter cannot save_distributed")
    init_backend(_DEVICE_TYPE)
    rank = dist.get_rank()
    reference = _reference_tensors()
    checkpoint_dir = _make_shared_temp_dir("test_dcp_torch_sharded_load_")

    try:
        _save_with_torch(checkpoint_dir, reference)
        dist.barrier()
        if rank == 0:
            written = sorted(path.name for path in checkpoint_dir.iterdir())
            assert len(written) > 1 and not any(name.endswith(".json") for name in written), \
                f"expected shard files from several ranks and no index, got {written}"
        _load_and_check(TorchShardedSafetensorsReader, checkpoint_dir, reference)
    finally:
        dist.barrier()
        if rank == 0:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
        dist.barrier()
