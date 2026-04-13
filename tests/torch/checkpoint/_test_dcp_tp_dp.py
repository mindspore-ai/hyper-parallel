# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""fully_shard + DCP save/load (default flatten_state_dict=True) and bytes; run via parallel_run or torchrun."""
# pylint: disable=C0413
import os
import re
import shutil
from pathlib import Path
from typing import Tuple

import numpy as np

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import torch
import torch.distributed as dist
import torch_npu  # pylint: disable=W0611

from hyper_parallel import SkipDTensorDispatch, init_device_mesh
from hyper_parallel.core.distributed_checkpoint import load, save
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import DTensor, distribute_module, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from tests.torch.common_net import FullyShardTestNet
from tests.torch.utils import init_dist

HIDDEN = 32
LAYERS = 2
BATCH = 4
# After load: train model and model2 in lockstep for this many steps; loss must match each step.
COMPARE_STEPS_AFTER_LOAD = 3
TP_SIZE = 2
MP = MixedPrecisionPolicy(
    param_dtype=torch.float32,
    reduce_dtype=torch.float32,
    output_dtype=torch.float32,
    cast_forward_inputs=True,
)

# Single context: zeros_like / ones_like use DTensor dispatch; other ops use the local path (forward, backward, step).
_DTENSOR_TRAIN_NO_SKIP = {torch.zeros_like, torch.ones_like}


def _save_unsharded_checkpoint(path: Path):
    """Save unsharded reference checkpoint before any sharding/distribute steps."""
    base_model = FullyShardTestNet(HIDDEN, LAYERS, has_bias=False, dense_weight_init=torch.rand)
    torch.save({"model": base_model.state_dict()}, path)


def _build_dp_tp_meshes(num_cards: int):
    """Build root (4,) mesh and logical dp/tp meshes (both split degree=2)."""
    assert num_cards == 4, f"this test requires world_size=4, but got {num_cards}"
    root_mesh_1d = init_device_mesh(
        device_type="npu",
        mesh_shape=(num_cards,),
        mesh_dim_names=("axis0",),
    )
    mesh_2d = DeviceMesh(
        device_type="npu",
        mesh=np.array(root_mesh_1d.rank_list, dtype=np.int32).reshape(2, 2),
        mesh_dim_names=("dp", "tp"),
    )
    dp_mesh = mesh_2d["dp"]
    tp_mesh = mesh_2d["tp"]
    return root_mesh_1d, mesh_2d, dp_mesh, tp_mesh


def _expected_layer1_offsets(root_mesh):
    """Return expected global offsets for layer1.weight local shard under (dp=2,tp=2), both on dim-0."""
    coord = root_mesh.get_coordinate()
    # layer1.weight global shape: (32, 32)
    # TP shards dim-0 by 2, then DP(fully_shard) shards dim-0 again by 2 => local shape (8, 32).
    # Under mesh dim order ("dp", "tp"), layer1 first shards by TP on dim-0,
    # then fully_shard(DP) shards dim-0 again inside each TP chunk.
    row_off = coord[1] * 16 + coord[0] * 8
    return row_off, 0


def _verify_pretrain_dcp_layout(
    *,
    rank: int,
    world: int,
    root_mesh,
    model,
    unsharded_path: Path,
    sharded_pretrain_path: Path,
):
    """
    Validate DCP pre-train save correctness and TP/DP shard placement using layer1.weight.
    """
    unsharded_state = torch.load(unsharded_path, map_location="cpu")
    full_w = _tensor_to_numpy_float(unsharded_state["model"]["dense_layers.layers.layer1.weight"])
    assert tuple(full_w.shape) == (32, 32), f"unexpected full layer1.weight shape: {tuple(full_w.shape)}"

    load_state = {"model": model.state_dict()}
    load(load_state, checkpoint_id=sharded_pretrain_path, use_collectives=True)
    local_w = _tensor_to_numpy_float(load_state["model"]["dense_layers.layers.layer1.weight"])
    assert tuple(local_w.shape) == (8, 32), f"unexpected local layer1.weight shape: {tuple(local_w.shape)}"

    row_off, col_off = _expected_layer1_offsets(root_mesh)
    expected_local = full_w[row_off: row_off + 8, col_off: col_off + 32]
    assert np.allclose(local_w, expected_local, rtol=1e-5, atol=1e-5), (
        f"rank{rank} layer1.weight shard mismatch at offset=({row_off},{col_off})"
    )

    gathered = [None for _ in range(world)]
    dist.all_gather_object(gathered, {"rank": rank, "offset": (row_off, col_off)})
    if rank == 0:
        rank_to_offset = {item["rank"]: item["offset"] for item in gathered}
        expected = {0: (0, 0), 1: (16, 0), 2: (8, 0), 3: (24, 0)}
        assert rank_to_offset == expected, (
            f"unexpected layer1.weight shard layout mapping: got {rank_to_offset}, expected {expected}"
        )
        print(f"[DCP Layout Check] layer1.weight offsets by rank: {rank_to_offset}")
        print("[DCP Layout Check] layout matches expected dp=2,tp=2 placement with TP+DP both sharding dim-0.")


def _assert_bytes_file_count(checkpoint_path: Path, world_size: int, *, expect_bytes: bool) -> None:
    """
    Assert bytes file count in checkpoint directory.

    Args:
        checkpoint_path (Path): Checkpoint directory path.
        world_size (int): Distributed world size.
        expect_bytes (bool): Whether BYTE_IO files are expected to exist.
    """
    bytes_files = sorted(file_path.name for file_path in checkpoint_path.glob("*.bytes"))
    if not expect_bytes:
        assert not bytes_files, (
            f"unexpected bytes files found in {checkpoint_path}: {bytes_files}"
        )
        return

    rank_pattern = re.compile(r"^_rank(\d+)_\.bytes$")
    rank_to_file: dict[int, str] = {}
    for file_name in bytes_files:
        match = rank_pattern.match(file_name)
        assert match is not None, f"unexpected bytes file naming: {file_name}"
        rank = int(match.group(1))
        assert 0 <= rank < world_size, f"bytes file rank out of range: {file_name}"
        assert rank not in rank_to_file, (
            f"found multiple bytes files for rank{rank}: {rank_to_file[rank]}, {file_name}"
        )
        rank_to_file[rank] = file_name

    assert rank_to_file, (
        f"expected bytes files in {checkpoint_path}, but none were found"
    )


def _fully_shard_model(num_cards: int) -> FullyShardTestNet:
    """Build a fully-sharded ``FullyShardTestNet`` on ``num_cards`` NPUs."""
    model = FullyShardTestNet(HIDDEN, LAYERS, has_bias=False, dense_weight_init=torch.rand)
    _, _, dp_mesh, tp_mesh = _build_dp_tp_meshes(num_cards)
    x_placements = tuple(Replicate() for _ in range(tp_mesh.ndim))
    layer0_w_placements = (Shard(1),) + tuple(Replicate() for _ in range(tp_mesh.ndim - 1))
    layer1_w_placements = (Shard(0),) + tuple(Replicate() for _ in range(tp_mesh.ndim - 1))

    def partition_fn(mod_name, module, device_mesh):
        # Root w1 and layer0 stay TP-replicated; layer1 keeps original TP-shard strategy.
        if mod_name == "":
            param = module.w1
            module.register_parameter(
                "w1",
                torch.nn.Parameter(
                    distribute_tensor(param.data, device_mesh, x_placements),
                    requires_grad=param.requires_grad,
                ),
            )
            return
        if mod_name == "dense_layers.layers.layer0":
            param = module.weight
            module.register_parameter(
                "weight",
                torch.nn.Parameter(
                    distribute_tensor(param.data, device_mesh, layer0_w_placements),
                    requires_grad=param.requires_grad,
                ),
            )
            return
        if mod_name == "dense_layers.layers.layer1":
            param = module.weight
            module.register_parameter(
                "weight",
                torch.nn.Parameter(
                    distribute_tensor(param.data, device_mesh, layer1_w_placements),
                    requires_grad=param.requires_grad,
                ),
            )

    def input_fn(mod, inputs, device_mesh):  # pylint: disable=unused-argument
        # Convert local input tensor to DTensor on TP mesh before forward.
        x = inputs[0]
        if isinstance(x, DTensor):
            return inputs
        return (DTensor.from_local(x, device_mesh, x_placements),) + tuple(inputs[1:])

    distribute_module(
        model,
        device_mesh=tp_mesh,
        partition_fn=partition_fn,
        input_fn=input_fn,
    )
    mesh = dp_mesh
    for dense_layer in model.dense_layers.layers:
        fully_shard(
            dense_layer,
            mesh=mesh,
            reshard_after_forward=True,
            mp_policy=MP,
        )
    fully_shard(model, mesh=mesh, reshard_after_forward=True, mp_policy=MP)
    # Align with TP + fully_shard e2e path: gradient communication uses SUM.
    model.set_reduce_op_type("sum")
    return model


def _build_model_and_adamw(
    num_cards: int,
    *,
    lr_g0: float,
    lr_g1: float,
    eps: float,
    betas: Tuple[float, float],
    weight_decay: float,
) -> Tuple[FullyShardTestNet, torch.optim.AdamW]:
    """Return a fully-sharded model and AdamW with two learning-rate groups."""
    model = _fully_shard_model(num_cards)
    params = list(model.parameters())
    assert len(params) >= 2, "need at least two parameter tensors for two param_groups"
    mid = len(params) // 2
    optimizer = torch.optim.AdamW(
        [
            {"params": params[:mid], "lr": lr_g0, "eps": eps, "betas": betas, "weight_decay": weight_decay},
            {"params": params[mid:], "lr": lr_g1, "eps": eps, "betas": betas, "weight_decay": weight_decay},
        ]
    )
    return model, optimizer


def _flatten_coordinate(mesh_shape, coordinate):
    """Flatten an n-D mesh coordinate into a row-major linear index."""
    flat_idx = 0
    for dim_size, dim_idx in zip(mesh_shape, coordinate):
        flat_idx = flat_idx * dim_size + dim_idx
    return flat_idx


def _get_local_batch_slice(x, dp_size, dp_idx):
    """Return the DP-local input slice for ``dp_idx``."""
    x_chunks = torch.chunk(x, dp_size, dim=0)
    return x_chunks[dp_idx].contiguous()

def _train_step(model, x, optimizer, n: int = 1, *, tp_reduce_size: int) -> float:
    "Execute training steps for model and optimizer, return the loss value."
    last = 0.0
    seed_device = x.to_local().device if isinstance(x, DTensor) else x.device
    for _ in range(n):
        loss = model(x)
        print(f"=========loss layout: {loss.layout}")
        loss.backward(torch.tensor(1.0 / tp_reduce_size, device=seed_device))
        with SkipDTensorDispatch(no_skip=_DTENSOR_TRAIN_NO_SKIP):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        last = float(loss.to_local().detach().cpu()) if isinstance(loss, DTensor) else loss.item()
    return last


def _tensor_to_numpy_float(t):
    # Normalize Tensor/DTensor to float32 numpy array for numerical comparisons.
    if isinstance(t, DTensor):
        return t.to_local().detach().cpu().float().numpy()
    return t.detach().cpu().float().numpy()


def _assert_sd_tensors_close(a, b, rtol=1e-5, atol=1e-5):
    """Recursively assert dict/list leaves are close (tensor or DTensor)."""
    if isinstance(a, (torch.Tensor, DTensor)) and isinstance(b, (torch.Tensor, DTensor)):
        print(f"dtensor layout: {a.layout}")
        if isinstance(a, DTensor):
            print(f"dtensor placements: {a.placements}")
        assert np.allclose(
            _tensor_to_numpy_float(a),
            _tensor_to_numpy_float(b),
            rtol=rtol,
            atol=atol,
        ), f"tensor mismatch {a} vs {b}"
        return
    if isinstance(a, dict) and isinstance(b, dict):
        assert set(a.keys()) == set(b.keys()), f"keys {set(a.keys())} vs {set(b.keys())}"
        for k in a:
            _assert_sd_tensors_close(a[k], b[k], rtol=rtol, atol=atol)
        return
    if isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _assert_sd_tensors_close(x, y, rtol=rtol, atol=atol)
        return
    if type(a) is type(b) and a == b:
        return
    assert a == b, f"mismatch {type(a)} {a!r} vs {type(b)} {b!r}"


def test_dcp_with_optimizer_tp_dp():
    """fully_shard + AdamW + DCP save/load + flatten_state_dict + bytes."""
    init_dist()
    world = dist.get_world_size()
    rank = dist.get_rank()
    assert world == 4, f"test_dcp_with_fully_shard_optimizer requires world_size=4, but got {world}"

    path_name = "dcp_with_fully_shard_optimizer"
    checkpoint_path = Path(path_name)
    unsharded_ckpt_path = Path("dcp_with_fully_shard_optimizer_unsharded.pt")
    sharded_pretrain_ckpt_path = Path("dcp_with_fully_shard_optimizer_pretrain_dcp")

    if rank == 0:
        os.makedirs(path_name, exist_ok=True)
    dist.barrier()
    # Keep unsharded reference init identical to the later sharded model init.
    torch.manual_seed(7)
    if rank == 0:
        _save_unsharded_checkpoint(unsharded_ckpt_path)
    dist.barrier()

    torch.manual_seed(7)
    shared_betas = (0.9, 0.999)
    shared_wd = 0.01
    lr_g0, lr_g1 = 1.7e-4, 2.3e-4
    save_eps = 1e-7
    model, optimizer = _build_model_and_adamw(
        world,
        lr_g0=lr_g0,
        lr_g1=lr_g1,
        eps=save_eps,
        betas=shared_betas,
        weight_decay=shared_wd,
    )
    _, root_mesh, dp_mesh, tp_mesh = _build_dp_tp_meshes(world)
    tp_reduce_size = tp_mesh.size()

    full_x = torch.randn(BATCH, HIDDEN).npu()
    dp_idx = _flatten_coordinate(dp_mesh.mesh_shape, dp_mesh.get_coordinate())
    x = _get_local_batch_slice(full_x, dp_mesh.size(), dp_idx)

    # Save sharded model checkpoint before any train step.
    pretrain_save_state = {"model": model.state_dict()}
    save(
        pretrain_save_state,
        checkpoint_id=sharded_pretrain_ckpt_path,
        use_collectives=True,
    )
    dist.barrier()
    if rank == 0:
        _assert_bytes_file_count(sharded_pretrain_ckpt_path, world, expect_bytes=False)
    _verify_pretrain_dcp_layout(
        rank=rank,
        world=world,
        root_mesh=root_mesh,
        model=model,
        unsharded_path=unsharded_ckpt_path,
        sharded_pretrain_path=sharded_pretrain_ckpt_path,
    )

    steps_save = 1
    _train_step(model, x, optimizer, steps_save, tp_reduce_size=tp_reduce_size)

    save_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    save(save_state, checkpoint_id=checkpoint_path, use_collectives=True)
    dist.barrier()
    if rank == 0:
        _assert_bytes_file_count(checkpoint_path, world, expect_bytes=True)

    wrong_lr_g0, wrong_lr_g1 = 5e-2, 8e-2
    wrong_eps = 1e-2
    model2, optimizer2 = _build_model_and_adamw(
        world,
        lr_g0=wrong_lr_g0,
        lr_g1=wrong_lr_g1,
        eps=wrong_eps,
        betas=shared_betas,
        weight_decay=shared_wd,
    )

    steps_load_wrong = 2
    _train_step(model2, x, optimizer2, steps_load_wrong, tp_reduce_size=tp_reduce_size)

    load_state = {
        "model": model2.state_dict(),
        "optimizer": optimizer2.state_dict(),
    }

    load(load_state, checkpoint_id=checkpoint_path, use_collectives=True)

    _assert_sd_tensors_close(save_state["model"], load_state["model"])
    _assert_sd_tensors_close(save_state["optimizer"], load_state["optimizer"])

    model2.load_state_dict(load_state["model"])
    optimizer2.load_state_dict(load_state["optimizer"])

    _assert_sd_tensors_close(model2.state_dict(), save_state["model"])
    _assert_sd_tensors_close(optimizer2.state_dict(), save_state["optimizer"])


    dist.barrier()
    if rank == 0:
        shutil.rmtree(checkpoint_path)
        shutil.rmtree(sharded_pretrain_ckpt_path)
        unsharded_ckpt_path.unlink(missing_ok=True)

    # model was not trained after save; same checkpoint as model2 after load — paired steps should match loss.
    for _ in range(COMPARE_STEPS_AFTER_LOAD):
        loss_m = _train_step(model, x, optimizer, tp_reduce_size=tp_reduce_size)
        loss_m2 = _train_step(model2, x, optimizer2, tp_reduce_size=tp_reduce_size)
        assert np.isclose(loss_m, loss_m2, rtol=0.0, atol=1e-4), (
            f"loss mismatch after load (paired train step): {loss_m} vs {loss_m2}"
        )
