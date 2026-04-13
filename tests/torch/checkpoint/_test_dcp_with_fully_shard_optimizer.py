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
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from tests.torch.common_net import FullyShardTestNet
from tests.torch.utils import init_dist

HIDDEN = 32
LAYERS = 2
BATCH = 4
# After load: train model and model2 in lockstep for this many steps; loss must match each step.
COMPARE_STEPS_AFTER_LOAD = 3
MP = MixedPrecisionPolicy(
    param_dtype=torch.float32,
    reduce_dtype=torch.float32,
    output_dtype=torch.float32,
    cast_forward_inputs=True,
)

# Single context: zeros_like / ones_like use DTensor dispatch; other ops use the local path (forward, backward, step).
_DTENSOR_TRAIN_NO_SKIP = {torch.zeros_like, torch.ones_like}


def _fully_shard_model(num_cards: int) -> FullyShardTestNet:
    """Build a fully-sharded ``FullyShardTestNet`` on ``num_cards`` NPUs."""
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(num_cards,),
        mesh_dim_names=("dp",),
    )
    model = FullyShardTestNet(HIDDEN, LAYERS, has_bias=False)
    for dense_layer in model.dense_layers.layers:
        fully_shard(
            dense_layer,
            mesh=mesh,
            reshard_after_forward=True,
            mp_policy=MP,
        )
    fully_shard(model, mesh=mesh, reshard_after_forward=True, mp_policy=MP)
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


def _train_step(model, x, optimizer, n: int = 1) -> float:
    last = 0.0
    for _ in range(n):
        with SkipDTensorDispatch(no_skip=_DTENSOR_TRAIN_NO_SKIP):
            loss = model(x)
            loss.backward(torch.ones_like(loss))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        last = loss.item()
    return last


def _tensor_to_numpy_float(t):
    if isinstance(t, DTensor):
        return t.to_local().detach().cpu().float().numpy()
    return t.detach().cpu().float().numpy()


def _assert_sd_tensors_close(a, b, rtol=1e-5, atol=1e-5):
    """Recursively assert dict/list leaves are close (tensor or DTensor)."""
    if isinstance(a, (torch.Tensor, DTensor)) and isinstance(b, (torch.Tensor, DTensor)):
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


def test_dcp_with_fully_shard_optimizer():
    """fully_shard + AdamW + DCP save/load + flatten_state_dict + bytes."""
    init_dist()
    world = dist.get_world_size()
    rank = dist.get_rank()

    path_name = "dcp_with_fully_shard_optimizer"
    checkpoint_path = Path(path_name)

    if rank == 0:
        os.makedirs(path_name, exist_ok=True)
    dist.barrier()

    torch.manual_seed(7 + rank)
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
    x = torch.randn(BATCH, HIDDEN).npu()

    steps_save = 1
    _train_step(model, x, optimizer, steps_save)

    save_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    save(save_state, checkpoint_id=checkpoint_path, use_collectives=True)
    dist.barrier()

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
    _train_step(model2, x, optimizer2, steps_load_wrong)

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

    # model was not trained after save; same checkpoint as model2 after load 鈥?paired steps should match loss.
    for _ in range(COMPARE_STEPS_AFTER_LOAD):
        loss_m = _train_step(model, x, optimizer)
        loss_m2 = _train_step(model2, x, optimizer2)
        assert np.isclose(loss_m, loss_m2, rtol=0.0, atol=1e-4), (
            f"loss mismatch after load (paired train step): {loss_m} vs {loss_m2}"
        )
