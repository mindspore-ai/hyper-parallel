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
"""Test: deferred (meta-device) initialization coupled with fully_shard and prefetch."""
import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

# pylint: disable=C0413
from typing import List, Tuple

import torch
from torch import nn, Tensor
from torch import optim
import torch.distributed as dist
import torch_npu  # pylint: disable=W0611

from hyper_parallel import init_device_mesh, SkipDTensorDispatch
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.init_weights import init_empty_weights
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from tests.torch.utils import init_dist
from tests.torch.common_net import MetaInitNet


# ---------------------------------------------------------------------------
# Test networks
# ---------------------------------------------------------------------------


class MultiLayerNet(nn.Module):
    """Multi-layer network to exercise the per-layer fully_shard + prefetch pattern."""

    def __init__(self, hidden_size: int = 32, num_layers: int = 3) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden_size, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.head(x).sum()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mp_policy() -> MixedPrecisionPolicy:
    return MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
        cast_forward_inputs=True,
    )

def train_steps(model: nn.Module, optimizer: optim.Optimizer, input_list: List[Tuple[Tensor, ...]]) -> None:
    """Run one training step per entry in *input_list* and verify optimizer param identity.

    After every optimizer.step(), checks that every parameter tensor held by the
    optimizer is the same object as the corresponding model parameter, ensuring that
    fully_shard has not silently replaced parameter objects during the forward/backward.

    Args:
        model: The fully_shard-wrapped model to train.
        optimizer: Optimizer constructed from model.parameters() before this call.
        input_list: One tuple of input tensors per training step.
    """
    for inputs in input_list:
        optimizer.zero_grad()
        loss = model(*inputs)
        loss.backward()
        after_fwd_bwd_params = list(model.parameters())
        opt_params = [p for group in optimizer.param_groups for p in group["params"]]
        optimizer.step()
        opt_ids = {id(p) for p in opt_params}
        model_ids = {id(p) for p in after_fwd_bwd_params}
        assert opt_ids == model_ids, (
            f"Optimizer param ids differ from model param ids after step: "
            f"opt_ids={sorted(opt_ids)}, model_ids={sorted(model_ids)}"
        )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_fully_shard_meta_init():
    """
    Feature: Test fully_shard with meta device initialization
    Description: Model is created on meta device, then materialized to NPU before training.
    This validates the lazy_init path: reset_sharded_param and _validate_no_meta_params.
    Expectation: run successfully
    """
    hidden_size = 32
    init_dist()
    world_size = dist.get_world_size()
    mesh = init_device_mesh(device_type="npu", mesh_shape=(world_size,), mesh_dim_names=("dp",))

    with torch.device("meta"):
        model = MetaInitNet(hidden_size)

    model = fully_shard(
        model,
        mesh=mesh,
        reshard_after_forward=True,
        mp_policy=_make_mp_policy()
    )
    model.to_empty(device="npu")
    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    input_list = [(torch.rand(4, hidden_size).npu(),) for _ in range(2)]
    optimizer = optim.Adam(model.parameters())
    with SkipDTensorDispatch():
        train_steps(model, optimizer, input_list)

def test_fully_shard_init_empty_weights_with_prefetch():
    """
    Feature: init_empty_weights + fully_shard + prefetch
    Description: Build a multi-layer model inside init_empty_weights context,
                 apply per-layer fully_shard with forward prefetch configured,
                 materialize local DTensor shards, then verify training succeeds.
    Expectation: run successfully
    """
    rank, _ = init_dist()
    world_size = dist.get_world_size()
    device = torch.device(f"npu:{rank}")
    hidden_size = 32

    mesh = init_device_mesh(device_type="npu", mesh_shape=(world_size,), mesh_dim_names=("dp",))
    mp_policy = _make_mp_policy()

    with init_empty_weights():
        model = MultiLayerNet(hidden_size, num_layers=3)

    fsdp_layers = []
    for layer in model.layers:
        fully_shard(layer, mesh=mesh, reshard_after_forward=True, mp_policy=mp_policy)
        fsdp_layers.append(layer)
    fully_shard(model.head, mesh=mesh, reshard_after_forward=True, mp_policy=mp_policy)
    fully_shard(model, mesh=mesh, reshard_after_forward=True, mp_policy=mp_policy)

    for i in range(len(fsdp_layers) - 1):
        fsdp_layers[i].set_modules_to_forward_prefetch([fsdp_layers[i + 1]])


    # Materialize: replaces each meta _local_tensor with a real torch.empty tensor.
    for p in model.parameters():
        if isinstance(p, DTensor):
            local = p._local_tensor  # pylint: disable=W0212
            if local.is_meta:
                p._local_tensor = torch.empty(  # pylint: disable=W0212
                    local.shape, dtype=local.dtype, device=device,requires_grad=p._local_tensor.requires_grad
                )

    # reset_parameters() fills values in-place; _local_tensor objects must not change.
    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    # Collect expected FQNs from the model before training triggers lazy_init.
    expected_fqns = {name for name, _ in model.named_parameters()}

    input_list = [(torch.rand(4, hidden_size, device=device),) for _ in range(2)]
    optimizer = optim.Adam(model.parameters())
    with SkipDTensorDispatch():
        train_steps(model, optimizer, input_list)

    # After the first forward pass, _init_params_fqn has run; verify every
    # managed param (hsdp_params + replicate_params) got a correct FQN.
    from hyper_parallel.core.fully_shard.hsdp_utils import get_hsdp_state  # pylint: disable=C0415
    from hyper_parallel.platform import get_platform as _get_platform  # pylint: disable=C0415
    platform_impl = _get_platform()

    assigned_fqns = set()
    for _, module in platform_impl.get_cells_and_names(model):
        hsdp_state = get_hsdp_state(module)
        if hsdp_state is None:
            continue
        for hsdp_param in hsdp_state._iter_managed_params():
            fqn = getattr(hsdp_param, "_param_fqn", None)
            assert fqn is not None, (
                f"_param_fqn not assigned for a parameter in module {type(module).__name__}"
            )
            assigned_fqns.add(fqn)

    assert assigned_fqns == expected_fqns, (
        f"_param_fqn mismatch after init_empty_weights + prefetch deferred init: "
        f"assigned={sorted(assigned_fqns)}, expected={sorted(expected_fqns)}"
    )
