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
"""init.md: init_empty_weights -> fully_shard (meta) -> to_empty -> reset_parameter loop."""

import os
from typing import Tuple, Union

import numpy as np

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

# pylint: disable=C0413
import mindspore._checkparam as Validator
import mindspore as ms
from mindspore import mint, nn
from mindspore.common import dtype
from mindspore.common.initializer import Normal, initializer
from mindspore.common.parameter import Parameter
from mindspore.common.api import _no_grad
import mindspore.communication.management as D

from hyper_parallel import init_device_mesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.init_weights import init_empty_weights
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from hyper_parallel.platform import get_platform

D.init()

def init_method_normal(sigma: float = 0.01, param_init_dtype: dtype = dtype.float32):
    """Mindspore style init method. `initializer` method independent of the generator's
    random management. """

    def init_(tensor_shape: Union[Tuple[int, ...], list]):
        return initializer(Normal(mean=0.0, sigma=sigma), tensor_shape, param_init_dtype)

    return init_


def init_mint_normal(dtensor):
    """Fill the tensor with random numbers sampled from the normal distribution."""
    if isinstance(dtensor, DTensor):
        with _no_grad():
            dtensor.normal_()
    else:
        raise ValueError(f"Only support dtensor type, got: {type(dtensor)}")


class VocabEmbedding(nn.Cell):
    """Vocabulary embedding for the empty-weights / reset_parameter flow."""

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        init_method=None,
    ):
        super().__init__()
        if init_method is None:
            init_method = init_method_normal()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = mint.gather
        self.tile = mint.tile
        self.reshape = mint.reshape

        self.weight = Parameter(init_method([self.num_embeddings, self.embedding_dim]), name="weight")

    def construct(self, input_):
        """
        Forward of vocab embedding.

        input_: (B, S)
        weight: (V, H)
        output: (B, S, H)
        """
        Validator.check_type_name("input_ids", input_.dtype, [dtype.int32, dtype.int64], self.cls_name)

        _, seq_len = input_.shape

        input_ = self.reshape(input_, (-1, 1))
        input_ = self.tile(input_, (1, self.embedding_dim))
        masked_input = input_

        output = self.embedding(self.weight, 0, masked_input)
        output = self.reshape(output, (-1, seq_len, self.embedding_dim))

        return output

    def reset_parameter(self):
        """Reset embedding weight using distributed random normal init."""
        init_mint_normal(self.weight)


class MyModel(nn.Cell):
    """Root model wrapping ``VocabEmbedding``."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        """Initialize wrapper model with a single embedding submodule."""
        super().__init__()
        self.embedding = VocabEmbedding(num_embeddings, embedding_dim)

    def construct(self, input_):
        """Forward pass."""
        return self.embedding(input_)


def _assert_shards_differ(model, rank, world_size):
    """Assert DTensor trainable params hold different shards across ranks."""
    for param in model.trainable_params():
        assert isinstance(param, DTensor), f"trainable param expected DTensor, got {type(param)}"
        local = param.to_local()
        gathered = [mint.zeros_like(local) for _ in range(world_size)]
        mint.distributed.all_gather(gathered, local)
        for other_rank in range(world_size):
            if other_rank != rank:
                assert not (local == gathered[other_rank]).all(), (
                    f"param '{param.name}' rank {rank} == rank {other_rank}"
                )


def _assert_reset_parameter_correctness(net):
    """Assert distributed random init correctness: finite, reproducible by seed, and seed-sensitive."""
    def _run_reset_once():
        for _, cell in net.cells_and_names():
            if hasattr(cell, "reset_parameter"):
                cell.reset_parameter()

    def _snapshot_local_params():
        return {p.name: p.to_local().asnumpy().copy() for p in net.trainable_params()}

    for p in net.trainable_params():
        local = p.to_local()
        assert local.dtype == ms.float32, f"param '{p.name}' dtype expected float32, got {local.dtype}"
        assert mint.isfinite(local).all(), f"param '{p.name}' contains inf/nan after reset"
        assert (local != 0).any(), f"param '{p.name}' should not be all zeros after reset"

    ms.manual_seed(2026)
    _run_reset_once()
    first = _snapshot_local_params()

    ms.manual_seed(2027)
    _run_reset_once()
    second = _snapshot_local_params()

    has_seed_effect = any(not np.array_equal(first[name], second[name]) for name in first)
    assert has_seed_effect, "reset_parameter should produce different values for different seeds"


def test_empty_weights_and_reset_parameter():
    """
    Feature: init.md flow — init_empty_weights, fully_shard under meta, to_empty, reset loop
    Description: Same as ``init.md`` pseudo-code (fully_shard + ``net.to_empty()`` + ``for cell in net.cells()``).
    Expectation: run successfully
    """
    plat = get_platform()
    rank = plat.get_rank()
    world_size = plat.get_world_size()
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(world_size,),
        mesh_dim_names=("dp",),
    )
    mp_policy = MixedPrecisionPolicy(
        param_dtype=ms.float32,
        reduce_dtype=ms.float32,
        output_dtype=ms.float32,
        cast_forward_inputs=True,
    )

    with init_empty_weights():
        net = MyModel(128, 32)

    with ms.DeviceCtx("meta"):
        net = fully_shard(net, mesh=mesh, reshard_after_forward=True, mp_policy=mp_policy)

    for p in net.trainable_params():
        assert isinstance(p, DTensor), f"param expected DTensor after fully_shard, got {type(p)}"
        local = p.to_local()
        assert getattr(local, "is_meta", False), (
            f"param expected on meta after init_empty_weights, got {getattr(local, 'device', local)}"
        )

    net.to_empty(device="npu")
    for _, cell in net.cells_and_names():
        if hasattr(cell, "reset_parameter"):
            cell.reset_parameter()

    for p in net.trainable_params():
        local = p.to_local()
        assert not getattr(local, "is_meta", False), "param should be materialized after to_empty and reset"

    _assert_reset_parameter_correctness(net)
    _assert_shards_differ(net, rank, world_size)
