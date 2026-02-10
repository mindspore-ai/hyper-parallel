# Copyright 2025 Huawei Technologies Co., Ltd
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
"""flash attention score shard in python"""

import math
import numpy as np
import mindspore as ms
import mindspore.communication.management as D
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore.ops import flash_attention_score
from hyper_parallel import init_device_mesh, shard_module, DTensor
from hyper_parallel.core.placement_types import Shard, Replicate
from hyper_parallel.core.shard.sharding_plan import ShardingPlan
from hyper_parallel.core.shard.ops.parallel_flash_attention_score import ParallelFlashAttention


def setup_module():
    """setup module"""
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend")
    D.init()


def generate_inputs(b, n1, n2, s1, s2, dqkv, input_layout, dtype, return_tensor=True):
    """generate inputs"""
    min_value = -1
    max_value = 1
    if input_layout == "BSH":
        query = np.random.uniform(min_value, max_value, [b, s1, n1 * dqkv])
        key = np.random.uniform(min_value, max_value, [b, s2, n2 * dqkv])
        value = np.random.uniform(min_value, max_value, [b, s2, n2 * dqkv])
    elif input_layout == "BNSD":
        query = np.random.uniform(min_value, max_value, [b, n1, s1, dqkv])
        key = np.random.uniform(min_value, max_value, [b, n2, s2, dqkv])
        value = np.random.uniform(min_value, max_value, [b, n2, s2, dqkv])
    elif input_layout == "SBH":
        query = np.random.uniform(min_value, max_value, [s1, b, n1 * dqkv])
        key = np.random.uniform(min_value, max_value, [s2, b, n2 * dqkv])
        value = np.random.uniform(min_value, max_value, [s2, b, n2 * dqkv])
    elif input_layout == "BSND":
        query = np.random.uniform(min_value, max_value, [b, s1, n1, dqkv])
        key = np.random.uniform(min_value, max_value, [b, s2, n2, dqkv])
        value = np.random.uniform(min_value, max_value, [b, s2, n2, dqkv])
    elif input_layout == "TND":
        query = np.random.uniform(min_value, max_value, [b * s1, n1, dqkv])
        key = np.random.uniform(min_value, max_value, [b * s2, n2, dqkv])
        value = np.random.uniform(min_value, max_value, [b * s2, n2, dqkv])
    else:
        raise ValueError("input_layout is invalid.")
    real_shift = None
    attn_mask = np.triu(np.ones([b, 1, s1, s2]))
    prefix = None
    if return_tensor:
        return Tensor(query, dtype=dtype), Tensor(key, dtype=dtype), Tensor(value, dtype=dtype), real_shift, \
            Tensor(attn_mask, dtype=mstype.uint8), prefix
    return query, key, value, real_shift, attn_mask, prefix


def test_flash_attention_score_model_parallel():
    '''
    Feature: FlashAttentionScore in python shard.
    Description: Test FlashAttentionScore model parallel in python shard.
    Expectation: Run success.
    '''
    ms.set_seed(1)
    b, n, s, d = 2, 16, 1024, 128
    input_layout = 'BNSD'
    sparse_mode = 0
    dtype = mstype.bfloat16
    scalar_value = 1.0 / math.sqrt(d)
    query, key, value, _, attn_mask, _ = generate_inputs(b, n, n, s, s, d, input_layout, dtype)

    # Standalone
    standalone_output = flash_attention_score(query=query,
                                              key=key,
                                              value=value,
                                              head_num=n,
                                              attn_mask=attn_mask,
                                              scalar_value=scalar_value,
                                              input_layout=input_layout,
                                              sparse_mode=sparse_mode)
    # Parallel
    mesh_shape = (2, 1, 4)
    alias_name = ("dp", "cp", "tp")

    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=mesh_shape,
        mesh_dim_names=alias_name
    )

    # Define placements using Placement format
    # query shape: [b, n, s, d] (BNSD)
    query_placements = (Shard(0), Shard(2), Shard(1))

    # key shape: [b, n, s, d] (BNSD)
    key_placements = (Shard(0), Replicate(), Shard(1))

    # value shape: [b, n, s, d] (BNSD)
    value_placements = (Shard(0), Replicate(), Shard(1))

    # attn_mask shape: [b, 1, s1, s2]
    attn_mask_placements = (Shard(0), Shard(2), Replicate())

    # output placements (same as query)
    output_placements = (Shard(0), Shard(2), Shard(1))

    query_local = DTensor.distribute_tensor(query, mesh, query_placements)
    key_local = DTensor.distribute_tensor(key, mesh, key_placements)
    value_local = DTensor.distribute_tensor(value, mesh, value_placements)
    attn_mask_local = DTensor.distribute_tensor(attn_mask, mesh, attn_mask_placements)

    parallel_net = ParallelFlashAttention(head_num=n,
                                          scalar_value=scalar_value,
                                          input_layout=input_layout,
                                          sparse_mode=sparse_mode)
    sharding_plan = ShardingPlan(
        input_plan={
            "input": (query_placements, key_placements, value_placements, attn_mask_placements),
        },
        output_plan={
            "output": (output_placements,),
        },
    )
    shard_module(parallel_net, device_mesh=mesh, sharding_plan=sharding_plan)
    parallel_output = parallel_net(query_local, key_local, value_local, attn_mask_local)

    # Validate
    parallel_output = parallel_output.full_tensor()
    assert np.allclose(standalone_output.astype(mstype.float32).asnumpy(),
                       parallel_output.astype(mstype.float32).asnumpy(),
                       1e-3, 1e-3)
