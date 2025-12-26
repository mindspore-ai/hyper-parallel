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
from hyper_parallel import Layout, shard
from hyper_parallel.core.tensor_parallel.ops.parallel_flash_attention_score import ParallelFlashAttention
from tests.mindspore.st.tensor_parallel.utils import global_to_local, local_to_global


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
    alias_name = ("dp", "cp", "mp")
    rank_list = list(range(8))
    layout = Layout(mesh_shape, alias_name, rank_list)
    query_layout = layout("dp", "mp", "cp", "None")
    key_layout = layout("dp", "mp", "None", "None")
    value_layout = layout("dp", "mp", "None", "None")
    attn_mask_layout = layout("dp", "None", "cp", "None")
    query_local = global_to_local(query, query_layout)
    key_local = global_to_local(key, key_layout)
    value_local = global_to_local(value, value_layout)
    attn_mask_local = global_to_local(attn_mask, attn_mask_layout)

    parallel_net = ParallelFlashAttention(head_num=n,
                                          scalar_value=scalar_value,
                                          input_layout=input_layout,
                                          sparse_mode=sparse_mode)
    stra = {"forward": {"input": (query_layout, key_layout, value_layout, attn_mask_layout),
                        "output": (query_layout,)}}
    shard(parallel_net, stra)
    parallel_output = parallel_net(query_local, key_local, value_local, attn_mask_local)

    # Validate
    parallel_output = local_to_global(parallel_output)
    assert np.allclose(standalone_output.astype(mstype.float32).asnumpy(),
                       parallel_output.astype(mstype.float32).asnumpy(),
                       1e-3, 1e-3)
