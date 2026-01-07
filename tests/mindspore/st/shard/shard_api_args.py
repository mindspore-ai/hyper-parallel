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
"""Test hyper_parallel.shard with args and kwargs"""

import numpy as np
import mindspore.communication.management as D
from mindspore import nn, Tensor
from hyper_parallel import Layout, shard, DTensor

class NetTestInput(nn.Cell):
    """
        Net for testing hyper_parallel.shard processing args and kwargs inputs
    """
    def construct(self, x, y, z, use_residual_add=False, return_residual_sign=True):
        temp = x + y
        result = temp * z
        if use_residual_add:
            result = x + result
        return result, use_residual_add if return_residual_sign else result


def test_shard_with_args_and_kwargs_non_dtensor_input():
    '''
    Feature: Test hyper_parallel.shard with non-dtensor inputs as both args and kwargs
    Description: Test hyper_parallel.shard with non-dtensor inputs as both args and kwargs
    Expectation: Run success
    '''
    D.init()
    mesh = Layout(mesh_shape=(2, 4), alias_name=("dp", "tp"))
    model = NetTestInput()
    input_layouts = (mesh("dp", "tp"), mesh("dp", "tp"), mesh("dp", "tp"), mesh(), mesh())
    output_layouts = (mesh("dp", "tp"), mesh())
    sharding_plan = {"forward": {"input": input_layouts, "output": output_layouts}}
    shard(model, sharding_plan)
    x = Tensor(np.ones((4, 8)))
    y = Tensor(np.zeros((4, 8)))
    z = Tensor(np.arange(32).reshape(4, 8))

    # All inputs all Non-DTensor, and has multiple outputs
    # test inputs for both args and kwargs
    result, _ = model(x, y, z=z, use_residual_add=True, return_residual_sign=True)
    assert not isinstance(result, DTensor)

    # All inputs all None-DTensor, and has single output
    # test inputs for both args and kwargs
    result = model(x, y, z=z, use_residual_add=True, return_residual_sign=False)
    assert not isinstance(result, DTensor)
