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
"""test redistribute"""
import numpy as np
import mindspore as ms
from mindspore import Tensor
import mindspore.communication.management as D
from hyper_parallel import DTensor, Layout


def test_shard_to_replicate():
    '''
    Feature: shard to replicate.
    Description:
    Expectation: Run success.
    '''
    D.init()

    layout = Layout((1, 8), ("dp", "tp"))
    x_layout = layout("None", "tp")
    dst_layout = layout("None", "None")
    x_slice = Tensor(np.ones([8, 1]), dtype=ms.float32)
    dist_x = DTensor.from_local(x_slice, x_layout)
    out = dist_x.redistribute(dst_layout)

    expect_out = Tensor(np.ones([8, 8]), dtype=ms.float32)

    assert np.allclose(expect_out.asnumpy(),
                       out.to_local().asnumpy(),  # use to_local()
                       0.001, 0.001)


def test_replicate_to_shard():
    '''
    Feature: replicate to shard.
    Description:
    Expectation: Run success.
    '''
    D.init()

    layout = Layout((1, 8), ("dp", "tp"))
    x_layout = layout("None", "None")
    dst_layout = layout("None", "tp")
    x_slice = Tensor(np.ones([8, 8]), dtype=ms.float32)
    dist_x = DTensor.from_local(x_slice, x_layout)
    out = dist_x.redistribute(dst_layout)

    expect_out = Tensor(np.ones([8, 1]), dtype=ms.float32)

    assert np.allclose(expect_out.asnumpy(),
                       out.to_local().asnumpy(),  # use to_local()
                       0.001, 0.001)


def test_different_mesh():
    '''
    Feature: different mesh.
    Description:
    Expectation: Run success.
    '''
    D.init()

    layout = Layout((1, 8, 1), ("dp", "tp", "sp"))
    dst_layout = layout("None", "tp")

    layout_1 = Layout((1, 8), ("dp", "tp"))
    x_layout = layout_1("None", "None")
    x_slice = Tensor(np.ones([8, 8]), dtype=ms.float32)
    dist_x = DTensor.from_local(x_slice, x_layout)
    out = dist_x.redistribute(dst_layout)

    expect_out = Tensor(np.ones([8, 1]), dtype=ms.float32)

    assert np.allclose(expect_out.asnumpy(),
                       out.to_local().asnumpy(),  # use to_local()
                       0.001, 0.001)
