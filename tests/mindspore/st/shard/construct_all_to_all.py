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
"""test construct_all_to_all"""
import numpy as np
import mindspore as ms
import mindspore.communication.management as D
from hyper_parallel.core.tensor_redistribution import TensorRedistribution

D.init()

tensor_redis = TensorRedistribution()
# pylint: disable=W0212
construct_all_to_all = tensor_redis._construct_all_to_all


def cal_expect_output(data, split_count, args):
    """cal_expect_output"""
    rank = D.get_rank()
    index = args[3].index(rank)
    split_size = data.shape[args[0]] // split_count
    split_data = data.split(split_size, args[0])[index]
    concat_data_list = [split_data for _ in args[3]]
    expect_output = ms.mint.concat(concat_data_list, args[1])
    return expect_output


def test_construct_all_to_all():
    """
    Feature: test construct_all_to_all communication function.
    Description: test construct_all_to_all communication function.
    Expectation: expect correct result.
    """
    data = ms.Tensor(np.arange(64).reshape([4, 4, 4]).astype(np.float16))
    args1 = (0, 0, 2, [0, 1])
    args2 = (0, 1, 2, [0, 1])
    args3 = (1, 0, 2, [0, 1])
    args4 = (1, 1, 2, [0, 1])
    args5 = (1, 2, 2, [0, 1])
    out1 = construct_all_to_all(data, *args1)
    assert np.all(out1.asnumpy() == cal_expect_output(data, 2, args1).asnumpy())
    out2 = construct_all_to_all(data, *args2)
    assert np.all(out2.asnumpy() == cal_expect_output(data, 2, args2).asnumpy())
    out3 = construct_all_to_all(data, *args3)
    assert np.all(out3.asnumpy() == cal_expect_output(data, 2, args3).asnumpy())
    out4 = construct_all_to_all(data, *args4)
    assert np.all(out4.asnumpy() == cal_expect_output(data, 2, args4).asnumpy())
    out5 = construct_all_to_all(data, *args5)
    assert np.all(out5.asnumpy() == cal_expect_output(data, 2, args5).asnumpy())


def test_construct_all_to_all_8p():
    """
    Feature: test construct_all_to_all communication function.
    Description: test construct_all_to_all communication function.
    Expectation: expect correct result.
    """
    data = ms.Tensor(np.arange(64 * 64).reshape([16, 16, 16]).astype(np.float16))
    args1 = (0, 0, 8, [0, 1, 2, 3, 4, 5, 6, 7])
    args2 = (0, 1, 8, [0, 1, 2, 3, 4, 5, 6, 7])
    args3 = (1, 0, 8, [0, 1, 2, 3, 4, 5, 6, 7])
    args4 = (1, 1, 8, [0, 1, 2, 3, 4, 5, 6, 7])
    args5 = (1, 2, 8, [0, 1, 2, 3, 4, 5, 6, 7])
    args6 = (2, 1, 8, [0, 1, 2, 3, 4, 5, 6, 7])
    out1 = construct_all_to_all(data, *args1)
    assert np.all(out1.asnumpy() == cal_expect_output(data, 8, args1).asnumpy())
    out2 = construct_all_to_all(data, *args2)
    assert np.all(out2.asnumpy() == cal_expect_output(data, 8, args2).asnumpy())
    out3 = construct_all_to_all(data, *args3)
    assert np.all(out3.asnumpy() == cal_expect_output(data, 8, args3).asnumpy())
    out4 = construct_all_to_all(data, *args4)
    assert np.all(out4.asnumpy() == cal_expect_output(data, 8, args4).asnumpy())
    out5 = construct_all_to_all(data, *args5)
    assert np.all(out5.asnumpy() == cal_expect_output(data, 8, args5).asnumpy())
    out6 = construct_all_to_all(data, *args6)
    assert np.all(out6.asnumpy() == cal_expect_output(data, 8, args6).asnumpy())
