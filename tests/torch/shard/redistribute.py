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
"""test torch redistribute"""
import numpy as np
import torch
import pytest
# pylint: disable=W0611
import torch_npu
from hyper_parallel import DTensor, Layout
from tests.torch.utils import init_dist


def test_shard_to_replicate():
    '''
    Feature: shard to replicate.
    Description:
    Expectation: Run success.
    '''
    init_dist()

    layout = Layout((1, 8), ("dp", "tp"))
    x_layout = layout("None", "tp")
    dst_layout = layout("None", "None")
    dist_x = DTensor.from_local(torch.ones(8, 1).npu(), x_layout)
    out = dist_x.redistribute(dst_layout)

    expect_out = torch.ones(8, 8)

    assert np.allclose(expect_out.cpu().detach().numpy(),
                       out.to_local().cpu().detach().numpy(),  # use to_local()
                       0.001, 0.001)


def test_replicate_to_shard():
    '''
    Feature: replicate to shard.
    Description:
    Expectation: Run success.
    '''
    init_dist()

    layout = Layout((1, 8), ("dp", "tp"))
    dst_layout = layout("None", "tp")
    x_layout = layout("None", "None")
    dist_x = DTensor.from_local(torch.ones(8, 8).npu(), x_layout)
    out = dist_x.redistribute(dst_layout)

    expect_out = torch.ones(8, 1)

    assert np.allclose(expect_out.cpu().detach().numpy(),
                       out.to_local().cpu().detach().numpy(),  # use to_local()
                       0.001, 0.001)


def test_different_mesh():
    '''
    Feature: different mesh, unsupported.
    Description:
    Expectation: Run success.
    '''
    init_dist()

    layout = Layout((1, 8, 1), ("dp", "tp", "sp"))
    dst_layout = layout("None", "tp")

    layout_1 = Layout((1, 8), ("dp", "tp"))
    x_layout = layout_1("None", "None")
    dist_x = DTensor.from_local(torch.ones(8, 8).npu(), x_layout)
    with pytest.raises(RuntimeError):
        dist_x.redistribute(dst_layout)
