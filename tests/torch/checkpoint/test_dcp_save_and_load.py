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
"""test checkpoint DCP save and load"""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

DCP_SAVE_AND_LOAD = "dcp_save_and_load.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dcp_save_and_load_group1():
    """
    Feature: parallel run case in checkpoint
    Description:
        1.test_dcp_save_and_load_with_dtensor_and_tensor_and_scalar
        2.test_dcp_async_save_and_load_with_dtensor_and_tensor_and_scalar
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DCP_SAVE_AND_LOAD, "test_dcp_save_and_load_with_dtensor_and_tensor_and_scalar", 12253, 4),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_dcp_save_and_load_group1_gloo():
    """
    Feature: parallel run case in checkpoint
    Description:
        1.test_dcp_save_and_load_with_dtensor_and_tensor_and_scalar
        2.test_dcp_async_save_and_load_with_dtensor_and_tensor_and_scalar
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DCP_SAVE_AND_LOAD, "test_dcp_save_and_load_with_dtensor_and_tensor_and_scalar", num_proc=4),
        TorchCase(DCP_SAVE_AND_LOAD, "test_dcp_async_save_and_load_with_dtensor_and_tensor_and_scalar", num_proc=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dcp_save_and_load_group2():
    """
    Feature: parallel run case in checkpoint
    Description:
        1.test_dcp_save_and_load_with_full_tensor
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DCP_SAVE_AND_LOAD, "test_dcp_save_and_load_with_full_tensor", 12255, 2),
        TorchCase(DCP_SAVE_AND_LOAD, "test_dcp_async_save_and_load_with_dtensor_and_tensor_and_scalar", 12256, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level3", card_mark="allcards", essential_mark="essential")
def test_dcp_save_and_load_group3():
    """
    Feature: parallel run case in checkpoint
    Description:
        1.test_dcp_save_and_load_save_8card_load_4card
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DCP_SAVE_AND_LOAD, "test_dcp_save_and_load_save_8card_load_4card", 12254, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dcp_save_and_load_group4():
    """
    Feature: parallel run case in checkpoint
    Description:
        1.test_dcp_save_and_load_with_static_dp_tp_pp
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DCP_SAVE_AND_LOAD, "test_dcp_save_and_load_with_static_dp_tp_pp", 12257, 8),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dcp_save_and_load_group4_gloo():
    """
    Feature: parallel run case in checkpoint (gloo CPU backend)
    Description:
        1.test_dcp_save_and_load_with_static_dp_tp_pp
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(DCP_SAVE_AND_LOAD, "test_dcp_save_and_load_with_static_dp_tp_pp", num_proc=8),
    ])




@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dcp_save_and_load_dynamic_tp_pp():
    """
    Feature: DCP save and load with dynamic TP+PP topology change.
    Description: Save on PP2×TP4 and load on PP4×TP2 using TopologyMapper.
    Expectation: Run success, loaded values match expected TP2 shards.
    """
    parallel_run([
        TorchCase(DCP_SAVE_AND_LOAD, "test_dcp_save_and_load_dynamic_tp_pp", 12258, 8),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dcp_save_and_load_dynamic_tp_pp_gloo():
    """
    Feature: DCP save and load with dynamic TP+PP topology change (gloo CPU backend).
    Description: Save on PP2×TP4 and load on PP4×TP2 using TopologyMapper.
    Expectation: Run success, loaded values match expected TP2 shards.
    """
    parallel_run([
        TorchCase(DCP_SAVE_AND_LOAD, "test_dcp_save_and_load_dynamic_tp_pp", num_proc=8),
    ])



@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dcp_save_and_load_hsdp_ep_moe():
    """
    Feature: DCP save and load with HSDP + EP (MoE) topology change.
    Description: Save on HSDP(rep=2,shard=2)×EP=2 and load on EP=4×TP=2
        using TopologyMapper with PP FQN mapping.
    Expectation: Run success, HSDP dedup correct, loaded expert/router/buffer
        values match expected slices.
    """
    parallel_run([
        TorchCase(DCP_SAVE_AND_LOAD, "test_dcp_save_and_load_hsdp_ep_moe", 12259, 8),
    ])


@arg_mark(plat_marks=["cpu_linux"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dcp_save_and_load_hsdp_ep_moe_gloo():
    """
    Feature: DCP save and load with HSDP + EP (MoE) topology change (gloo CPU backend).
    Description: Save on HSDP(rep=2,shard=2)×EP=2 and load on EP=4×TP=2
        using TopologyMapper with PP FQN mapping.
    Expectation: Run success, HSDP dedup correct, loaded expert/router/buffer
        values match expected slices.
    """
    parallel_run([
        TorchCase(DCP_SAVE_AND_LOAD, "test_dcp_save_and_load_hsdp_ep_moe", num_proc=8),
    ])
