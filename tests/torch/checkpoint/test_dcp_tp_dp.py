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
"""Launch _test_dcp_with_fully_shard_optimizer.py (same pattern as parallel_run + TorchCase under tests/torch/fully_shard)."""
from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

_TEST_DCP_TP_DP = "_test_dcp_tp_dp.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_dcp_with_optimizer_tp_dp():
    """
    Feature: fully_shard + DCP + flatten_state_dict + bytes (multi-GPU).

    Same style as test_fully_shard_auto_grad.
    Description: test_dcp_with_optimizer_tp_dp
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_DCP_TP_DP, "test_dcp_with_optimizer_tp_dp", 12502, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_dcp_tp_fsdp_model_state_roundtrip():
    """
    Feature: TP + FSDP + DCP sync save/load round-trip for model state only.

    Description: test_dcp_tp_fsdp_model_state_roundtrip
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_DCP_TP_DP, "test_dcp_tp_fsdp_model_state_roundtrip", 12504, 4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_dcp_async_save_with_optimizer_tp_dp():
    """
    Feature: fully_shard + DCP + flatten_state_dict + bytes (multi-GPU).

    Same style as test_fully_shard_auto_grad.
    Description: test_dcp_async_save_with_optimizer_tp_dp
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_DCP_TP_DP, "test_dcp_async_save_with_optimizer_tp_dp", 12503, 4),
    ])
