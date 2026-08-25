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
"""launch _test_fully_shard_auto_grad.py cases"""
import os

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

_TEST_FULLY_SHARD_AUTO_GRAD = os.path.join(os.path.dirname(__file__), "_test_fully_shard_auto_grad.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_fully_shard_auto_grad():
    """
    Feature: parallel run case in fully_shard auto grad
    Description:
        Run chunked output autograd, single-rank FSDP and HSDP standalone
        parity, and four-rank HSDP forward-forward-backward-backward parity.
    Expectation: Every case completes with matching outputs and gradients.
    """
    # Independent multi-rank HCCL jobs may bind the same NPU NIC socket even
    # when parallel_run assigns disjoint devices, so run them in separate waves.
    parallel_run([
        TorchCase(_TEST_FULLY_SHARD_AUTO_GRAD, "test_chunked_output_fully_shard", 12501, 2),
        TorchCase(_TEST_FULLY_SHARD_AUTO_GRAD, "test_single_rank_fsdp_autograd_parity", 12502, 1),
        TorchCase(_TEST_FULLY_SHARD_AUTO_GRAD, "test_single_rank_hsdp_autograd_parity", 12503, 1),
    ])
    parallel_run([
        TorchCase(_TEST_FULLY_SHARD_AUTO_GRAD, "test_hsdp_ffbb_autograd_parity", 12504, 4),
    ])
