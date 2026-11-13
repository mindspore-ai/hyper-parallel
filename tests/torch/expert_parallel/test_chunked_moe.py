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
"""Thin NPU launchers for complete Chunk MoE training gradients."""
from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark

_WORKER = str(Path(__file__).with_name("_test_chunked_moe.py"))


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
          essential_mark="essential")
def test_chunked_moe_single():
    """
    Feature: chunked moe single
    Description: Validate the fused expert backward on one NPU.
    Expectation: The documented output, gradient and state assertions pass.
    """
    torchrun_case(_WORKER, "test_chunked_moe_single", num_proc=1)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards",
          essential_mark="essential")
def test_chunked_moe_ep4():
    """
    Feature: chunked moe ep4
    Description: Validate four expert owners including empty receiving ranks.
    Expectation: The documented output, gradient and state assertions pass.
    """
    torchrun_case(_WORKER, "test_chunked_moe_ep4", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards",
          essential_mark="essential")
def test_chunked_moe_ep2_replicas():
    """
    Feature: chunked moe ep2 replicas
    Description: Validate expert replica gradients independently of dense gradients.
    Expectation: The documented output, gradient and state assertions pass.
    """
    torchrun_case(_WORKER, "test_chunked_moe_ep2_replicas", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards",
          essential_mark="essential")
def test_score_after_ep4():
    """
    Feature: score after ep4
    Description: Regress score-after probability dispatch for the unchanged whole-input path.
    Expectation: The documented output, gradient and state assertions pass.
    """
    torchrun_case(_WORKER, "test_score_after_ep4", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards",
          essential_mark="essential")
def test_chunked_moe_fsdp2_ep2():
    """
    Feature: chunked moe fsdp2 ep2
    Description: Validate parent FSDP reshard, EP ownership and repeated optimizer updates.
    Expectation: The documented output, gradient and state assertions pass.
    """
    torchrun_case(_WORKER, "test_chunked_moe_fsdp2_ep2", num_proc=4)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard",
          essential_mark="essential")
def test_receive_indices_npu():
    """
    Feature: NPU receive permutation
    Description: Cover direct-map selection, short tails, empty segments and identity maps.
    Expectation: Integer maps and row gradients agree exactly with independent references.
    """
    torchrun_case(_WORKER, "test_receive_indices_npu", num_proc=1)
