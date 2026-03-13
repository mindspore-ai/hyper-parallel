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
"""test clip_grad_norm_ API (torchrun runner)"""

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase

_TEST_CLIP_GRAD = "_test_clip_grad.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_clip_grad_norm_comprehensive():
    """
    Feature: parallel run case in clip_grad
    Description:
        1.test_clip_grad_norm_comprehensive
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_CLIP_GRAD, "test_clip_grad_norm_comprehensive", 12360, 8)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_clip_grad_norm_partial_shard():
    """
    Feature: parallel run case in clip_grad
    Description:
        1.test_clip_grad_norm_partial_shard
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_CLIP_GRAD, "test_clip_grad_norm_partial_shard", 12361, 8)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_clip_grad_norm_edge_cases():
    """
    Feature: parallel run case in clip_grad
    Description:
        1.test_clip_grad_norm_edge_cases
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_CLIP_GRAD, "test_clip_grad_norm_edge_cases", 12362, 1)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_clip_grad_norm_empty_grads():
    """
    Feature: parallel run case in clip_grad
    Description:
        1.test_clip_grad_norm_empty_grads
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_CLIP_GRAD, "test_clip_grad_norm_empty_grads", 12363, 8)
    ])
