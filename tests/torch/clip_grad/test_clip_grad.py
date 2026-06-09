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
def test_clip_grad_norm_training_5step():
    """
    Feature: FSDP2-aligned clip_grad_norm_ precision
    Description:
        5-step HSDP training loop. At each step, verify clipped grads match
        torch.nn.utils reference on full gradients. Guarantees loss alignment.
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_CLIP_GRAD, "test_clip_grad_norm_training_5step", 12360, 8)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_clip_grad_norm_main_grad():
    """
    Feature: main_grad support for clip_grad_norm_
    Description:
        Verify clip_grad_norm_ reads param.main_grad when param.grad is None.
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_CLIP_GRAD, "test_clip_grad_norm_main_grad", 12364, 8)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_clip_grad_norm_frozen_params():
    """
    Feature: frozen param handling for clip_grad_norm_
    Description:
        Verify clip_grad_norm_ correctly skips frozen params.
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_CLIP_GRAD, "test_clip_grad_norm_frozen_params", 12365, 8)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_clip_grad_norm_replicate_params():
    """
    Feature: clip_grad_norm_ with mixed FSDP-sharded + replicate_params
    Description:
        Replicate-grad norms must not be all-reduced over the shard group.
        Otherwise replicate norm² is multiplied by shard_world_size,
        inflating the reported global norm.
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_CLIP_GRAD, "test_clip_grad_norm_replicate_params", 12367, 8)
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_clip_grad_norm_multi_group():
    """
    Feature: multi-grad-group parameter ordering stability
    Description:
        8-layer model wrapped per-layer with fully_shard creates multiple
        grad_groups. Verify norm stacking uses original parameter order
        (not grad_group iteration order) to avoid float32 non-associative
        ULP diffs on non-rank-0 ranks.
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_TEST_CLIP_GRAD, "test_clip_grad_norm_multi_group", 12366, 8)
    ])
