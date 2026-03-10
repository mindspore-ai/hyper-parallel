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
from tests.torch.utils import torchrun_case


@arg_mark(
    plat_marks=["platform_ascend910b"], level_mark="level0",
    card_mark="allcards", essential_mark="essential",
)
def test_clip_grad_norm_comprehensive():
    """
    Feature: clip_grad_norm_ comprehensive semantics + E2E SGD closed loop
    Description: Verify norm & clipped-grad parity across HSDP 2D (L2,
        Linf) and FSDP 1D (L2) meshes with model and parameters() APIs.
        Combo 1 additionally verifies post-SGD-step parameter parity.
    Expectation: run successfully
    """
    torchrun_case(
        "_test_clip_grad.py",
        "test_clip_grad_norm_comprehensive",
        12360,
    )


@arg_mark(
    plat_marks=["platform_ascend910b"], level_mark="level0",
    card_mark="allcards", essential_mark="essential",
)
def test_clip_grad_norm_partial_shard():
    """
    Feature: clip_grad_norm_ with TP+FSDP Partial + Shard placements
    Description: Manually construct DTensor params with [Partial(SUM),
        Shard(0)] on a (tp=2, dp=4) mesh. Verify total_norm and clipped
        grads match nn.utils reference on the full gradient, and all
        ranks agree on the norm.
    Expectation: run successfully
    """
    torchrun_case(
        "_test_clip_grad.py",
        "test_clip_grad_norm_partial_shard",
        12361,
    )


@arg_mark(
    plat_marks=["platform_ascend910b"], level_mark="level1",
    card_mark="allcards", essential_mark="essential",
)
def test_clip_grad_norm_edge_cases():
    """
    Feature: clip_grad_norm_ edge cases (single card)
    Description: Verify unusual norm_types (0/1/-inf), foreach variants
        (None/False/True), error_if_nonfinite with inf/nan injection,
        and mixed fp16/fp32 dtype promotion on plain non-DTensor params.
    Expectation: run successfully
    """
    torchrun_case(
        "_test_clip_grad.py",
        "test_clip_grad_norm_edge_cases",
        12362,
        num_proc=1,
    )


@arg_mark(
    plat_marks=["platform_ascend910b"], level_mark="level1",
    card_mark="allcards", essential_mark="essential",
)
def test_clip_grad_norm_empty_grads():
    """
    Feature: clip_grad_norm_ empty/sparse gradient handling
    Description: Verify no deadlock when all or some ranks have grad=None.
        All-None returns norm=0.0 in float32; sparse grads produce
        consistent finite norm across all ranks.  Includes Partial
        placement deadlock regression (asymmetric grad=None in tp group).
    Expectation: run successfully
    """
    torchrun_case(
        "_test_clip_grad.py",
        "test_clip_grad_norm_empty_grads",
        12363,
    )
