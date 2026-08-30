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
"""launch _test_fully_shard_auto_grad.py cases (MindSpore)"""
import os

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import MindSporeCase, parallel_run

_TEST_FILE = os.path.join(os.path.dirname(__file__), "_test_fully_shard_auto_grad.py")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_fully_shard_autograd_01():
    """
    Feature: fully_shard autograd (MindSpore)
    Description: Run two fully_shard autograd cases concurrently over an 8-card budget --
        the first on the front 4 cards, the second on the back 4 cards:
        - test_chunked_output_fully_shard: a fully_shard-wrapped OutputLayer called multiple
          times in a for-loop with results concatenated and a single backward.
        - test_input_requires_grad_fully_shard_grad_parity: fully_shard wraps only inner
          blocks (root left unwrapped) and the wrapped block is fed an input that requires
          grad, so its PostBackwardFunction drives scheduler_state==BACKWARD -- the boundary
          case where the root backward hook must still drain the last reduce-scatter. Each
          block mixes a sharded Dense (reduce-scatter path) and a replicate_params sub-layer
          (all-reduce path); grads are compared against a single-card standalone baseline at
          an equivalent global batch size (sharded all-gathered, replicate compared directly).
    Expectation: both cases run success; all grads non-None and matching the baseline.
    """
    parallel_run([
        MindSporeCase(_TEST_FILE, "test_chunked_output_fully_shard", worker_num=4, local_worker_num=4),
        MindSporeCase(_TEST_FILE, "test_input_requires_grad_fully_shard_grad_parity",
                      worker_num=4, local_worker_num=4),
    ])


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_fully_shard_size_one_autograd():
    """
    Feature: size-one fully_shard autograd with an unused forward output.
    Description: Run the MindSpore port of PyTorch's ``test_unused_forward_output``
        on a one-rank FSDP mesh for ten optimizer iterations.
    Expectation: Losses, gradients, and parameters match standalone while the
        communication storage and logical parameters retain their leaf invariants.
    """
    parallel_run([
        MindSporeCase(
            _TEST_FILE,
            "test_single_rank_unused_forward_output_autograd",
            worker_num=1,
            local_worker_num=1,
        ),
    ], global_num_proc=1)
