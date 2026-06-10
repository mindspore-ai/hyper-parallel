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
"""msrun launcher for MindSpore 1F1B activation swap memory comparison."""
from contextlib import contextmanager
import os
from typing import Optional

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import MindSporeCase, parallel_run

_TEST_FILE = os.path.join(os.path.dirname(__file__), "_test_pp_swap_1f1b.py")
_MEMORY_ACTIVATION_MB = os.environ.get("PP_SWAP_1F1B_TEST_ACTIVATION_MB", "200")
_ACCURACY_TOKENS_PER_MICRO = os.environ.get("PP_SWAP_1F1B_ACCURACY_TOKENS_PER_MICRO", "256")


@contextmanager
def _updated_env(env_updates: dict[str, Optional[str]]):
    """Temporarily apply environment overrides for one test phase."""
    old_env = {key: os.environ.get(key) for key in env_updates}
    try:
        for key, value in env_updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1",
          card_mark="allcards", essential_mark="essential")
def test_1f1b_pipeline_swap():
    """
    Feature: 1F1B activation swap device memory comparison.
    Description: Run a four-rank 1F1B schedule (8 micro-batches, 2048 hidden,
        4-layer DeepStage) with 200 MB activation per layer by default. First
        record baseline device peak memory without swap, then wrap every rank's
        stage for swap and compare device peak memory. Run a smaller accuracy
        pass separately so the serial reference does not pollute peak memory.
    Expectation: ranks with swap windows reduce device peak memory, ranks
        without swap windows do not regress, and correctness check passes.
    """
    memory_env = {"PP_SWAP_ACTIVATION_MB": _MEMORY_ACTIVATION_MB}
    accuracy_env = {
        "PP_SWAP_ACTIVATION_MB": None,
        "PP_SWAP_TOKENS_PER_MICRO": _ACCURACY_TOKENS_PER_MICRO,
    }

    with _updated_env(memory_env):
        parallel_run([
            MindSporeCase(_TEST_FILE, "test_1f1b_no_swap", worker_num=4, local_worker_num=4),
        ])
        parallel_run([
            MindSporeCase(_TEST_FILE, "test_1f1b_swap_memory", worker_num=4, local_worker_num=4),
        ])
    with _updated_env(accuracy_env):
        parallel_run([
            MindSporeCase(_TEST_FILE, "test_1f1b_swap_accuracy", worker_num=4, local_worker_num=4),
        ])
