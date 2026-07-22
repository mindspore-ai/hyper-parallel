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
"""msrun launcher for Pipeline Parallel + fully_shard activation-swap tests."""
from contextlib import contextmanager
import os
import tempfile
import uuid

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import MindSporeCase, parallel_run

_TEST_FILE = os.path.join(os.path.dirname(__file__), "_test_pp_composite.py")
_BASELINE_FILE_PREFIX = os.path.join(tempfile.gettempdir(), "pp_swap_composite")


@contextmanager
def _updated_env(env_updates: dict):
    """Temporarily apply environment overrides for one distributed phase."""
    old_env = {key: os.environ.get(key) for key in env_updates}
    try:
        os.environ.update(env_updates)
        yield
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _cleanup_baselines(tag: str, case_name: str, worker_num: int) -> None:
    """Remove the temporary no-swap records for one paired scenario."""
    spec_name = {
        "test_fully_shard_pp_gpipe": "pp_fsdp_gpipe",
        "test_fully_shard_pp_vpp": "pp_fsdp_vpp",
    }[case_name]
    for rank in range(worker_num):
        path = f"{_BASELINE_FILE_PREFIX}_{tag}_{spec_name}_rank{rank}.pkl"
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def _run_fsdp_swap_group() -> None:
    """Run four-card GPipe and VPP cases in parallel for each swap phase."""
    tag = uuid.uuid4().hex
    worker_num = 4
    cases = (
        ("test_fully_shard_pp_gpipe", 18301),
        ("test_fully_shard_pp_vpp", 18303),
    )
    common_env = {
        "PP_SWAP_COMPOSITE_BASELINE_TAG": tag,
        "PP_SWAP_COMPOSITE_ACTIVATION_MB": os.environ.get(
            "PP_SWAP_COMPOSITE_TEST_ACTIVATION_MB", "16"
        ),
        "PP_SWAP_COMPOSITE_MICROBATCHES": "4",
    }
    try:
        for case_name, _ in cases:
            _cleanup_baselines(tag, case_name, worker_num)
        with _updated_env({
            **common_env,
            "PP_SWAP_COMPOSITE_ENABLE_SWAP": "0",
        }):
            parallel_run(
                [
                    MindSporeCase(_TEST_FILE, case_name, master_port, worker_num, worker_num)
                    for case_name, master_port in cases
                ],
            )
        with _updated_env({
            **common_env,
            "PP_SWAP_COMPOSITE_ENABLE_SWAP": "1",
        }):
            parallel_run(
                [
                    MindSporeCase(_TEST_FILE, case_name, master_port + 100, worker_num, worker_num)
                    for case_name, master_port in cases
                ],
            )
    finally:
        for case_name, _ in cases:
            _cleanup_baselines(tag, case_name, worker_num)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_pp_plus_fully_shard_swap():
    """
    Feature: Pipeline swap composed with fully_shard under GPipe and VPP.
    Description: Run the existing four-card GPipe and Interleaved 1F1B
                 composite workers concurrently. The no-swap phase runs first,
                 followed by the swap phase. Both retain five-step serial
                 loss and gradient-shard checks and compare peak memory.
    Expectation: Losses and gradients match both references and swap reduces
                 peak memory on every rank with a swap window.
    """
    _run_fsdp_swap_group()
