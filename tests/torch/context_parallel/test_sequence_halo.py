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
"""Thin CP8 launchers for migrated halo and CSA regressions."""

from pathlib import Path

from tests.common.distributed_launcher import torchrun_case
from tests.common.mark_utils import arg_mark


_ROOT = Path(__file__).resolve().parent


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_sequence_halo_gloo():
    """Eight ranks verify variable splits, repeated owners and gradients."""
    torchrun_case(str(_ROOT / "_test_sequence_halo.py"), "test_halo_and_gather_adjoint_gloo",
                  master_port=14983, num_proc=8)


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_csa_halo_gloo():
    """Eight ranks compare CP1, AG and halo for mixed CSA shared chains."""
    torchrun_case(str(_ROOT / "_test_csa_halo.py"), "test_csa_halo_shared_chain_gloo",
                  master_port=14984, num_proc=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_sequence_halo_hccl():
    """Eight NPUs verify empty/variable splits and exact returned gradients."""
    torchrun_case(str(_ROOT / "_test_sequence_halo.py"), "test_halo_and_gather_adjoint_hccl",
                  master_port=14985, num_proc=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_csa_halo_hccl():
    """Eight NPUs exercise the combined halo, native Indexer/KL and Omni attention."""
    torchrun_case(str(_ROOT / "_test_csa_halo.py"), "test_csa_halo_shared_chain_hccl",
                  master_port=14986, num_proc=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_qwen_sum_reduce_scatter_hccl():
    """The shared collective preserves the other asynchronous attention caller."""
    torchrun_case(str(_ROOT / "_test_qwen_rs.py"), "test_qwen_async_rs_cp8",
                  master_port=14987, num_proc=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_v41_training_step_hccl():
    """Compare the combined six-layer training path across packed-boundary updates."""
    torchrun_case(str(_ROOT / "_test_v41_training.py"), "test_v41_training_step_cp8",
                  master_port=14988, num_proc=8)
