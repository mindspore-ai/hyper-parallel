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
"""Pytest launcher for PP optimizer state dict tests.

Spawns torchrun workers:
  - P1, P2: 2-card PP-only tests (run in parallel, 2+2=4 <= 8)
  - P3-P7: 8-card PP+HSDP tests (one per group, each needs all 8 cards)
"""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.parallel_case import parallel_run, TorchCase
from tests.torch.utils import torchrun_case

_WORKER = str(Path(__file__).resolve().parent / "_test_pp_optim_state_dict.py")


# -----------------------------------------------------------------------
# PP-only tests (2 cards each, run in parallel: 2+2=4 <= 8)
# -----------------------------------------------------------------------
@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_pp_only_optim_state_dict_group1():
    """
    Feature: get/set_optim_state_dict for PP (no FSDP)
    Description:
        1.test_p1_pp_only_optim_state_dict_fqn_roundtrip
        2.test_p2_pp_only_cpu_offload_roundtrip
    Expectation: Run success.
    """
    parallel_run([
        TorchCase(_WORKER, "test_p1_pp_only_optim_state_dict_fqn_roundtrip", 13840, 2),
        TorchCase(_WORKER, "test_p2_pp_only_cpu_offload_roundtrip", 13841, 2),
    ])


# -----------------------------------------------------------------------
# PP+HSDP tests (8 cards each — one per group)
# -----------------------------------------------------------------------
@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_pp_hsdp_optim_state_dict_p3():
    """
    Feature: get/set_optim_state_dict FQN roundtrip for PP+HSDP
    Description:
        1.test_p3_pp_hsdp_optim_state_dict_fqn_roundtrip
    Expectation: Run success.
    """
    torchrun_case(
        _WORKER,
        "test_p3_pp_hsdp_optim_state_dict_fqn_roundtrip",
        master_port=13842,
        num_proc=8,
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_pp_hsdp_optim_state_dict_p4():
    """
    Feature: full_state_dict + cpu_offload restore to correct device for PP+HSDP
    Description:
        1.test_p4_pp_hsdp_full_cpu_restore_to_device
    Expectation: Run success.
    """
    torchrun_case(
        _WORKER,
        "test_p4_pp_hsdp_full_cpu_restore_to_device",
        master_port=13843,
        num_proc=8,
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_pp_hsdp_optim_state_dict_p5():
    """
    Feature: local shape correctness for PP+HSDP
    Description:
        1.test_p5_pp_hsdp_local_shape_correctness
    Expectation: Run success.
    """
    torchrun_case(
        _WORKER,
        "test_p5_pp_hsdp_local_shape_correctness",
        master_port=13844,
        num_proc=8,
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_pp_hsdp_optim_state_dict_p6():
    """
    Feature: DCP save/load with optim state dict under PP+HSDP
    Description:
        1.test_p6_pp_hsdp_dcp_save_load_nested
    Expectation: Run success.
    """
    torchrun_case(
        _WORKER,
        "test_p6_pp_hsdp_dcp_save_load_nested",
        master_port=13845,
        num_proc=8,
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards",
    essential_mark="essential",
)
def test_pp_hsdp_optim_state_dict_p7():
    """
    Feature: flatten roundtrip for PP+HSDP
    Description:
        1.test_p7_pp_hsdp_flatten_roundtrip
    Expectation: Run success.
    """
    torchrun_case(
        _WORKER,
        "test_p7_pp_hsdp_flatten_roundtrip",
        master_port=13846,
        num_proc=8,
    )
