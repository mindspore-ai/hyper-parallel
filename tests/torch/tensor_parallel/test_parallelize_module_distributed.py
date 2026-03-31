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
"""Pytest entry points: ``parallelize_module`` under torchrun (2-rank NPU).

Worker: ``parallelize_module_distributed.py`` in this directory.
Run from ``tests/torch/tensor_parallel/`` so ``torchrun_case`` resolves the worker file.
"""
from tests.common.mark_utils import arg_mark
from tests.torch.utils import torchrun_case

_FILE = "parallelize_module_distributed.py"
_MASTER_PORT_BASE = 10460


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_parallelize_module_returns_same_instance_npu():
    """
    Feature: parallelize_module returns same module instance on NPU
    Description: pytest spawns 2-rank torchrun to run worker case with verify style on Linear
    Expectation: worker asserts out is m; torchrun_case exit code 0
    """
    torchrun_case(
        _FILE,
        "test_parallelize_module_returns_same_instance_npu",
        master_port=_MASTER_PORT_BASE,
        num_proc=2,
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_parallelize_module_mesh_aligned_with_process_group_npu():
    """
    Feature: parallelize_module mesh aligns with HCCL process group on NPU
    Description: pytest spawns 2-rank torchrun to run worker mesh vs process group check
    Expectation: worker _VerifyMeshParallelStyle apply passes; torchrun_case exit code 0
    """
    torchrun_case(
        _FILE,
        "test_parallelize_module_mesh_aligned_with_process_group_npu",
        master_port=_MASTER_PORT_BASE + 1,
        num_proc=2,
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_parallelize_module_dict_fnmatch_npu():
    """
    Feature: parallelize_module dict plan with fnmatch wildcard on NPU
    Description: pytest spawns 2-rank torchrun to run worker net* plan on MLP
    Expectation: worker asserts style.count and per-child counters; torchrun_case exit code 0
    """
    torchrun_case(
        _FILE,
        "test_parallelize_module_dict_fnmatch_npu",
        master_port=_MASTER_PORT_BASE + 2,
        num_proc=2,
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_parallelize_module_src_data_rank_npu():
    """
    Feature: parallelize_module propagates src_data_rank on NPU
    Description: pytest spawns 2-rank torchrun to run worker src_data_rank 1 then None
    Expectation: worker asserts ranks on styles; torchrun_case exit code 0
    """
    torchrun_case(
        _FILE,
        "test_parallelize_module_src_data_rank_npu",
        master_port=_MASTER_PORT_BASE + 3,
        num_proc=2,
    )


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
def test_parallelize_module_single_style_root_npu():
    """
    Feature: parallelize_module single ParallelStyle on root module on NPU
    Description: pytest spawns 2-rank torchrun to run worker single CountingParallelStyle on Linear
    Expectation: worker asserts style.count==1 and root apply count; torchrun_case exit code 0
    """
    torchrun_case(
        _FILE,
        "test_parallelize_module_single_style_root_npu",
        master_port=_MASTER_PORT_BASE + 4,
        num_proc=2,
    )
