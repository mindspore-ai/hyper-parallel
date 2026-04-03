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
"""Pytest entry-points for ``parallelize_module`` NPU distributed tests.

Each test spawns worker processes via torchrun and delegates to the
corresponding test function in parallelize_module_distributed.py (NPU/hccl).

Run from ``tests/torch/tensor_parallel/`` so the worker module path resolves (same
pattern as ``tests/torch/context_parallel/test_cp_npu.py``).

Port allocation:
  10460–10469  8-card tests (single-node ``num_proc=8``)
"""
from tests.torch.utils import torchrun_case
from tests.common.mark_utils import arg_mark

_FILE = "parallelize_module_distributed.py"


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_parallelize_module_mesh_aligned_with_process_group_npu():
    """Mesh rank list matches hccl world size (8-rank).

    Feature: parallelize_module under distributed
    Description: Style apply sees a 1-D mesh consistent with process group.
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_parallelize_module_mesh_aligned_with_process_group_npu", master_port=10460, num_proc=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_parallelize_module_dict_fnmatch_npu():
    """Dict plan with fnmatch on 8 ranks.

    Feature: parallelize_module path patterns
    Description: ``net*`` matches multiple children on each rank.
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_parallelize_module_dict_fnmatch_npu", master_port=10461, num_proc=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_parallelize_module_src_data_rank_npu():
    """src_data_rank propagated to ParallelStyle (8-rank).

    Feature: parallelize_module keyword args
    Description: ``src_data_rank`` set on style before apply.
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_parallelize_module_src_data_rank_npu", master_port=10462, num_proc=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_parallelize_module_single_style_root_npu():
    """Single ParallelStyle on root module (8-rank).

    Feature: parallelize_module single-style form
    Description: One apply per rank on the root module.
    Expectation: Run success.
    """
    torchrun_case(_FILE, "test_parallelize_module_single_style_root_npu", master_port=10463, num_proc=8)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_parallelize_module_colwise_linear_precision_vs_pytorch_ref_npu():
    """Colwise-style linear vs PyTorch TP reference (8-rank).

    Feature: parallelize_module numerical parity
    Description: Compare sharded linear forward to reference.
    Expectation: Run success.
    """
    torchrun_case(
        _FILE,
        "test_parallelize_module_colwise_linear_precision_vs_pytorch_ref_npu",
        master_port=10464,
        num_proc=8,
    )


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
def test_parallelize_module_rowwise_linear_precision_vs_pytorch_ref_npu():
    """Rowwise-style linear vs PyTorch TP reference (8-rank).

    Feature: parallelize_module numerical parity
    Description: Compare sharded linear forward to reference.
    Expectation: Run success.
    """
    torchrun_case(
        _FILE,
        "test_parallelize_module_rowwise_linear_precision_vs_pytorch_ref_npu",
        master_port=10465,
        num_proc=8,
    )
