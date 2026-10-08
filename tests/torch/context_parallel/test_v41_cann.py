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
"""Lightweight single-NPU launchers for the CSA CANN adapters."""

from importlib import import_module

from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_li_ratio_packed_and_nonlast_cp_rank():
    """Select packed compressed keys with CP-local queries."""
    import_module("tests.torch.context_parallel._test_v41_cann").test_li_ratio_packed_and_nonlast_cp_rank()


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_kl_fused_and_partial_rows_share_the_same_global_mean():
    """Compare native and reference KL including padding and outer gradient seed."""
    worker = import_module("tests.torch.context_parallel._test_v41_cann")
    worker.test_kl_fused_and_partial_rows_share_the_same_global_mean()
