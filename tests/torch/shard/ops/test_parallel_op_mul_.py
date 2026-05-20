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
"""test base dtensor mul_"""
from tests.torch.utils import torchrun_case
from tests.common.mark_utils import arg_mark

_IMPL_FILE = "parallel_op_mul_.py"

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_mul_inplace_basic():
    """
    Feature: test parallel op mul_
    Description: test parallel op mul_ with identical sharding layouts.
    Expectation: Run success.
    """
    torchrun_case(_IMPL_FILE, "test_mul_inplace_basic")

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_mul_inplace_basic():
    """
    Feature: test parallel op mul_ (gloo cpu)
    Description: test parallel op mul_ with identical sharding layouts.
    Expectation: Run success.
    """
    torchrun_case(_IMPL_FILE, "test_mul_inplace_basic")

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_distributed_mul_inplace_broadcast():
    """
    Feature: test parallel op mul_
    Description: test parallel op mul_ with broadcastable shapes.
    Expectation: Run success.
    """
    torchrun_case(_IMPL_FILE, "test_mul_inplace_broadcast")

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_mul_inplace_broadcast():
    """
    Feature: test parallel op mul_ (gloo cpu)
    Description: test parallel op mul_ with broadcastable shapes.
    Expectation: Run success.
    """
    torchrun_case(_IMPL_FILE, "test_mul_inplace_broadcast")

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_distributed_mul_inplace_scalar():
    """
    Feature: test parallel op mul_
    Description: test parallel op mul_ with scalar inputs.
    Expectation: Run success.
    """
    torchrun_case(_IMPL_FILE, "test_mul_inplace_scalar")

@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_gloo_mul_inplace_scalar():
    """
    Feature: test parallel op mul_ (gloo cpu)
    Description: test parallel op mul_ with scalar inputs.
    Expectation: Run success.
    """
    torchrun_case(_IMPL_FILE, "test_mul_inplace_scalar")
