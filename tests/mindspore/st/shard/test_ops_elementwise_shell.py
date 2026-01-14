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
"""parallel_elementwise_shell test"""

from tests.common.mark_utils import arg_mark
from tests.mindspore.st.utils import msrun_case


class ElementWiseShellBase:
    """Base class for elementwise ST shell tests."""

    file_name = "elementwise_shard_in_python.py"
    case_name = None
    master_port = None
    glog_v = 2

    def _run(self):
        if self.case_name is None or self.master_port is None:
            raise ValueError("case_name/master_port must be set in subclass.")
        msrun_case(self.glog_v, self.file_name, self.case_name, self.master_port)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
class TestMinimumSameShape(ElementWiseShellBase):
    """Shell launcher for Minimum same-shape elementwise ST case."""

    case_name = "test_minimum_same_shape_parallel_1"
    master_port = 11300

    def test_minimum_same_shape_parallel_1(self):
        """
        Feature: Minimum operator.
        Description: Test same-shape inputs with full parallel in python shard.
        Expectation: Run success.
        """
        self._run()


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
class TestLessEqualSameShape(ElementWiseShellBase):
    """Shell launcher for LessEqual same-shape elementwise ST case."""

    case_name = "test_less_equal_same_shape_parallel_2"
    master_port = 11301

    def test_less_equal_same_shape_parallel_2(self):
        """
        Feature: LessEqual operator.
        Description: Test same-shape inputs with bool output cast in python shard.
        Expectation: Run success.
        """
        self._run()


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
class TestGreaterEqualSameShape(ElementWiseShellBase):
    """Shell launcher for GreaterEqual same-shape elementwise ST case."""

    case_name = "test_greater_equal_same_shape_parallel_3"
    master_port = 11302

    def test_greater_equal_same_shape_parallel_3(self):
        """
        Feature: GreaterEqual operator.
        Description: Test same-shape inputs in python shard.
        Expectation: Run success.
        """
        self._run()


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
class TestLogicalOrSameShape(ElementWiseShellBase):
    """Shell launcher for LogicalOr same-shape elementwise ST case."""

    case_name = "test_logical_or_same_shape_parallel_4"
    master_port = 11303

    def test_logical_or_same_shape_parallel_4(self):
        """
        Feature: LogicalOr operator.
        Description: Test same-shape bool inputs in python shard.
        Expectation: Run success.
        """
        self._run()


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
class TestMinimumBroadcastDim0(ElementWiseShellBase):
    """Shell launcher for Minimum broadcasting on dimension 0."""

    case_name = "test_minimum_broadcast_dim0_parallel_5"
    master_port = 11304

    def test_minimum_broadcast_dim0_parallel_5(self):
        """
        Feature: Minimum operator with broadcasting.
        Description: Test broadcasting on dimension 0 (y: [1, 256, 128]) in python shard.
        Expectation: Run success.
        """
        self._run()


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
class TestMinimumBroadcastDim1(ElementWiseShellBase):
    """Shell launcher for Minimum broadcasting on dimension 1."""

    case_name = "test_minimum_broadcast_dim1_parallel_6"
    master_port = 11305

    def test_minimum_broadcast_dim1_parallel_6(self):
        """
        Feature: Minimum operator with broadcasting.
        Description: Test broadcasting on dimension 1 (y: [16, 1, 128]) in python shard.
        Expectation: Run success.
        """
        self._run()


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
class TestMinimumBroadcastDim2(ElementWiseShellBase):
    """Shell launcher for Minimum broadcasting on dimension 2."""

    case_name = "test_minimum_broadcast_dim2_parallel_7"
    master_port = 11306

    def test_minimum_broadcast_dim2_parallel_7(self):
        """
        Feature: Minimum operator with broadcasting.
        Description: Test broadcasting on dimension 2 (y: [16, 256, 1]) in python shard.
        Expectation: Run success.
        """
        self._run()


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
class TestMinimumBroadcastRankMismatch(ElementWiseShellBase):
    """Shell launcher for Minimum broadcasting with rank mismatch."""

    case_name = "test_minimum_broadcast_rank_mismatch_parallel_8"
    master_port = 11307

    def test_minimum_broadcast_rank_mismatch_parallel_8(self):
        """
        Feature: Minimum operator with broadcasting.
        Description: Test rank mismatch broadcasting (y: [256, 128] -> [16, 256, 128]) in python shard.
        Expectation: Run success.
        """
        self._run()


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
class TestMinimumBroadcastScalarLike(ElementWiseShellBase):
    """Shell launcher for Minimum broadcasting with scalar-like input."""

    case_name = "test_minimum_broadcast_scalar_like_parallel_9"
    master_port = 11308

    def test_minimum_broadcast_scalar_like_parallel_9(self):
        """
        Feature: Minimum operator with broadcasting.
        Description: Test scalar-like broadcasting (y: [128]) in python shard.
        Expectation: Run success.
        """
        self._run()


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
class TestLessEqualBroadcastMultiDim(ElementWiseShellBase):
    """Shell launcher for LessEqual broadcasting on multiple dimensions."""

    case_name = "test_less_equal_broadcast_multi_dim_parallel_10"
    master_port = 11309

    def test_less_equal_broadcast_multi_dim_parallel_10(self):
        """
        Feature: LessEqual operator with broadcasting.
        Description: Test multi-dimension broadcasting (y: [1, 256, 1]) in python shard.
        Expectation: Run success.
        """
        self._run()


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="allcards",
    essential_mark="essential",
)
class TestMinimumPartialShard(ElementWiseShellBase):
    """Shell launcher for Minimum with partial sharding."""

    case_name = "test_minimum_partial_shard_parallel_11"
    master_port = 11310

    def test_minimum_partial_shard_parallel_11(self):
        """
        Feature: Minimum operator with partial sharding.
        Description: Test partial sharding (only one dimension sharded) in python shard.
        Expectation: Run success.
        """
        self._run()
