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
"""Thin pytest entry for MindSpore autograd compatibility ST (lazy import)."""
from __future__ import annotations

import importlib

from tests.common.mark_utils import arg_mark

_IMPL = "tests.mindspore.st._autograd_compat_impl"


def _run(name: str):
    # pylint: disable=C0415
    return getattr(importlib.import_module(_IMPL), name)()


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_scalar_backward_grad_correctness():
    """See ``_autograd_compat_impl.test_scalar_backward_grad_correctness``."""
    return _run("test_scalar_backward_grad_correctness")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_backward_with_sens_grad_correctness():
    """See ``_autograd_compat_impl.test_backward_with_sens_grad_correctness``."""
    return _run("test_backward_with_sens_grad_correctness")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_retain_grad_keeps_non_leaf_grad_correctness():
    """See ``_autograd_compat_impl.test_retain_grad_keeps_non_leaf_grad_correctness``."""
    return _run("test_retain_grad_keeps_non_leaf_grad_correctness")
