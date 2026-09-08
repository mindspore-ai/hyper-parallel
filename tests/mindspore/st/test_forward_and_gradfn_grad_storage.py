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
"""Thin pytest entry (lazy-import). Impl: ``tests.mindspore.st._forward_and_gradfn_grad_storage_impl``."""
from __future__ import annotations

import importlib

from tests.common.mark_utils import arg_mark  # noqa: F401

_IMPL = "tests.mindspore.st._forward_and_gradfn_grad_storage_impl"


def _run(name: str):
    # pylint: disable=C0415
    return getattr(importlib.import_module(_IMPL), name)()


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_forward_and_gradfn_populates_parameter_grad():
    """See impl ``test_forward_and_gradfn_populates_parameter_grad``."""
    return _run("test_forward_and_gradfn_populates_parameter_grad")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_forward_and_gradfn_parameter_grad_accumulates():
    """See impl ``test_forward_and_gradfn_parameter_grad_accumulates``."""
    return _run("test_forward_and_gradfn_parameter_grad_accumulates")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_accumulate_grad_falls_back_for_shared_non_leaf_weight():
    """See impl ``test_accumulate_grad_falls_back_for_shared_non_leaf_weight``."""
    return _run("test_accumulate_grad_falls_back_for_shared_non_leaf_weight")
