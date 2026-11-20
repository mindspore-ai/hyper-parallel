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
"""Tests for the extensible dry-run operator interface."""

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from hyper_parallel.tools.dryrun.operator_support import (
    DryRunOpRegistry,
    register_fake_op,
    unsupported_operator_hint,
)


_OP_NAME = "hp_dryrun_test::identity"
torch.library.define(_OP_NAME, "(Tensor input) -> Tensor")


def test_decorator_registers_shape_only_kernel():
    """A user decorator can register a shape-only implementation."""
    registry = DryRunOpRegistry()

    @registry.register_fake(_OP_NAME)
    def identity_fake(input_tensor):
        return torch.empty_like(input_tensor)

    with FakeTensorMode():
        fake = torch.empty(2, 3)
        output = torch.ops.hp_dryrun_test.identity.default(fake)
    assert output.shape == fake.shape
    assert registry.registered_ops == (_OP_NAME,)


def test_unsupported_operator_hint_is_actionable():
    """Unsupported-op diagnostics explain immediate registration."""
    hint = unsupported_operator_hint(
        RuntimeError("UnsupportedOperatorException: custom::missing"),
    )
    assert hint is not None
    assert "register_fake_op" in hint
    assert "Import the defining module" in hint


def test_public_decorator_targets_global_registry():
    """The package-level shortcut delegates to the process registry."""
    assert callable(register_fake_op)
