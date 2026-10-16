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
"""Verify local-op execution scopes and checkpoint replay on expanded operators."""
import importlib
from unittest.mock import MagicMock

import numpy as np
import pytest

from tests.ut.platform.mindspore._ensure_mindspore_platform import ensure_mindspore_platform_default

ms = pytest.importorskip("mindspore")
ensure_mindspore_platform_default()

from hyper_parallel.core.activation_checkpoint import checkpoint_wrapper, checkpoint_exclude_wrapper
from hyper_parallel.core.shard.local_op_context import LOCAL_OP_EXECUTOR
from hyper_parallel.platform.mindspore.autograd_compat import enable_mindspore_backward_compat

dispatch_module = importlib.import_module("hyper_parallel.core.shard._op_dispatch")
exclude_module = importlib.import_module(
    "hyper_parallel.platform.mindspore.activation_checkpoint.checkpoint_exclude_wrapper"
)
enable_mindspore_backward_compat()


def _dispatcher(monkeypatch, expand):
    """Use real layout dispatch and caching with a synthetic distributed expansion."""
    operator = MagicMock()
    operator.preprocess.side_effect = lambda args, kwargs: (args, kwargs, [])
    operator.infer_layout.return_value = ((None,), None)
    operator.get_expand_impl.side_effect = expand
    operator.wrap_output.side_effect = lambda result, layouts: result
    cache = MagicMock()
    cache.distributed_op.return_value = operator
    cache.get_layout_cache.return_value = {}
    monkeypatch.setattr(dispatch_module.LayoutCacheManager, "get_instance", lambda: cache)
    dispatcher = object.__new__(dispatch_module.OpDispatcher)
    dispatcher.layout_infer_ops = {"TestOp": {}}
    return dispatcher


def test_local_executor_cache_isolation_and_exception_restore(monkeypatch):
    """A cached expansion must not leak executors or intercept its helpers."""
    def expand(kernel, *_):
        def call(value):
            assert LOCAL_OP_EXECUTOR.get() is None
            return kernel(value + 1) + kernel(value + 2)
        return call

    dispatcher = _dispatcher(monkeypatch, expand)
    calls = []

    class ScopedExecutor:
        def run(self, kernel, *args, **kwargs):
            calls.append(args)
            return kernel(*args, **kwargs)

    executor = ScopedExecutor().run

    raw = lambda value: value * 2
    token = LOCAL_OP_EXECUTOR.set(executor)
    try:
        assert dispatcher._dispatch_layout_infer("TestOp", raw, (1,), {}) == 10
        assert LOCAL_OP_EXECUTOR.get() is executor
        with pytest.raises(ValueError, match="Missing"):
            dispatcher._dispatch_layout_infer("Missing", raw, (1,), {})
        assert LOCAL_OP_EXECUTOR.get() is executor
    finally:
        LOCAL_OP_EXECUTOR.reset(token)
    assert calls == [(2,), (3,)]
    assert dispatcher._dispatch_layout_infer("TestOp", raw, (1,), {}) == 10
    assert calls == [(2,), (3,)]


def test_local_executor_covers_multiple_distributed_ops(monkeypatch):
    """Local-op scope covers every dispatched operator in the region."""
    dispatcher = _dispatcher(monkeypatch, lambda kernel, *_: kernel)
    dispatcher.layout_infer_ops["OtherOp"] = {}
    calls = []

    class ScopedExecutor:
        def run(self, kernel, *args, **kwargs):
            calls.append(args)
            return kernel(*args, **kwargs)

    executor = ScopedExecutor().run
    token = LOCAL_OP_EXECUTOR.set(executor)
    try:
        assert dispatcher._dispatch_layout_infer("OtherOp", lambda value: value + 1, (1,), {}) == 2
        assert LOCAL_OP_EXECUTOR.get() is executor
        assert dispatcher._dispatch_layout_infer("TestOp", lambda value: value + 2, (1,), {}) == 3
    finally:
        LOCAL_OP_EXECUTOR.reset(token)
    assert calls == [(1,), (1,)]


@pytest.mark.parametrize("scope", [None, False, True])
def test_cell_and_function_scope_defaults_and_overrides(scope):
    """Explicit scope overrides callable-type defaults independently of preservation."""
    class Router(ms.nn.Cell):
        def construct(self, value):
            return value * 2

    for preserve in (False, True):
        cell = checkpoint_exclude_wrapper(Router(), exclude_single_op=scope, preserve_inputs=preserve)
        function = checkpoint_exclude_wrapper(lambda value: value * 2,
                                             exclude_single_op=scope, preserve_inputs=preserve)
        assert cell.exclude_single_op is (False if scope is None else scope)
        assert function.exclude_single_op is (True if scope is None else scope)


@pytest.mark.parametrize("scope", [0, 1, "True", []])
def test_invalid_scope_rejected(scope):
    with pytest.raises(ValueError, match="exclude_single_op"):
        checkpoint_exclude_wrapper(lambda value: value, exclude_single_op=scope)


@pytest.mark.parametrize("cell_target", [False, True])
def test_region_scope_with_dtensor_input(monkeypatch, cell_target):
    """A router-like region must never switch scopes just because one input is a DTensor."""
    calls = []

    def region(value, bias):
        calls.append(1)
        return value * bias

    original_collect = exclude_module._collect_tensor_inputs
    marker = MagicMock(spec=exclude_module.DTensor)

    def collect(args, kwargs):
        leaves = original_collect(args, kwargs)
        return leaves + [((("arg", 1),), marker)] if len(args) == 2 else leaves

    monkeypatch.setattr(exclude_module, "_collect_tensor_inputs", collect)
    class Router(ms.nn.Cell):
        def construct(self, value, bias):
            return region(value, bias)

    excluded = checkpoint_exclude_wrapper(Router()) if cell_target else checkpoint_exclude_wrapper(
        region, exclude_single_op=False
    )
    value = ms.Tensor([1., 2., 3.])
    value.requires_grad = True

    def block(input_value):
        return excluded(input_value * 2, ms.Tensor([3., 3., 3.])).sum()

    output = checkpoint_wrapper(block)(value)
    output.backward()
    np.testing.assert_allclose(value.grad.asnumpy(), [6., 6., 6.])
    assert calls == [1]


@pytest.mark.parametrize("cell_target", [False, True])
def test_expansion_replayed_while_local_kernels_are_excluded(monkeypatch, cell_target):
    """Real hooks rebuild copied prefixes and replay cat without reexecuting kernels."""
    calls = {"expand": 0, "kernel": 0}

    def expand(kernel, *_):
        def call(value):
            calls["expand"] += 1
            assert LOCAL_OP_EXECUTOR.get() is None
            # Copying with index_select deliberately creates non-alias inputs.
            first = value.index_select(0, ms.Tensor([0, 1], ms.int32))
            second = value.index_select(0, ms.Tensor([2, 3], ms.int32))
            return ms.ops.cat((kernel(first), kernel(second)))
        return call

    dispatcher = _dispatcher(monkeypatch, expand)

    def kernel(value):
        calls["kernel"] += 1
        return value * value

    def distributed(value):
        return dispatcher._dispatch_layout_infer("TestOp", kernel, (value,), {})

    # The UT harness imports DTensor with the Torch base before collection.
    # Mock only its type test and registered target resolution here.
    original_collect = exclude_module._collect_tensor_inputs
    marker = MagicMock(spec=exclude_module.DTensor)

    def collect(args, kwargs):
        if args and args[0].shape == (4,):
            return [((("arg", 0),), marker)]
        return original_collect(args, kwargs)

    monkeypatch.setattr(exclude_module, "_collect_tensor_inputs", collect)
    class DistributedCell(ms.nn.Cell):
        def construct(self, value):
            return distributed(value)

    excluded = checkpoint_exclude_wrapper(DistributedCell(), exclude_single_op=True) if cell_target else (
        checkpoint_exclude_wrapper(distributed)
    )
    tensor = ms.Tensor([1., 2., 3., 4.])
    tensor.requires_grad = True

    def block(value):
        return excluded(value * 2).sum()

    output = checkpoint_wrapper(block)(tensor)
    output.backward()
    np.testing.assert_allclose(output.asnumpy(), 120.)
    np.testing.assert_allclose(tensor.grad.asnumpy(), [8., 16., 24., 32.])
    assert calls == {"expand": 2, "kernel": 2}
    assert LOCAL_OP_EXECUTOR.get() is None
    tensor.grad = None
    checkpoint_wrapper(block)(tensor).backward()
    np.testing.assert_allclose(tensor.grad.asnumpy(), [8., 16., 24., 32.])
    assert calls == {"expand": 4, "kernel": 4}
    assert LOCAL_OP_EXECUTOR.get() is None


def test_unexpanded_local_kernel_and_kernel_exception(monkeypatch):
    """Direct kernels use the executor too, and exceptions restore its scope."""
    dispatcher = _dispatcher(monkeypatch, lambda *_: None)
    calls = []

    class ScopedExecutor:
        def run(self, kernel, *args, **kwargs):
            assert LOCAL_OP_EXECUTOR.get() is None
            calls.append(args)
            return kernel(*args, **kwargs)

    executor = ScopedExecutor().run

    def fail(value):
        raise RuntimeError(str(value))

    token = LOCAL_OP_EXECUTOR.set(executor)
    try:
        assert dispatcher._dispatch_layout_infer("TestOp", lambda value: value + 1, (3,), {}) == 4
        with pytest.raises(RuntimeError, match="3"):
            dispatcher._dispatch_layout_infer("TestOp", fail, (3,), {})
        assert LOCAL_OP_EXECUTOR.get() is executor
    finally:
        LOCAL_OP_EXECUTOR.reset(token)
    assert calls == [(3,), (3,)]
