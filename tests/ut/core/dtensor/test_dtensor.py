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
"""Unit tests for :func:`hyper_parallel.core.dtensor.dtensor.distribute_module`.

Tests mirror PyTorch ``torch.distributed.tensor`` API tests in
``test/distributed/tensor/test_api.py`` (notably ``test_distribute_module`` and
``test_distribute_module_input_fn_output_fn*``) using mocks so they run without
multi-device hardware.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
import unittest
import warnings
from unittest.mock import MagicMock, Mock, patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.device_mesh import DeviceMesh, _mesh_resources
from hyper_parallel.core.dtensor import dtensor as _hp_dtensor_mod
from hyper_parallel.core.dtensor.dtensor import DTensorBase, distribute_module
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard


class _FakeModule:
    """Minimal nn.Module-like object for ``distribute_module`` tests.

    Do not use ``MagicMock`` as the root module: ``getattr(mock, "_distribute_module_applied", False)``
    returns a truthy child mock for a missing attribute, so the guard fires on the first call.
    """

    def __init__(self, params: dict | None = None, buffers: dict | None = None):
        self._parameters = dict(params) if params is not None else {}
        self._buffers = dict(buffers) if buffers is not None else {}
        self.register_parameter = MagicMock()
        self.register_forward_pre_hook = MagicMock()
        self.register_forward_hook = MagicMock()
        self._modules_list: list = []
        self._named_list: list = []

    def modules(self):
        """Return an iterator over this fake tree (default: only ``self``)."""
        return iter(self._modules_list if self._modules_list else [self])

    def named_modules(self):
        """Return an iterator of ``(name, module)`` pairs (default: ``("", self)``)."""
        return iter(self._named_list if self._named_list else [("", self)])


class TestDistributeModule(unittest.TestCase):
    """Tests for :func:`distribute_module`."""

    def tearDown(self):
        """Reset thread-local device mesh stack after each test."""
        _mesh_resources.mesh_stack.clear()

    def _mock_mesh(self, ndim: int = 2) -> MagicMock:
        mesh = MagicMock()
        mesh.ndim = ndim
        return mesh

    def _leaf_module(self, params: dict | None = None, buffers: dict | None = None) -> _FakeModule:
        return _FakeModule(params=params, buffers=buffers)

    def test_raises_when_called_twice(self):
        """Calling ``distribute_module`` twice on the same module raises ``RuntimeError``."""
        mod = self._leaf_module()
        mesh = self._mock_mesh(1)
        with patch("hyper_parallel.core.dtensor.dtensor.distribute_tensor") as mock_dt:
            mock_dt.return_value = MagicMock(spec=DTensorBase)
            distribute_module(mod, device_mesh=mesh, partition_fn=None)
        with self.assertRaisesRegex(
            RuntimeError,
            "distribute_module should only be called once",
        ):
            distribute_module(mod, device_mesh=mesh, partition_fn=None)

    def test_uses_current_mesh_when_device_mesh_none(self):
        """When ``device_mesh`` is ``None``, the active mesh from ``_mesh_resources`` is used."""
        mod = self._leaf_module()
        mesh = self._mock_mesh(1)
        _mesh_resources.mesh_stack.append(mesh)
        with patch("hyper_parallel.core.dtensor.dtensor.distribute_tensor") as mock_dt:
            mock_dt.return_value = MagicMock(spec=DTensorBase)
            distribute_module(mod, device_mesh=None, partition_fn=None)
        mock_dt.assert_not_called()

    def test_partition_fn_none_replicates_plain_parameters(self):
        """Without ``partition_fn``, plain parameters are replicated via ``distribute_tensor``."""
        root = self._leaf_module({})
        child = self._leaf_module({})
        p_data = MagicMock()
        p_data.shape = (2, 3)
        param = MagicMock()
        param.data = p_data
        param.requires_grad = True
        child._parameters = {"weight": param}

        root._modules_list = [root, child]
        root._named_list = [("", root), ("child", child)]

        mesh = self._mock_mesh(2)
        fake_dt = MagicMock(spec=DTensorBase)
        fake_new_param = MagicMock()

        with patch("hyper_parallel.core.dtensor.dtensor.distribute_tensor") as mock_dt:
            mock_dt.return_value = fake_dt
            with patch("hyper_parallel.core.dtensor.dtensor.platform.Parameter", return_value=fake_new_param):
                out = distribute_module(root, device_mesh=mesh, partition_fn=None)

        self.assertIs(out, root)
        self.assertTrue(getattr(root, "_distribute_module_applied", False))
        self.assertEqual(mock_dt.call_count, 1)
        mock_dt.assert_called_once()
        args, kwargs = mock_dt.call_args
        self.assertIs(args[0], p_data)
        self.assertIs(args[1], mesh)
        self.assertEqual(args[2], [Replicate(), Replicate()])
        child.register_parameter.assert_called_once_with("weight", fake_new_param)

    def test_partition_fn_invoked_for_each_named_module_then_replicate(self):
        """``partition_fn`` runs once per ``named_modules`` entry before replicate."""
        root = self._leaf_module({})
        child = self._leaf_module({})
        param = MagicMock()
        param.data = MagicMock()
        param.data.shape = (1,)
        param.requires_grad = False
        child._parameters = {"w": param}
        root._modules_list = [root, child]
        root._named_list = [("", root), ("c", child)]

        mesh = self._mock_mesh(1)
        partition_fn = MagicMock()
        fake_dt = MagicMock(spec=DTensorBase)

        with patch("hyper_parallel.core.dtensor.dtensor.distribute_tensor", return_value=fake_dt):
            with patch("hyper_parallel.core.dtensor.dtensor.platform.Parameter", return_value=MagicMock()):
                distribute_module(root, device_mesh=mesh, partition_fn=partition_fn)

        self.assertEqual(partition_fn.call_count, 2)
        partition_fn.assert_any_call("", root, mesh)
        partition_fn.assert_any_call("c", child, mesh)

    def test_skips_none_parameter(self):
        """``None`` entries in ``_parameters`` are not passed to ``distribute_tensor``."""
        mod = self._leaf_module({"bias": None, "w": MagicMock()})
        mod._parameters["w"].data = MagicMock()
        mod._parameters["w"].data.shape = (1,)
        mod._parameters["w"].requires_grad = True
        mesh = self._mock_mesh(1)

        with patch("hyper_parallel.core.dtensor.dtensor.distribute_tensor") as mock_dt:
            mock_dt.return_value = MagicMock(spec=DTensorBase)
            with patch("hyper_parallel.core.dtensor.dtensor.platform.Parameter", return_value=MagicMock()):
                distribute_module(mod, device_mesh=mesh, partition_fn=None)

        self.assertEqual(mock_dt.call_count, 1)

    def test_replicates_buffers(self):
        """Buffers in ``_buffers`` are converted with ``distribute_tensor`` (replicate)."""
        mod = self._leaf_module({})
        buf = MagicMock()
        buf.shape = (4,)
        mod._buffers = {"running": buf}
        mesh = self._mock_mesh(1)
        fake_dt = MagicMock(spec=DTensorBase)

        with patch("hyper_parallel.core.dtensor.dtensor.distribute_tensor", return_value=fake_dt):
            with patch("hyper_parallel.core.dtensor.dtensor.platform.Parameter", return_value=MagicMock()):
                distribute_module(mod, device_mesh=mesh, partition_fn=None)

        self.assertIs(mod._buffers["running"], fake_dt)

    def test_input_fn_invalid_arity_raises(self):
        """``input_fn`` with arity other than 2 or 3 raises ``ValueError``."""
        mod = self._leaf_module({})
        mesh = self._mock_mesh(1)

        def bad_input_fn(arg):
            return arg

        with self.assertRaisesRegex(ValueError, "input_fn should take in 2 or 3 arguments"):
            distribute_module(mod, device_mesh=mesh, partition_fn=None, input_fn=bad_input_fn)

    def test_output_fn_invalid_arity_raises(self):
        """``output_fn`` with arity other than 2 or 3 raises ``ValueError``."""
        mod = self._leaf_module({})
        mesh = self._mock_mesh(1)

        def bad_out(first, second, third, fourth):
            return fourth

        with self.assertRaisesRegex(ValueError, "output_fn should take in 2 or 3 arguments"):
            distribute_module(mod, device_mesh=mesh, partition_fn=None, output_fn=bad_out)

    def test_input_fn_three_arg_registers_pre_hook(self):
        """A three-argument ``input_fn`` is registered as ``register_forward_pre_hook``."""
        mod = self._leaf_module({})
        mesh = self._mock_mesh(1)
        captured = {}

        def input_fn(m, inputs, dm):
            captured["m"] = m
            captured["inputs"] = inputs
            captured["dm"] = dm
            return inputs

        with patch("hyper_parallel.core.dtensor.dtensor.distribute_tensor", return_value=MagicMock(spec=DTensorBase)):
            with patch("hyper_parallel.core.dtensor.dtensor.platform.Parameter", return_value=MagicMock()):
                distribute_module(mod, device_mesh=mesh, partition_fn=None, input_fn=input_fn)

        mod.register_forward_pre_hook.assert_called_once()
        hook = mod.register_forward_pre_hook.call_args[0][0]
        hook(mod, (1, 2))
        self.assertIs(captured["m"], mod)
        self.assertEqual(captured["inputs"], (1, 2))
        self.assertIs(captured["dm"], mesh)

    def test_input_fn_two_arg_deprecated_warning(self):
        """Two-argument ``input_fn`` emits a ``FutureWarning`` (PyTorch deprecation parity)."""
        mod = self._leaf_module({})
        mesh = self._mock_mesh(1)

        def input_fn(inputs, dm):
            return inputs

        with patch("hyper_parallel.core.dtensor.dtensor.distribute_tensor", return_value=MagicMock(spec=DTensorBase)):
            with patch("hyper_parallel.core.dtensor.dtensor.platform.Parameter", return_value=MagicMock()):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    distribute_module(mod, device_mesh=mesh, partition_fn=None, input_fn=input_fn)
        self.assertTrue(any(issubclass(x.category, FutureWarning) for x in w))

    def test_output_fn_three_arg_registers_forward_hook(self):
        """A three-argument ``output_fn`` is registered as ``register_forward_hook``."""
        mod = self._leaf_module({})
        mesh = self._mock_mesh(1)

        def output_fn(m, outputs, dm):
            return outputs

        with patch("hyper_parallel.core.dtensor.dtensor.distribute_tensor", return_value=MagicMock(spec=DTensorBase)):
            with patch("hyper_parallel.core.dtensor.dtensor.platform.Parameter", return_value=MagicMock()):
                distribute_module(mod, device_mesh=mesh, partition_fn=None, output_fn=output_fn)

        mod.register_forward_hook.assert_called_once()
        hook = mod.register_forward_hook.call_args[0][0]
        out = hook(mod, None, 42)
        self.assertEqual(out, 42)

    # --- The following 8 tests align with distribute_module logic in PyTorch test/distributed/tensor/test_api.py ---

    def test_partition_fn_call_order_root_empty_string_first(self):
        """Aligns with PyTorch ``named_modules()``: root name is ``""`` and precedes children (same order as ``test_distribute_module``)."""
        root = self._leaf_module({})
        a = self._leaf_module({})
        b = self._leaf_module({})
        root._modules_list = [root, a, b]
        root._named_list = [("", root), ("a", a), ("a.b", b)]
        mesh = self._mock_mesh(1)
        names = []

        def partition_fn(name, submod, dm):
            names.append(name)
            self.assertIs(dm, mesh)

        with patch("hyper_parallel.core.dtensor.dtensor.distribute_tensor", return_value=MagicMock(spec=DTensorBase)):
            with patch("hyper_parallel.core.dtensor.dtensor.platform.Parameter", return_value=MagicMock()):
                distribute_module(root, device_mesh=mesh, partition_fn=partition_fn)

        self.assertEqual(names, ["", "a", "a.b"])

    def test_partition_fn_invoked_for_submodule_with_no_direct_parameters(self):
        """``partition_fn`` is still invoked for submodules with no direct ``_parameters`` (PyTorch calls it for every ``named_modules`` node)."""
        root = self._leaf_module({})
        empty = self._leaf_module({})
        root._modules_list = [root, empty]
        root._named_list = [("", root), ("empty", empty)]
        mesh = self._mock_mesh(1)
        seen_empty = False

        def partition_fn(name, submod, dm):
            nonlocal seen_empty
            if name == "empty":
                seen_empty = True
                self.assertEqual(submod._parameters, {})

        with patch("hyper_parallel.core.dtensor.dtensor.distribute_tensor", return_value=MagicMock(spec=DTensorBase)):
            with patch("hyper_parallel.core.dtensor.dtensor.platform.Parameter", return_value=MagicMock()):
                distribute_module(root, device_mesh=mesh, partition_fn=partition_fn)

        self.assertTrue(seen_empty)

    def test_replicate_passes_requires_grad_to_parameter(self):
        """Aligns with PyTorch: the replicate path preserves ``requires_grad=False`` when creating a new ``Parameter``."""
        mod = self._leaf_module({})
        p_data = MagicMock()
        p_data.shape = (2,)
        param = MagicMock()
        param.data = p_data
        param.requires_grad = False
        mod._parameters = {"w": param}
        mesh = self._mock_mesh(1)
        fake_dt = MagicMock(spec=DTensorBase)

        with patch("hyper_parallel.core.dtensor.dtensor.distribute_tensor", return_value=fake_dt):
            with patch("hyper_parallel.core.dtensor.dtensor.platform.Parameter") as mock_param_cls:
                mock_param_cls.return_value = MagicMock()
                distribute_module(mod, device_mesh=mesh, partition_fn=None)

        mock_param_cls.assert_called_once()
        _, kwargs = mock_param_cls.call_args
        self.assertFalse(kwargs.get("requires_grad", True))

    def test_typeerror_when_submodule_cannot_register_parameter(self):
        """Submodule has ``_parameters`` but lacks ``register_parameter`` / ``_params``; should raise ``TypeError`` (PyTorch requires writable parameters)."""

        class _BadLeaf:
            def __init__(self):
                p = MagicMock()
                p.data = MagicMock()
                p.data.shape = (1,)
                p.requires_grad = True
                self._parameters = {"w": p}
                self._buffers = {}

        root = self._leaf_module({})
        bad = _BadLeaf()
        root._modules_list = [root, bad]
        root._named_list = [("", root), ("bad", bad)]
        mesh = self._mock_mesh(1)

        with patch("hyper_parallel.core.dtensor.dtensor.distribute_tensor", return_value=MagicMock(spec=DTensorBase)):
            with patch("hyper_parallel.core.dtensor.dtensor.platform.Parameter", return_value=MagicMock()):
                with self.assertRaisesRegex(
                    TypeError,
                    "distribute_module expects nn.Module-like objects",
                ):
                    distribute_module(root, device_mesh=mesh, partition_fn=None)

    def test_output_fn_two_arg_deprecated_future_warning(self):
        """Aligns with ``test_distribute_module_input_fn_output_fn_warning``: two-arg ``output_fn`` emits ``FutureWarning``."""
        mod = self._leaf_module({})
        mesh = self._mock_mesh(1)

        def output_fn(outputs, dm):
            return outputs

        with patch("hyper_parallel.core.dtensor.dtensor.distribute_tensor", return_value=MagicMock(spec=DTensorBase)):
            with patch("hyper_parallel.core.dtensor.dtensor.platform.Parameter", return_value=MagicMock()):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    distribute_module(mod, device_mesh=mesh, partition_fn=None, output_fn=output_fn)

        self.assertTrue(any(issubclass(x.category, FutureWarning) for x in w))
        self.assertTrue(any("output_fn" in str(x.message) for x in w))

    def test_input_fn_and_output_fn_three_arg_both_registered(self):
        """Aligns with ``test_distribute_module_input_fn_output_fn``: root module registers both pre-hook and forward hook."""
        mod = self._leaf_module({})
        mesh = self._mock_mesh(1)

        def input_fn(m, inputs, dm):
            return inputs

        def output_fn(m, outputs, dm):
            return outputs

        with patch("hyper_parallel.core.dtensor.dtensor.distribute_tensor", return_value=MagicMock(spec=DTensorBase)):
            with patch("hyper_parallel.core.dtensor.dtensor.platform.Parameter", return_value=MagicMock()):
                distribute_module(
                    mod,
                    device_mesh=mesh,
                    partition_fn=None,
                    input_fn=input_fn,
                    output_fn=output_fn,
                )

        mod.register_forward_pre_hook.assert_called_once()
        mod.register_forward_hook.assert_called_once()

    def test_deprecated_two_arg_input_and_output_both_emit_future_warning(self):
        """Same as PyTorch ``test_distribute_module_input_fn_output_fn_warning``: two-arg ``input_fn`` and ``output_fn`` each emit a deprecation warning."""
        mod = self._leaf_module({})
        mesh = self._mock_mesh(1)

        def input_fn(inputs, dm):
            return inputs

        def output_fn(outputs, dm):
            return outputs

        with patch("hyper_parallel.core.dtensor.dtensor.distribute_tensor", return_value=MagicMock(spec=DTensorBase)):
            with patch("hyper_parallel.core.dtensor.dtensor.platform.Parameter", return_value=MagicMock()):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    distribute_module(
                        mod,
                        device_mesh=mesh,
                        partition_fn=None,
                        input_fn=input_fn,
                        output_fn=output_fn,
                    )

        future = [x for x in w if issubclass(x.category, FutureWarning)]
        self.assertGreaterEqual(len(future), 2)
        msgs = " ".join(str(x.message) for x in future)
        self.assertIn("input_fn", msgs)
        self.assertIn("output_fn", msgs)

    def test_partition_fn_shards_named_linear_like_pytorch_partial(self):
        """Aligns with ``test_distribute_module`` partial-shard scenario: ``partition_fn`` shards a named submodule via ``Shard`` + ``distribute_tensor`` + ``register_parameter``; the rest are replicated."""
        root = self._leaf_module({})
        shard_m = self._leaf_module({})
        repl_m = self._leaf_module({})
        p_shard = MagicMock()
        p_shard.data = MagicMock()
        p_shard.data.shape = (4, 4)
        p_shard.requires_grad = True
        p_repl = MagicMock()
        p_repl.data = MagicMock()
        p_repl.data.shape = (3,)
        p_repl.requires_grad = True
        shard_m._parameters = {"weight": p_shard}
        repl_m._parameters = {"weight": p_repl}
        root._modules_list = [root, shard_m, repl_m]
        root._named_list = [("", root), ("seq.0", shard_m), ("seq.1", repl_m)]
        mesh = self._mock_mesh(1)
        fake_dt_shard = MagicMock(spec=DTensorBase)
        fake_dt_repl = MagicMock(spec=DTensorBase)

        with patch("hyper_parallel.core.dtensor.dtensor.distribute_tensor") as mock_dt:
            mock_dt.side_effect = [fake_dt_shard, fake_dt_repl, fake_dt_repl]

            def real_partition_fn(name, module, dm):
                if name == "seq.0" and module is shard_m:
                    dist = mock_dt(p_shard.data, dm, [Shard(0)])
                    module.register_parameter(
                        "weight",
                        _hp_dtensor_mod.platform.Parameter(dist, requires_grad=True),
                    )

            with patch("hyper_parallel.core.dtensor.dtensor.platform.Parameter") as mock_param_cls:
                mock_param_cls.side_effect = lambda *args, **kwargs: MagicMock()
                distribute_module(root, device_mesh=mesh, partition_fn=real_partition_fn)

        # Mock returns are not real DTensors, so replicate runs again on seq.0 (differs from real PyTorch DTensor behaviour, but covers the "partition Shard + subsequent replicate" path)
        self.assertEqual(mock_dt.call_count, 3)
        shard_calls = [c for c in mock_dt.call_args_list if c[0][2] == [Shard(0)]]
        repl_calls = [c for c in mock_dt.call_args_list if c[0][2] == [Replicate()]]
        self.assertEqual(len(shard_calls), 1)
        self.assertEqual(len(repl_calls), 2)


# ============================================================================
# In-place op contracts: DTensor.copy_ / DTensor.zero_ / DTensor.fill_
# ============================================================================
#
# Verifies the contracts documented on DTensor.copy_:
#   * type / mesh / placement / shape checks raise synchronously;
#   * scalar src (numel == 1) relaxes the placement-match requirement;
#   * matching-placement copies forward to ``_local_tensor.data.copy_``;
#   * zero_ / fill_ forward to ``_local_tensor.data.zero_/fill_`` and return self.

import pytest

from hyper_parallel.core.dtensor.dtensor import DTensor, _is_broadcastable


# Bypass C-level isinstance(self, Tensor) check by invoking the unbound
# Python function directly. Same workaround as test_dtensor_to.py.
_copy_fn = DTensor.__dict__["copy_"]
_zero_fn = DTensor.__dict__["zero_"]
_fill_fn = DTensor.__dict__["fill_"]


def _inplace_numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def _inplace_make_self(local_shape=(4, 4), placements=None, mesh="fake_mesh"):
    """Build a fake ``self`` DTensor (SimpleNamespace) for the unbound method."""
    if placements is None:
        placements = (Shard(0),)
    local = Mock(name="local_tensor")
    local.shape = local_shape
    local.numel = Mock(return_value=_inplace_numel(local_shape))
    return SimpleNamespace(
        _local_tensor=local,
        _device_mesh=mesh,
        _placements=tuple(placements),
    ), local


def _inplace_make_src(local_shape=(4, 4), placements=None, mesh="fake_mesh", numel=None):
    """Build a fake ``src`` DTensor that passes ``isinstance(src, DTensor)``.

    Uses ``MagicMock(spec=DTensor)`` so the isinstance check returns True
    without inheriting from the C-level Tensor.
    """
    if placements is None:
        placements = (Shard(0),)
    src = MagicMock(spec=DTensor)
    src._local_tensor = Mock(name="src_local")
    src._local_tensor.shape = local_shape
    src._local_tensor.numel = Mock(
        return_value=numel if numel is not None else _inplace_numel(local_shape)
    )
    src._device_mesh = mesh
    src._placements = tuple(placements)
    # copy_() reads src through public accessors (device_mesh / placements /
    # to_local); mirror the protected-member values onto them so the mock stays
    # consistent with the public API.
    src.device_mesh = mesh
    src.placements = tuple(placements)
    src.to_local = Mock(return_value=src._local_tensor)
    return src


class TestIsBroadcastable:
    """Validate the broadcast-compatibility helper."""

    @pytest.mark.parametrize("src, dst", [
        ((), (4, 4)),
        ((1,), (4, 4)),
        ((4,), (4, 4)),
        ((1, 4), (4, 4)),
        ((4, 1), (4, 4)),
        ((4, 4), (4, 4)),
        ((1, 1), (4, 4)),
    ])
    def test_broadcastable(self, src, dst):
        """_is_broadcastable should return True for broadcast-compatible shapes."""
        assert _is_broadcastable(src, dst)

    @pytest.mark.parametrize("src, dst", [
        ((8, 8), (4, 4)),     # different non-1 size
        ((2, 4), (4, 4)),     # leading dim mismatch
        ((4, 4, 4), (4, 4)),  # src has more dims
        ((5,), (4, 4)),       # last dim mismatch
    ])
    def test_not_broadcastable(self, src, dst):
        assert not _is_broadcastable(src, dst)


class TestCopySamePlacement:
    """Happy path: matching mesh + placements + shape."""

    def test_returns_self(self):
        dst, _ = _inplace_make_self(local_shape=(4, 4), placements=[Shard(0)])
        src = _inplace_make_src(local_shape=(4, 4), placements=[Shard(0)])

        ret = _copy_fn(dst, src)

        assert ret is dst

    def test_forwards_to_local_copy(self):
        dst, local = _inplace_make_self(local_shape=(4, 4), placements=[Shard(0)])
        src = _inplace_make_src(local_shape=(4, 4), placements=[Shard(0)])

        _copy_fn(dst, src, non_blocking=True)

        local.copy_.assert_called_once_with(
            src._local_tensor, non_blocking=True
        )

    def test_default_non_blocking_is_false(self):
        dst, local = _inplace_make_self()
        src = _inplace_make_src()

        _copy_fn(dst, src)

        _, kwargs = local.copy_.call_args
        assert kwargs["non_blocking"] is False


class TestCopyContractViolations:
    """Type / mesh / placement / shape contract enforcement."""

    def test_plain_tensor_src_raises_type_error(self):
        dst, _ = _inplace_make_self()
        with pytest.raises(TypeError, match="src should be a DTensor"):
            _copy_fn(dst, object())

    def test_none_src_raises_type_error(self):
        dst, _ = _inplace_make_self()
        with pytest.raises(TypeError, match="src should be a DTensor"):
            _copy_fn(dst, None)

    def test_mesh_mismatch_raises_value_error(self):
        dst, _ = _inplace_make_self(mesh="mesh_a")
        src = _inplace_make_src(mesh="mesh_b")

        with pytest.raises(ValueError, match="DeviceMesh"):
            _copy_fn(dst, src)

    def test_equivalent_mesh_instances_are_allowed(self):
        """Copy accepts distinct DeviceMesh instances with equivalent topology."""
        dst_mesh = MagicMock(spec=DeviceMesh)
        dst_mesh.device_type = "npu"
        dst_mesh.to_hash.return_value = ((2,), ("dp",), (0, 1))
        src_mesh = MagicMock(spec=DeviceMesh)
        src_mesh.device_type = "npu"
        src_mesh.to_hash.return_value = ((2,), ("dp",), (0, 1))
        dst, _ = _inplace_make_self(mesh=dst_mesh)
        src = _inplace_make_src(mesh=src_mesh)

        result = _copy_fn(dst, src)

        assert result is dst, (
            f"Equivalent DeviceMesh instances should be accepted: expected={dst!r}, got={result!r}"
        )

    def test_equivalent_topology_with_different_device_type_raises(self):
        """Copy rejects equivalent mesh topology when device types differ."""
        dst_mesh = MagicMock(spec=DeviceMesh)
        dst_mesh.device_type = "npu"
        dst_mesh.to_hash.return_value = ((2,), ("dp",), (0, 1))
        src_mesh = MagicMock(spec=DeviceMesh)
        src_mesh.device_type = "cpu"
        src_mesh.to_hash.return_value = ((2,), ("dp",), (0, 1))
        dst, _ = _inplace_make_self(mesh=dst_mesh)
        src = _inplace_make_src(mesh=src_mesh)

        with pytest.raises(ValueError, match="DeviceMesh"):
            _copy_fn(dst, src)

    def test_placement_mismatch_non_scalar_raises(self):
        dst, _ = _inplace_make_self(local_shape=(4, 4), placements=[Shard(0)])
        # Same local shape but different placement and >1 element — disallowed.
        src = _inplace_make_src(
            local_shape=(4, 4), placements=[Replicate()], numel=16
        )

        with pytest.raises(ValueError, match="src.placements should equal"):
            _copy_fn(dst, src)

    def test_shape_not_broadcastable_raises(self):
        dst, _ = _inplace_make_self(local_shape=(4, 4), placements=[Shard(0)])
        # Same placement so the placement check passes, but shapes differ.
        src = _inplace_make_src(local_shape=(8, 8), placements=[Shard(0)])

        with pytest.raises(ValueError, match="broadcastable"):
            _copy_fn(dst, src)


class TestCopyScalarBroadcastRelaxation:
    """Single-element src bypasses placement match."""

    def test_zero_dim_scalar_across_placement(self):
        dst, local = _inplace_make_self(local_shape=(4, 4), placements=[Shard(0)])
        src = _inplace_make_src(local_shape=(), placements=[Replicate()], numel=1)

        ret = _copy_fn(dst, src)

        assert ret is dst
        local.copy_.assert_called_once()

    def test_shape_one_tuple_scalar_across_placement(self):
        dst, local = _inplace_make_self(local_shape=(4, 4), placements=[Shard(0)])
        src = _inplace_make_src(local_shape=(1,), placements=[Replicate()], numel=1)

        ret = _copy_fn(dst, src)

        assert ret is dst
        local.copy_.assert_called_once()

    def test_shape_1_1_scalar_across_placement(self):
        dst, local = _inplace_make_self(local_shape=(4, 4), placements=[Shard(0)])
        src = _inplace_make_src(local_shape=(1, 1), placements=[Replicate()], numel=1)

        ret = _copy_fn(dst, src)

        assert ret is dst
        local.copy_.assert_called_once()


class TestZero:
    """Unit tests for DTensor.zero_."""

    def test_returns_self(self):
        dst, _ = _inplace_make_self()
        ret = _zero_fn(dst)
        assert ret is dst

    def test_forwards_to_local_zero(self):
        dst, local = _inplace_make_self()
        _zero_fn(dst)
        local.zero_.assert_called_once_with()


class TestFill:
    """Unit tests for DTensor.fill_."""

    def test_returns_self(self):
        dst, _ = _inplace_make_self()
        ret = _fill_fn(dst, 3.14)
        assert ret is dst

    def test_forwards_value_to_local_fill(self):
        dst, local = _inplace_make_self()
        _fill_fn(dst, 42.0)
        local.fill_.assert_called_once_with(42.0)


if __name__ == "__main__":
    unittest.main()
