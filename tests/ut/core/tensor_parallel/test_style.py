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
"""Unit tests for ``hyper_parallel.core.tensor_parallel.style``.

Tests for :class:`ParallelStyle` abstract base class and its contract:
- Abstract class cannot be instantiated directly
- Subclasses must implement ``apply`` method
- ``src_data_rank`` default value and mutability
- ``apply`` method signature and return value contract

:class:`PrepareModuleInput` / :class:`PrepareModuleOutput` /
:class:`PrepareModuleInputOutput` cases mirror PyTorch
``test/distributed/tensor/parallel/test_parallelize_api.py`` (``TensorParallelAPITests``)
where applicable; mesh size is 1 with ``init_backend=False`` for CPU-only UT.
"""
import os
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.dtensor.device_mesh import init_device_mesh, _DEVICE_MESH_MAP
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.core.tensor_parallel.api import parallelize_module
from hyper_parallel.core.tensor_parallel.style import (
    ParallelStyle,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    PrepareModuleOutput,
)
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS, PlatformType


def _patch_platform_rank_for_dtensor_redistribute(world_size: int = 1):
    """Patch ``torch.distributed`` rank helpers so ``tensor_redistribution`` runs without ``init_process_group``.

    Instance-level patches to ``platform.get_rank`` do not reliably override
    ``TorchPlatform``'s ``@staticmethod`` implementations; patching ``torch.distributed``
    APIs directly keeps behavior correct when running the full ``tests/ut`` suite.
    """
    return patch.multiple(
        "torch.distributed",
        get_rank=MagicMock(return_value=0),
        get_world_size=MagicMock(return_value=world_size),
    )


class ConcreteParallelStyle(ParallelStyle):
    """A concrete implementation of ParallelStyle for testing."""

    def __init__(self):
        super().__init__()
        self.apply_called = False
        self.last_module = None
        self.last_device_mesh = None

    def apply(self, module, device_mesh):
        """Apply the parallel style and record the call."""
        self.apply_called = True
        self.last_module = module
        self.last_device_mesh = device_mesh
        return module


class WrapperParallelStyle(ParallelStyle):
    """A wrapper style that wraps the module in an identity wrapper."""

    class IdentityWrapper(nn.Module):
        """Simple wrapper for testing."""

        def __init__(self, module):
            super().__init__()
            self.wrapped = module

        def forward(self, *args, **kwargs):
            return self.wrapped(*args, **kwargs)

    def apply(self, module, device_mesh):
        """Wrap the module."""
        return WrapperParallelStyle.IdentityWrapper(module)


class TestParallelStyle(unittest.TestCase):
    """Tests for ParallelStyle abstract base class."""

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def _setup_mock_platform(self, mock_platform, world_size: int = 4):
        mock_platform.platform_type = PlatformType.PYTORCH
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else __import__("numpy").array(t)
        )

    def _make_1d_mesh(self, mock_platform, size: int = 4):
        self._setup_mock_platform(mock_platform, world_size=size)
        return init_device_mesh(
            device_type="cpu",
            mesh_shape=(size,),
            mesh_dim_names=("tp",),
            init_backend=False,
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_cannot_instantiate_abstract_class(self, mock_platform):
        """
        Feature: ParallelStyle abstract base class instantiation
        Description: call ParallelStyle() without subclassing
        Expectation: throw TypeError whose message mentions abstract
        """
        self._setup_mock_platform(mock_platform)
        with self.assertRaises(TypeError) as ctx:
            ParallelStyle()
        self.assertIn("abstract", str(ctx.exception).lower())

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_concrete_subclass_can_be_instantiated(self, mock_platform):
        """
        Feature: concrete ParallelStyle subclass instantiation
        Description: instantiate ConcreteParallelStyle which implements apply
        Expectation: object is not None and apply_called is False before apply
        """
        self._setup_mock_platform(mock_platform)
        style = ConcreteParallelStyle()
        self.assertIsNotNone(style)
        self.assertFalse(style.apply_called)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_src_data_rank_default_value(self, mock_platform):
        """
        Feature: ParallelStyle src_data_rank default value
        Description: new ConcreteParallelStyle without assigning src_data_rank
        Expectation: src_data_rank equals 0
        """
        self._setup_mock_platform(mock_platform)
        style = ConcreteParallelStyle()
        self.assertEqual(style.src_data_rank, 0)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_src_data_rank_can_be_set(self, mock_platform):
        """
        Feature: ParallelStyle src_data_rank modification
        Description: assign src_data_rank to integer 2 on ConcreteParallelStyle
        Expectation: reading src_data_rank returns 2
        """
        self._setup_mock_platform(mock_platform)
        style = ConcreteParallelStyle()
        style.src_data_rank = 2
        self.assertEqual(style.src_data_rank, 2)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_src_data_rank_can_be_none(self, mock_platform):
        """
        Feature: ParallelStyle src_data_rank may be None
        Description: set src_data_rank to None on ConcreteParallelStyle
        Expectation: src_data_rank is None
        """
        self._setup_mock_platform(mock_platform)
        style = ConcreteParallelStyle()
        style.src_data_rank = None
        self.assertIsNone(style.src_data_rank)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_apply_receives_correct_arguments(self, mock_platform):
        """
        Feature: ParallelStyle.apply receives module and mesh
        Description: call ConcreteParallelStyle.apply(Linear, 1-D DeviceMesh)
        Expectation: apply_called True; last_module and last_device_mesh match; return is module
        """
        mesh = self._make_1d_mesh(mock_platform)
        style = ConcreteParallelStyle()
        module = nn.Linear(4, 4)

        result = style.apply(module, mesh)

        self.assertTrue(style.apply_called)
        self.assertIs(style.last_module, module)
        self.assertIs(style.last_device_mesh, mesh)
        self.assertIs(result, module)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_apply_can_return_wrapped_module(self, mock_platform):
        """
        Feature: ParallelStyle.apply may return wrapped module
        Description: WrapperParallelStyle.apply wraps Linear in IdentityWrapper
        Expectation: result is not original module; result.wrapped is original module
        """
        mesh = self._make_1d_mesh(mock_platform)
        style = WrapperParallelStyle()
        module = nn.Linear(4, 4)

        result = style.apply(module, mesh)

        self.assertIsNot(result, module)
        self.assertIsInstance(result, nn.Module)
        self.assertIs(result.wrapped, module)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_apply_called_multiple_times(self, mock_platform):
        """
        Feature: ParallelStyle.apply may run multiple times
        Description: same style applies to two different Linear modules on same mesh
        Expectation: last_module tracks second module; apply_called remains True
        """
        mesh = self._make_1d_mesh(mock_platform)
        style = ConcreteParallelStyle()
        module1 = nn.Linear(4, 4)
        module2 = nn.Linear(3, 3)

        style.apply(module1, mesh)
        self.assertIs(style.last_module, module1)

        style.apply(module2, mesh)
        self.assertIs(style.last_module, module2)
        self.assertTrue(style.apply_called)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_different_styles_have_independent_src_data_rank(self, mock_platform):
        """
        Feature: independent src_data_rank per ParallelStyle instance
        Description: two ConcreteParallelStyle instances with src_data_rank 1 and 2
        Expectation: style1 reads 1 and style2 reads 2 with no cross mutation
        """
        self._setup_mock_platform(mock_platform)
        style1 = ConcreteParallelStyle()
        style2 = ConcreteParallelStyle()

        style1.src_data_rank = 1
        style2.src_data_rank = 2

        self.assertEqual(style1.src_data_rank, 1)
        self.assertEqual(style2.src_data_rank, 2)


class TestParallelStyleWithMockMesh(unittest.TestCase):
    """Additional tests for ParallelStyle using mock mesh."""

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def _setup_mock_platform(self, mock_platform, world_size: int = 4):
        mock_platform.platform_type = PlatformType.PYTORCH
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else __import__("numpy").array(t)
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_apply_with_different_mesh_dimensions(self, mock_platform):
        """
        Feature: ParallelStyle.apply with varying 1-D mesh world sizes
        Description: loop mesh sizes 1,2,4,8; each time apply Linear with new mesh
        Expectation: each iteration return is module; last_device_mesh equals current mesh
        """
        for size in [1, 2, 4, 8]:
            EXISTING_COMM_GROUPS.clear()
            _DEVICE_MESH_MAP.clear()
            self._setup_mock_platform(mock_platform, world_size=size)
            mesh = init_device_mesh(
                device_type="cpu",
                mesh_shape=(size,),
                mesh_dim_names=("tp",),
                init_backend=False,
            )
            style = ConcreteParallelStyle()
            module = nn.Linear(4, 4)

            result = style.apply(module, mesh)

            self.assertIs(result, module)
            self.assertIs(style.last_device_mesh, mesh)


class _DummyIdentityModule(nn.Module):
    """Identity module (same idea as PyTorch ``DummyModule`` in ``test_parallelize_api``)."""

    def forward(self, x):
        return x


class _DupOutModule(nn.Module):
    """Returns two values for multi-output layout tests."""

    def forward(self, x):
        return x, torch.tensor(1.0, dtype=x.dtype, device=x.device)


class TestPrepareModuleInput(unittest.TestCase):
    """Tests aligned with PyTorch ``TensorParallelAPITests.test_prepare_module_input``."""

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def _setup_mock_platform(self, mock_platform, world_size: int = 1):
        mock_platform.platform_type = PlatformType.PYTORCH
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else __import__("numpy").array(t)
        )

    def _make_1d_mesh(self, mock_platform, size: int = 1):
        self._setup_mock_platform(mock_platform, world_size=size)
        return init_device_mesh(
            device_type="cpu",
            mesh_shape=(size,),
            mesh_dim_names=("tp",),
            init_backend=False,
        )

    def test_is_subclass_of_parallel_style(self):
        """PrepareModuleInput is a ParallelStyle."""
        self.assertIsInstance(PrepareModuleInput(), ParallelStyle)

    def test_constructor_wraps_single_placement_as_tuple(self):
        """Single Placement for input_layouts is normalized to a one-tuple."""
        style = PrepareModuleInput(
            input_layouts=Shard(0),
            desired_input_layouts=Replicate(),
        )
        self.assertEqual(style.input_layouts, (Shard(0),))
        self.assertEqual(style.desired_input_layouts, (Replicate(),))

    def test_constructor_raises_when_desired_none_but_input_given(self):
        """PyTorch parity: input_layouts set implies desired_input_layouts must be set."""
        with self.assertRaises(AssertionError):
            PrepareModuleInput(
                input_layouts=(Replicate(),),
                desired_input_layouts=None,
            )

    def test_constructor_raises_on_layout_length_mismatch(self):
        with self.assertRaises(AssertionError):
            PrepareModuleInput(
                input_layouts=(Replicate(), Replicate()),
                desired_input_layouts=(Replicate(),),
            )

    def test_constructor_raises_on_kwarg_layout_length_mismatch(self):
        with self.assertRaises(AssertionError):
            PrepareModuleInput(
                input_kwarg_layouts={"a": Replicate()},
                desired_input_kwarg_layouts={},
            )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_forward_matches_pytorch_prepare_module_input_case(self, mock_platform):
        """Same layout contract as PyTorch ``test_prepare_module_input`` (rank-agnostic check)."""
        mesh = self._make_1d_mesh(mock_platform, size=1)
        module = _DummyIdentityModule()
        PrepareModuleInput(
            input_layouts=Shard(0),
            desired_input_layouts=Replicate(),
            use_local_output=False,
        ).apply(module, mesh)
        inp = torch.rand(5, 7)
        with _patch_platform_rank_for_dtensor_redistribute(world_size=1):
            output = module(inp)
            self.assertIsInstance(output, DTensor)
            restored = output.redistribute(mesh, [Shard(0)]).to_local()
        self.assertTrue(torch.allclose(inp, restored))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_parallelize_module_single_style_like_pytorch(self, mock_platform):
        """``parallelize_module(m, mesh, PrepareModuleInput(...))`` applies hooks on root."""
        mesh = self._make_1d_mesh(mock_platform, size=1)
        module = _DummyIdentityModule()
        parallelize_module(
            module,
            mesh,
            PrepareModuleInput(
                input_layouts=Shard(0),
                desired_input_layouts=Replicate(),
                use_local_output=False,
            ),
        )
        inp = torch.rand(4, 6)
        with _patch_platform_rank_for_dtensor_redistribute(world_size=1):
            out = module(inp)
            restored = out.redistribute(mesh, [Shard(0)]).to_local()
        self.assertTrue(torch.allclose(inp, restored))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_use_local_output_true_passes_local_tensor_to_forward(self, mock_platform):
        """With ``use_local_output=True``, forward receives plain tensors, not DTensor."""
        mesh = self._make_1d_mesh(mock_platform, size=1)
        module = _DummyIdentityModule()
        PrepareModuleInput(
            input_layouts=Replicate(),
            desired_input_layouts=Replicate(),
            use_local_output=True,
        ).apply(module, mesh)
        inp = torch.ones(2, 3)
        out = module(inp)
        self.assertIsInstance(out, torch.Tensor)
        self.assertTrue(torch.allclose(inp, out))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_value_error_when_forward_arity_mismatches_input_layouts(self, mock_platform):
        """Raise ``ValueError`` when the number of forward args does not match ``input_layouts``."""
        mesh = self._make_1d_mesh(mock_platform, size=1)

        class TwoIn(nn.Module):
            def forward(self, x, y):
                return x + y

        module = TwoIn()
        PrepareModuleInput(
            input_layouts=(Replicate(),),
            desired_input_layouts=(Replicate(),),
        ).apply(module, mesh)
        with self.assertRaises(ValueError):
            module(torch.randn(2, 2), torch.randn(2, 2))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_none_placeholder_leaves_input_unchanged(self, mock_platform):
        """A ``None`` slot in layout tuples skips preparing that positional argument."""
        mesh = self._make_1d_mesh(mock_platform, size=1)

        class PickSecond(nn.Module):
            def forward(self, x, y):
                return y

        module = PickSecond()
        PrepareModuleInput(
            input_layouts=(None, Replicate()),
            desired_input_layouts=(None, Replicate()),
            use_local_output=True,
        ).apply(module, mesh)
        x = torch.randn(2, 2)
        y = torch.randn(2, 2)
        out = module(x, y)
        self.assertTrue(torch.all(torch.eq(out, y)))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_with_kwargs_prepares_kwarg_tensor(self, mock_platform):
        """``input_kwarg_layouts`` prepares keyword tensor arguments like positional ones."""
        mesh = self._make_1d_mesh(mock_platform, size=1)

        class M(nn.Module):
            def forward(self, x, scale=None):
                return (x, scale)

        m = M()
        PrepareModuleInput(
            input_layouts=(Replicate(),),
            desired_input_layouts=(Replicate(),),
            input_kwarg_layouts={"scale": Replicate()},
            desired_input_kwarg_layouts={"scale": Replicate()},
            use_local_output=True,
        ).apply(m, mesh)
        x = torch.ones(2, 3)
        s = torch.tensor(2.0)
        out_x, out_s = m(x, scale=s)
        self.assertIsInstance(out_x, torch.Tensor)
        self.assertIsInstance(out_s, torch.Tensor)

    def test_repr_contains_key_fields(self):
        style = PrepareModuleInput(
            input_layouts=Shard(0),
            desired_input_layouts=Replicate(),
            use_local_output=False,
        )
        r = repr(style)
        self.assertIn("PrepareModuleInput", r)
        self.assertIn("use_local_output=False", r)

    def test_prepare_input_fn_returns_inputs_when_layouts_none(self):
        """
        Feature: PrepareModuleInput._prepare_input_fn with no layouts
        Description: default constructor leaves input_layouts as None
        Expectation: inputs tuple is returned unchanged
        """
        style = PrepareModuleInput()
        mesh = MagicMock()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))
        self.assertIs(style._prepare_input_fn(inputs, mesh), inputs)

    def test_prepare_input_fn_wraps_non_tuple_input(self):
        """
        Feature: PrepareModuleInput._prepare_input_fn normalizes scalar args
        Description: forward passes a single tensor instead of a one-tuple
        Expectation: tensor is prepared like the sole positional argument
        """
        style = PrepareModuleInput(
            input_layouts=Replicate(),
            desired_input_layouts=Replicate(),
            use_local_output=True,
        )
        mesh = MagicMock()
        tensor = torch.ones(2, 3)
        with patch.object(style, "_prepare_input_arg", return_value=tensor) as mock_arg:
            out = style._prepare_input_fn(tensor, mesh)
            mock_arg.assert_called_once_with(tensor, mesh, Replicate(), Replicate())
            self.assertEqual(out, (tensor,))

    def test_prepare_input_fn_raises_when_desired_layouts_none_at_runtime(self):
        """
        Feature: PrepareModuleInput._prepare_input_fn validates desired_input_layouts
        Description: input_layouts set at runtime while desired_input_layouts is None
        Expectation: AssertionError from defensive check in _prepare_input_fn
        """
        style = PrepareModuleInput()
        style.input_layouts = (Replicate(),)
        style.desired_input_layouts = None
        mesh = MagicMock()
        with self.assertRaises(AssertionError):
            style._prepare_input_fn((torch.randn(1, 1),), mesh)

    def test_prepare_input_arg_passes_through_existing_dtensor(self):
        """
        Feature: PrepareModuleInput._prepare_input_arg DTensor branch
        Description: input is already a DTensor with a layout that needs redistribution
        Expectation: redistribute is called; result is local when use_local_output=True
        """
        style = PrepareModuleInput(
            input_layouts=Replicate(),
            desired_input_layouts=Shard(0),
            use_local_output=True,
        )
        mesh = MagicMock()
        mock_dt = MagicMock(spec=DTensor)
        redistributed = MagicMock(spec=DTensor)
        mock_dt.redistribute.return_value = redistributed
        local = torch.randn(2, 2)
        redistributed.to_local.return_value = local

        result = style._prepare_input_arg(mock_dt, mesh, Replicate(), Shard(0))
        mock_dt.redistribute.assert_called_once_with(mesh, (Shard(0),))
        self.assertTrue(torch.allclose(result, local))

    @patch("hyper_parallel.core.tensor_parallel.style.platform")
    def test_prepare_input_arg_raises_for_non_tensor(self, mock_platform):
        """
        Feature: PrepareModuleInput._prepare_input_arg type validation
        Description: non-tensor, non-DTensor positional value with a layout set
        Expectation: AssertionError
        """
        mock_platform.is_tensor.return_value = False
        style = PrepareModuleInput(
            input_layouts=Replicate(),
            desired_input_layouts=Replicate(),
        )
        with self.assertRaises(AssertionError):
            style._prepare_input_arg(42, MagicMock(), Replicate(), Replicate())


class TestPrepareModuleOutput(unittest.TestCase):
    """Tests aligned with PyTorch ``TensorParallelAPITests.test_prepare_module_output``."""

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def _setup_mock_platform(self, mock_platform, world_size: int = 1):
        mock_platform.platform_type = PlatformType.PYTORCH
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else __import__("numpy").array(t)
        )

    def _make_1d_mesh(self, mock_platform, size: int = 1):
        self._setup_mock_platform(mock_platform, world_size=size)
        return init_device_mesh(
            device_type="cpu",
            mesh_shape=(size,),
            mesh_dim_names=("tp",),
            init_backend=False,
        )

    def test_constructor_raises_on_layout_length_mismatch(self):
        with self.assertRaises(AssertionError):
            PrepareModuleOutput(
                output_layouts=(Replicate(), Replicate()),
                desired_output_layouts=(Shard(0),),
            )

    def test_repr_contains_key_fields(self):
        """``repr`` includes class name and layout settings."""
        style = PrepareModuleOutput(
            output_layouts=Replicate(),
            desired_output_layouts=Shard(0),
            use_local_output=False,
        )
        r = repr(style)
        self.assertIn("PrepareModuleOutput", r)
        self.assertIn("use_local_output=False", r)

    def test_prepare_out_fn_raises_on_output_length_mismatch(self):
        """
        Feature: PrepareModuleOutput._prepare_out_fn arity validation
        Description: module returns more tensors than output_layouts entries
        Expectation: ValueError mentioning same length
        """
        style = PrepareModuleOutput(
            output_layouts=(Replicate(),),
            desired_output_layouts=(Replicate(),),
        )
        mesh = MagicMock()
        with self.assertRaises(ValueError) as ctx:
            style._prepare_out_fn((torch.randn(1), torch.randn(1)), mesh)
        self.assertIn("same length", str(ctx.exception))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_forward_matches_pytorch_prepare_module_output_case(self, mock_platform):
        """Same as PyTorch ``test_prepare_module_output`` (Replicate -> Shard(0) on hook)."""
        mesh = self._make_1d_mesh(mock_platform, size=1)
        module = _DummyIdentityModule()
        PrepareModuleOutput(
            output_layouts=Replicate(),
            desired_output_layouts=Shard(0),
            use_local_output=True,
        ).apply(module, mesh)
        torch.manual_seed(15)
        inp = torch.rand(16, 7)
        dtensor = DTensor.from_local(inp, mesh, [Replicate()])
        with _patch_platform_rank_for_dtensor_redistribute(world_size=1):
            output = module(dtensor)
            expected = dtensor.redistribute(mesh, [Shard(0)]).to_local()
        self.assertTrue(torch.allclose(expected, output))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_none_placeholder_skips_second_output(self, mock_platform):
        """``None`` in ``output_layouts`` leaves the matching return value untouched."""
        mesh = self._make_1d_mesh(mock_platform, size=1)
        module = _DupOutModule()
        PrepareModuleOutput(
            output_layouts=(Replicate(), None),
            desired_output_layouts=(Shard(0), None),
            use_local_output=True,
        ).apply(module, mesh)
        with _patch_platform_rank_for_dtensor_redistribute(world_size=1):
            a, b = module(torch.randn(3, 4))
        self.assertIsInstance(a, torch.Tensor)
        self.assertIsInstance(b, torch.Tensor)
        self.assertEqual(b.item(), 1.0)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_use_local_output_false_returns_dtensor(self, mock_platform):
        """With ``use_local_output=False``, the hook leaves outputs as :class:`DTensor`."""
        mesh = self._make_1d_mesh(mock_platform, size=1)
        module = _DummyIdentityModule()
        PrepareModuleOutput(
            output_layouts=Replicate(),
            desired_output_layouts=Replicate(),
            use_local_output=False,
        ).apply(module, mesh)
        inp = torch.ones(2, 2)
        out = module(inp)
        self.assertIsInstance(out, DTensor)


class TestPrepareModuleInputOutput(unittest.TestCase):
    """Tests aligned with PyTorch ``TensorParallelAPITests.test_prepare_module_input_output``."""

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def _setup_mock_platform(self, mock_platform, world_size: int = 1):
        mock_platform.platform_type = PlatformType.PYTORCH
        mock_platform.get_rank.return_value = 0
        mock_platform.get_world_size.return_value = world_size
        mock_platform.tensor_to_numpy.side_effect = (
            lambda t: t.numpy() if hasattr(t, "numpy") else __import__("numpy").array(t)
        )

    def _make_1d_mesh(self, mock_platform, size: int = 1):
        self._setup_mock_platform(mock_platform, world_size=size)
        return init_device_mesh(
            device_type="cpu",
            mesh_shape=(size,),
            mesh_dim_names=("tp",),
            init_backend=False,
        )

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_forward_matches_pytorch_prepare_module_input_output_case(self, mock_platform):
        """Same end-to-end tensor equality as PyTorch ``test_prepare_module_input_output``."""
        mesh = self._make_1d_mesh(mock_platform, size=1)
        module = _DummyIdentityModule()
        PrepareModuleInputOutput(
            input_layouts=Shard(0),
            desired_input_layouts=Replicate(),
            output_layouts=Replicate(),
            desired_output_layouts=Shard(1),
        ).apply(module, mesh)
        inp = torch.rand(5, 7)
        with _patch_platform_rank_for_dtensor_redistribute(world_size=1):
            output = module(inp)
            expected = (
                DTensor.from_local(inp, mesh, [Shard(0)])
                .redistribute(mesh, [Shard(1)])
                .to_local()
            )
        self.assertTrue(torch.allclose(expected, output))

    def test_delegates_to_prepare_substyles_in_repr(self):
        """``repr`` exposes nested PrepareModuleInput / PrepareModuleOutput settings."""
        style = PrepareModuleInputOutput(
            input_layouts=Replicate(),
            desired_input_layouts=Shard(0),
            output_layouts=Shard(0),
            desired_output_layouts=Replicate(),
            use_local_input=True,
            use_local_output=False,
        )
        r = repr(style)
        self.assertIn("PrepareModuleInputOutput", r)
        self.assertIn("use_local_input=True", r)
        self.assertIn("use_local_output=False", r)


if __name__ == "__main__":
    unittest.main()
