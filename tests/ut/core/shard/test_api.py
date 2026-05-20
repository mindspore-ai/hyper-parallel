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
"""Unit tests for hyper_parallel.core.shard.api"""
import os
import unittest
from unittest.mock import MagicMock, patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "mindspore"

import numpy as np

from hyper_parallel.core.dtensor.dtensor import DTensor, _build_layout
from hyper_parallel.core.dtensor.placement_types import Shard, Replicate
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh, _DEVICE_MESH_MAP
from hyper_parallel.core.shard.sharding_plan import ShardingPlan
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS
from hyper_parallel.core.shard import api as shard_api


def _make_mesh(mock_platform, mesh_shape, mesh_dim_names):
    EXISTING_COMM_GROUPS.clear()
    _DEVICE_MESH_MAP.clear()
    mock_platform.get_rank.return_value = 0
    mock_platform.get_world_size.return_value = int(np.prod(mesh_shape))
    mock_platform.tensor_to_numpy.side_effect = (
        lambda t: t.numpy() if hasattr(t, "numpy") else np.array(t)
    )
    return init_device_mesh(
        device_type="npu",
        mesh_shape=mesh_shape,
        mesh_dim_names=mesh_dim_names,
        init_backend=False,
    )


class TestHasKwargs(unittest.TestCase):
    """
    Feature: _has_kwargs function
    Description: Detect whether a function has keyword arguments with defaults.
    Expectation: Returns True when any param has a default value, False otherwise.
    """

    def test_no_kwargs_returns_false(self):
        """Function with only positional args returns False."""
        def f(x, y):
            pass
        self.assertFalse(shard_api._has_kwargs(f))

    def test_with_kwargs_returns_true(self):
        """Function with default arg returns True."""
        def f(x, y=1):
            pass
        self.assertTrue(shard_api._has_kwargs(f))

    def test_all_kwargs_returns_true(self):
        """All-keyword function returns True."""
        def f(x=1, y=2):
            pass
        self.assertTrue(shard_api._has_kwargs(f))

    def test_no_args_returns_false(self):
        """Zero-argument function returns False."""
        def f():
            pass
        self.assertFalse(shard_api._has_kwargs(f))


class TestGetParamName(unittest.TestCase):
    """
    Feature: _get_param_name function
    Description: Extract parameter names from a function signature.
    Expectation: Returns list of parameter names in declaration order.
    """

    def test_simple(self):
        """Basic positional parameter extraction."""
        def f(a, b, c):
            pass
        self.assertEqual(shard_api._get_param_name(f), ["a", "b", "c"])

    def test_with_defaults(self):
        """Parameters with defaults are included in the name list."""
        def f(x, y=0, z=None):
            pass
        self.assertEqual(shard_api._get_param_name(f), ["x", "y", "z"])

    def test_empty(self):
        """Zero-argument function returns empty list."""
        def f():
            pass
        self.assertEqual(shard_api._get_param_name(f), [])


class TestConvertShardingPlan(unittest.TestCase):
    """
    Feature: _convert_sharding_plan function
    Description: Convert placement tuples and dicts to Layout objects.
    Expectation: Placement tuples become Layout; None values are preserved.
    """

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_parameter_placement_tuple_converted(self, mock_platform):
        """Placement tuple in 'parameter' key is converted to Layout."""
        from hyper_parallel.core.dtensor.layout import Layout
        mesh = _make_mesh(mock_platform, (2, 2), ("dp", "tp"))
        plan = {"parameter": {"w": (Replicate(), Shard(1))}}
        result = shard_api._convert_sharding_plan(plan, mesh)
        self.assertIn("parameter", result)
        self.assertIsInstance(result["parameter"]["w"], Layout)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_none_value_preserved(self, mock_platform):
        """None value in plan is kept as None."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        plan = {"parameter": {"w": None}}
        result = shard_api._convert_sharding_plan(plan, mesh)
        self.assertIsNone(result["parameter"]["w"])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_forward_input_wrapped_in_list(self, mock_platform):
        """A single placement tuple under 'forward'/'input' is wrapped in a list."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        plan = {"forward": {"input": (Shard(0),)}}
        result = shard_api._convert_sharding_plan(plan, mesh)
        forward = result.get("forward", {})
        self.assertIn("input", forward)
        self.assertIsInstance(forward["input"], list)
        self.assertEqual(len(forward["input"]), 1)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_forward_output_wrapped_in_list(self, mock_platform):
        """A single placement tuple under 'forward'/'output' is wrapped in a list."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        plan = {"forward": {"output": (Replicate(),)}}
        result = shard_api._convert_sharding_plan(plan, mesh)
        forward = result.get("forward", {})
        self.assertIn("output", forward)
        self.assertIsInstance(forward["output"], list)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_alias_string_placement_converted(self, mock_platform):
        """Alias string tuple is also converted to Layout."""
        from hyper_parallel.core.dtensor.layout import Layout
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        plan = {"parameter": {"w": ("dp",)}}
        result = shard_api._convert_sharding_plan(plan, mesh)
        self.assertIsInstance(result["parameter"]["w"], Layout)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_list_of_placements_in_forward_input(self, mock_platform):
        """A list of placement tuples under forward/input is converted element-wise."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        plan = {"forward": {"input": [(Shard(0),), (Replicate(),)]}}
        result = shard_api._convert_sharding_plan(plan, mesh)
        forward = result.get("forward", {})
        self.assertIsInstance(forward["input"], list)
        self.assertEqual(len(forward["input"]), 2)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_dict_format_in_forward_input(self, mock_platform):
        """Dict format for forward/input (kwargs) is converted to dict of Layouts."""
        from hyper_parallel.core.dtensor.layout import Layout
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        plan = {"forward": {"input": {"x": (Shard(0),), "y": (Replicate(),)}}}
        result = shard_api._convert_sharding_plan(plan, mesh)
        forward = result.get("forward", {})
        self.assertIsInstance(forward["input"], dict)
        self.assertIsInstance(forward["input"]["x"], Layout)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_non_placement_value_returned_as_is(self, mock_platform):
        """A non-placement scalar value is returned unchanged."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        plan = {"parameter": {"scale": 1.0}}
        result = shard_api._convert_sharding_plan(plan, mesh)
        self.assertEqual(result["parameter"]["scale"], 1.0)


class TestParallelIn(unittest.TestCase):
    """
    Feature: _parallel_in function
    Description: Redistribute DTensor args/kwargs to specified layouts before forward.
    Expectation: DTensors are redistributed; non-DTensors pass through unchanged.
    """

    def test_invalid_layout_type_raises_value_error(self):
        """layouts must be list, tuple, or dict."""
        def f(x):
            pass
        with self.assertRaises(ValueError):
            shard_api._parallel_in(f, (MagicMock(spec=DTensor),), {}, "bad_layout")

    def test_non_dtensor_arg_passes_through(self):
        """Non-DTensor positional args are passed through unchanged."""
        def f(x):
            pass
        result_args, result_kwargs = shard_api._parallel_in(f, (42,), {}, [None])
        self.assertEqual(result_args, (42,))

    def test_dtensor_arg_is_redistributed(self):
        """DTensor positional arg is redistributed to target layout."""
        from hyper_parallel.core.dtensor.layout import Layout

        target_layout = MagicMock(spec=Layout)
        target_layout.mesh = MagicMock()
        target_layout.alias_placements = (Replicate(),)

        local_t = np.array([1.0, 2.0])
        dt = MagicMock(spec=DTensor)
        dt.redistribute.return_value = dt

        def f(x):
            pass

        result_args, _ = shard_api._parallel_in(f, (dt,), {}, [target_layout])
        dt.redistribute.assert_called_once_with(
            target_layout.mesh, target_layout.alias_placements
        )

    def test_dict_layout_applies_to_kwargs(self):
        """Dict layout applies redistribution to matching kwargs."""
        from hyper_parallel.core.dtensor.layout import Layout

        target_layout = MagicMock(spec=Layout)
        target_layout.mesh = MagicMock()
        target_layout.alias_placements = (Replicate(),)

        dt = MagicMock(spec=DTensor)
        dt.redistribute.return_value = dt

        def f(x, key=None):
            pass

        _, result_kwargs = shard_api._parallel_in(
            f, (), {"key": dt}, {"key": target_layout}
        )
        dt.redistribute.assert_called_once()

    def test_non_dtensor_kwarg_passes_through(self):
        """Non-DTensor kwargs are passed through unchanged when layout is None."""
        def f(x, y=None):
            pass
        _, result_kwargs = shard_api._parallel_in(
            f, (), {"y": 99}, {"y": None}
        )
        self.assertEqual(result_kwargs["y"], 99)


class TestParallelOut(unittest.TestCase):
    """
    Feature: _parallel_out function
    Description: Redistribute DTensor outputs to specified layouts after forward.
    Expectation: Each output is redistributed to its corresponding layout.
    """

    def test_invalid_layout_type_raises_value_error(self):
        """layouts must be list or tuple."""
        from hyper_parallel.core.dtensor.layout import Layout
        dt = MagicMock(spec=DTensor)
        with self.assertRaises(ValueError):
            shard_api._parallel_out(dt, "bad")

    def test_tuple_output_size_mismatch_raises_value_error(self):
        """Mismatch between output tuple size and layout list size raises ValueError."""
        from hyper_parallel.core.dtensor.layout import Layout
        layout = MagicMock(spec=Layout)
        dt1 = MagicMock(spec=DTensor)
        dt2 = MagicMock(spec=DTensor)
        with self.assertRaises(ValueError):
            shard_api._parallel_out((dt1, dt2), [layout])

    def test_single_dtensor_output_redistributed(self):
        """Single DTensor output is redistributed to layouts[0]."""
        from hyper_parallel.core.dtensor.layout import Layout

        layout = MagicMock(spec=Layout)
        layout.mesh = MagicMock()
        layout.alias_placements = (Replicate(),)

        dt = MagicMock(spec=DTensor)
        dt.redistribute.return_value = dt

        shard_api._parallel_out(dt, [layout])
        dt.redistribute.assert_called_once_with(layout.mesh, layout.alias_placements)

    def test_single_output_with_multiple_layouts_raises(self):
        """Single non-tuple output with multiple layouts raises ValueError."""
        from hyper_parallel.core.dtensor.layout import Layout
        layout1 = MagicMock(spec=Layout)
        layout2 = MagicMock(spec=Layout)
        dt = MagicMock(spec=DTensor)
        with self.assertRaises(ValueError):
            shard_api._parallel_out(dt, [layout1, layout2])

    def test_non_dtensor_output_passes_through(self):
        """Non-DTensor single output passes through unchanged."""
        from hyper_parallel.core.dtensor.layout import Layout
        layout = MagicMock(spec=Layout)
        layout.mesh = MagicMock()
        layout.alias_placements = (Replicate(),)
        result = shard_api._parallel_out(42, [layout])
        self.assertEqual(result, 42)

    def test_tuple_output_redistributed_per_layout(self):
        """Tuple of DTensor outputs, each redistributed to its layout."""
        from hyper_parallel.core.dtensor.layout import Layout

        layout1 = MagicMock(spec=Layout)
        layout1.mesh = MagicMock()
        layout1.alias_placements = (Replicate(),)
        layout2 = MagicMock(spec=Layout)
        layout2.mesh = MagicMock()
        layout2.alias_placements = (Shard(0),)

        dt1 = MagicMock(spec=DTensor)
        dt1.redistribute.return_value = dt1
        dt2 = MagicMock(spec=DTensor)
        dt2.redistribute.return_value = dt2

        result = shard_api._parallel_out((dt1, dt2), [layout1, layout2])
        self.assertIsInstance(result, tuple)
        dt1.redistribute.assert_called_once_with(layout1.mesh, layout1.alias_placements)
        dt2.redistribute.assert_called_once_with(layout2.mesh, layout2.alias_placements)

    def test_tuple_output_non_dtensor_element_passes_through(self):
        """Non-DTensor elements inside tuple output are kept unchanged."""
        from hyper_parallel.core.dtensor.layout import Layout
        layout = MagicMock(spec=Layout)
        result = shard_api._parallel_out((None,), [layout])
        self.assertIsInstance(result, tuple)
        self.assertIsNone(result[0])


class TestShardCallable(unittest.TestCase):
    """
    Feature: _shard_callable function
    Description: Wrap a callable with input/output redistribution.
    Expectation: Returned function applies parallel_in/parallel_out around the original call.
    """

    def test_no_forward_plan_returns_original(self):
        """When forward key is missing, returns the original function."""
        def f(x):
            return x

        wrapped = shard_api._shard_callable(f, {})
        self.assertIs(wrapped, f)

    def test_forward_plan_wraps_function(self):
        """With forward plan, a wrapper function is returned."""
        def f(x):
            return x

        mock_layout = MagicMock()
        mock_layout.mesh = MagicMock()
        mock_layout.alias_placements = ()

        plan = {"forward": {"input": [mock_layout], "output": [mock_layout]}}
        wrapped = shard_api._shard_callable(f, plan)
        self.assertIsNot(wrapped, f)
        self.assertTrue(callable(wrapped))


class TestShardModule(unittest.TestCase):
    """
    Feature: shard_module function
    Description: Test the shard_module API for world_size==1 early return and type validation.
    Expectation: Returns None when world_size==1; raises TypeError for invalid sharding_plan.
    """

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    @patch("hyper_parallel.core.shard.api.platform")
    def test_world_size_one_returns_none(self, mock_api_platform):
        """When world_size==1, shard_module returns None immediately."""
        mock_api_platform.get_world_size.return_value = 1
        model = MagicMock()
        mesh = MagicMock()
        plan = ShardingPlan()
        result = shard_api.shard_module(model, mesh, plan)
        self.assertIsNone(result)

    @patch("hyper_parallel.core.shard.api.platform")
    def test_invalid_sharding_plan_type_raises(self, mock_api_platform):
        """Passing a dict instead of ShardingPlan raises TypeError."""
        mock_api_platform.get_world_size.return_value = 2
        mock_api_platform.Module = object

        model = MagicMock()
        mesh = MagicMock()

        with self.assertRaises(TypeError):
            shard_api.shard_module(model, mesh, {"w": (Shard(0),)})

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    @patch("hyper_parallel.core.shard.api.platform")
    def test_callable_is_wrapped_when_not_module(self, mock_api_platform, mock_mesh_platform):
        """When model is not a Module, _shard_callable wraps it."""
        mock_api_platform.get_world_size.return_value = 2
        mock_api_platform.Module = type(None)

        _make_mesh(mock_mesh_platform, (2,), ("dp",))
        mesh = list(_DEVICE_MESH_MAP.values())[0]

        def my_func(x):
            return x

        plan = ShardingPlan()
        result = shard_api.shard_module(my_func, mesh, plan)
        self.assertIs(result, my_func)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    @patch("hyper_parallel.core.shard.api.platform")
    def test_input_plan_must_be_dict(self, mock_api_platform, mock_mesh_platform):
        """Non-dict input_plan raises TypeError."""
        mock_api_platform.get_world_size.return_value = 2
        mock_api_platform.Module = object

        _make_mesh(mock_mesh_platform, (2,), ("dp",))
        mesh = list(_DEVICE_MESH_MAP.values())[0]

        model = MagicMock(spec=object)
        plan = ShardingPlan(input_plan="bad_input")

        with self.assertRaises(TypeError):
            shard_api.shard_module(model, mesh, plan)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    @patch("hyper_parallel.core.shard.api.platform")
    def test_output_plan_must_be_dict(self, mock_api_platform, mock_mesh_platform):
        """Non-dict output_plan raises TypeError."""
        mock_api_platform.get_world_size.return_value = 2
        mock_api_platform.Module = object

        _make_mesh(mock_mesh_platform, (2,), ("dp",))
        mesh = list(_DEVICE_MESH_MAP.values())[0]

        model = MagicMock(spec=object)
        plan = ShardingPlan(output_plan="bad_output")

        with self.assertRaises(TypeError):
            shard_api.shard_module(model, mesh, plan)


class TestRegisterHook(unittest.TestCase):
    """
    Feature: _register_hook function
    Description: Register forward hooks on submodules according to sharding plan.
    Expectation: ValueError raised when key does not end with 'input' or 'output'.
    """

    def test_invalid_key_suffix_raises(self):
        """Keys not ending in 'input'/'output' raise ValueError."""
        model = MagicMock()
        model.named_modules = MagicMock(return_value=[("", model)])

        with patch("hyper_parallel.core.shard.api.platform") as mock_platform:
            mock_platform.get_cells_and_names.return_value = [("", model)]
            mock_layout = MagicMock()
            plan = {"bad_key": mock_layout}
            with self.assertRaises(ValueError):
                shard_api._register_hook(model, plan)


class TestConvertShardingPlanEdgePaths(unittest.TestCase):
    """
    Feature: _convert_sharding_plan edge paths
    Description: Cover _is_placement_tuple nested-tuple branches, _convert_value list/tuple
                 fallbacks, _convert_forward_plan None path, and top-level input/output keys.
    Expectation: Correct conversion or False from _is_placement_tuple for each case.
    """

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_nested_string_tuple_is_valid_placement(self, mock_platform):
        """Nested tuple of strings (multi-axis alias) is a valid placement element."""
        from hyper_parallel.core.dtensor.layout import Layout
        mesh = _make_mesh(mock_platform, (2, 2), ("dp", "tp"))
        plan = {"parameter": {"w": (("dp", "tp"),)}}
        result = shard_api._convert_sharding_plan(plan, mesh)
        self.assertIsInstance(result["parameter"]["w"], Layout)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_nested_placement_tuple_is_not_placement(self, mock_platform):
        """Tuple of placement tuples is NOT a placement spec — falls to _convert_value tuple branch."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        plan = {"parameter": {"w": ((Shard(0),), (Shard(1),))}}
        result = shard_api._convert_sharding_plan(plan, mesh)
        self.assertIsInstance(result["parameter"]["w"], list)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_empty_nested_tuple_makes_outer_not_placement(self, mock_platform):
        """Outer tuple with empty inner tuple is not a placement spec."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        plan = {"parameter": {"w": ((),)}}
        result = shard_api._convert_sharding_plan(plan, mesh)
        self.assertIsInstance(result["parameter"]["w"], list)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_non_placement_item_in_tuple_not_placement(self, mock_platform):
        """Tuple containing non-str/Placement item is not a placement spec."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        plan = {"parameter": {"w": (1,)}}
        result = shard_api._convert_sharding_plan(plan, mesh)
        self.assertIsInstance(result["parameter"]["w"], list)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_convert_value_list_case(self, mock_platform):
        """_convert_value processes list values by recursing on each element."""
        from hyper_parallel.core.dtensor.layout import Layout
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        plan = {"forward": {"input": {"x": [(Shard(0),), (Replicate(),)]}}}
        result = shard_api._convert_sharding_plan(plan, mesh)
        items = result["forward"]["input"]["x"]
        self.assertIsInstance(items, list)
        self.assertIsInstance(items[0], Layout)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_convert_value_non_placement_tuple_case(self, mock_platform):
        """_convert_value processes tuple-of-tuples by recursing on each element."""
        from hyper_parallel.core.dtensor.layout import Layout
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        plan = {"forward": {"input": {"x": ((Shard(0),), (Replicate(),))}}}
        result = shard_api._convert_sharding_plan(plan, mesh)
        items = result["forward"]["input"]["x"]
        self.assertIsInstance(items, list)
        self.assertIsInstance(items[0], Layout)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_convert_forward_plan_none_returns_none(self, mock_platform):
        """_convert_sharding_plan({'forward': None}, mesh) keeps forward as None."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        plan = {"forward": None}
        result = shard_api._convert_sharding_plan(plan, mesh)
        self.assertIsNone(result.get("forward"))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_forward_input_value_none_preserved(self, mock_platform):
        """None value for a forward input key is preserved as None."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        plan = {"forward": {"input": None}}
        result = shard_api._convert_sharding_plan(plan, mesh)
        self.assertIsNone(result["forward"]["input"])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_forward_non_input_output_key_converted(self, mock_platform):
        """A forward key not ending in input/output is converted via _convert_value."""
        from hyper_parallel.core.dtensor.layout import Layout
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        plan = {"forward": {"mask": (Shard(0),)}}
        result = shard_api._convert_sharding_plan(plan, mesh)
        self.assertIsInstance(result["forward"]["mask"], Layout)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_top_level_input_key_none(self, mock_platform):
        """Top-level 'input' key with None value is preserved."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        plan = {"input": None}
        result = shard_api._convert_sharding_plan(plan, mesh)
        self.assertIsNone(result["input"])

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_top_level_input_key_placement_tuple(self, mock_platform):
        """Top-level 'input' key with placement tuple is wrapped in a list."""
        from hyper_parallel.core.dtensor.layout import Layout
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        plan = {"input": (Shard(0),)}
        result = shard_api._convert_sharding_plan(plan, mesh)
        self.assertIsInstance(result["input"], list)
        self.assertIsInstance(result["input"][0], Layout)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_top_level_input_key_dict(self, mock_platform):
        """Top-level 'input' key with dict converts each placement value."""
        from hyper_parallel.core.dtensor.layout import Layout
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        plan = {"input": {"x": (Shard(0),)}}
        result = shard_api._convert_sharding_plan(plan, mesh)
        self.assertIsInstance(result["input"], dict)
        self.assertIsInstance(result["input"]["x"], Layout)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_top_level_input_key_list_of_tuples(self, mock_platform):
        """Top-level 'input' key with list of placement tuples converts element-wise."""
        from hyper_parallel.core.dtensor.layout import Layout
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        plan = {"input": [(Shard(0),), (Replicate(),)]}
        result = shard_api._convert_sharding_plan(plan, mesh)
        self.assertIsInstance(result["input"], list)
        self.assertEqual(len(result["input"]), 2)
        self.assertIsInstance(result["input"][0], Layout)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    def test_forward_input_scalar_passes_through(self, mock_platform):
        """Forward input with non-standard scalar uses line 164 path; scalar returned as-is."""
        mesh = _make_mesh(mock_platform, (2,), ("dp",))
        plan = {"forward": {"input": 42}}
        result = shard_api._convert_sharding_plan(plan, mesh)
        self.assertEqual(result["forward"]["input"], 42)


class TestParallelInDictLayout(unittest.TestCase):
    """
    Feature: _parallel_in with dict layouts
    Description: When layout is a dict, _get_layout uses param name as key.
    Expectation: DTensor args are redistributed using the named layout.
    """

    def test_dtensor_arg_with_dict_layout(self):
        """DTensor arg with dict layout uses _get_layout(index, is_list=False)."""
        from hyper_parallel.core.dtensor.layout import Layout

        target_layout = MagicMock(spec=Layout)
        target_layout.mesh = MagicMock()
        target_layout.alias_placements = (Replicate(),)

        dt = MagicMock(spec=DTensor)
        dt.redistribute.return_value = dt

        def f(x):
            pass

        result_args, _ = shard_api._parallel_in(f, (dt,), {}, {"x": target_layout})
        dt.redistribute.assert_called_once_with(
            target_layout.mesh, target_layout.alias_placements
        )


class TestHookFunctions(unittest.TestCase):
    """
    Feature: Forward hook functions
    Description: _forward_pre_hook / _forward_hook process DTensor layouts when set.
    Expectation: When cell.in_layout / out_layout is not None, redistribution occurs.
    """

    @patch("hyper_parallel.core.shard.api.platform")
    def test_forward_pre_hook_with_layout(self, mock_platform):
        """_forward_pre_hook processes args when cell.in_layout is set."""
        cell = MagicMock()
        cell.in_layout = [None]

        def construct_fn(x):
            pass

        mock_platform.get_cell_construct.return_value = construct_fn
        result = shard_api._forward_pre_hook(cell, (42,))
        self.assertEqual(result, (42,))

    @patch("hyper_parallel.core.shard.api.platform")
    def test_forward_pre_hook_no_layout_returns_args(self, mock_platform):
        """_forward_pre_hook returns args unchanged when in_layout is None."""
        cell = MagicMock()
        cell.in_layout = None
        result = shard_api._forward_pre_hook(cell, (1, 2))
        self.assertEqual(result, (1, 2))

    @patch("hyper_parallel.core.shard.api.platform")
    def test_forward_pre_with_kwargs_hook_with_layout(self, mock_platform):
        """_forward_pre_with_kwargs_hook processes args/kwargs when in_layout is set."""
        cell = MagicMock()
        cell.in_layout = [None]

        def construct_fn(x):
            pass

        mock_platform.get_cell_construct.return_value = construct_fn
        args_out, kwargs_out = shard_api._forward_pre_with_kwargs_hook(cell, (42,), {"k": 1})
        self.assertEqual(args_out, (42,))

    @patch("hyper_parallel.core.shard.api.platform")
    def test_forward_pre_with_kwargs_hook_no_layout(self, mock_platform):
        """_forward_pre_with_kwargs_hook returns unchanged when in_layout is None."""
        cell = MagicMock()
        cell.in_layout = None
        args_out, kwargs_out = shard_api._forward_pre_with_kwargs_hook(cell, (5,), {"a": 7})
        self.assertEqual(args_out, (5,))
        self.assertEqual(kwargs_out, {"a": 7})

    def test_forward_hook_with_out_layout(self):
        """_forward_hook redistributes outputs when out_layout is set."""
        from hyper_parallel.core.dtensor.layout import Layout
        layout = MagicMock(spec=Layout)
        layout.mesh = MagicMock()
        layout.alias_placements = (Replicate(),)

        cell = MagicMock()
        cell.out_layout = [layout]

        dt = MagicMock(spec=DTensor)
        dt.redistribute.return_value = dt

        result = shard_api._forward_hook(cell, (), dt)
        dt.redistribute.assert_called_once()

    def test_forward_hook_no_layout_returns_outputs(self):
        """_forward_hook returns outputs unchanged when out_layout is None."""
        cell = MagicMock()
        cell.out_layout = None
        result = shard_api._forward_hook(cell, (), "output_value")
        self.assertEqual(result, "output_value")

    def test_forward_with_kwargs_hook_delegates(self):
        """_forward_with_kwargs_hook delegates to _forward_hook."""
        cell = MagicMock()
        cell.out_layout = None
        result = shard_api._forward_with_kwargs_hook(cell, (), {}, "output_value")
        self.assertEqual(result, "output_value")


class TestRegisterHookValid(unittest.TestCase):
    """
    Feature: _register_hook valid paths
    Description: Registers input and output hooks on cells found in model.
    Expectation: register_forward_pre_hook / register_forward_hook called correctly.
    """

    @patch("hyper_parallel.core.shard.api.platform")
    def test_register_input_hook_no_kwargs(self, mock_platform):
        """Registers forward_pre_hook (no-kwargs variant) for input layout."""
        mock_cell = MagicMock()

        def construct_fn(x):
            pass

        mock_platform.get_cells_and_names.return_value = [("", mock_cell)]
        mock_platform.get_cell_construct.return_value = construct_fn

        mock_layout = MagicMock()
        shard_api._register_hook(mock_cell, {"input": [mock_layout]})

        mock_cell.register_forward_pre_hook.assert_called_once_with(
            shard_api._forward_pre_hook, with_kwargs=False
        )
        self.assertEqual(mock_cell.in_layout, [mock_layout])

    @patch("hyper_parallel.core.shard.api.platform")
    def test_register_output_hook_no_kwargs(self, mock_platform):
        """Registers forward_hook (no-kwargs variant) for output layout."""
        mock_cell = MagicMock()

        def construct_fn(x):
            pass

        mock_platform.get_cells_and_names.return_value = [("", mock_cell)]
        mock_platform.get_cell_construct.return_value = construct_fn

        mock_layout = MagicMock()
        shard_api._register_hook(mock_cell, {"output": [mock_layout]})

        mock_cell.register_forward_hook.assert_called_once_with(
            shard_api._forward_hook, with_kwargs=False
        )
        self.assertEqual(mock_cell.out_layout, [mock_layout])

    @patch("hyper_parallel.core.shard.api.platform")
    def test_register_hook_with_kwargs_construct(self, mock_platform):
        """Registers kwargs-variant hooks when construct has default args."""
        mock_cell = MagicMock()

        def construct_fn(x, y=None):
            pass

        mock_platform.get_cells_and_names.return_value = [("", mock_cell)]
        mock_platform.get_cell_construct.return_value = construct_fn

        mock_layout = MagicMock()
        shard_api._register_hook(mock_cell, {"input": [mock_layout]})

        mock_cell.register_forward_pre_hook.assert_called_once_with(
            shard_api._forward_pre_with_kwargs_hook, with_kwargs=True
        )

    @patch("hyper_parallel.core.shard.api.platform")
    def test_register_hook_none_value_skipped(self, mock_platform):
        """None value in plan is skipped without registering any hook."""
        mock_cell = MagicMock()
        mock_platform.get_cells_and_names.return_value = [("", mock_cell)]

        shard_api._register_hook(mock_cell, {"input": None})

        mock_cell.register_forward_pre_hook.assert_not_called()
        mock_cell.register_forward_hook.assert_not_called()

    @patch("hyper_parallel.core.shard.api.platform")
    def test_register_hook_submodule_key(self, mock_platform):
        """Plan key with dot prefix finds named sub-cell."""
        root_cell = MagicMock()
        sub_cell = MagicMock()

        def construct_fn(x):
            pass

        mock_platform.get_cells_and_names.return_value = [
            ("", root_cell), ("layer", sub_cell)
        ]
        mock_platform.get_cell_construct.return_value = construct_fn

        mock_layout = MagicMock()
        shard_api._register_hook(root_cell, {"layer.input": [mock_layout]})

        sub_cell.register_forward_pre_hook.assert_called_once()
        self.assertEqual(sub_cell.in_layout, [mock_layout])


class TestRegisterLocalTensorHook(unittest.TestCase):
    """
    Feature: _register_local_tensor_hook
    Description: Registers a hook that converts DTensor outputs to local tensors.
    Expectation: Hook is registered; DTensor outputs unwrapped, others passed through.
    """

    @patch("hyper_parallel.core.shard.api.platform")
    def test_hook_registered_on_named_cell(self, mock_platform):
        """Hook is registered on the specified sub-cell."""
        root = MagicMock()
        sub = MagicMock()
        mock_platform.get_cells_and_names.return_value = [("", root), ("sub", sub)]

        shard_api._register_local_tensor_hook(root, ["sub"])

        sub.register_forward_hook.assert_called_once()

    @patch("hyper_parallel.core.shard.api.platform")
    def test_hook_func_unwraps_dtensor(self, mock_platform):
        """Registered hook converts DTensor output to local tensor."""
        root = MagicMock()
        sub = MagicMock()
        captured_hook = []

        def fake_register(hook_fn):
            captured_hook.append(hook_fn)

        sub.register_forward_hook = fake_register
        mock_platform.get_cells_and_names.return_value = [("", root), ("sub", sub)]

        shard_api._register_local_tensor_hook(root, ["sub"])
        self.assertTrue(captured_hook)

        dt = MagicMock(spec=DTensor)
        dt.to_local.return_value = "local_tensor"
        result = captured_hook[0](sub, (), dt)
        self.assertEqual(result, "local_tensor")

    @patch("hyper_parallel.core.shard.api.platform")
    def test_hook_func_passes_non_dtensor(self, mock_platform):
        """Registered hook passes through non-DTensor scalar output."""
        root = MagicMock()
        sub = MagicMock()
        captured_hook = []

        sub.register_forward_hook = captured_hook.append
        mock_platform.get_cells_and_names.return_value = [("", root), ("sub", sub)]

        shard_api._register_local_tensor_hook(root, ["sub"])
        result = captured_hook[0](sub, (), 42)
        self.assertEqual(result, 42)

    @patch("hyper_parallel.core.shard.api.platform")
    def test_hook_func_unwraps_tuple_output(self, mock_platform):
        """Registered hook recursively unwraps DTensor in tuple output."""
        root = MagicMock()
        sub = MagicMock()
        captured_hook = []

        sub.register_forward_hook = captured_hook.append
        mock_platform.get_cells_and_names.return_value = [("", root), ("sub", sub)]

        shard_api._register_local_tensor_hook(root, ["sub"])

        dt = MagicMock(spec=DTensor)
        dt.to_local.return_value = "local"
        result = captured_hook[0](sub, (), (dt, 99))
        self.assertIsInstance(result, tuple)
        self.assertEqual(result[0], "local")
        self.assertEqual(result[1], 99)


class TestShardCallableExecution(unittest.TestCase):
    """
    Feature: _shard_callable wrapper execution
    Description: The returned _shard_wrapper actually calls the original function
                 and applies _parallel_out when output_layout is set.
    Expectation: Original function is called; non-DTensor output passes through unchanged.
    """

    def test_wrapper_calls_function_with_output_layout(self):
        """_shard_wrapper calls func and applies _parallel_out for non-DTensor output."""
        call_log = []

        def f(x):
            call_log.append(x)
            return x * 2

        mock_layout = MagicMock()
        mock_layout.mesh = MagicMock()
        mock_layout.alias_placements = ()

        plan = {"forward": {"output": [mock_layout]}}
        wrapped = shard_api._shard_callable(f, plan)

        result = wrapped(5)
        self.assertEqual(call_log, [5])
        self.assertEqual(result, 10)

    def test_wrapper_with_input_and_output_layout_non_dtensor(self):
        """_shard_wrapper applies _parallel_in then _parallel_out for non-DTensors."""
        call_log = []

        def f(x):
            call_log.append(x)
            return x + 1

        mock_layout = MagicMock()
        mock_layout.mesh = MagicMock()
        mock_layout.alias_placements = ()

        plan = {"forward": {"input": [None], "output": [mock_layout]}}
        wrapped = shard_api._shard_callable(f, plan)

        result = wrapped(3)
        self.assertEqual(call_log, [3])
        self.assertEqual(result, 4)


class TestShardModuleWithPlan(unittest.TestCase):
    """
    Feature: shard_module with non-empty ShardingPlan fields
    Description: Test branches for plan, input_plan, output_plan, return_local_tensor,
                 and parameter sharding error paths.
    Expectation: Correct plan normalization and errors for bad parameter names.
    """

    def setUp(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    def tearDown(self):
        EXISTING_COMM_GROUPS.clear()
        _DEVICE_MESH_MAP.clear()

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    @patch("hyper_parallel.core.shard.api.platform")
    def test_callable_with_full_plan(self, mock_api, mock_mesh):
        """shard_module with plan + input_plan + output_plan + return_local_tensor (callable)."""
        mock_api.get_world_size.return_value = 2
        mock_api.Module = type(None)

        _make_mesh(mock_mesh, (2,), ("dp",))
        mesh = list(_DEVICE_MESH_MAP.values())[0]

        def my_func(x):
            return x

        plan = ShardingPlan(
            plan={"w": (Shard(0),)},
            input_plan={"input": (Shard(0),)},
            output_plan={"output": (Replicate(),)},
            return_local_tensor=["layer"],
        )
        result = shard_api.shard_module(my_func, mesh, plan)
        self.assertTrue(callable(result))

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    @patch("hyper_parallel.core.shard.api.platform")
    @patch("hyper_parallel.core.shard.api.Module", object)
    def test_param_not_found_raises_value_error(self, mock_api, mock_mesh):
        """shard_module raises ValueError when param not found in model."""
        mock_api.get_world_size.return_value = 2
        mock_api.search_parameter_by_name.return_value = None

        _make_mesh(mock_mesh, (2,), ("dp",))
        mesh = list(_DEVICE_MESH_MAP.values())[0]

        model = object()
        plan = ShardingPlan(plan={"missing.weight": (Shard(0),)})
        with self.assertRaises(ValueError):
            shard_api.shard_module(model, mesh, plan)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    @patch("hyper_parallel.core.shard.api.platform")
    @patch("hyper_parallel.core.shard.api.Module", object)
    def test_param_layout_not_layout_raises_value_error(self, mock_api, mock_mesh):
        """shard_module raises ValueError when layout is not a Layout instance."""
        mock_api.get_world_size.return_value = 2

        _make_mesh(mock_mesh, (2,), ("dp",))
        mesh = list(_DEVICE_MESH_MAP.values())[0]

        model = object()
        plan = ShardingPlan(plan={"w": 42})
        with self.assertRaises((ValueError, TypeError)):
            shard_api.shard_module(model, mesh, plan)

    @patch("hyper_parallel.core.dtensor.device_mesh.platform")
    @patch("hyper_parallel.core.shard.api.platform")
    @patch("hyper_parallel.core.shard.api.Module", object)
    def test_param_found_and_applied(self, mock_api, mock_mesh):
        """shard_module applies layout to found parameter and returns model."""
        from hyper_parallel.core.dtensor.layout import Layout as _Layout

        mock_api.get_world_size.return_value = 2
        _make_mesh(mock_mesh, (2,), ("dp",))
        mesh = list(_DEVICE_MESH_MAP.values())[0]

        mock_param = MagicMock()
        mock_param.dim.return_value = 1
        mock_api.search_parameter_by_name.return_value = ("", "w", mock_param)
        mock_api.set_layout_into_parameter.return_value = mock_param
        mock_api.update_parameter_by_name.return_value = None

        model = object()
        plan = ShardingPlan(plan={"w": (Shard(0),)})
        result = shard_api.shard_module(model, mesh, plan)
        self.assertIs(result, model)
        mock_api.set_layout_into_parameter.assert_called_once()


if __name__ == "__main__":
    unittest.main()
