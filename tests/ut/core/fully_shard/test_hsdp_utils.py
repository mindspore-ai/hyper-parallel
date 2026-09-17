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
"""Unit tests for fully_shard hsdp_utils (no NPU required).

Covers ParamModuleInfo, _get_param_module_infos, and _named_parameters_with_duplicates
from hyper_parallel.core.fully_shard.hsdp_utils. All tests use CPU and simple nn.Modules.
"""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

# Force torch platform before any hyper_parallel imports
os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

# pylint: disable=C0413
import torch
from torch import nn

from hyper_parallel.core.fully_shard.hsdp_utils import (
    FullyShardParamMode,
    ParamModuleInfo,
    _get_param_module_infos,
    _named_parameters_with_duplicates,
    get_dtensor_managed_mesh,
    get_hsdp_state,
    get_managed_modules_parameters,
    get_rank_list_for_axes,
    get_split_rank_lists_for_axes,
    infer_fully_shard_param_mode,
    is_dtensor_managed_param,
    unwrap_dtensor_param,
)


class SimpleLinear(nn.Module):
    """Simple linear module for testing."""

    def __init__(self, in_features=4, out_features=4):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x @ self.weight + self.bias


class SharedParamModule(nn.Module):
    """Module with shared parameter (tied weights)."""

    def __init__(self, dim=4):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dim))
        self.linear = nn.Linear(dim, dim)
        self.linear.weight = self.weight  # shared

    def forward(self, x):
        return self.linear(x)


class TestParamModuleInfo(unittest.TestCase):
    """Unit tests for ParamModuleInfo (module, param_name, shared_modules)."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        self.device = torch.device("cpu")

    def test_basic_param_module_info(self):
        """ParamModuleInfo stores module and param_name correctly.

        description: Create ParamModuleInfo(module, param_name='weight') with no shared.
        expectation: module and param_name match; shared_modules/names are empty.
        feature: hsdp_utils ParamModuleInfo.
        """
        # Arrange
        mod = SimpleLinear(4, 4)
        # Act
        info = ParamModuleInfo(module=mod, param_name="weight")
        # Assert
        self.assertIs(info.module, mod)
        self.assertEqual(info.param_name, "weight")
        self.assertEqual(info.shared_modules, [])
        self.assertEqual(info.shared_param_names, [])

    def test_param_module_info_with_shared(self):
        """ParamModuleInfo tracks shared modules.

        description: Create ParamModuleInfo with shared_modules and shared_param_names.
        expectation: shared_modules and shared_param_names stored and match.
        feature: hsdp_utils ParamModuleInfo shared parameter tracking.
        """
        # Arrange
        mod1 = SimpleLinear(4, 4)
        mod2 = SimpleLinear(4, 4)
        # Act
        info = ParamModuleInfo(
            module=mod1,
            param_name="weight",
            shared_modules=[mod2],
            shared_param_names=["weight"],
        )
        # Assert
        self.assertEqual(len(info.shared_modules), 1)
        self.assertIs(info.shared_modules[0], mod2)
        self.assertEqual(info.shared_param_names, ["weight"])


class TestNamedParametersWithDuplicates(unittest.TestCase):
    """Unit tests for _named_parameters_with_duplicates (name, param) iterator."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        self.device = torch.device("cpu")

    def test_returns_named_params(self):
        """_named_parameters_with_duplicates returns (name, param) tuples.

        description: Call with recurse=False on SimpleLinear; collect names.
        expectation: List of (name, param); 'weight' and 'bias' in names.
        feature: hsdp_utils _named_parameters_with_duplicates.
        """
        # Arrange
        mod = SimpleLinear(4, 4)
        # Act
        result = _named_parameters_with_duplicates(mod, recurse=False)
        # Assert
        self.assertIsInstance(result, list)
        names = [name for name, _ in result]
        self.assertIn("weight", names)
        self.assertIn("bias", names)

    def test_remove_duplicate_raises(self):
        """_named_parameters_with_duplicates raises if remove_duplicate in kwargs.

        description: Call with remove_duplicate=True (disallowed).
        expectation: AssertionError mentioning 'remove_duplicate'.
        feature: hsdp_utils _named_parameters_with_duplicates API constraint.
        """
        # Arrange
        mod = SimpleLinear(4, 4)
        # Act & Assert
        with self.assertRaises(AssertionError) as ctx:
            _named_parameters_with_duplicates(mod, remove_duplicate=True)
        self.assertIn("remove_duplicate", str(ctx.exception))

    def test_falls_back_for_modules_without_remove_duplicate_support(self):
        """Modules that reject remove_duplicate should still expose their local params."""

        class LegacyNamedParameters(SimpleLinear):
            """Module double that mimics an older named_parameters signature."""

            def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
                if "remove_duplicate" in locals() and remove_duplicate is False:
                    raise AssertionError("remove_duplicate unsupported")
                return super().named_parameters(prefix=prefix, recurse=recurse)

        mod = LegacyNamedParameters(4, 4)

        result = _named_parameters_with_duplicates(mod, recurse=False)

        self.assertEqual([name for name, _ in result], ["weight", "bias"])


class TestGetParamModuleInfos(unittest.TestCase):
    """Unit tests for _get_param_module_infos (param -> ParamModuleInfo mapping)."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        self.device = torch.device("cpu")

    def test_single_module_single_param(self):
        """_get_param_module_infos returns correct info for single param.

        description: Pass one module and its params; call _get_param_module_infos.
        expectation: One ParamModuleInfo per param; each info.module is the module.
        feature: hsdp_utils _get_param_module_infos.
        """
        # Arrange
        mod = SimpleLinear(4, 4)
        params = list(mod.parameters())
        # Act
        infos = _get_param_module_infos(params, (mod,))
        # Assert
        self.assertEqual(len(infos), len(params))
        for info in infos:
            self.assertIsInstance(info, ParamModuleInfo)
            self.assertIs(info.module, mod)

    def test_param_not_in_module_tree_raises(self):
        """_get_param_module_infos raises when param not in module tree.

        description: Pass an orphan parameter not belonging to the given modules.
        expectation: AssertionError containing 'not in the module tree'.
        feature: hsdp_utils _get_param_module_infos validation.
        """
        # Arrange
        mod = SimpleLinear(4, 4)
        orphan_param = nn.Parameter(torch.ones(2, 2))
        # Act & Assert
        with self.assertRaises(AssertionError) as ctx:
            _get_param_module_infos([orphan_param], (mod,))
        self.assertIn("not in the module tree", str(ctx.exception))

    def test_multiple_modules(self):
        """_get_param_module_infos works with multiple modules.

        description: Pass two modules and combined params; call _get_param_module_infos.
        expectation: Number of infos equals number of params.
        feature: hsdp_utils _get_param_module_infos multi-module.
        """
        # Arrange
        mod1 = SimpleLinear(4, 4)
        mod2 = SimpleLinear(4, 4)
        params = list(mod1.parameters()) + list(mod2.parameters())
        # Act
        infos = _get_param_module_infos(params, (mod1, mod2))
        # Assert
        self.assertEqual(len(infos), len(params))

    def test_shared_parameter_records_alias_owner(self):
        """Shared parameters should remember every module/name pair that owns them."""
        mod = SharedParamModule()

        info = _get_param_module_infos([mod.weight], (mod,))[0]

        self.assertIs(info.module, mod)
        self.assertEqual(info.param_name, "weight")
        self.assertEqual(info.shared_modules, [mod.linear])
        self.assertEqual(info.shared_param_names, ["weight"])


class TestManagedParamDiscovery(unittest.TestCase):
    """Unit tests for managed-parameter discovery."""

    def test_get_managed_modules_parameters_skips_ignored_duplicates_and_initialized(self):
        """Only unique, non-ignored, non-HSDP-initialized params should be managed."""
        mod = SharedParamModule()
        mod.linear.bias._hsdp_param_initialized = True

        params = get_managed_modules_parameters((mod,), ignored_params=(mod.weight,))

        self.assertEqual(params, [])


class TestDTensorParamHelpers(unittest.TestCase):
    """Unit tests for DTensor metadata detection without distributed init."""

    class FakeDTensor:
        """Small class used only to exercise isinstance-based DTensor branches."""

        def __init__(self, mesh="mesh"):
            self.device_mesh = mesh

    def test_unwrap_dtensor_accepts_payload_directly_or_through_data(self):
        """DTensor payloads may be carried by the param itself or by param.data."""
        direct = self.FakeDTensor("direct-mesh")
        wrapped = SimpleNamespace(data=self.FakeDTensor("data-mesh"))

        with patch("hyper_parallel.core.fully_shard.hsdp_utils.DTensor", self.FakeDTensor):
            self.assertIs(unwrap_dtensor_param(direct), direct)
            self.assertIs(unwrap_dtensor_param(wrapped), wrapped.data)
            self.assertTrue(is_dtensor_managed_param(wrapped))
            self.assertEqual(get_dtensor_managed_mesh(wrapped), "data-mesh")

    def test_unwrap_dtensor_accepts_minimal_layout_payload(self):
        """Param-like objects with DTensor layout fields should be treated as managed."""
        payload = SimpleNamespace(
            _device_mesh="private-mesh",
            _placements=("shard",),
            _local_tensor=torch.ones(2),
        )

        self.assertIs(unwrap_dtensor_param(payload), payload)
        self.assertEqual(get_dtensor_managed_mesh(payload), "private-mesh")
        self.assertIsNone(get_dtensor_managed_mesh(torch.nn.Parameter(torch.ones(2))))

    def test_infer_fully_shard_param_mode_from_dtensor_presence_and_mesh(self):
        """Param mode should reflect whether fully_shard adds a new mesh."""
        local_param = torch.nn.Parameter(torch.ones(2))
        dtensor_payload = SimpleNamespace(
            _device_mesh="tp-mesh",
            _placements=("tp",),
            _local_tensor=torch.ones(2),
        )

        self.assertEqual(infer_fully_shard_param_mode(None, [local_param]), FullyShardParamMode.LOCAL_PARAM)
        self.assertEqual(infer_fully_shard_param_mode(None, [dtensor_payload]), FullyShardParamMode.DTENSOR_COMPAT)
        self.assertEqual(infer_fully_shard_param_mode("fsdp-mesh", [dtensor_payload]), FullyShardParamMode.DTENSOR_UNIFIED)


class TestMeshRankHelpers(unittest.TestCase):
    """Unit tests for mesh axis to rank-list conversion."""

    def _mesh(self):
        return SimpleNamespace(rank=5, rank_list=list(range(8)), mesh_shape=(2, 2, 2))

    def test_get_rank_list_for_axes_handles_empty_axes_and_missing_rank(self):
        """Rank selection should keep the current rank when no axes vary."""
        mesh = self._mesh()

        self.assertEqual(get_rank_list_for_axes(mesh, [], rank=5), [5])
        with self.assertRaisesRegex(ValueError, "not found"):
            get_rank_list_for_axes(mesh, [0], rank=99)

    def test_get_rank_list_for_axes_varies_only_requested_axes(self):
        """Only requested axes should vary while complementary coordinates stay fixed."""
        mesh = self._mesh()

        self.assertEqual(get_rank_list_for_axes(mesh, [0, 2], rank=5), [0, 1, 4, 5])

    def test_get_split_rank_lists_for_axes_handles_empty_full_and_partial_axes(self):
        """Split rank lists should reflect the complementary mesh coordinates."""
        mesh = self._mesh()

        self.assertEqual(get_split_rank_lists_for_axes(mesh, []), [list(range(8))])
        self.assertEqual(get_split_rank_lists_for_axes(mesh, [0, 1, 2]), [list(range(8))])
        self.assertEqual(
            get_split_rank_lists_for_axes(mesh, [1]),
            [[0, 2], [1, 3], [4, 6], [5, 7]],
        )


class TestGetHSDPState(unittest.TestCase):
    """Unit tests for resolving a fully_shard state from a managed module."""

    def test_get_hsdp_state_returns_scheduler_state_for_hsdp_module(self):
        """Managed modules should expose the state attached to their scheduler."""
        from hyper_parallel.core.fully_shard.api import HSDPModule

        class ManagedModule(HSDPModule):
            """Minimal managed module with an installed scheduler."""

        module = ManagedModule()
        module.hsdp_scheduler = SimpleNamespace(hsdp_state="state")

        self.assertEqual(get_hsdp_state(module), "state")
        module.hsdp_scheduler = None
        with self.assertRaisesRegex(AssertionError, "contains 'hsdp_scheduler'"):
            get_hsdp_state(module)
        self.assertIsNone(get_hsdp_state(object()))


if __name__ == "__main__":
    unittest.main()
