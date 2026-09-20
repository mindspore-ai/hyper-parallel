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
"""Unit tests for model state rebuilt after meta materialization."""
# pylint: disable=wrong-import-position

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import torch
from torch import nn

from hyper_parallel import DeviceMesh, Shard, distribute_tensor
from hyper_parallel.models._transformers.checkpoint_loader import (
    LoadReport,
    _finalize_model_loading,
)
from hyper_parallel.models._transformers.model_builder import (
    _apply_materialization_adapter,
    _materialize_and_load_model,
)
from hyper_parallel.models.materialization import (
    MaterializationContext,
    rebuild_materialized_state,
    register_materialized_state_hook,
    register_rebuildable_buffer,
)
from tests.common.mark_utils import arg_mark


class _DerivedBufferModule(nn.Module):
    """Small model exposing both declarative rebuild recipes."""

    def __init__(self) -> None:
        """Register constant and factory-derived runtime buffers."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))
        self.factory_contexts = []
        register_rebuildable_buffer(self, "constant", value=torch.arange(4))
        register_rebuildable_buffer(self, "generated", factory=self._build_generated)

    def _build_generated(self, context: MaterializationContext) -> torch.Tensor:
        self.factory_contexts.append(context)
        return torch.full((3,), 7, dtype=torch.long)


class _InvalidHookModule(nn.Module):
    """Module whose hook violates the protected-parameter contract."""

    def __init__(self) -> None:
        """Create one parameter that the invalid hook will mutate."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))

    @torch.no_grad()
    def rebuild_materialized_state_(self, context: MaterializationContext) -> None:
        """Deliberately violate the hook contract for DFX validation."""
        del context
        self.weight.zero_()


class _CheckpointModel(nn.Module):
    """Minimal checkpoint finalization model with derived runtime state."""

    def __init__(self) -> None:
        """Create checkpoint-owned weight and non-persistent runtime state."""
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([3.0]))
        self.register_buffer("runtime", torch.tensor([5.0]), persistent=False)
        self.initialize_calls = 0

    @torch.no_grad()
    def initialize_weights(self) -> None:
        """Record and expose any unintended model initialization call."""
        self.initialize_calls += 1
        self.weight.fill_(99.0)


class _NativeHfStyleModule(nn.Module):
    """Unmodified-model stand-in whose existing buffer is adapter-registered."""

    def __init__(self) -> None:
        """Create native model metadata and one official runtime buffer."""
        super().__init__()
        self.config = SimpleNamespace(
            model_type="native_hf_probe",
            architectures=["NativeHfProbeModel"],
        )
        self.weight = nn.Parameter(torch.ones(1))
        self.register_buffer("runtime_ids", torch.arange(5), persistent=False)
        self.register_buffer("hook_ids", torch.arange(3) + 10, persistent=False)


class TestMaterializedState(unittest.TestCase):
    """Tests for declarative buffers and the post-materialization lifecycle."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_builder_rebuilds_non_persistent_buffers_after_to_empty(self) -> None:
        """The scratch build path restores constant and factory-derived state."""
        model = _DerivedBufferModule().to(device="meta")
        model.factory_contexts.clear()

        _materialize_and_load_model(
            model,
            is_meta_device=True,
            device=torch.device("cpu"),
            load_base_model=False,
            pretrained_path=None,
            weights_mapping=None,
        )

        self.assertTrue(torch.equal(model.constant, torch.arange(4)))
        self.assertTrue(torch.equal(model.generated, torch.full((3,), 7, dtype=torch.long)))
        self.assertEqual(len(model.factory_contexts), 1)
        self.assertEqual(model.factory_contexts[0].reason, "random_init")
        self.assertEqual(model.factory_contexts[0].device, torch.device("cpu"))
        self.assertNotIn("constant", model.state_dict())
        self.assertNotIn("generated", model.state_dict())

        rebuild_materialized_state(
            model,
            MaterializationContext(
                reason="checkpoint_load",
                device=torch.device("cpu"),
                strict=True,
            ),
        )
        self.assertTrue(torch.equal(model.constant, torch.arange(4)))
        self.assertTrue(torch.equal(model.generated, torch.full((3,), 7, dtype=torch.long)))
        self.assertEqual(model.factory_contexts[-1].reason, "checkpoint_load")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    @patch("hyper_parallel.models._transformers.model_builder.get_model_adapter")
    def test_native_hf_adapter_registers_existing_buffer(self, mock_get_adapter) -> None:
        """A family adapter supports an HF class without changing its inheritance."""
        model = _NativeHfStyleModule()

        def _register_native_state(root: nn.Module) -> None:
            register_rebuildable_buffer(root, "runtime_ids")
            hook_source = root.get_buffer("hook_ids").detach().cpu().clone()

            def _restore_hook_ids(module: nn.Module, context: MaterializationContext) -> None:
                del context
                target = module.get_buffer("hook_ids")
                target.copy_(hook_source.to(device=target.device))

            register_materialized_state_hook(root, _restore_hook_ids)

        mock_get_adapter.return_value = SimpleNamespace(
            materialization=_register_native_state,
        )
        _apply_materialization_adapter(model)
        model.to(device="meta")

        _materialize_and_load_model(
            model,
            is_meta_device=True,
            device=torch.device("cpu"),
            load_base_model=False,
            pretrained_path=None,
            weights_mapping=None,
        )

        self.assertIs(type(model), _NativeHfStyleModule)
        self.assertTrue(torch.equal(model.runtime_ids, torch.arange(5)))
        self.assertTrue(torch.equal(model.hook_ids, torch.arange(3) + 10))
        self.assertNotIn("runtime_ids", model.state_dict())
        self.assertNotIn("hook_ids", model.state_dict())

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_strict_hook_rejects_parameter_mutation(self) -> None:
        """A model hook cannot silently overwrite checkpoint-owned parameters."""
        model = _InvalidHookModule()
        context = MaterializationContext(
            reason="checkpoint_load",
            device=torch.device("cpu"),
            strict=True,
        )

        with self.assertRaisesRegex(RuntimeError, "changed protected parameter:weight"):
            rebuild_materialized_state(model, context)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    @patch("hyper_parallel.models._transformers.model_builder.HuggingFaceCheckpointer")
    def test_builder_rebuilds_buffers_after_checkpoint_load(self, mock_checkpointer_cls) -> None:
        """The pretrained path rebuilds runtime state after load finalization."""
        model = _DerivedBufferModule().to(device="meta")
        model.factory_contexts.clear()

        def _load(path, state, **kwargs):
            """Stand in for the checkpointer: fill the weight and report it loaded."""
            del path, kwargs
            with torch.no_grad():
                model.weight.fill_(3.0)
            state["load_report"] = LoadReport(
                loaded_keys=("weight",),
                missing_keys=(),
                unexpected_keys=(),
            )
            return state

        mock_checkpointer_cls.return_value.load.side_effect = _load

        _materialize_and_load_model(
            model,
            is_meta_device=True,
            device=torch.device("cpu"),
            load_base_model=True,
            pretrained_path="unused",
            weights_mapping=None,
        )

        self.assertTrue(torch.equal(model.constant, torch.arange(4)))
        self.assertTrue(torch.equal(model.generated, torch.full((3,), 7, dtype=torch.long)))
        self.assertEqual(len(model.factory_contexts), 1)
        self.assertEqual(model.factory_contexts[0].reason, "checkpoint_load")
        self.assertTrue(torch.equal(model.weight, torch.full((2,), 3.0)))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    @patch("hyper_parallel.core.dtensor.device_mesh.dist.get_rank", return_value=0)
    def test_rebuild_preserves_dtensor_layout(self, mock_get_rank) -> None:
        """A global source is sliced into existing rank-local DTensor storage."""
        del mock_get_rank
        model = _DerivedBufferModule()
        mesh = DeviceMesh(
            "cpu",
            [0, 1],
            mesh_dim_names=("tp",),
            _init_backend=False,
        )
        target = distribute_tensor(model.constant, mesh, (Shard(0),))
        model._buffers["constant"] = target  # pylint: disable=protected-access
        target.to_local().fill_(-1)
        expected_layout = target.layout

        rebuild_materialized_state(
            model,
            MaterializationContext(
                reason="random_init",
                device=torch.device("cpu"),
                strict=True,
            ),
        )

        self.assertIs(model.constant, target)
        self.assertIs(model.constant.layout, expected_layout)
        self.assertEqual(tuple(model.constant.shape), (4,))
        self.assertTrue(torch.equal(model.constant.to_local(), torch.tensor([0, 1])))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_checkpoint_finalizer_does_not_initialize_non_persistent_buffers(self) -> None:
        """Runtime buffers do not expand pretrained finalization into random init."""
        model = _CheckpointModel()
        report = LoadReport(
            loaded_keys=("weight",),
            missing_keys=(),
            unexpected_keys=(),
        )

        _finalize_model_loading(model, report, strict=True)

        self.assertEqual(model.initialize_calls, 0)
        self.assertTrue(torch.equal(model.weight, torch.tensor([3.0])))


if __name__ == "__main__":
    unittest.main()
