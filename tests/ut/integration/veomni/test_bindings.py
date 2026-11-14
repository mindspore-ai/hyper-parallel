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
"""CPU checks of explicit VeOmni construction bindings and prefetch."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from hyper_parallel.integration.veomni.adapter import (
    _build_construction_bindings,
    _build_parallelize_binding,
    _configure_block_prefetch,
)


class TestConstructionBindings(unittest.TestCase):
    """Check copy-in-place restoration without importing optional VeOmni."""

    def test_block_prefetch_depth_and_switch(self):
        """Depth counts native VeOmni blocks, not parameters or individual children."""
        blocks = [
            SimpleNamespace(
                _fsdp_modules=[object()],
                set_modules_to_forward_prefetch=MagicMock(),
                set_modules_to_backward_prefetch=MagicMock(),
            )
            for _ in range(4)
        ]
        config = SimpleNamespace(
            forward_prefetch=True,
            hyper_parallel=SimpleNamespace(forward_prefetch_depth=2),
        )
        _configure_block_prefetch(blocks, config)
        blocks[0].set_modules_to_forward_prefetch.assert_called_once_with(
            [blocks[1]._fsdp_modules[0], blocks[2]._fsdp_modules[0]]
        )
        blocks[2].set_modules_to_backward_prefetch.assert_called_once_with(
            [blocks[1]._fsdp_modules[0]]
        )
        config.forward_prefetch = False
        _configure_block_prefetch(blocks, config)
        blocks[0].set_modules_to_forward_prefetch.assert_called_with([])

    def test_backward_prefetch_depth_two_uses_nearest_block_first(self):
        """Backward depth is independent and clips targets at the first block."""
        blocks = [
            SimpleNamespace(
                _fsdp_modules=[object()],
                set_modules_to_forward_prefetch=MagicMock(),
                set_modules_to_backward_prefetch=MagicMock(),
            )
            for _ in range(4)
        ]
        config = SimpleNamespace(
            forward_prefetch=False,
            hyper_parallel=SimpleNamespace(
                forward_prefetch_depth=1, backward_prefetch_depth=2
            ),
        )
        _configure_block_prefetch(blocks, config)
        for index, block in enumerate(blocks):
            expected = [
                blocks[target]._fsdp_modules[0]
                for target in range(index - 1, max(-1, index - 3), -1)
            ]
            block.set_modules_to_backward_prefetch.assert_called_once_with(expected)
            block.set_modules_to_forward_prefetch.assert_called_once_with([])

    def test_block_build_reuses_native_pipeline_and_backward_reshard(self):
        """Original discovery, initialization and loading execute once and own the model."""
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
        model._no_split_modules = ["Linear"]
        events = []

        def native_build(model, basic_modules=None, **kwargs):
            events.append(kwargs)
            model.set_reshard_after_backward = MagicMock()
            for module in model:
                module._fsdp_modules = [module]
                module.set_modules_to_forward_prefetch = MagicMock()
                module.set_modules_to_backward_prefetch = MagicMock()
            return model

        parallel = SimpleNamespace(
            parallelize_model_fsdp2=native_build,
            sort_fqn_by_submodule_first=lambda names: names,
        )
        config = SimpleNamespace(
            forward_prefetch=True,
            reshard_after_backward=False,
            hyper_parallel=SimpleNamespace(
                forward_prefetch_depth=2, reduce_comm_interval=1
            ),
        )
        _, wrapper = _build_parallelize_binding(parallel, config)
        self.assertIs(wrapper(model, enable_forward_prefetch=True), model)
        self.assertEqual(events, [{"enable_forward_prefetch": False}])
        model.set_reshard_after_backward.assert_called_once_with(False)
        model[0].set_modules_to_forward_prefetch.assert_called_once_with([model[1]])

    def test_fusion_configuration_reaches_native_hyper_builder(self):
        """Only Hyper-specific fusion values are added to the existing builder kwargs."""

        def build_optimizer(model, optimizer_type="adamw", param_groups=None):
            return model

        parallel = SimpleNamespace(
            fully_shard=MagicMock(), FSDPModule=type("FSDP", (), {})
        )
        checkpoint = SimpleNamespace(
            get_model_state_dict=MagicMock(),
            set_model_state_dict=MagicMock(),
            OptimizerState=type("OptimizerState", (), {}),
        )
        modules = {
            "veomni.distributed.torch_parallelize": parallel,
            "veomni.optim.optimizer": SimpleNamespace(build_optimizer=build_optimizer),
            "veomni.checkpoint.dcp_checkpointer": checkpoint,
        }
        config = SimpleNamespace(
            low_precision_reduce_scatter_comm=False,
            hyper_parallel=SimpleNamespace(
                wrap_policy="block",
                comm_fusion=True,
                comm_fusion_zero_copy=False,
                reduce_comm_interval=1,
            ),
        )
        with (
            patch(
                "hyper_parallel.integration.veomni.adapter.import_module",
                side_effect=modules.__getitem__,
            ),
            patch(
                "hyper_parallel.integration.veomni.adapter._build_parallelize_binding",
                return_value=(object(), object()),
            ),
        ):
            bindings = dict(_build_construction_bindings(fsdp_config=config))
        model = torch.nn.Linear(2, 2)
        with patch("hyper_parallel.integration.veomni.adapter.fully_shard") as builder:
            bindings[parallel.fully_shard](
                model, mesh="mesh", reshard_after_forward=True
            )
        builder.assert_called_once_with(
            model,
            mesh="mesh",
            reshard_after_forward=True,
            comm_fusion=True,
            comm_fusion_zero_copy=False,
        )
        config.hyper_parallel.reduce_comm_interval = 2
        with (
            patch(
                "hyper_parallel.integration.veomni.adapter.import_module",
                side_effect=modules.__getitem__,
            ),
            self.assertRaisesRegex(NotImplementedError, "set_reduce_comm_interval"),
        ):
            _build_construction_bindings(fsdp_config=config)

    def test_fsdp_and_optimizer_selection_are_independent(self):
        """An optimizer-only activation must not replace sharding or model DCP."""
        for fsdp in (False, True):
            for muon in (False, True):
                for hyper_parameters in (False, True):
                    with self.subTest(
                        fsdp=fsdp, muon=muon, hyper_parameters=hyper_parameters
                    ):

                        def native_optimizer(
                            model,
                            optimizer_type="muon",
                            optimizer_config=None,
                            param_groups=None,
                        ):
                            return "native"

                        parallel = SimpleNamespace(
                            fully_shard=MagicMock(), FSDPModule=type("FSDP", (), {})
                        )
                        checkpoint = SimpleNamespace(
                            get_model_state_dict=MagicMock(),
                            set_model_state_dict=MagicMock(),
                            OptimizerState=type("OptimizerState", (), {}),
                        )
                        modules = {
                            "veomni.distributed.torch_parallelize": parallel,
                            "veomni.optim.optimizer": SimpleNamespace(
                                build_optimizer=native_optimizer
                            ),
                            "veomni.checkpoint.dcp_checkpointer": checkpoint,
                        }
                        with patch(
                            "hyper_parallel.integration.veomni.adapter.import_module",
                            side_effect=modules.__getitem__,
                        ):
                            bindings = dict(
                                _build_construction_bindings(fsdp=fsdp, muon=muon)
                            )
                        self.assertEqual(parallel.fully_shard in bindings, fsdp)
                        self.assertEqual(
                            checkpoint.get_model_state_dict in bindings, fsdp
                        )
                        model = torch.nn.Linear(2, 2)
                        with (
                            patch(
                                "hyper_parallel.integration.veomni.optimizer.uses_hyper",
                                return_value=hyper_parameters,
                            ),
                            patch(
                                "hyper_parallel.integration.veomni.optimizer._build_hyper_muon_adapter",
                                return_value="hyper",
                            ) as hyper_factory,
                            patch(
                                "hyper_parallel.integration.veomni.optimizer._build_native_muon_adapter",
                                return_value="native-view",
                            ) as view_factory,
                        ):
                            actual = bindings[native_optimizer](model)
                        self.assertEqual(
                            actual,
                            "hyper"
                            if muon
                            else "native-view"
                            if hyper_parameters
                            else "native",
                        )
                        self.assertEqual(hyper_factory.call_count, int(muon))
                        self.assertEqual(
                            view_factory.call_count, int(not muon and hyper_parameters)
                        )

    def _setter(self):
        """Create the native factory and module aliases used by binding tests."""

        def build_optimizer(model, optimizer_type="adamw", param_groups=None):
            """Return the input model so factory binding remains observable."""
            return model

        parallel = SimpleNamespace(
            fully_shard=MagicMock(), FSDPModule=type("NativeFSDP", (), {})
        )
        checkpoint = SimpleNamespace(
            get_model_state_dict=MagicMock(),
            set_model_state_dict=MagicMock(),
            OptimizerState=type("OptimizerState", (), {}),
        )
        modules = {
            "veomni.distributed.torch_parallelize": parallel,
            "veomni.optim.optimizer": SimpleNamespace(build_optimizer=build_optimizer),
            "veomni.checkpoint.dcp_checkpointer": checkpoint,
        }
        with patch(
            "hyper_parallel.integration.veomni.adapter.import_module",
            side_effect=modules.__getitem__,
        ):
            bindings = _build_construction_bindings()
        return dict(bindings)[checkpoint.set_model_state_dict], checkpoint

    def test_restore_keeps_optimizer_parameter_identity(self):
        """Do not inherit Hyper core assign=True at the framework boundary."""
        setter, _ = self._setter()
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        parameter = model.weight
        restored = {
            name: torch.ones_like(value) for name, value in model.state_dict().items()
        }
        with patch(
            "hyper_parallel.integration.veomni.adapter.uses_hyper", return_value=True
        ):
            setter(model, restored)
        self.assertIs(model.weight, parameter)
        self.assertIs(optimizer.param_groups[0]["params"][0], parameter)
        torch.testing.assert_close(model.weight, torch.ones_like(model.weight))

    def test_plain_torch_restore_delegates_unchanged(self):
        """Actual tensor type, not adapter presence, selects the setter."""
        setter, checkpoint = self._setter()
        model = torch.nn.Linear(2, 2)
        state = model.state_dict()
        setter(model, state)
        checkpoint.set_model_state_dict.assert_called_once_with(
            model, model_state_dict=state
        )

    def test_unsupported_full_restore_fails_before_mutation(self):
        """Reject unsupported restore semantics rather than copying locally."""
        setter, _ = self._setter()
        model = torch.nn.Linear(2, 2)
        before = model.weight.detach().clone()
        with (
            patch(
                "hyper_parallel.integration.veomni.adapter.uses_hyper",
                return_value=True,
            ),
            self.assertRaisesRegex(NotImplementedError, "sharded DCP"),
        ):
            setter(model, {}, options=SimpleNamespace(full_state_dict=True))
        torch.testing.assert_close(model.weight, before)
