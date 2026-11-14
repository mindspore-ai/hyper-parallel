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
"""CPU checks of VeOmni optimizer construction and checkpoint ownership."""

import math
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch.distributed.tensor import DTensor as TorchDTensor
from torch.distributed.tensor import Shard as TorchShard

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.core.fully_shard.api import HSDPModule
from hyper_parallel.integration.veomni.optimizer import (
    HyperOptimizerAdapter,
    _build_hyper_muon_adapter,
    _muon_layout_options,
    _NativeMuonAdapter,
    _optimizer_leaf_from_flat,
    _restore_hyper_state_view,
    build_optimizer_binding,
    build_optimizer_state_binding,
    uses_hyper,
)


class TestHyperOptimizerAdapter(unittest.TestCase):
    """Exercise checkpoint ownership and native optimizer delegation."""

    def test_adamw_groups_live_parameters_without_torch_views(self):
        """Mesh grouping must retain original parameters and native decay rules."""
        model = torch.nn.Module()
        model.first = torch.nn.Linear(2, 2, bias=False)
        model.second = torch.nn.Linear(2, 2, bias=False)
        model.second.weight = model.first.weight
        parameter = model.first.weight
        parameter.device_mesh = SimpleNamespace(mesh_dim_names=("ep_fsdp",))

        def native(
            model,
            optimizer_type="adamw",
            param_groups=None,
            lr=0.01,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
            fused=False,
            no_decay_modules=None,
            no_decay_params=None,
        ):
            raise AssertionError(
                "EP type inspection must not call the Torch-only factory"
            )

        def native_groups(model_arg, parameters, weight_decay, modules, names):
            self.assertIs(model_arg, model)
            self.assertIs(parameters[0], parameter)
            self.assertEqual(weight_decay, 0.1)
            return [{"params": parameters, "weight_decay": 0.0}]

        _, factory = build_optimizer_binding(
            SimpleNamespace(
                build_optimizer=native,
                _should_build_extra_parallel_aware=lambda model: True,
                get_parallel_state=lambda: SimpleNamespace(
                    extra_parallel_names=("ep",)
                ),
                filter_empty_param_groups=lambda groups: groups,
                _make_param_groups_for_subset=native_groups,
            ),
            use_hyper_optimizer=False,
        )
        with (
            patch(
                "hyper_parallel.integration.veomni.optimizer.uses_hyper",
                return_value=True,
            ),
            patch(
                "hyper_parallel.integration.veomni.optimizer.to_torch_view",
                side_effect=AssertionError("No views"),
            ),
        ):
            adapter = factory(
                model,
                param_groups=[{"params": [parameter], "lr": 0.02, "weight_decay": 0.0}],
            )
        self.assertIs(model.first.weight, parameter)
        self.assertIs(adapter.param_groups[0]["params"][0], parameter)
        self.assertEqual(adapter.param_groups[0]["lr"], 0.02)
        self.assertEqual(adapter.param_groups[0]["weight_decay"], 0.0)
        self.assertIs(model._extra_parallel_param_groups["ep"][0], parameter)
        self.assertEqual(model._ep_replicated_lora_param_ids, set())
        parameter.grad = torch.ones_like(parameter)
        adapter.step()
        leaf = adapter.optimizers_dict["ep"]
        self.assertIn(parameter, leaf.state)
        self.assertIs(model.second.weight, parameter)

    def test_adamw_binding_unwraps_native_multi_optimizer_leaves(self):
        """Keep native EP leaves and groups instead of nesting their container."""
        model = torch.nn.Linear(2, 2)
        leaves = {
            "ep": torch.optim.AdamW([model.weight], lr=0.01, weight_decay=0.1),
            "non_extra_parallel": torch.optim.AdamW(
                [model.bias], lr=0.02, weight_decay=0.0
            ),
        }
        container = SimpleNamespace(_is_multi_optimizer=True, optimizers_dict=leaves)

        def native(model, optimizer_type="adamw", param_groups=None):
            return container

        _, factory = build_optimizer_binding(
            SimpleNamespace(
                build_optimizer=native,
                _should_build_extra_parallel_aware=lambda model: False,
            ),
            use_hyper_optimizer=False,
        )
        with patch(
            "hyper_parallel.integration.veomni.optimizer.uses_hyper", return_value=True
        ):
            adapter = factory(model)
        self.assertEqual(adapter.key_names, list(leaves))
        for name, leaf in leaves.items():
            self.assertIs(adapter.optimizers_dict[name], leaf)
        self.assertEqual([group["lr"] for group in adapter.param_groups], [0.01, 0.02])
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        adapter.step()
        saved = deepcopy(adapter.state_dict())
        adapter.zero_grad(set_to_none=True)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        for leaf in leaves.values():
            leaf.state.clear()
        adapter.load_state_dict(saved)
        self.assertEqual(leaves["ep"].state[model.weight]["step"].item(), 1)
        self.assertEqual(
            leaves["non_extra_parallel"].state[model.bias]["step"].item(), 1
        )

    def test_dcp_marks_only_load_templates_without_modifying_saved_options(self):
        """Partial loads must leave detectable holes instead of usable current defaults."""
        state = {
            "param_groups.weight.ns_implementation": "gram",
            "param_groups.weight.ns_epsilon": 1e-7,
            "param_groups.weight.ns_coefficients": [(3.4, -4.7, 2.0)],
            "param_groups.weight.gram_ns_reset_iterations": (2,),
            "param_groups.weight.lr": 0.01,
        }

        class NativeState:
            def __init__(self, optimizer, load):
                self.optimizer, self._load = optimizer, load

            def state_dict(self):
                return state

        _, bound = build_optimizer_state_binding(NativeState)
        adapter = HyperOptimizerAdapter(torch.nn.Linear(2, 2), {})
        self.assertIs(bound(adapter, False).state_dict(), state)
        self.assertIs(bound(object(), True).state_dict(), state)
        loaded = bound(adapter, True).state_dict()
        self.assertNotEqual(loaded["param_groups.weight.ns_implementation"], "gram")
        self.assertTrue(math.isnan(loaded["param_groups.weight.ns_epsilon"]))
        self.assertTrue(math.isnan(loaded["param_groups.weight.ns_coefficients"][0][0]))
        self.assertNotEqual(
            loaded["param_groups.weight.gram_ns_reset_iterations"], (2,)
        )
        self.assertEqual(loaded["param_groups.weight.lr"], 0.01)
        self.assertEqual(state["param_groups.weight.ns_implementation"], "gram")
        with self.assertRaisesRegex(ValueError, "NS configuration keys"):
            adapter.confirm_legacy_muon_checkpoint_options(lr=0.01)

    def test_restore_state_retains_ceil_chunk_mesh_and_placements(self):
        """Both owner and empty ranks keep their live FSDP geometry."""
        for rows in (0, 1):
            with self.subTest(local_rows=rows):
                local = torch.ones(rows, 128)
                parameter = MagicMock(spec=DTensor)
                parameter.shape = (1, 128)
                parameter.placements = (Shard(0, uneven_shard=True),)
                parameter.to_local.return_value = local
                parameter.device_mesh.mesh = torch.arange(4)
                parameter.layout.tensor_stride = (128, 1)
                view = MagicMock(spec=TorchDTensor)
                view.shape = parameter.shape
                view.to_local.return_value = local
                view.placements = (TorchShard(0),)
                view.device_mesh.mesh = torch.arange(4)
                with patch.object(DTensor, "from_local") as factory:
                    _restore_hyper_state_view(view, parameter)
                factory.assert_called_once_with(
                    local,
                    parameter.device_mesh,
                    parameter.placements,
                    shape=parameter.shape,
                    stride=(128, 1),
                )

    def test_restore_state_rejects_another_rank_partition(self):
        """Restoring the flag must not blindly relabel differently sharded data."""
        for mismatch in ("global_shape", "mesh", "placement", "local_shape"):
            with self.subTest(mismatch=mismatch):
                parameter = MagicMock(spec=DTensor)
                parameter.shape = (4, 3)
                parameter.placements = (Shard(0),)
                parameter.to_local.return_value = torch.ones(2, 3)
                parameter.device_mesh.mesh = torch.arange(2)
                view = MagicMock(spec=TorchDTensor)
                view.shape = parameter.shape
                view.to_local.return_value = torch.ones(2, 3)
                view.placements = (TorchShard(0),)
                view.device_mesh.mesh = torch.arange(2)
                if mismatch == "global_shape":
                    view.shape = (6, 3)
                elif mismatch == "mesh":
                    view.device_mesh.mesh = torch.tensor([1, 0])
                elif mismatch == "placement":
                    view.placements = (TorchShard(1),)
                else:
                    view.to_local.return_value = torch.ones(1, 3)
                with (
                    patch.object(DTensor, "from_local") as factory,
                    self.assertRaisesRegex(ValueError, "parameter layout"),
                ):
                    _restore_hyper_state_view(view, parameter)
                factory.assert_not_called()

    def test_original_lr_scale_uses_block_shape_without_changing_storage(self):
        """Head blocks use native original scaling and share the NS input storage."""
        model = torch.nn.Linear(8, 24, bias=False)
        model.weight.model_name = "weight"
        native = SimpleNamespace(
            infer_head_block_counts=MagicMock(return_value={"weight": 4})
        )
        groups, reshape = _muon_layout_options(
            model,
            [model.weight],
            {
                "adjust_lr_fn": "original",
                "head_group_size": 1,
                "head_split_modules": ["weight"],
            },
            native,
        )
        slices = reshape("weight", model.weight)
        self.assertEqual(len(slices), 4)
        self.assertEqual(groups[0]["head_blocks"], 4)
        for block in slices:
            self.assertEqual(tuple(block.shape), (6, 8))
            self.assertEqual(
                block.untyped_storage().data_ptr(),
                model.weight.untyped_storage().data_ptr(),
            )
            self.assertAlmostEqual(
                groups[0]["matched_adamw_rms"] * math.sqrt(max(block.shape)), 1.0
            )
        self.assertIs(reshape("other", model.weight)[0], model.weight)

    def test_original_lr_scale_handles_tall_wide_and_singleton_expert_shapes(self):
        """Encode the native last-two-dimension LR scale in Hyper parameter groups."""
        for shape in ((24, 8), (8, 24), (4, 6, 8), (16, 1, 8)):
            with self.subTest(shape=shape):
                parameter = torch.nn.Parameter(torch.zeros(shape))
                parameter.model_name = "weight"
                groups, reshape = _muon_layout_options(
                    torch.nn.Module(),
                    [parameter],
                    {"adjust_lr_fn": "original"},
                    SimpleNamespace(),
                )
                logical = shape[-2:]
                actual = groups[0]["matched_adamw_rms"] * math.sqrt(max(logical))
                self.assertAlmostEqual(
                    actual, math.sqrt(max(1.0, shape[-2] / shape[-1]))
                )
                if len(shape) == 3 and shape[1] == 1:
                    slices = reshape("weight", parameter)
                    self.assertEqual(len(slices), shape[0])
                    for block in slices:
                        self.assertEqual(tuple(block.shape), shape[-2:])
                        self.assertEqual(
                            block.untyped_storage().data_ptr(),
                            parameter.untyped_storage().data_ptr(),
                        )
                else:
                    self.assertIsNone(reshape)

    def test_muon_layout_rejects_invalid_head_and_lr_configuration(self):
        """Reject invalid settings before constructing optimizer state."""
        for options in (
            {"head_group_size": -1},
            {"head_group_size": True},
            {"head_group_size": 1},
            {"adjust_lr_fn": "unknown"},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                _muon_layout_options(torch.nn.Module(), [], options, SimpleNamespace())

    def test_muon_restore_rejects_changed_head_layout_or_lr_policy(self):
        """Checkpoint metadata must agree with the active reshape and LR policy."""
        model = torch.nn.Linear(8, 16, bias=False)
        leaf = SimpleNamespace(
            is_muon=True,
            param_groups=[
                {"params": [model.weight], "head_blocks": 4, "adjust_lr_fn": "original"}
            ],
        )
        adapter = HyperOptimizerAdapter(model, {"muon": leaf})
        for state, message in (
            ({"param_groups.weight.adjust_lr_fn": "original"}, "head_blocks"),
            ({"param_groups.weight.head_blocks": 4}, "LR policy"),
        ):
            with self.subTest(state=state), self.assertRaisesRegex(ValueError, message):
                adapter.load_state_dict(state)

    def test_muon_restore_checks_ns_configuration_before_loading_state(self):
        """Reject closure/group mismatches, including backend and Gram reset settings."""
        model = torch.nn.Linear(2, 2, bias=False)
        settings = {
            "ns_implementation": "gram",
            "ns_coefficients": [(3.4445, -4.775, 2.0315)] * 5,
            "ns_epsilon": 1e-7,
            "ns_steps": 5,
            "ns_variant": "custom",
            "gram_ns_reset_iterations": (2,),
        }
        leaf = SimpleNamespace(
            is_muon=True, param_groups=[{"params": [model.weight], **settings}]
        )
        adapter = HyperOptimizerAdapter(model, {"muon": leaf})
        state = {f"param_groups.weight.{key}": value for key, value in settings.items()}
        for key, value in (
            ("ns_coefficients", [(3.0, -4.0, 2.0)] * 5),
            ("ns_epsilon", 0.01),
            ("ns_steps", 3),
            ("ns_variant", "asym5"),
            ("ns_implementation", "std"),
            ("gram_ns_reset_iterations", (1,)),
        ):
            with (
                self.subTest(key=key),
                patch(
                    "hyper_parallel.integration.veomni.optimizer.set_optim_state_dict"
                ) as restore,
                self.assertRaisesRegex(ValueError, key),
            ):
                adapter.load_state_dict({**state, f"param_groups.weight.{key}": value})
            restore.assert_not_called()
        missing = dict(state)
        missing.pop("param_groups.weight.ns_implementation")
        with self.assertRaisesRegex(ValueError, "missing NS configuration"):
            adapter.load_state_dict(missing)
        state["param_groups.weight.ns_coefficients"] = [
            list(row) for row in settings["ns_coefficients"]
        ]
        state["param_groups.weight.gram_ns_reset_iterations"] = [2]
        with patch(
            "hyper_parallel.integration.veomni.optimizer.set_optim_state_dict"
        ) as restore:
            adapter.load_state_dict(state)
        restore.assert_called_once()

    def test_confirmed_ns_options_reach_the_actual_restore_payload(self):
        """Validated legacy values must replace markers in the loaded groups too."""
        model = torch.nn.Linear(2, 2, bias=False)
        settings = {
            "ns_implementation": "gram",
            "ns_coefficients": [(3.4445, -4.775, 2.0315)] * 5,
            "ns_epsilon": 1e-7,
            "ns_steps": 5,
            "ns_variant": "custom",
            "gram_ns_reset_iterations": (2,),
        }
        leaf = SimpleNamespace(
            is_muon=True, param_groups=[{"params": [model.weight], **settings}]
        )
        adapter = HyperOptimizerAdapter(model, {"muon": leaf})
        state = {f"param_groups.weight.{key}": value for key, value in settings.items()}
        marked = adapter.dcp_load_template(state)
        for key in ("ns_implementation", "gram_ns_reset_iterations"):
            state[f"param_groups.weight.{key}"] = marked[f"param_groups.weight.{key}"]
        adapter.confirm_legacy_muon_checkpoint_options(
            ns_implementation="gram", gram_ns_reset_iterations=[2]
        )
        with patch(
            "hyper_parallel.integration.veomni.optimizer.set_optim_state_dict"
        ) as restore:
            adapter.load_state_dict(state)
        group = restore.call_args.args[2]["param_groups"][0]
        self.assertEqual(group["ns_implementation"], "gram")
        self.assertEqual(tuple(group["gram_ns_reset_iterations"]), (2,))

    def test_optimizer_binding_rejects_missing_or_incompatible_parameters(self):
        """Fail at activation, not with a dictionary KeyError during construction."""

        def positional_only(model, optimizer_type, param_groups, /):
            return model

        for factory in (
            None,
            lambda model: model,
            lambda **kwargs: kwargs,
            positional_only,
        ):
            with (
                self.subTest(factory=factory),
                self.assertRaisesRegex(
                    RuntimeError, "veomni.optim.optimizer.build_optimizer"
                ),
            ):
                build_optimizer_binding(
                    SimpleNamespace(build_optimizer=factory), use_hyper_optimizer=True
                )

    def test_optimizer_binding_accepts_keyword_only_and_additional_parameters(self):
        """Check only the dependency contract, not the entire native signature."""

        def native(
            *, model, optimizer_type="adamw", param_groups=None, extra_option=None
        ):
            return model, extra_option

        original, wrapper = build_optimizer_binding(
            SimpleNamespace(build_optimizer=native), use_hyper_optimizer=False
        )
        model = torch.nn.Linear(2, 2)
        self.assertIs(original, native)
        self.assertEqual(
            wrapper(model=model, extra_option="preserved"), (model, "preserved")
        )

    def test_native_muon_refreshes_parameter_views_before_step(self):
        """The view adapter retains live storage and delegates the native update."""
        model = torch.nn.Linear(2, 2)
        view_model = torch.nn.Linear(2, 2)
        optimizer = MagicMock()
        adapter = _NativeMuonAdapter(model, view_model, optimizer)
        adapter.step()
        optimizer.step.assert_called_once_with()
        for parameter, view in zip(model.parameters(), view_model.parameters()):
            self.assertEqual(parameter.data_ptr(), view.data_ptr())
            self.assertIsNone(view.grad)

    def test_hyper_muon_matches_native_vlm_grouping_contract(self):
        """VLM groups are re-split by Muon, just as in the native factory."""

        def native(model, optimizer_type="muon", param_groups=None):
            return model

        logger = MagicMock()
        native_module = SimpleNamespace(build_optimizer=native, logger=logger)
        _, wrapper = build_optimizer_binding(native_module, use_hyper_optimizer=True)
        model = torch.nn.Linear(2, 2)
        groups = [{"params": list(model.parameters()), "lr": 0.01}]
        with patch(
            "hyper_parallel.integration.veomni.optimizer._build_hyper_muon_adapter"
        ) as build:
            self.assertIs(wrapper(model=model, param_groups=groups), build.return_value)
        logger.warning_rank0.assert_called_once()
        self.assertIs(build.call_args.args[0], model)
        self.assertIs(build.call_args.args[1]["param_groups"], groups)

    def test_unsharded_hyper_module_remains_recognizable(self):
        """Evaluation exposes plain parameters but must retain the Hyper boundary."""

        class Wrapped(torch.nn.Linear, HSDPModule):
            pass

        self.assertTrue(uses_hyper(Wrapped(2, 2)))
        self.assertFalse(uses_hyper(torch.nn.Linear(2, 2)))

    def test_restore_releases_replica_transport_state(self):
        """Restore first, then apply Hyper's existing owner-state cleanup."""
        model = torch.nn.Linear(2, 2)
        leaf = MagicMock()
        leaf.param_groups = []
        adapter = HyperOptimizerAdapter(model, {"muon": leaf})
        events = []
        leaf.cleanup_synced_state.side_effect = lambda: events.append("cleanup")
        payload = {"state": {}, "param_groups": []}
        with patch(
            "hyper_parallel.integration.veomni.optimizer.set_optim_state_dict",
            side_effect=lambda *args: events.append("restore"),
        ) as restore:
            adapter.load_state_dict({})
        restore.assert_called_once_with(model, leaf, payload)
        self.assertEqual(events, ["restore", "cleanup"])

    def test_mismatched_names_fail_before_restore(self):
        """A partial optimizer set must not silently load a different leaf."""
        adapter = HyperOptimizerAdapter(torch.nn.Linear(2, 2), {"muon": MagicMock()})
        with (
            patch(
                "hyper_parallel.integration.veomni.optimizer.set_optim_state_dict"
            ) as restore,
            self.assertRaisesRegex(ValueError, "flattened"),
        ):
            adapter.load_state_dict({"adamw": {}})
        restore.assert_not_called()

    def test_native_adamw_update_and_zero_grad(self):
        """Composition retains native leaf updates and scheduler groups."""
        model = torch.nn.Linear(2, 2)
        leaf = torch.optim.AdamW(model.parameters(), lr=0.01)
        adapter = HyperOptimizerAdapter(model, {"adamw": leaf})
        before = model.weight.detach().clone()
        model(torch.ones(1, 2)).sum().backward()
        adapter.step()
        self.assertFalse(torch.equal(model.weight, before))
        self.assertIs(adapter.param_groups[0], leaf.param_groups[0])
        adapter.zero_grad()
        self.assertIsNone(model.weight.grad)

    def test_flat_checkpoint_retains_leaf_groups_and_filters_other_states(self):
        """Equal LR values must not merge groups with different decay policies."""
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(
            [
                {"params": [model.weight], "weight_decay": 0.1},
                {"params": [model.bias], "weight_decay": 0.0},
            ],
            lr=0.01,
        )
        moment = torch.ones_like(model.weight)
        flat = {
            "state.weight.exp_avg": moment,
            "state.other_leaf.exp_avg": torch.ones(1),
            "param_groups.weight.lr": 0.02,
            "param_groups.bias.lr": 0.03,
        }
        actual = _optimizer_leaf_from_flat(
            flat, optimizer, dict(model.named_parameters())
        )
        self.assertIs(actual["state"]["weight"]["exp_avg"], moment)
        self.assertNotIn("other_leaf", actual["state"])
        self.assertEqual(
            [group["params"] for group in actual["param_groups"]],
            [["weight"], ["bias"]],
        )
        self.assertEqual(
            [group["lr"] for group in actual["param_groups"]], [0.02, 0.03]
        )
        self.assertEqual(
            [group["weight_decay"] for group in actual["param_groups"]], [0.1, 0.0]
        )

    def test_flat_checkpoint_roundtrip_restores_groups_and_lazy_state(self):
        """Resume changed groups and an unused parameter like ordinary AdamW."""
        model = torch.nn.Linear(2, 2)
        model.register_parameter("unused", torch.nn.Parameter(torch.ones(2)))
        leaf = torch.optim.AdamW(
            [
                {"params": [model.weight], "weight_decay": 0.1},
                {"params": [model.bias, model.unused], "weight_decay": 0.0},
            ],
            lr=0.01,
        )
        adapter = HyperOptimizerAdapter(model, {"adamw": leaf})
        for parameter in (model.weight, model.bias):
            parameter.grad = torch.ones_like(parameter)
        adapter.step()
        saved = deepcopy(adapter.state_dict())
        reference = deepcopy(model)
        reference_leaf = torch.optim.AdamW(
            [
                {"params": [reference.weight]},
                {"params": [reference.bias, reference.unused]},
            ]
        )
        reference_leaf.load_state_dict(deepcopy(leaf.state_dict()))
        leaf.state.clear()
        for group in leaf.param_groups:
            group.update(lr=1.0, weight_decay=1.0)
        adapter.load_state_dict(saved)
        self.assertEqual([group["lr"] for group in leaf.param_groups], [0.01, 0.01])
        self.assertEqual(
            [group["weight_decay"] for group in leaf.param_groups], [0.1, 0.0]
        )
        self.assertEqual(leaf.state[model.unused]["step"].item(), 0)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        for parameter in reference.parameters():
            parameter.grad = torch.ones_like(parameter)
        adapter.step()
        reference_leaf.step()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected)

    def test_dense_muon_template_includes_lazy_group_step(self):
        """Native DCP must see the counter before the first local step."""
        leaf = SimpleNamespace(is_muon=True)
        adapter = HyperOptimizerAdapter(torch.nn.Linear(2, 2), {"muon": leaf})
        state = {"state": {"weight": {}}, "param_groups": [{"params": ["weight"]}]}
        with patch(
            "hyper_parallel.integration.veomni.optimizer.get_optim_state_dict",
            return_value=state,
        ):
            actual = adapter.state_dict()
        self.assertEqual(actual["param_groups.weight.step"], 0)

    def test_ns_callback_accepts_hyper_steps_keyword(self):
        """Forward Hyper's callback keyword without invoking optional CUDA kernels."""
        model = torch.nn.Linear(2, 2, bias=False)
        matrix = torch.ones(2, 2)
        factory, run_ns = MagicMock(), MagicMock(return_value=matrix)
        modules = {
            "hyper_parallel.core.optimizer.utils": SimpleNamespace(
                _install_logger_methods=MagicMock()
            ),
            "hyper_parallel.core.optimizer.dtensor_compat": SimpleNamespace(
                detect_dtensor_backend=MagicMock()
            ),
            "hyper_parallel.core.optimizer.muon": SimpleNamespace(Muon=factory),
            "veomni.optim.muon": SimpleNamespace(
                split_muon_adamw_params=MagicMock(
                    return_value=([model.weight], [], [], [])
                ),
                run_newton_schulz=run_ns,
            ),
        }
        for requested in ("std", "gram", "gram_quack", None):
            implementation = requested or "gram_quack"
            with self.subTest(implementation=implementation):
                values = {
                    "no_decay_modules": [],
                    "no_decay_params": [],
                    "lr": 1e-4,
                    "muon_kwargs": {"ns_implementation": requested}
                    if requested
                    else {},
                }
                with patch(
                    "hyper_parallel.integration.veomni.optimizer.import_module",
                    side_effect=modules.__getitem__,
                ):
                    _build_hyper_muon_adapter(model, values, SimpleNamespace())
                callback = factory.call_args.kwargs["zeropower_fn"]
                if implementation == "std":
                    self.assertIsNone(callback)
                    run_ns.assert_not_called()
                else:
                    self.assertIs(callback(matrix, steps=3), matrix)
                    run_ns.assert_called_once_with(
                        matrix,
                        ns_steps=3,
                        ns_coefficients=(3.4445, -4.775, 2.0315),
                        eps=1e-7,
                        ns_implementation=implementation,
                        gram_ns_reset_iterations=(2,),
                    )
                run_ns.reset_mock()
