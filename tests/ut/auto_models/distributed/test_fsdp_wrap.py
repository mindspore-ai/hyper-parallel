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
"""CPU-only checks for explicit FSDP child units and root reshard policy."""
# pylint: disable=protected-access

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from torch import nn
import yaml

from hyper_parallel.distributed._builder import fsdp_adapter
from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager
from hyper_parallel.models.build_options import FSDP2Config
from hyper_parallel.trainer.config.parser import parse_training_args


def _make_model(tied: bool = False) -> nn.Module:
    """Build the module layout of a text decoder without loading weights."""
    model = nn.Module()
    model.model = nn.Module()
    model.model.gradient_checkpointing = False
    model.model.embed_tokens = nn.Embedding(13, 5)
    model.model.layers = nn.ModuleList([nn.Linear(5, 5), nn.Linear(5, 5)])
    model.model.norm = nn.LayerNorm(5)
    model.lm_head = nn.Linear(5, 13, bias=False)
    if tied:
        model.lm_head.weight = model.model.embed_tokens.weight
    return model


class TestFSDPRootWrap(unittest.TestCase):
    """Check explicit child ownership and the shared root/child YAML policy."""

    def _manager(self, **kwargs) -> FSDP2Manager:
        return FSDP2Manager(FSDP2Config(**kwargs), SimpleNamespace(fsdp_moe_mesh=None, tp_size=1))

    def test_manual_classes_preserve_expert_units(self):
        """Feature: manual wrap. Description: select a parent with EP. Expectation: retain experts."""
        model = _make_model()
        model.model.layers[0].experts = nn.Linear(5, 5)
        manager = self._manager(wrapper_class=["ModuleList", "Embedding"])
        manager.mesh_context.fsdp_moe_mesh = object()
        metadata = {model.model.layers[0].experts.weight:
                    SimpleNamespace(mesh=SimpleNamespace(mesh_dim_names=("edp_shard", "ep")))}
        units = manager._find_wrap_modules(model, metadata)
        self.assertEqual([unit.fqn for unit in units],
                         ["model.embed_tokens", "model.layers", "model.layers.0.experts"])
        owners = manager._resolve_parameter_owners(model, units)
        self.assertIs(owners[model.model.layers[0].experts.weight], model.model.layers[0].experts)
        self.assertIs(owners[model.model.layers[1].weight], model.model.layers)
        self.assertIs(owners[model.lm_head.weight], model)
        dense_mesh, expert_mesh = object(), object()
        def _mesh_for_unit(expert=False):
            return expert_mesh if expert else dense_mesh
        def _source_for_unit(fsdp_manager, module, *_args):
            self.assertIs(fsdp_manager, manager)
            return metadata if module is model.model.layers[0].experts else None
        with patch.object(manager, "_build_fsdp_actual_mesh", side_effect=_mesh_for_unit), \
                patch.object(fsdp_adapter, "_build_source_shard_info_by_param", return_value=metadata), \
                patch.object(fsdp_adapter, "_build_managed_source_shard_info", side_effect=_source_for_unit), \
                patch.object(fsdp_adapter, "_source_infos_for_fully_shard", return_value=None), \
                patch.object(manager, "_configure_source_layout_gradient_scaling", return_value=False), \
                patch.object(manager, "_configure_prefetch"), \
                patch.object(fsdp_adapter, "fully_shard") as shard:
            manager.parallelize(model)
        calls = shard.call_args_list
        self.assertIs(calls[0].args[0], model.model.layers[0].experts)
        self.assertIs(calls[0].kwargs["mesh"], expert_mesh)
        self.assertIs(calls[-1].args[0], model)
        self.assertTrue(all(call.kwargs["mesh"] is dense_mesh for call in calls[1:]))

    def test_manual_classes_validate_and_handle_checkpoint_wrapper(self):
        """Feature: class selection. Description: invalid and wrapped inputs. Expectation: safe selection."""
        for names in ("Linear", [""], [7]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                FSDP2Config(wrapper_class=names)
        with self.assertRaisesRegex(ValueError, "matched no child"):
            self._manager(wrapper_class=["MissingClass"])._find_wrap_modules(_make_model())
        with self.assertRaisesRegex(ValueError, "shares parameters"):
            self._manager(wrapper_class=["Embedding"])._find_wrap_modules(_make_model(tied=True))
        model = _make_model()
        wrapper = nn.Module()
        wrapper._checkpoint_wrapped_module = model.model.layers[0]
        model.model.layers[0] = wrapper
        units = self._manager(wrapper_class=["Linear", "Linear"])._find_wrap_modules(model)
        self.assertEqual([unit.fqn for unit in units], ["model.layers.0", "model.layers.1", "lm_head"])
        self.assertIs(units[0].module, wrapper)

    def test_default_owners_unchanged(self):
        """Old configs retain embedding, head and final norm on the root."""
        model = _make_model()
        manager = self._manager()
        units = manager._find_wrap_modules(model)
        self.assertEqual([unit.fqn for unit in units], ["model.layers.0", "model.layers.1"])
        owners = manager._resolve_parameter_owners(model, units)
        for parameter in (model.model.embed_tokens.weight, model.lm_head.weight, model.model.norm.weight):
            self.assertIs(owners[parameter], model)

    def test_named_visual_children_keep_real_fqns_and_owners(self):
        """Named visual projections must not acquire synthetic ModuleList-style indices."""
        model = _make_model()
        model.visual = nn.Module()
        model.visual.gradient_checkpointing = False
        model.visual.patch_embed = nn.Module()
        model.visual.patch_embed.proj = nn.Linear(5, 5)
        model.visual.merger = nn.Module()
        model.visual.merger.norm = nn.LayerNorm(5)
        model.visual.merger.linear_fc1 = nn.Linear(5, 5)
        model.visual.merger.act_fn = nn.GELU()
        model.visual.merger.linear_fc2 = nn.Linear(5, 5)
        manager = self._manager()
        units = manager._find_wrap_modules(model)
        owners = manager._resolve_parameter_owners(model, units)
        for unit in units:
            self.assertIs(model.get_submodule(unit.fqn), unit.module)
            self.assertEqual({param for param, owner in owners.items() if owner is unit.module},
                             set(unit.module.parameters()))
        self.assertEqual([unit.fqn for unit in units if unit.fqn.startswith("model.layers.")],
                         ["model.layers.0", "model.layers.1"])
        self.assertIs(owners[model.visual.patch_embed.proj.weight], model.visual.patch_embed.proj)

    def test_explicit_units_own_embedding_and_head(self):
        """Extra units take exclusive ownership; root keeps only the final norm."""
        model = _make_model()
        manager = self._manager(wrapper_class=["Embedding", "Linear"])
        units = manager._find_wrap_modules(model)
        self.assertEqual(len(units), 4)
        owners = manager._resolve_parameter_owners(model, units)
        self.assertIs(owners[model.model.embed_tokens.weight], model.model.embed_tokens)
        self.assertIs(owners[model.lm_head.weight], model.lm_head)
        self.assertEqual({name for name, param in model.named_parameters() if owners[param] is model},
                         {"model.norm.weight", "model.norm.bias"})
        self.assertEqual([unit.fqn for unit in manager._order_wrap_modules(model, units)],
                         ["model.embed_tokens", "model.layers.0", "model.layers.1", "lm_head"])

    def test_root_and_child_share_reshard_policy(self):
        """One configured flag reaches both root and child units unchanged."""
        for reshard in (True, False):
            with self.subTest(reshard=reshard):
                manager = self._manager(reshard_after_forward=reshard)
                child_kwargs, root_kwargs = manager._build_fully_shard_kwargs(object())
                self.assertIs(child_kwargs["reshard_after_forward"], reshard)
                self.assertIs(root_kwargs["reshard_after_forward"], reshard)

    def test_embedding_only_leaves_head_on_root(self):
        """Chunk loss can access the head while embedding has its own FSDP unit."""
        model = _make_model()
        manager = self._manager(wrapper_class=["Embedding"])
        units = manager._find_wrap_modules(model)
        owners = manager._resolve_parameter_owners(model, units)
        self.assertIs(owners[model.model.embed_tokens.weight], model.model.embed_tokens)
        self.assertIs(owners[model.lm_head.weight], model)
        self.assertNotIn("lm_head", [unit.fqn for unit in units])

    def test_parallelize_passes_policy_to_actual_wrap_calls(self):
        """Child wrapping precedes root wrapping and prefetch includes the extra units."""
        model = _make_model()
        manager = self._manager(wrapper_class=["Embedding", "Linear"],
                                reshard_after_forward=True)
        with patch.object(manager, "_build_fsdp_actual_mesh", return_value=object()), \
                patch.object(manager, "_configure_prefetch") as prefetch, \
                patch.object(fsdp_adapter, "fully_shard") as shard:
            self.assertIs(manager.parallelize(model), model)
        self.assertEqual(len(shard.call_args_list), 5)
        self.assertIs(shard.call_args_list[-1].args[0], model)
        self.assertTrue(all(call.kwargs["reshard_after_forward"] for call in shard.call_args_list))
        self.assertEqual(prefetch.call_args.args[0],
                         [model.model.embed_tokens, *model.model.layers, model.lm_head])

    def test_tied_embedding_head_rejected_for_separate_units(self):
        """Shared weights keep their old root owner unless independent wrapping is requested."""
        model = _make_model(tied=True)
        manager = self._manager()
        owners = manager._resolve_parameter_owners(model, manager._find_wrap_modules(model))
        self.assertIs(owners[model.lm_head.weight], model)
        with self.assertRaisesRegex(ValueError, "shares parameters"):
            self._manager(wrapper_class=["Embedding", "Linear"])._find_wrap_modules(model)

    def test_config_rejects_invalid_types(self):
        """Catch malformed lists and root policy before touching the model."""
        with self.assertRaisesRegex(TypeError, "reshard_after_forward"):
            FSDP2Config(reshard_after_forward="true")

    def test_cli_overrides_reach_fsdp_config(self):
        """The Trainer parser applies CLI overrides to a self-contained FSDP config."""
        raw = {
            "model": {"_target_": "torch.nn.Linear", "in_features": 4, "out_features": 4},
            "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.01},
            "fsdp_config": {"wrapper_class": ["Embedding"], "reshard_after_forward": True},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fsdp.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            yaml_config = parse_training_args([str(path)])
            config = parse_training_args([
                str(path),
                "--fsdp_config.reshard_after_forward=false",
                "--fsdp_config.wrapper_class=[Linear,Embedding]",
            ])
        self.assertEqual(yaml_config.fsdp_config.wrapper_class, ["Embedding"])
        self.assertEqual(config.fsdp_config.wrapper_class, ["Linear", "Embedding"])
        self.assertTrue(yaml_config.fsdp_config.reshard_after_forward)
        self.assertFalse(config.fsdp_config.reshard_after_forward)
        self.assertIsNone(config.loss_fn)
