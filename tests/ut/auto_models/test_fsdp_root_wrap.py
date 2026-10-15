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

from hyper_parallel.components.losses import ChunkedCausalLMLoss
from hyper_parallel.core.fully_shard.api import HSDPModule
from hyper_parallel.distributed._builder import fsdp_adapter
from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager
from hyper_parallel.models.build_options import FSDP2Config
from hyper_parallel.trainer.config.manager import parse_training_args


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


class _ShardedHead(nn.Linear, HSDPModule):
    """Represent the wrapped-head interface without distributed initialization."""


class TestFSDPRootWrap(unittest.TestCase):
    """Check explicit child ownership and the shared root/child YAML policy."""

    def _manager(self, **kwargs) -> FSDP2Manager:
        return FSDP2Manager(FSDP2Config(**kwargs), SimpleNamespace(fsdp_moe_mesh=None, tp_size=1))

    def test_default_owners_unchanged(self):
        """Old configs retain embedding, head and final norm on the root."""
        model = _make_model()
        manager = self._manager()
        units = manager._find_wrap_modules(model)
        self.assertEqual([unit.fqn for unit in units], ["model.layers.0", "model.layers.1"])
        owners = manager._resolve_parameter_owners(model, units)
        for parameter in (model.model.embed_tokens.weight, model.lm_head.weight, model.model.norm.weight):
            self.assertIs(owners[parameter], model)

    def test_explicit_units_own_embedding_and_head(self):
        """Extra units take exclusive ownership; root keeps only the final norm."""
        model = _make_model()
        manager = self._manager(root_wrap_modules=["model.embed_tokens", "lm_head", "lm_head"])
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

    def test_parallelize_passes_policy_to_actual_wrap_calls(self):
        """Child wrapping precedes root wrapping and prefetch includes the extra units."""
        model = _make_model()
        manager = self._manager(root_wrap_modules=["model.embed_tokens", "lm_head"],
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

    def test_invalid_module_fqn_rejected(self):
        """Unknown modules and parameter names are not silently ignored."""
        for name in ("missing", "lm_head.weight"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "module FQN"):
                self._manager(root_wrap_modules=[name])._find_wrap_modules(_make_model())

    def test_tied_embedding_head_rejected_for_separate_units(self):
        """Shared weights keep their old root owner unless independent wrapping is requested."""
        model = _make_model(tied=True)
        manager = self._manager()
        owners = manager._resolve_parameter_owners(model, manager._find_wrap_modules(model))
        self.assertIs(owners[model.lm_head.weight], model)
        with self.assertRaisesRegex(ValueError, "shares parameters"):
            self._manager(root_wrap_modules=["model.embed_tokens", "lm_head"])._find_wrap_modules(model)

    def test_config_rejects_invalid_types(self):
        """Catch malformed lists and root policy before touching the model."""
        for names in ("lm_head", [""], [None], ["model..embed_tokens"]):
            with self.subTest(names=names), self.assertRaisesRegex(ValueError, "root_wrap_modules"):
                FSDP2Config(root_wrap_modules=names)
        with self.assertRaisesRegex(TypeError, "reshard_after_forward"):
            FSDP2Config(reshard_after_forward="true")

    def test_cli_overrides_reach_fsdp_config(self):
        """The real Trainer parser accepts both new fields without changing YAML defaults."""
        yaml_path = Path(__file__).resolve().parents[3] / "examples/qwen3_8/train_online.yaml"
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        # Exercise the real config types without importing dataset native helpers.
        raw = {key: raw[key] for key in ("model", "optimizer", "fsdp_config")}
        raw["fsdp_config"].update(root_wrap_modules=["model.embed_tokens"], reshard_after_forward=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fsdp.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            yaml_config = parse_training_args([str(path)])
            config = parse_training_args([
                str(path), "--fsdp_config.root_wrap_modules=[model.embed_tokens,lm_head]",
                "--fsdp_config.reshard_after_forward=false",
            ])
        self.assertEqual(yaml_config.fsdp_config.root_wrap_modules, ["model.embed_tokens"])
        self.assertTrue(yaml_config.fsdp_config.reshard_after_forward)
        self.assertEqual(config.fsdp_config.root_wrap_modules, ["model.embed_tokens", "lm_head"])
        self.assertFalse(config.fsdp_config.reshard_after_forward)
        self.assertIsNone(config.loss_fn)

    def test_chunk_loss_rejects_independent_head(self):
        """Direct head-weight access cannot bypass an independent FSDP unit's hooks."""
        model = _make_model()
        model.lm_head = _ShardedHead(5, 13, bias=False)
        with self.assertRaisesRegex(NotImplementedError, "independently FSDP-wrapped lm_head"):
            ChunkedCausalLMLoss().bind_model(model)
