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
"""CPU coverage of logical HF loading into HP replacement layouts."""

import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import torch
from rl.unified_execution.runtimes.vllm.weight_loader import LogicalWeightLoader
from rl.weight_sync.model_adapter import ModelWeightAdapter
from rl.weight_sync.packed_weight import (
    build_packed_weight_buckets,
    materialize_packed_weight_bucket,
    unpack_packed_weights,
)
from rl.workers.engine.model import ModelRegistration, resolve_vllm_model
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from hyper_parallel.components.checkpoint.conversion_ops import ConcatenateWithSections, InterleaveQKV
from hyper_parallel.components.checkpoint.weight_conversion import WeightConverter, WeightRenaming
from hyper_parallel.components.modules.swiglu_mlp import SwiGLUMLP
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import _LAYOUT_CACHE, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Shard
from hyper_parallel.models.qwen3.adapter.conversion.module_replacement import replace_qwen3_flash_attention


class _GroupedProjection(nn.Module):
    """Minimal storage with the production GQA conversion descriptor."""

    def __init__(self) -> None:
        """Allocate the fused parameter under the production projection name."""
        super().__init__()
        self.linear_qkv = nn.Linear(3, 16, bias=False)
        self.kv_heads = 2

    def make_transforms(self) -> list:
        """Describe two KV groups, two query heads per group, head size two."""
        return [WeightConverter(
            ["q_proj.weight", "k_proj.weight", "v_proj.weight"],
            "linear_qkv.weight", [InterleaveQKV(self.kv_heads, 2, 2, 2, False)],
        )]


class _FusedMLP(nn.Module):
    """Minimal storage with production SwiGLU conversion descriptors."""

    def __init__(self) -> None:
        """Allocate fused Gate/Up and renamed Down parameters."""
        super().__init__()
        self.linear_fc1 = nn.Linear(3, 10, bias=False)
        self.linear_fc2 = nn.Linear(5, 3, bias=False)

    @staticmethod
    def make_transforms() -> list:
        """Describe the gate/up merge and down-projection rename."""
        return [
            WeightConverter(["gate_proj.weight", "up_proj.weight"], "linear_fc1.weight",
                            [ConcatenateWithSections((5, 5))]),
            WeightRenaming("down_proj.weight", "linear_fc2.weight"),
        ]


class TestUnifiedWeightLoader(unittest.TestCase):
    """Publication completeness must include every logical projection."""

    def setUp(self) -> None:
        """Prepare distinct logical values that expose row-order mistakes."""
        self.addCleanup(_LAYOUT_CACHE.clear)
        self.model = nn.Module()
        self.model.attn = _GroupedProjection()
        self.model.mlp = _FusedMLP()
        self.loader = LogicalWeightLoader(self.model)
        self.weights = {
            "attn.q_proj.weight": torch.arange(24, dtype=torch.float32).reshape(8, 3),
            "attn.k_proj.weight": torch.arange(12, dtype=torch.float32).reshape(4, 3) + 100,
            "attn.v_proj.weight": torch.arange(12, dtype=torch.float32).reshape(4, 3) + 200,
            "mlp.gate_proj.weight": torch.full((5, 3), 3.0),
            "mlp.up_proj.weight": torch.full((5, 3), 4.0),
            "mlp.down_proj.weight": torch.full((3, 5), 5.0),
        }

    def _check_values(self, increment: float = 0.0) -> None:
        query, key, value = (self.weights[f"attn.{name}_proj.weight"] for name in ("q", "k", "v"))
        expected = torch.cat((query[:4], key[:2], value[:2], query[4:], key[2:], value[2:]))
        torch.testing.assert_close(self.model.attn.linear_qkv.weight, expected + increment)
        torch.testing.assert_close(
            self.model.mlp.linear_fc1.weight[:5], self.weights["mlp.gate_proj.weight"] + increment,
        )
        torch.testing.assert_close(self.model.mlp.linear_fc1.weight[5:], self.weights["mlp.up_proj.weight"] + increment)
        torch.testing.assert_close(
            self.model.mlp.linear_fc2.weight, self.weights["mlp.down_proj.weight"] + increment,
        )

    def test_initial_checkpoint_then_independent_bucket_publication(self) -> None:
        """One iterator and reordered single-tensor buckets load identical layouts."""
        loaded = self.loader.load(self.weights.items())
        self.assertEqual(loaded, {"attn.linear_qkv.weight", "mlp.linear_fc1.weight", "mlp.linear_fc2.weight"})
        self._check_values()
        self.loader.begin()
        for name in reversed(self.weights):
            self.assertTrue(self.loader.load([(name, self.weights[name] + 7)]))
        self.loader.finish()
        self._check_values(7)

    def test_duplicate_logical_source_rejected_across_buckets(self) -> None:
        """A repeated source cannot overwrite a previously accepted bucket."""
        self.loader.begin()
        item = next(iter(self.weights.items()))
        self.loader.load([item])
        with self.assertRaisesRegex(ValueError, "Duplicate logical weight"):
            self.loader.load([item])

    def test_bad_shape_and_unknown_name_rejected(self) -> None:
        """Do not silently accept checkpoint names or shapes outside the plan."""
        self.loader.begin()
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            self.loader.load([("attn.q_proj.weight", torch.zeros(3, 8))])
        with self.assertRaisesRegex(ValueError, "Unknown logical"):
            self.loader.load([("attn.linear_qkv.weight", torch.zeros(16, 3))])

    def test_tied_alias_coverage_and_conflicts(self) -> None:
        """One alias covers tied storage; two aliases must carry equal values."""
        model = nn.Module()
        model.embed_tokens = nn.Embedding(4, 3)
        model.lm_head = nn.Linear(3, 4, bias=False)
        model.lm_head.weight = model.embed_tokens.weight
        loader = LogicalWeightLoader(model)
        value = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        self.assertEqual(loader.load([("embed_tokens.weight", value)]),
                         {"embed_tokens.weight", "lm_head.weight"})
        loader.load([("embed_tokens.weight", value), ("lm_head.weight", value.clone())])
        loader.begin()
        loader.load([("lm_head.weight", value)])
        with self.assertRaisesRegex(ValueError, "Conflicting tied"):
            loader.load([("embed_tokens.weight", value + 1)])

    def test_parallel_projection_loads_only_owned_fused_rows(self) -> None:
        """Both TP ranks receive correct interleaved QKV rows across buckets."""
        query, key, value = (self.weights[f"attn.{name}_proj.weight"] for name in ("q", "k", "v"))
        expected = torch.cat((query[:4], key[:2], value[:2], query[4:], key[2:], value[2:]))
        for rank in range(2):
            with patch("hyper_parallel.core.dtensor.device_mesh.dist.get_rank", return_value=rank), \
                    patch("hyper_parallel.core.dtensor.device_mesh.dist.is_initialized", return_value=True):
                _LAYOUT_CACHE.clear()
                mesh = DeviceMesh("cpu", [0, 1], mesh_dim_names=("tp",), _init_backend=False)
                model = _GroupedProjection()
                model.linear_qkv.weight = nn.Parameter(distribute_tensor(
                    torch.zeros(16, 3), mesh, [Shard(0)],
                ))
                model._rl_logical_weight_transforms = model.make_transforms()  # pylint: disable=protected-access
                model.kv_heads = 1
                loader = LogicalWeightLoader(model)
                loader.begin()
                for name in ("v", "q", "k"):
                    loader.load([(f"{name}_proj.weight", self.weights[f"attn.{name}_proj.weight"])])
                loader.finish()
                torch.testing.assert_close(model.linear_qkv.weight.to_local(), expected[rank * 8:(rank + 1) * 8])
        _LAYOUT_CACHE.clear()

    def test_parallel_projection_slices_columns(self) -> None:
        """Row-parallel projection loading follows its actual HP column layout."""
        with patch("hyper_parallel.core.dtensor.device_mesh.dist.get_rank", return_value=1), \
                patch("hyper_parallel.core.dtensor.device_mesh.dist.is_initialized", return_value=True):
            _LAYOUT_CACHE.clear()
            mesh = DeviceMesh("cpu", [0, 1], mesh_dim_names=("tp",), _init_backend=False)
            model = nn.Linear(6, 4, bias=False)
            model.weight = nn.Parameter(distribute_tensor(torch.zeros(4, 6), mesh, [Shard(1)]))
            value = torch.arange(24, dtype=torch.float32).reshape(4, 6)
            LogicalWeightLoader(model).load([("weight", value)])
            torch.testing.assert_close(model.weight.to_local(), value[:, 3:])
        _LAYOUT_CACHE.clear()

    def test_production_local_parameters_use_training_layout(self) -> None:
        """Production local parameters load both interleaved rows and column shards."""
        query, key, value = (self.weights[f"attn.{name}_proj.weight"] for name in ("q", "k", "v"))
        expected = torch.cat((query[:4], key[:2], value[:2], query[4:], key[2:], value[2:]))
        expected = expected.repeat(1, 2)
        for rank in range(2):
            for axis in (0, 1):
                with self.subTest(rank=rank, axis=axis), \
                        patch("hyper_parallel.core.dtensor.device_mesh.dist.get_rank", return_value=rank), \
                        patch("hyper_parallel.core.dtensor.device_mesh.dist.is_initialized", return_value=True):
                    _LAYOUT_CACHE.clear()
                    mesh = DeviceMesh("cpu", [0, 1], mesh_dim_names=("tp",), _init_backend=False)
                    model = _GroupedProjection()
                    model._rl_logical_weight_transforms = model.make_transforms()  # pylint: disable=protected-access
                    shard = distribute_tensor(torch.zeros(16, 6), mesh, [Shard(axis)])
                    parameter = nn.Parameter(shard.to_local())
                    parameter._sharding_spec = shard.layout  # pylint: disable=protected-access
                    model.linear_qkv.weight = parameter
                    model.kv_heads = 1
                    loader = LogicalWeightLoader(model)
                    loader.begin()
                    for name in ("v", "q", "k"):
                        source = self.weights[f"attn.{name}_proj.weight"].repeat(1, 2)
                        loader.load([(f"{name}_proj.weight", source)])
                        source.fill_(-1)
                    loader.finish()
                    torch.testing.assert_close(parameter, expected.chunk(2, dim=axis)[rank], rtol=0, atol=0)

    def test_production_tied_column_shards_preserve_alias_checks(self) -> None:
        """Local tensor loading honors HP placement and compares tied local storage."""
        with patch("hyper_parallel.core.dtensor.device_mesh.dist.get_rank", return_value=1), \
                patch("hyper_parallel.core.dtensor.device_mesh.dist.is_initialized", return_value=True):
            mesh = DeviceMesh("cpu", [0, 1], mesh_dim_names=("tp",), _init_backend=False)
            shard = distribute_tensor(torch.zeros(4, 6), mesh, [Shard(1)])
            parameter = nn.Parameter(shard.to_local())
            parameter._sharding_spec = shard.layout  # pylint: disable=protected-access
            model = nn.Module()
            model.register_parameter("weight", parameter)
            model.register_parameter("alias", parameter)
            source = torch.arange(24, dtype=torch.float32).reshape(4, 6)
            loader = LogicalWeightLoader(model)
            self.assertEqual(loader.load([("weight", source), ("alias", source.clone())]), {"weight", "alias"})
            torch.testing.assert_close(parameter, source[:, 3:], rtol=0, atol=0)
            loader.begin()
            loader.load([("weight", source)])
            with self.assertRaisesRegex(ValueError, "Conflicting tied"):
                loader.load([("alias", source + 1)])


if __name__ == "__main__":
    unittest.main()


def _registration(implementation: str) -> object:
    identity = ModelRegistration(
        "qwen3", "qwen3", "/unused", "/unused", "Qwen3ForCausalLM", "qwen3", "qwen3", False,
    )
    return resolve_vllm_model(identity, implementation)


def _source(bias: bool) -> nn.Module:
    """Construct independent logical source weights for publication tests."""
    model = nn.Module()
    model.config = SimpleNamespace(model_type="qwen3", hidden_act="silu")
    model.model = nn.Module()
    layer = nn.Module()
    layer.mlp = nn.Module()
    layer.mlp.config = model.config
    layer.mlp.gate_proj = nn.Linear(4, 6, bias=bias)
    layer.mlp.up_proj = nn.Linear(4, 6, bias=bias)
    layer.mlp.down_proj = nn.Linear(6, 4, bias=bias)
    model.model.layers = nn.ModuleList([layer])
    return model.to(dtype=torch.bfloat16)


def _publish(source: nn.Module, implementation: str) -> tuple[nn.Module, dict]:
    """Publish a complete logical weight stream through the real adapter."""
    receiver = deepcopy(source)
    with torch.no_grad():
        for parameter in receiver.parameters():
            parameter.zero_()
    loader = LogicalWeightLoader(receiver)
    adapter = ModelWeightAdapter(_registration(implementation))
    adapter.bind_source(source)
    state = adapter.map_local_state_dict(source.state_dict())
    logical_received = {}
    loader.begin()
    with patch("rl.weight_sync.packed_weight.dist.get_rank", return_value=0):
        for bucket in build_packed_weight_buckets(state, 32):
            packed = materialize_packed_weight_bucket(state, bucket, device=torch.device("cpu"))
            logical = unpack_packed_weights(packed, adapter.packed_metadata(bucket.worker_metadata()))
            logical_received.update((name, value.clone()) for name, value in logical)
            loader.load(logical)
            # Prove the loader owns a copy after the transport reuses its bucket.
            packed.zero_()
    loader.finish()
    return receiver, logical_received


class TestUnifiedPublication(unittest.TestCase):
    """The sender must recover HF names from actual HP conversion descriptors."""

    def test_fused_mlp_round_trip_through_actual_transfer_contract(self) -> None:
        """Recover logical HF projections and reproduce every fused parameter bit."""
        for bias in (False, True):
            for implementation in ("native", "unified"):
                with self.subTest(bias=bias, implementation=implementation):
                    source = _source(bias)
                    logical_reference = {name: value.clone() for name, value in source.state_dict().items()}
                    layer = source.model.layers[0]
                    layer.mlp = SwiGLUMLP(module=layer.mlp)
                    physical_reference = {name: value.clone() for name, value in source.state_dict().items()}
                    receiver, logical_received = _publish(source, implementation)
                    self.assertEqual(logical_received.keys(), logical_reference.keys())
                    for name, expected in logical_reference.items():
                        self.assertTrue(torch.equal(logical_received[name], expected), name)
                    for name, actual in receiver.state_dict().items():
                        self.assertTrue(torch.equal(actual, physical_reference[name]), name)

    def test_source_conversion_uses_logical_shapes_after_tp_localization(self) -> None:
        """Source metadata remains global when production TP exposes plain local parameters."""
        source = _source(False)
        mlp = SwiGLUMLP(module=source.model.layers[0].mlp)
        source.model.layers[0].mlp = mlp
        source._rl_logical_parameter_shapes = {  # pylint: disable=protected-access
            name: tuple(value.shape) for name, value in source.named_parameters()
        }
        mlp._rl_logical_weight_transforms = mlp.make_transforms()  # pylint: disable=protected-access
        full_state = {name: value.clone() for name, value in source.state_dict().items()}
        mlp.linear_fc1.weight = nn.Parameter(mlp.linear_fc1.weight[:6].clone())
        adapter = ModelWeightAdapter(_registration("unified"))
        adapter.bind_source(source)
        bucket = build_packed_weight_buckets(full_state, 4096)[0]
        metadata = adapter.packed_metadata(bucket.worker_metadata())
        entry = next(item for item in metadata if item["name"].endswith("linear_fc1.weight"))
        self.assertEqual([row["length"] for row in entry["canonical_rows"]], [6, 6])

    def test_qkv_bias_and_mlp_survive_source_to_receiver_publication(self) -> None:
        """Round-trip real Qwen3 grouped QKV weights and biases alongside fused MLP."""
        config = Qwen3Config.from_dict({
            "vocab_size": 32, "hidden_size": 16, "intermediate_size": 24, "num_hidden_layers": 1,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 4, "attention_bias": True,
            "tie_word_embeddings": False,
        })
        source = Qwen3ForCausalLM(config).to(dtype=torch.bfloat16)
        logical_reference = {name: value.clone() for name, value in source.state_dict().items()}
        layer = source.model.layers[0]
        layer.self_attn = replace_qwen3_flash_attention(
            module=layer.self_attn, module_fqn="model.layers.0.self_attn", context={},
        )
        layer.mlp = SwiGLUMLP(module=layer.mlp)
        physical_reference = {name: value.clone() for name, value in source.state_dict().items()}
        receiver, logical_received = _publish(source, "unified")
        self.assertEqual(logical_received.keys(), logical_reference.keys())
        for name, expected in logical_reference.items():
            self.assertTrue(torch.equal(logical_received[name], expected), name)
        for name, actual in receiver.state_dict().items():
            self.assertTrue(torch.equal(actual, physical_reference[name]), name)


def _expert_model() -> nn.Module:
    """Build a small expert model with the requested local parameter layout."""
    model = nn.Module()
    model.config = SimpleNamespace(model_type="qwen3_moe", num_experts=3, hidden_size=4,
                                   moe_intermediate_size=2, tie_word_embeddings=False)
    model.model = nn.Module()
    layer = nn.Module()
    layer.mlp = nn.Module()
    layer.mlp.experts = nn.Module()
    layer.mlp.experts.gate_up_proj = nn.Parameter(torch.zeros(3, 4, 4))
    layer.mlp.experts.down_proj = nn.Parameter(torch.zeros(3, 2, 4))
    model.model.layers = nn.ModuleList([layer])
    return model


class TestExpertWeights(unittest.TestCase):
    """Shared expert identity and publication contracts."""

    def setUp(self) -> None:
        """Keep metadata-only mesh caches isolated between tests."""
        self.addCleanup(_LAYOUT_CACHE.clear)

    def test_streamed_expert_loading_and_publication_completeness(self) -> None:
        """Independent gate/up/down deliveries reconstruct the actual expert matrices."""
        model = _expert_model()
        loader = LogicalWeightLoader(model)
        weights = {}
        for expert in range(3):
            for offset, projection in enumerate(("gate_proj", "up_proj", "down_proj")):
                shape = (4, 2) if projection == "down_proj" else (2, 4)
                weights[f"model.layers.0.mlp.experts.{expert}.{projection}.weight"] = (
                    torch.arange(8, dtype=torch.float32).reshape(shape) + 100 * expert + 10 * offset
                )
        loader.begin()
        for name, value in reversed(list(weights.items())):
            loader.load([(name, value)])
        loader.finish()
        experts = model.model.layers[0].mlp.experts
        for expert in range(3):
            prefix = f"model.layers.0.mlp.experts.{expert}."
            expected = torch.cat((weights[prefix + "gate_proj.weight"], weights[prefix + "up_proj.weight"]), dim=0)
            torch.testing.assert_close(experts.gate_up_proj[expert], expected.T, atol=0, rtol=0)
            torch.testing.assert_close(
                experts.down_proj[expert], weights[prefix + "down_proj.weight"].T, atol=0, rtol=0,
            )
        loader.begin()
        loader.load(list(weights.items())[:-1])
        with self.assertRaisesRegex(ValueError, "Incomplete unified weight publication"):
            loader.finish()

    def test_expert_shards_load_only_owned_regions(self) -> None:
        """Global checkpoint pieces reconstruct three distinct local expert shards."""
        for rank in range(3):
            with self.subTest(rank=rank), \
                    patch("hyper_parallel.core.dtensor.device_mesh.dist.get_rank", return_value=rank), \
                    patch("hyper_parallel.core.dtensor.device_mesh.dist.is_initialized", return_value=True):
                _LAYOUT_CACHE.clear()
                model = _expert_model()
                mesh = DeviceMesh("cpu", [0, 1, 2], mesh_dim_names=("ep",), _init_backend=False)
                experts = model.model.layers[0].mlp.experts
                for name in ("gate_up_proj", "down_proj"):
                    shard = distribute_tensor(experts.get_parameter(name).detach(), mesh, [Shard(0)])
                    parameter = nn.Parameter(shard.to_local())
                    parameter._sharding_spec = shard.layout  # pylint: disable=protected-access
                    setattr(experts, name, parameter)
                loader = LogicalWeightLoader(model)
                weights = [(f"model.layers.0.mlp.experts.{expert}.{projection}.weight",
                            torch.full((4, 2) if projection == "down_proj" else (2, 4), float(10 * expert + offset)))
                           for expert in range(3)
                           for offset, projection in enumerate(("gate_proj", "up_proj", "down_proj"))]
                loader.load(reversed(weights))
                expected = torch.cat((torch.full((1, 4, 2), float(rank * 10)),
                                      torch.full((1, 4, 2), float(rank * 10 + 1))), dim=-1)
                torch.testing.assert_close(experts.gate_up_proj, expected, atol=0, rtol=0)
                torch.testing.assert_close(experts.down_proj, torch.full((1, 2, 4), float(rank * 10 + 2)),
                                           atol=0, rtol=0)
