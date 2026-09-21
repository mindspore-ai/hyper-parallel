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
"""Numerical and build-contract tests for the shared Qwen3 RL integration."""
# pylint: disable=forbidden-backend-import,protected-access

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch_npu
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.distributed import ShardingPlanner as SharedPlanner
from hyper_parallel.distributed._builder import parameter_sharding
from hyper_parallel.distributed.mesh import DistributedSetup, MeshContext
from hyper_parallel.distributed.tensor_parallel.param_role import ParameterClassifier
from hyper_parallel.models._transformers import auto_model, model_builder
from hyper_parallel.models._transformers.checkpoint_loader import CheckpointManager
from hyper_parallel.models._transformers.model_builder import _move_model_to_device
from hyper_parallel.models.build_options import FSDP2Config
from hyper_parallel.models.replacement import apply_module_replacements, compile_module_replacements
from rl.roles import qwen3_builder as qwen3
from rl.roles.model_setup import ModelRegistration, resolve_vllm_model
from rl.roles.qwen3_builder import Qwen3ShardingPlanner as ShardingPlanner
from rl.roles.qwen3_builder import get_module_replacements, get_parallel_overrides
from rl.roles.qwen3_builder import restore_tied_parameter as _replicate_tied_weights
from rl.roles.weight_sync import packed_weight
from rl.roles.weight_sync.layout import (
    build_direct_reshard_plan,
    pack_direct_bucket,
    resolve_destination_layouts,
    resolve_source_layouts,
)
from rl.roles.weight_sync.model_adapter import ModelWeightAdapter, rollout_tensor_descriptions
from rl.roles.weight_sync.packed_weight import (
    build_packed_weight_buckets,
    materialize_packed_weight_bucket,
    unpack_packed_weights,
)
from tests.ut.auto_models.distributed.conftest import FakeDeviceMesh


def tiny_config() -> Qwen3Config:
    """Keep explicit head_dim different from hidden_size / query heads."""
    return Qwen3Config(
        vocab_size=64, hidden_size=24, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=64, tie_word_embeddings=True, use_cache=False,
        architectures=["Qwen3ForCausalLM"],
    )


@pytest.fixture(name="cpu_kernels")
def _cpu_kernels(monkeypatch):
    """Use independent CPU mathematical oracles for only the NPU primitives."""
    def rms(x, weight, epsilon):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + epsilon)
        return (normalized.to(x.dtype) * weight,)

    def rotary(x, cos, sin, **_kwargs):
        first, second = x.chunk(2, dim=-1)
        return x * cos + torch.cat((-second, first), dim=-1) * sin

    def attention(q, k, v, *, atten_mask, sparse_mode, keep_prob, scale, **_kwargs):
        allowed = None if sparse_mode == 2 else ~atten_mask
        ratio = q.shape[1] // k.shape[1]
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k.repeat_interleave(ratio, dim=1), v.repeat_interleave(ratio, dim=1),
            attn_mask=allowed, is_causal=sparse_mode == 2,
            dropout_p=1 - keep_prob, scale=scale,
        )
        return (out,)

    monkeypatch.setattr(torch_npu, "npu_rms_norm", rms)
    monkeypatch.setattr(torch_npu, "npu_rotary_mul", rotary)
    monkeypatch.setattr(torch_npu, "npu_fusion_attention", attention)


def replace_model(model):
    """Apply the production declarations with the real replacement executor."""
    plan = compile_module_replacements(model, get_module_replacements())
    return apply_module_replacements(model, plan, weights_mapping=[])[0]


def test_shared_loader_preserves_checkpoint_logits_and_gradients(tmp_path, monkeypatch, cpu_kernels):
    """The public loader keeps real HF weights, padding semantics and gradients."""
    del cpu_kernels
    torch.manual_seed(42)
    reference = Qwen3ForCausalLM(tiny_config()).eval()
    reference.save_pretrained(tmp_path)
    monkeypatch.setattr(auto_model, "_current_device", lambda: torch.device("cpu"))
    setup = DistributedSetup()
    model = qwen3.from_pretrained(
        str(tmp_path), distributed_setup=setup, local_files_only=True,
        torch_dtype="float32", attn_implementation="sdpa", force_hf=True,
    ).eval()
    assert isinstance(model.model.layers[0].mlp, Qwen3MLP)
    assert setup.module_replacements is None
    assert setup.plan_overrides is None
    assert model.lm_head.weight is model.model.embed_tokens.weight
    tokens = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 0, 0]])
    mask = tokens.ne(0)
    actual = model(tokens, attention_mask=mask, use_cache=False).logits
    expected = reference(tokens, attention_mask=mask, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(
        model.model.embed_tokens.weight.grad, reference.model.embed_tokens.weight.grad,
        atol=2e-5, rtol=2e-5,
    )


def test_tp_plan_keeps_mlp_boundary_and_local_head_counts():
    """Fusion must not create a colwise leaf boundary or miss local head metadata."""
    model = replace_model(Qwen3ForCausalLM(tiny_config()))
    mesh = FakeDeviceMesh((2,), ("tp",))
    plan = ShardingPlanner(plan_overrides=get_parallel_overrides()).plan(
        model, mesh, tp_size=2, sequence_parallel=False,
    )
    attention = plan.modules["model.layers.0.self_attn"]
    assert set(attention._tp_local_attr_plan.auto_divide) == {"num_heads", "num_key_value_heads"}
    mlp = plan.modules["model.layers.0.mlp"]
    assert set(mlp.params) == {"gate_proj.weight", "up_proj.weight", "down_proj.weight"}
    assert "model.layers.0.mlp.gate_up_proj" not in plan.modules
    assert "lm_head" in plan.modules
    assert plan.modules["lm_head"].params["weight"] == plan.modules["model.embed_tokens"].params["weight"]


def test_native_mlp_tp_shards_match_original_forward_and_input_gradients():
    """The RL recipe retains the native dense MLP's ordinary TP contract."""
    source = Qwen3ForCausalLM(tiny_config()).model.layers[0].mlp
    hidden = torch.randn(2, 5, 24, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    outputs = []
    for rank in range(2):
        shard = deepcopy(source)
        for name in ("gate_proj", "up_proj"):
            getattr(shard, name).weight = torch.nn.Parameter(getattr(source, name).weight.chunk(2, dim=0)[rank].clone())
        shard.down_proj.weight = torch.nn.Parameter(source.down_proj.weight.chunk(2, dim=1)[rank].clone())
        outputs.append(shard(hidden))
    actual = sum(outputs)
    expected = source(reference_hidden)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, atol=1e-8, rtol=1e-5)


def weight_adapter(config):
    """Build the same source adapter used by both publication strategies."""
    identity = ModelRegistration("qwen", "qwen3", "/unused", "/unused", "Qwen3ForCausalLM", "qwen3", "qwen3", True)
    adapter = ModelWeightAdapter(resolve_vllm_model(identity, "hyper"))
    adapter.bind_source(SimpleNamespace(config=config))
    return adapter


def test_full_gather_restores_every_hf_weight(monkeypatch):
    """All QKV/gate/up values and tied aliases survive a packed publication."""
    monkeypatch.setattr(packed_weight.dist, "get_rank", lambda: 0)
    original = Qwen3ForCausalLM(tiny_config())
    expected = {name: value.clone() for name, value in original.state_dict().items()}
    replaced = replace_model(original)
    adapter = weight_adapter(original.config)
    state = adapter.map_local_state_dict(replaced.state_dict())
    actual = {}
    for bucket in build_packed_weight_buckets(state, 256):
        packed = materialize_packed_weight_bucket(state, bucket)
        actual.update(unpack_packed_weights(packed, adapter.packed_metadata(bucket.worker_metadata())))
    expected.pop("lm_head.weight")
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)

@pytest.mark.parametrize("trainer_tp,dp_shards", [(1, 1), (1, 3), (2, 1), (2, 3)])
@pytest.mark.parametrize("rollout_tp", [1, 2])
@pytest.mark.parametrize("is_hyper", [True, False])
def test_direct_reshard_fused_sources_match_canonical_destinations(trainer_tp, dp_shards, rollout_tp, is_hyper):
    """Copy actual fused shards through the production direct bucket packer."""

    config = tiny_config()
    model = Qwen3ForCausalLM(config)
    expected = {name: value.detach().clone() for name, value in model.named_parameters()}
    state = replace_model(model).state_dict()
    state.pop("lm_head.weight")
    adapter = weight_adapter(config)
    states = []
    descriptions = []

    class Mesh:
        """Expose explicit DP and TP coordinates to the layout planner."""
        ndim = 2

        def __init__(self, dp_rank, tp_rank):
            self.coordinate = (dp_rank, tp_rank)

        def size(self, axis):
            return (dp_shards, trainer_tp)[axis]

        def get_coordinate(self):
            return self.coordinate

    class LocalShard:
        def __init__(self, full, local, placements, mesh):
            self.shape = full.shape
            self.local = local
            self.placements = placements
            self.device_mesh = mesh

        def to_local(self):
            return self.local

    for dp_rank in range(dp_shards):
        for tp_rank in range(trainer_tp):
            rank = dp_rank * trainer_tp + tp_rank
            mesh = Mesh(dp_rank, tp_rank)
            local_state = {}
            for name, full in state.items():
                if "norm" in name:
                    tp_placement = Replicate()
                    tp_tensor = full
                else:
                    axis = 1 if name.endswith(("o_proj.weight", "down_proj.weight")) else 0
                    tp_placement = Shard(axis)
                    tp_tensor = full.tensor_split(trainer_tp, dim=axis)[tp_rank]
                local = tp_tensor.tensor_split(dp_shards, dim=0)[dp_rank].contiguous()
                local_state[name] = LocalShard(full, local, (Shard(0), tp_placement), mesh)
            states.append(local_state)
            descriptions.append(adapter.direct_source_descriptions(local_state, rank))
    sources = resolve_source_layouts(descriptions)

    destination_maps = []
    worker_descriptions = []
    expected_maps = []
    for tp_rank in range(rollout_tp):
        parameters = {}
        placements = {}
        expected_map = {}
        for name, full in expected.items():
            if "norm" in name:
                placement = Replicate()
                local = full
            else:
                axis = 1 if name.endswith(("o_proj.weight", "down_proj.weight")) else 0
                placement = Shard(axis)
                local = full.chunk(rollout_tp, dim=axis)[tp_rank]
            placements[name] = (placement,)
            expected_map[name] = local
        if not is_hyper:
            for layer in range(config.num_hidden_layers):
                prefix = f"model.layers.{layer}."
                qkv = [expected_map.pop(prefix + "self_attn." + projection + ".weight")
                       for projection in ("q_proj", "k_proj", "v_proj")]
                gate_up = [expected_map.pop(prefix + "mlp." + projection + ".weight")
                           for projection in ("gate_proj", "up_proj")]
                expected_map[prefix + "self_attn.qkv_proj.weight"] = torch.cat(qkv)
                expected_map[prefix + "mlp.gate_up_proj.weight"] = torch.cat(gate_up)
        parameters = {name: torch.zeros_like(value) for name, value in expected_map.items()}
        worker_model = SimpleNamespace(
            named_parameters=lambda values=parameters: values.items(), _tp_placements=placements,
        )
        worker_descriptions.append({
            "tp_rank": tp_rank, "tp_size": rollout_tp,
            "tensors": rollout_tensor_descriptions(
                worker_model, config, is_hyper=is_hyper, tp_rank=tp_rank, tp_size=rollout_tp,
            ),
        })
        destination_maps.append(parameters)
        expected_maps.append(expected_map)

    destinations = resolve_destination_layouts(
        worker_descriptions, {name: tuple(value.shape) for name, value in expected.items()},
    )
    plan = build_direct_reshard_plan(
        sources, destinations, source_world_size=len(states), bucket_size_bytes=128,
    )
    _assert_direct_bucket_copy(plan, states, destination_maps, expected_maps)


def test_shared_model_hf_export_round_trip(tmp_path, monkeypatch, cpu_kernels):
    """A live single-rank build exports reversible HF weights with tied aliases."""
    del cpu_kernels
    reference = Qwen3ForCausalLM(tiny_config()).eval()
    source_dir = tmp_path / "source"
    reference.save_pretrained(source_dir)
    monkeypatch.setattr(auto_model, "_current_device", lambda: torch.device("cpu"))
    model = qwen3.from_pretrained(str(source_dir), torch_dtype="float32", local_files_only=True)
    destination = tmp_path / "export"
    assert CheckpointManager(model).save_pretrained(destination)
    restored = Qwen3ForCausalLM.from_pretrained(destination)
    assert restored.lm_head.weight is restored.model.embed_tokens.weight
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, atol=0, rtol=0)

def test_tp_tied_parameter_identity_survives_storage_replacement():
    """Both aliases must follow future FSDP storage/parameter transitions."""
    model = Qwen3ForCausalLM(tiny_config())
    embedding = model.model.embed_tokens.weight
    model.lm_head.weight = torch.nn.Parameter(embedding.detach().clone())
    _replicate_tied_weights(model)
    assert model.lm_head.weight is embedding
    with torch.no_grad():
        model.model.embed_tokens.weight.add_(1)
    torch.testing.assert_close(model.lm_head.weight, embedding, atol=0, rtol=0)
    model.lm_head.weight = torch.nn.Parameter(torch.zeros(3, 24))
    with pytest.raises(ValueError, match="matching shapes"):
        _replicate_tied_weights(model)


def test_meta_materialization_preserves_tied_parameter_identity():
    """The meta build must retain tied aliases before checkpoint finalization."""
    with torch.device("meta"):
        model = Qwen3ForCausalLM(tiny_config())
    qwen3.adapt_materialization(model)
    assert model.lm_head.weight is model.model.embed_tokens.weight
    _move_model_to_device(model, True, torch.device("cpu"))
    assert not model.lm_head.weight.is_meta
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_runtime_adapter_does_not_patch_shared_code_or_other_models():
    """Qwen3 adapters must not mutate any shared function, class, or HF instance."""

    original = (SharedPlanner.plan, ParameterClassifier.classify,
                parameter_sharding._broadcast_tied_param, model_builder._move_model_to_device)
    model = replace_model(Qwen3ForCausalLM(tiny_config()))
    other = Qwen3ForCausalLM(tiny_config())
    native_names = list(other.named_parameters())
    before = ParameterClassifier().classify(model)
    ShardingPlanner(plan_overrides=get_parallel_overrides()).plan(
        model, FakeDeviceMesh((2,), ("tp",)), tp_size=2, sequence_parallel=False,
    )
    qwen3.adapt_materialization(model)
    assert ParameterClassifier().classify(model) == before
    assert list(other.named_parameters()) == native_names
    assert "to_empty" not in vars(other)
    assert original == (SharedPlanner.plan, ParameterClassifier.classify,
                        parameter_sharding._broadcast_tied_param, model_builder._move_model_to_device)


def test_scoped_fsdp_retie_precedes_shared_manager(monkeypatch):
    """Shared FSDP sees one tied parameter; its own implementation is reused."""
    model = Qwen3ForCausalLM(tiny_config())
    model.lm_head.weight = torch.nn.Parameter(model.model.embed_tokens.weight.detach().clone())
    source_info = object()
    received = []

    def parallelize(unused_manager, value, source_shard_info=None):
        del unused_manager
        assert value.lm_head.weight is value.model.embed_tokens.weight
        received.append(source_shard_info)
        return value

    monkeypatch.setattr(qwen3.FSDP2Manager, "parallelize", parallelize)
    manager = qwen3._Qwen3FSDP2Manager(FSDP2Config(), MeshContext())
    assert manager.parallelize(model, source_info) is model
    assert received == [source_info]


def test_scoped_runtime_rejects_other_model_families():
    """Explicit opt-in cannot accidentally adapt a non-Qwen3 model."""
    model = Qwen3ForCausalLM(tiny_config())
    model.config.model_type = "llama"
    with pytest.raises(ValueError, match="model_type='qwen3'"):
        qwen3.adapt_materialization(model)
    with pytest.raises(ValueError, match="model_type='qwen3'"):
        ShardingPlanner().plan(model, FakeDeviceMesh((2,), ("tp",)), tp_size=2)


def _assert_direct_bucket_copy(plan, states, destination_maps, expected_maps):
    """Copy every bounded bucket and compare all destination tensors exactly."""
    for (source_rank, tp_rank), buckets in plan.buckets.items():
        for bucket in buckets:
            assert bucket.total_bytes <= 128
            packed = pack_direct_bucket(states[source_rank], bucket, device=torch.device("cpu"))
            for entry in bucket.entries:
                value = packed.narrow(0, entry.buffer_offset, entry.num_bytes).view(torch.float32).view(entry.lengths)
                target_slice = tuple(slice(start, start + size)
                                     for start, size in zip(entry.destination_starts, entry.lengths))
                destination_maps[tp_rank][entry.target_name][target_slice] = value
    for actual, reference in zip(destination_maps, expected_maps):
        for name in reference:
            torch.testing.assert_close(actual[name], reference[name], atol=0, rtol=0)
