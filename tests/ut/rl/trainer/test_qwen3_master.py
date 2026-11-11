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
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch_npu
import yaml
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

import rl
from rl.workers.adapters import qwen3
from rl.workers.adapters.qwen3 import Qwen3ShardingPlanner as ShardingPlanner
from rl.workers.adapters.qwen3 import restore_tied_parameter as _replicate_tied_weights
from rl.workers.engine.model import ModelRegistration, resolve_vllm_model
from rl.weight_sync import packed_weight
from rl.weight_sync.model_adapter import ModelWeightAdapter
from rl.weight_sync.packed_weight import (
    build_packed_weight_buckets, materialize_packed_weight_bucket, unpack_packed_weights,
)

from hyper_parallel.distributed import ShardingPlanner as SharedPlanner
from hyper_parallel.distributed._builder import parameter_sharding
from hyper_parallel.distributed.mesh import DistributedSetup, MeshContext
from hyper_parallel.distributed.tensor_parallel.param_role import ParameterClassifier
from hyper_parallel.models._transformers import auto_model, model_builder
from hyper_parallel.models._transformers.checkpoint_loader import CheckpointManager
from hyper_parallel.models._transformers.model_builder import _move_model_to_device
from hyper_parallel.models.build_options import FSDP2Config
from hyper_parallel.models.replacement import apply_module_replacements, compile_module_replacements
from hyper_parallel.trainer.config import PlanOverride, normalize_distributed_setup_overrides
from hyper_parallel.trainer.config.resolver import resolve_component
from tests.ut.auto_models.distributed.conftest import FakeDeviceMesh



def declared_setup() -> DistributedSetup:
    """Exercise the shipped RL YAML through the actual HP normalization path."""
    recipe = Path(rl.__file__).resolve().parents[1] / "examples/gsm8k/configs/qwen3_4b_gsm8k_base.yaml"
    raw = yaml.safe_load(recipe.read_text())["train"]["plan_overrides"]
    rules = resolve_component(raw, annotation=list[PlanOverride], path="train.plan_overrides")
    setup = DistributedSetup()
    normalize_distributed_setup_overrides(setup, SimpleNamespace(plan_overrides=rules))
    return setup


def tiny_config() -> Qwen3Config:
    """Keep explicit head_dim different from hidden_size / query heads."""
    return Qwen3Config.from_dict({
        "vocab_size": 64, "hidden_size": 24, "intermediate_size": 32, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
        "max_position_embeddings": 64, "tie_word_embeddings": True, "use_cache": False,
        "architectures": ["Qwen3ForCausalLM"],
    })


@pytest.fixture(name="cpu_kernels")
def _cpu_kernels(monkeypatch):
    """Use independent CPU mathematical oracles for only the NPU primitives."""
    def rms(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> tuple[torch.Tensor]:
        """Compute a CPU RMS reference."""
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + epsilon)
        return (normalized.to(x.dtype) * weight,)

    def rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, **_kwargs: Any) -> torch.Tensor:
        """Compute the independent CPU rotary reference."""
        first, second = x.chunk(2, dim=-1)
        return x * cos + torch.cat((-second, first), dim=-1) * sin

    def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, atten_mask: Any,
                  sparse_mode: int, keep_prob: float, scale: float, **_kwargs: Any) -> tuple[torch.Tensor]:
        """Use CPU SDPA as the NPU primitive reference."""
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


def replace_model(model: torch.nn.Module) -> torch.nn.Module:
    """Apply the production declarations with the real replacement executor."""
    plan = compile_module_replacements(model, declared_setup().module_replacements)
    return apply_module_replacements(model, plan, weights_mapping=[])[0]


def test_shared_loader_preserves_checkpoint_logits_and_gradients(tmp_path, monkeypatch, cpu_kernels):
    """The public loader keeps real HF weights, padding semantics and gradients."""
    del cpu_kernels
    torch.manual_seed(42)
    reference = Qwen3ForCausalLM(tiny_config()).eval()
    reference.save_pretrained(tmp_path)
    monkeypatch.setattr(auto_model, "_current_device", lambda: torch.device("cpu"))
    setup = declared_setup()
    model = qwen3.from_pretrained(
        str(tmp_path), distributed_setup=setup, local_files_only=True,
        torch_dtype="float32", attn_implementation="sdpa", force_hf=True,
    ).eval()
    assert isinstance(model.model.layers[0].mlp, Qwen3MLP)
    assert setup.module_replacements
    assert setup.plan_overrides
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
    plan = ShardingPlanner(plan_overrides=declared_setup().plan_overrides).plan(
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


def weight_adapter(config: Qwen3Config) -> ModelWeightAdapter:
    """Build the same source adapter used by both publication strategies."""
    identity = ModelRegistration("qwen", "qwen3", "/unused", "/unused", "Qwen3ForCausalLM", "qwen3", "qwen3", True)
    adapter = ModelWeightAdapter(resolve_vllm_model(identity, "native"))
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
        packed = materialize_packed_weight_bucket(state, bucket, device=torch.device("cpu"))
        actual.update(unpack_packed_weights(packed, adapter.packed_metadata(bucket.worker_metadata())))
    expected.pop("lm_head.weight")
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)



def test_shared_model_hf_export_round_trip(tmp_path, monkeypatch, cpu_kernels):
    """A live single-rank build exports reversible HF weights with tied aliases."""
    del cpu_kernels
    reference = Qwen3ForCausalLM(tiny_config()).eval()
    source_dir = tmp_path / "source"
    reference.save_pretrained(source_dir)
    monkeypatch.setattr(auto_model, "_current_device", lambda: torch.device("cpu"))
    model = qwen3.from_pretrained(
        str(source_dir), distributed_setup=declared_setup(), torch_dtype="float32", local_files_only=True,
    )
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
    ShardingPlanner(plan_overrides=declared_setup().plan_overrides).plan(
        model, FakeDeviceMesh((2,), ("tp",)), tp_size=2, sequence_parallel=False,
    )
    qwen3.adapt_materialization(model)
    assert ParameterClassifier().classify(model) == before
    assert list(other.named_parameters()) == native_names
    assert "to_empty" not in vars(other)
    assert original == (SharedPlanner.plan, ParameterClassifier.classify,
                        parameter_sharding._broadcast_tied_param, model_builder._move_model_to_device)


@pytest.mark.parametrize("exclude_embedding", [False, True])
def test_scoped_fsdp_retie_precedes_shared_manager(monkeypatch, exclude_embedding):
    """Shared FSDP sees one tied parameter; its own implementation is reused."""
    model = Qwen3ForCausalLM(tiny_config())
    model.lm_head.weight = torch.nn.Parameter(model.model.embed_tokens.weight.detach().clone())
    source_info = object()
    received = []

    ignored = {model.model.embed_tokens.weight} if exclude_embedding else None

    def parallelize(unused_manager: Any, value: torch.nn.Module, source_shard_info: Any = None,
                    ignored_params: Any = None) -> torch.nn.Module:
        """Check tied identity at the shared manager boundary."""
        del unused_manager
        assert value.lm_head.weight is value.model.embed_tokens.weight
        received.append(source_shard_info)
        assert ignored_params is ignored
        return value

    monkeypatch.setattr(qwen3.FSDP2Manager, "parallelize", parallelize)
    manager = qwen3._Qwen3FSDP2Manager(FSDP2Config(), MeshContext())
    assert manager.parallelize(model, source_info, ignored_params=ignored) is model
    assert received == [source_info]


def test_scoped_runtime_rejects_other_model_families():
    """Explicit opt-in cannot accidentally adapt a non-Qwen3 model."""
    model = Qwen3ForCausalLM(tiny_config())
    model.config.model_type = "llama"
    with pytest.raises(ValueError, match="requires a supported Qwen3 family config"):
        qwen3.adapt_materialization(model)
    with pytest.raises(ValueError, match="requires a supported Qwen3 family config"):
        ShardingPlanner().plan(model, FakeDeviceMesh((2,), ("tp",)), tp_size=2)
