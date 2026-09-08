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
"""Shared MoE leaf, public TP/EP plan and Trainer combine contracts."""
# pylint: disable=protected-access,wrong-import-position

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import AutoModelForCausalLM, DeepseekV3Config, Qwen3MoeConfig

pytest.importorskip("vllm")
pytest.importorskip("vllm_ascend")

from rl.algorithm import build_algorithm
from rl.config import (
    _trainer_ep_size,
    _validate_model_implementation,
    _validate_moe_ep1_topology,
    build_model_registration,
    build_runtime_config,
    model_trust_remote_code,
    tokenizer_trust_remote_code,
    trainer_attention_implementation,
    validate_config,
)
from rl.roles.model import QWEN3_30B_A3B_CONFIG
from rl.roles.rollout import vllm_deepseek_v3, vllm_moe, vllm_qwen3, vllm_qwen3_moe  # noqa: E402
from rl.roles.rollout.vllm_deepseek_v3 import (  # noqa: E402
    _validate_public_mla_layout,
    _VLLMDeepseekV3RoutedExperts,
)
from rl.roles.rollout.vllm_moe import (  # noqa: E402
    HyperExpertPlacement,
    HyperLocalFusedExperts,
    local_routed_forward,
    qwen3_combine,
)
from rl.roles.rollout.vllm_moe_parallel import build_moe_tp_plan, pad_moe_tp_tokens

from examples.train_rl import load_config
from hyper_parallel import Replicate, Shard  # noqa: E402
from hyper_parallel.auto_models.components.distributed import ShardingPlanner, ep_compute
from hyper_parallel.auto_models.components.distributed.ep_utils import qwen3_moe_combine
from hyper_parallel.core.dtensor.layout import Layout  # noqa: E402
from hyper_parallel.platform import get_platform  # noqa: E402


def test_qwen3_families_share_one_paged_attention_adapter() -> None:
    """Dense and MoE adapters must not fork the paged-attention bridge."""
    assert vllm_qwen3.Qwen3PagedAttention is vllm_qwen3_moe.Qwen3PagedAttention


@pytest.mark.parametrize("num_experts", [64, 128])
@pytest.mark.parametrize("ep_size", [1, 2, 4])
def test_expert_placement_uses_public_layout(
    monkeypatch: pytest.MonkeyPatch, num_experts: int, ep_size: int,
) -> None:
    """Both families consume public EP placement on the runtime-owned group."""
    monkeypatch.setattr(get_platform(), "get_rank", lambda: 0)
    mesh = Layout((ep_size,), ("ep",), init_backend=False).mesh
    group = SimpleNamespace(world_size=ep_size, rank_in_group=0, device_group=object())
    monkeypatch.setattr(vllm_moe, "get_ep_group", lambda: group)
    monkeypatch.setattr(vllm_moe.DeviceMesh, "from_group", lambda *_args, **_kwargs: mesh)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_expert_parallel=True),
        model_config=SimpleNamespace(enforce_eager=True),
        device_config=SimpleNamespace(device=torch.device("cpu")),
    )
    for rank in range(ep_size):
        group.rank_in_group = rank
        placement = HyperExpertPlacement.from_config(config, num_experts)
        assert placement.mesh is mesh
        assert placement.local_count == num_experts // ep_size
        assert placement.global_start == rank * (num_experts // ep_size)

    # A changed public rule must not silently leave a hard-coded RL split active.
    monkeypatch.setattr(vllm_moe.TEMPLATES["moe_mlp"], "moe_expert_placement", Replicate())
    with pytest.raises(ValueError, match="public expert-axis"):
        HyperExpertPlacement.from_config(config, num_experts)


def test_qwen3_moe_small_config_is_not_a_supported_checkpoint() -> None:
    """Small component fixtures must not expand the supported model contract."""
    hf_config = SimpleNamespace(
        model_type="qwen3_moe",
        tie_word_embeddings=False,
        hidden_size=64,
        moe_intermediate_size=32,
        num_attention_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=2,
        num_key_value_heads=2,
        attention_dropout=0.0,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config, dtype=torch.bfloat16),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            enable_expert_parallel=False,
            enable_eplb=False,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        quant_config=None,
    )

    with pytest.raises(ValueError, match="official Qwen3-30B-A3B"):
        vllm_qwen3_moe._validate_adapter_config(vllm_config)


def _loader_only_experts(*, physical: bool = False) -> HyperLocalFusedExperts:
    """Build expert storage without constructing an NPU fused kernel."""
    experts = object.__new__(HyperLocalFusedExperts)
    torch.nn.Module.__init__(experts)
    experts.local_expert_count = 2
    experts.global_expert_start = 1
    experts._hyper_hidden_size = 4
    experts._hyper_intermediate_size = 3
    if physical:
        experts.w13_weight = torch.nn.Parameter(torch.zeros(2, 4, 6))
        experts.w2_weight = torch.nn.Parameter(torch.zeros(2, 3, 4))
    else:
        experts.w13_weight = torch.nn.Parameter(torch.zeros(2, 6, 4))
        experts.w2_weight = torch.nn.Parameter(torch.zeros(2, 4, 3))

    def process_weights_after_loading(layer: HyperLocalFusedExperts) -> None:
        """Model the unquantized Ascend post-load transpose on CPU."""
        layer.w13_weight = torch.nn.Parameter(
            layer.w13_weight.data.transpose(-2, -1).contiguous(),
            requires_grad=False,
        )
        layer.w2_weight = torch.nn.Parameter(
            layer.w2_weight.data.transpose(-2, -1).contiguous(),
            requires_grad=False,
        )

    experts.quant_method = SimpleNamespace(
        process_weights_after_loading=process_weights_after_loading
    )
    return experts


@pytest.mark.parametrize("physical", [False, True])
def test_packed_loader_targets_the_owned_expert_range(physical: bool) -> None:
    """Canonical packed tensors load into either vLLM physical layout."""
    experts = _loader_only_experts(physical=physical)
    gate_up = torch.arange(4 * 6 * 4, dtype=torch.float32).view(4, 6, 4)
    down = torch.arange(4 * 4 * 3, dtype=torch.float32).view(4, 4, 3)

    experts.load_gate_up(gate_up)
    experts.load_down(down)

    expected_gate_up = gate_up[1:3]
    expected_down = down[1:3]
    if physical:
        expected_gate_up = expected_gate_up.transpose(-2, -1)
        expected_down = expected_down.transpose(-2, -1)
    torch.testing.assert_close(experts.w13_weight, expected_gate_up)
    torch.testing.assert_close(experts.w2_weight, expected_down)


def test_per_expert_loader_builds_transformers_packed_storage() -> None:
    """Official per-expert keys share the same family-neutral leaf loader."""
    experts = _loader_only_experts()
    gate = torch.full((3, 4), 1.0)
    up = torch.full((3, 4), 2.0)
    down = torch.full((4, 3), 3.0)

    assert experts.load_expert_projection(gate, "gate_proj", 1) is True
    assert experts.load_expert_projection(up, "up_proj", 1) is True
    assert experts.load_expert_projection(down, "down_proj", 1) is True
    assert experts.load_expert_projection(gate, "gate_proj", 0) is False

    torch.testing.assert_close(experts.w13_weight[0, :3], gate)
    torch.testing.assert_close(experts.w13_weight[0, 3:], up)
    torch.testing.assert_close(experts.w2_weight[0], down)
    assert torch.count_nonzero(experts.w13_weight[1]) == 0


@pytest.mark.parametrize("checkpoint_format", ["packed", "per_expert"])
def test_qwen3_moe_loader_validates_the_post_replacement_schema(
    checkpoint_format: str,
) -> None:
    """Both official expert formats satisfy one canonical expected schema."""
    experts = _loader_only_experts()
    experts.global_expert_start = 0
    dense_weight = torch.nn.Parameter(torch.zeros(4))
    base = "model.layers.0.mlp.experts"
    model = SimpleNamespace(
        num_logical_experts=2,
        _expert_loaders={
            f"{base}.gate_up_proj": (experts, "gate_up"),
            f"{base}.down_proj": (experts, "down"),
        },
        _expected_weight_names={
            "model.norm.weight",
            f"{base}.gate_up_proj",
            f"{base}.down_proj",
        },
        named_parameters=lambda: (("model.norm.weight", dense_weight),),
    )
    dense = torch.arange(4, dtype=torch.float32)
    gate_up = torch.arange(48, dtype=torch.float32).view(2, 6, 4)
    down = torch.arange(24, dtype=torch.float32).view(2, 4, 3)
    if checkpoint_format == "packed":
        weights = [
            ("model.norm.weight", dense),
            (f"{base}.gate_up_proj", gate_up),
            (f"{base}.down_proj", down),
        ]
    else:
        weights = [("model.norm.weight", dense)]
        for expert_index in range(2):
            weights.extend(
                (
                    (
                        f"{base}.{expert_index}.gate_proj.weight",
                        gate_up[expert_index, :3],
                    ),
                    (
                        f"{base}.{expert_index}.up_proj.weight",
                        gate_up[expert_index, 3:],
                    ),
                    (
                        f"{base}.{expert_index}.down_proj.weight",
                        down[expert_index],
                    ),
                )
            )

    loaded = vllm_qwen3_moe.HyperQwen3MoeForCausalLM.load_weights(model, weights)

    assert loaded == model._expected_weight_names
    torch.testing.assert_close(dense_weight, dense)
    torch.testing.assert_close(experts.w13_weight, gate_up)
    torch.testing.assert_close(experts.w2_weight, down)


def test_sleep_wake_layout_restores_ascend_physical_storage() -> None:
    """Canonical wake storage is restored to grouped-matmul orientation."""
    experts = _loader_only_experts()
    canonical_w13 = torch.detach(experts.w13_weight).clone()
    canonical_w2 = torch.detach(experts.w2_weight).clone()

    experts.ensure_physical_weight_layout()

    assert tuple(experts.w13_weight.shape) == (2, 4, 6)
    assert tuple(experts.w2_weight.shape) == (2, 3, 4)
    torch.testing.assert_close(experts.w13_weight, canonical_w13.transpose(-2, -1))
    torch.testing.assert_close(experts.w2_weight, canonical_w2.transpose(-2, -1))


def test_ep1_topk_forward_does_not_call_all_to_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both HF-style and family-adapter routing stay communication-free."""
    experts = _loader_only_experts()
    experts.global_expert_start = 0

    def fake_assignments(
        _experts: HyperLocalFusedExperts,
        hidden_states: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Supply deterministic expert assignments for the routing test."""
        return hidden_states * (indices.reshape(-1, 1).to(hidden_states.dtype) + 1)

    def reject_collective(*_args: object, **_kwargs: object) -> None:
        """Reject communication from the local-only expert leaf."""
        raise AssertionError("EP1 must not call all-to-all")

    monkeypatch.setattr(HyperLocalFusedExperts, "_forward_assignments", fake_assignments)
    monkeypatch.setattr(torch.distributed, "all_to_all", reject_collective)
    monkeypatch.setattr(torch.distributed, "all_to_all_single", reject_collective)
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    indices = torch.tensor([[0, 1], [1, 0]])
    weights = torch.tensor([[0.25, 0.75], [0.5, 0.5]])

    qwen_output = experts(hidden, indices, weights)
    module = torch.nn.Module()
    module.experts = experts

    def router_fn(
        _module: torch.nn.Module,
        _hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the deterministic router fixture outputs."""
        return indices, weights

    deepseek_output = local_routed_forward(
        module,
        hidden.view(1, 2, 2),
        router_fn=router_fn,
    ).view(2, 2)

    expected = torch.stack((hidden[0] * 1.75, hidden[1] * 1.5))
    torch.testing.assert_close(qwen_output, expected)
    torch.testing.assert_close(deepseek_output, expected)


def test_deepseek_ep1_keeps_router_scale_bias_and_shared_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Moonlight family semantics remain outside the common expert leaf."""
    experts = _loader_only_experts()
    experts.global_expert_start = 0

    def fake_assignments(
        _experts: HyperLocalFusedExperts,
        hidden_states: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Supply deterministic expert assignments for the routing test."""
        return hidden_states * (indices.reshape(-1, 1).to(hidden_states.dtype) + 1)

    monkeypatch.setattr(HyperLocalFusedExperts, "_forward_assignments", fake_assignments)
    moe = object.__new__(_VLLMDeepseekV3RoutedExperts)
    torch.nn.Module.__init__(moe)
    moe.config = SimpleNamespace(
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        routed_scaling_factor=2.0,
    )
    moe.gate = torch.nn.Linear(2, 2, bias=False)
    moe.gate.weight.data.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    moe.gate.register_buffer("e_score_correction_bias", torch.tensor([-0.2, 0.3]))
    moe.experts = experts
    moe.shared_experts = torch.nn.Identity()
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 1.0]]])

    actual = moe(hidden)

    scores = (hidden.reshape(-1, 2) @ moe.gate.weight.T).sigmoid()
    selected = (scores + moe.gate.e_score_correction_bias).topk(2, dim=-1, sorted=False)[1]
    weights = scores.gather(1, selected)
    weights = weights / weights.sum(dim=-1, keepdim=True) * 2.0
    routed = []
    for token, token_indices, token_weights in zip(hidden.view(-1, 2), selected, weights):
        routed.append(
            sum(
                token * (int(expert) + 1) * weight
                for expert, weight in zip(token_indices, token_weights)
            )
        )
    expected = torch.stack(routed).view_as(hidden) + hidden
    torch.testing.assert_close(actual, expected)


def test_hyper_moe_tp_plan_preserves_public_dense_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only local expert storage and its token interface are adapted."""
    monkeypatch.setattr(get_platform(), "get_rank", lambda: 0)
    mesh = Layout((2, 2), ("dp", "tp"), init_backend=False).mesh
    config = Qwen3MoeConfig.from_dict({
        'hidden_size': 32,
        'intermediate_size': 64,
        'moe_intermediate_size': 16,
        'num_hidden_layers': 2,
        'num_attention_heads': 4,
        'num_key_value_heads': 2,
        'head_dim': 8,
        'num_experts': 8,
        'num_experts_per_tok': 2,
        'vocab_size': 64,
    })
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    reference = ShardingPlanner().plan(model, mesh, tp_size=2, ep_size=4, sequence_parallel=False)
    adapted = build_moe_tp_plan(model, mesh, tp_size=2, ep_size=4)
    assert adapted.mesh_dim_names == reference.mesh_dim_names == ("tp",)
    for name, expected in reference.modules.items():
        actual = adapted.modules[name]
        if not name.endswith(".mlp"):
            assert actual.params == expected.params
            assert actual.in_src == expected.in_src
            assert actual.in_dst == expected.in_dst
            assert actual.out_src == expected.out_src
            assert actual.out_dst == expected.out_dst
        else:
            assert actual.params == {
                key: axes for key, axes in expected.params.items() if not key.startswith("experts.")
            }
            assert next(iter(actual.in_src.values()))["tp"] == Replicate()
            assert next(iter(actual.in_dst.values()))["tp"] == Shard(1)
            assert next(iter(actual.out_src.values()))["tp"] == Shard(1)
            assert next(iter(actual.out_dst.values()))["tp"] == Replicate()
            assert actual.region_dispatch is False


@pytest.mark.parametrize("wrong_shape", [False, True])
def test_mla_storage_must_match_public_tp_slice(monkeypatch: pytest.MonkeyPatch, wrong_shape: bool) -> None:
    """Cache-aware leaf ownership does not exempt MLA from the public slicing contract."""
    monkeypatch.setattr(get_platform(), "get_rank", lambda: 0)
    mesh = Layout((2,), ("tp",), init_backend=False).mesh
    name = "model.layers.0.self_attn.q_proj.weight"
    parameters = {name: torch.empty(8 if wrong_shape else 4, 4)}
    model = SimpleNamespace(named_parameters=parameters.items)
    if wrong_shape:
        with pytest.raises(ValueError, match="violates public TP storage contract"):
            _validate_public_mla_layout(model, {name: (Shard(0),)}, {name: (8, 4)}, mesh)
    else:
        _validate_public_mla_layout(model, {name: (Shard(0),)}, {name: (8, 4)}, mesh)


def test_shared_experts_inherit_public_sp_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shared experts must communicate over the MoE's local token chunk, not mismatched replicas."""
    monkeypatch.setattr(get_platform(), "get_rank", lambda: 0)
    mesh = Layout((2, 2), ("dp", "tp"), init_backend=False).mesh
    config = DeepseekV3Config.from_dict({
        'hidden_size': 32,
        'intermediate_size': 64,
        'moe_intermediate_size': 16,
        'num_hidden_layers': 2,
        'num_attention_heads': 4,
        'num_key_value_heads': 4,
        'q_lora_rank': None,
        'kv_lora_rank': 8,
        'qk_nope_head_dim': 8,
        'qk_rope_head_dim': 4,
        'v_head_dim': 8,
        'n_routed_experts': 8,
        'n_shared_experts': 2,
        'n_group': 1,
        'topk_group': 1,
        'num_experts_per_tok': 2,
        'first_k_dense_replace': 1,
        'vocab_size': 64,
    })
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    expected = ShardingPlanner().plan(model, mesh, tp_size=2, ep_size=4, sequence_parallel=True)
    adapted = build_moe_tp_plan(model, mesh, tp_size=2, ep_size=4)
    name = "model.layers.1.mlp.shared_experts"
    for field in ("params", "in_src", "in_dst", "out_src", "out_dst"):
        assert getattr(adapted.modules[name], field) == getattr(expected.modules[name], field)
    assert next(iter(adapted.modules[name].in_src.values()))["tp"] == Shard(1)
    assert "model.layers.0.mlp" in adapted.modules


def test_moonlight_plugin_alias_keeps_canonical_planner_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the adapter up to allocation without bypassing public parameter coverage."""
    monkeypatch.setattr(get_platform(), "get_rank", lambda: 0)
    config = DeepseekV3Config.from_dict({
        'architectures': ["HyperDeepseekV3ForCausalLM"],
        'hidden_size': 32,
        'intermediate_size': 64,
        'moe_intermediate_size': 16,
        'num_hidden_layers': 2,
        'num_attention_heads': 4,
        'num_key_value_heads': 4,
        'q_lora_rank': None,
        'kv_lora_rank': 8,
        'qk_nope_head_dim': 8,
        'qk_rope_head_dim': 4,
        'v_head_dim': 8,
        'n_routed_experts': 8,
        'n_shared_experts': 2,
        'n_group': 1,
        'topk_group': 1,
        'num_experts_per_tok': 2,
        'first_k_dense_replace': 1,
        'vocab_size': 64,
    })
    runtime = SimpleNamespace(model_config=SimpleNamespace(hf_config=config),
                              parallel_config=SimpleNamespace(tensor_parallel_size=2),
                              device_config=SimpleNamespace(device="cpu"))
    monkeypatch.setattr(vllm_deepseek_v3, "_validate_adapter_config", lambda _: None)
    monkeypatch.setattr(vllm_deepseek_v3, "get_ep_group",
                        lambda: SimpleNamespace(world_size=4, ranks=[0, 1, 2, 3]))

    def stop_before_allocation(self: Any, **_kwargs: Any) -> Any:
        """Check canonical planning without allocating physical model weights."""
        assert self.config.architectures == ["DeepseekV3ForCausalLM"]
        raise RuntimeError("canonical plan passed; stop before physical allocation")

    monkeypatch.setattr(vllm_deepseek_v3.HyperDeepseekV3ForCausalLM, "to_empty", stop_before_allocation)
    with pytest.raises(RuntimeError, match="canonical plan passed"):
        vllm_deepseek_v3.HyperDeepseekV3ForCausalLM(vllm_config=runtime)
    assert config.architectures == ["HyperDeepseekV3ForCausalLM"]


@pytest.mark.parametrize("tokens", [0, 1, 2, 3, 7])
def test_moe_padding_preserves_real_tokens_and_gradients(tokens: int) -> None:
    """Padding stays inside a token-independent region and never changes its output extent."""
    seen = []

    class Region(torch.nn.Module):
        def forward(self, hidden_states: Any) -> Any:
            """Evaluate the token-local region and preserve real token outputs."""
            seen.append(hidden_states.shape[1])
            return hidden_states * 3

    module = Region()
    pad_moe_tp_tokens(module, 2)
    values = torch.randn(1, tokens, 4, requires_grad=True)
    output = module(values)
    torch.testing.assert_close(output, values * 3, rtol=0, atol=0)
    assert seen[0] > 0 and seen[0] % 2 == 0
    output.sum().backward()
    torch.testing.assert_close(values.grad, torch.full_like(values, 3), rtol=0, atol=0)


@pytest.mark.parametrize("hf_combine", [False, True])
def test_public_qwen_factory_preserves_default_and_shares_combiner(hf_combine: bool) -> None:
    """Existing consumers keep the old call; RL opts into the same public function."""
    module = SimpleNamespace(gate=object(), experts=object())
    mesh = MagicMock()
    mesh.__getitem__.return_value.size.return_value = 4
    hidden = torch.randn(1, 2, 4)
    with patch.object(ep_compute, "bind_local_expert_forward") as bind, patch.object(
        ep_compute, "ep_routed_forward", return_value=hidden,
    ) as routed:
        factory = ep_compute.qwen3moe_ep_compute_fn(
            module=module, mesh=None, tp_mesh=None, cp_mesh=None, ep_mesh=mesh,
            **({"hf_combine": True} if hf_combine else {}),
        )
        assert factory(module, hidden) is hidden
        bind.assert_called_once_with(module, 4)
        if hf_combine:
            assert routed.call_args.kwargs["combine_fn"] is qwen3_moe_combine
        else:
            assert "combine_fn" not in routed.call_args.kwargs
    assert qwen3_combine is qwen3_moe_combine


@pytest.mark.parametrize("ep,family,dp,tp", [
    (True, "qwen3_moe", 2, 2), (2, "qwen3_moe", 2, 2),
    (4, "qwen3", 2, 2), (4, "unknown", 2, 2), (4, "qwen3_moe", 1, 2),
])
def test_trainer_ep_rejects_unimplemented_topologies(ep: int, family: str, dp: int, tp: int) -> None:
    """Do not silently reinterpret EP or claim unimplemented model combinations."""
    with pytest.raises(ValueError):
        _trainer_ep_size({"ep": ep, "dp_shard": dp, "tp": tp}, family)


@pytest.mark.parametrize("implementation", ["native", "hyper"])
@pytest.mark.parametrize("trainer_tp,trainer_ep", [(1, 1), (1, 4), (2, 4)])
def test_qwen_moe_tp2_ep4_full_configuration(
    tmp_path: Path, implementation: str, trainer_tp: int, trainer_ep: int,
) -> None:
    """The real launcher must pass every configuration guard, not only the MoE-specific helper."""
    model_config = dict(QWEN3_30B_A3B_CONFIG)
    model_config.update(model_type="qwen3_moe", architectures=["Qwen3MoeForCausalLM"], tie_word_embeddings=False)
    (tmp_path / "config.json").write_text(json.dumps(model_config), encoding="utf-8")
    data_path = tmp_path / "data.parquet"
    data_path.touch()
    recipe = Path(__file__).parents[1] / "examples/configs/moonlight_16b_a3b_gsm8k_native_vllm.yaml"
    config = load_config(str(recipe), [
        f"--model.weights_path={tmp_path}", f"--model.tokenizer_path={tmp_path}",
        f"--model.registry_name=qwen_moe_tp2_{implementation}", "--model.name=qwen3_moe",
        "--model.attention_implementation=null", f"--rollout.vllm.model_implementation={implementation}",
        "--rollout.vllm.tensor_parallel_size=2", "--rollout.vllm.data_parallel_size=2",
        "--rollout.vllm.enable_expert_parallel=true",
        f"--train.accelerator.tp={trainer_tp}", f"--train.accelerator.ep={trainer_ep}",
        f"--train.accelerator.dp_shard={4 // trainer_tp}",
        f"--data.train_path={data_path}", "--evaluation.enabled=false",
    ])
    validate_config(config, build_algorithm(config["algorithm"]))
    runtime = build_runtime_config(config)
    assert runtime.accelerator.ep_size == trainer_ep
    assert runtime.accelerator.tp_size == trainer_tp
    assert len(runtime.plan_overrides) == int(trainer_ep > 1)
    if trainer_ep > 1:
        assert runtime.plan_overrides[0].local_compute_fn.hf_combine is True


@pytest.mark.parametrize("implementation", ["native", "hyper"])
@pytest.mark.parametrize("trainer_tp", [1, 2])
def test_moonlight_trainer_ep_and_rollout_tp_configuration(
    tmp_path: Path, implementation: str, trainer_tp: int,
) -> None:
    """Inject EP only into the MoE layer range from the fixed HF definition."""
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "deepseek_v3", "architectures": ["DeepseekV3ForCausalLM"],
        "q_lora_rank": None, "first_k_dense_replace": 1, "num_hidden_layers": 3,
    }), encoding="utf-8")
    data_path = tmp_path / "data.parquet"
    data_path.touch()
    recipe = Path(__file__).parents[1] / "examples/configs/moonlight_16b_a3b_gsm8k_native_vllm.yaml"
    config = load_config(str(recipe), [
        f"--model.weights_path={tmp_path}", f"--model.tokenizer_path={tmp_path}",
        f"--rollout.vllm.model_implementation={implementation}",
        "--rollout.vllm.tensor_parallel_size=2", "--rollout.vllm.data_parallel_size=2",
        "--rollout.vllm.enable_expert_parallel=true", "--train.accelerator.ep=4",
        f"--train.accelerator.tp={trainer_tp}", f"--train.accelerator.dp_shard={4 // trainer_tp}",
        f"--data.train_path={data_path}", "--evaluation.enabled=false",
    ])
    validate_config(config, build_algorithm(config["algorithm"]))
    runtime = build_runtime_config(config)
    assert runtime.accelerator.ep_size == 4
    assert [entry.match for entry in runtime.plan_overrides] == ["model.layers.1.mlp", "model.layers.2.mlp"]
    assert all(entry.local_compute_fn._target_path.endswith("deepseekv3_ep_compute_fn")
               for entry in runtime.plan_overrides)


def test_moonlight_base_config_exposes_explicit_ep_override() -> None:
    """The real-model EP launcher must pass strict CLI parsing before torchrun."""
    config_path = Path(__file__).parents[1] / "examples/configs/moonlight_16b_a3b_gsm8k_native_vllm.yaml"
    config = load_config(str(config_path), ["--rollout.vllm.enable_expert_parallel=true"])
    assert config["rollout"]["vllm"]["enable_expert_parallel"] is True


def test_moonlight_registration_uses_deepseek_v3_builtin_model(tmp_path: Path) -> None:
    """Moonlight resolves from checkpoint identity without remote model code."""
    hf_config = {
        "architectures": ["DeepseekV3ForCausalLM"],
        "model_type": "deepseek_v3",
        "q_lora_rank": None,
        "tie_word_embeddings": False,
    }
    (tmp_path / "config.json").write_text(json.dumps(hf_config), encoding="utf-8")
    model = {
        "registry_name": "moonlight_16b_a3b_instruct",
        "name": "deepseek_v3",
        "weights_path": str(tmp_path),
        "tokenizer_path": str(tmp_path),
        "trust_remote_code": False,
        "tokenizer_trust_remote_code": True,
        "attention_implementation": "transformers_builtin",
    }

    registration = build_model_registration({"model": model})

    assert registration.family == "deepseek_v3"
    assert registration.hf_architecture == "DeepseekV3ForCausalLM"
    assert registration.model_type == "deepseek_v3"
    assert registration.q_lora_rank is None
    assert registration.tie_word_embeddings is False
    assert model_trust_remote_code(model) is False
    assert tokenizer_trust_remote_code(model) is True
    assert trainer_attention_implementation(model) == "sdpa"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"trust_remote_code": True}, "requires model.trust_remote_code=false"),
        (
            {"attention_implementation": "hyper_mla"},
            "Unsupported model.attention_implementation",
        ),
        (
            {"attention_implementation": None},
            "requires model.attention_implementation='transformers_builtin'",
        ),
    ],
)
def test_moonlight_registration_rejects_incompatible_trainer_selection(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    """DeepSeek-V3 cannot silently select remote code or an incompatible MLA leaf."""
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV3ForCausalLM"],
                "model_type": "deepseek_v3",
                "q_lora_rank": None,
            }
        ),
        encoding="utf-8",
    )
    model = {
        "registry_name": "moonlight",
        "name": "deepseek_v3",
        "weights_path": str(tmp_path),
        "tokenizer_path": str(tmp_path),
        "trust_remote_code": False,
        "attention_implementation": "transformers_builtin",
    }
    model.update(overrides)

    with pytest.raises(ValueError, match=message):
        build_model_registration({"model": model})


def test_qwen3_moe_registration_uses_official_checkpoint_identity(tmp_path: Path) -> None:
    """Qwen3-30B-A3B is selected by config identity rather than directory name."""
    config = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "tie_word_embeddings": False,
        "hidden_size": 2048,
        "moe_intermediate_size": 768,
        "num_attention_heads": 32,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 48,
        "num_key_value_heads": 4,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    registration = build_model_registration(
        {
            "model": {
                "registry_name": "arbitrary-local-name",
                "name": "qwen3_moe",
                "weights_path": str(tmp_path),
                "tokenizer_path": str(tmp_path),
            }
        }
    )

    assert registration.family == "qwen3_moe"
    assert registration.hf_architecture == "Qwen3MoeForCausalLM"
    assert _validate_model_implementation(
        {"model_implementation": "hyper"}, registration
    ).architecture == "HyperQwen3MoeForCausalLM"


def test_qwen3_moe_registration_rejects_non_official_shape(tmp_path: Path) -> None:
    """The initial adapter cannot silently accept an unverified MoE variant."""
    config = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "tie_word_embeddings": False,
        "hidden_size": 1024,
        "moe_intermediate_size": 768,
        "num_attention_heads": 32,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 48,
        "num_key_value_heads": 4,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="official Qwen3-30B-A3B"):
        build_model_registration(
            {
                "model": {
                    "registry_name": "qwen3-moe",
                    "name": "qwen3_moe",
                    "weights_path": str(tmp_path),
                    "tokenizer_path": str(tmp_path),
                }
            }
        )


def test_qwen3_moe_rejects_unsupported_checkpoint_shape(tmp_path: Path) -> None:
    """Component test dimensions must not relax the production model contract."""
    config = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "tie_word_embeddings": False,
        "hidden_size": 64,
        "moe_intermediate_size": 32,
        "num_attention_heads": 4,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    model_config = {
        "model": {
            "registry_name": "qwen3-moe-p8-acceptance",
            "name": "qwen3_moe",
            "weights_path": str(tmp_path),
            "tokenizer_path": str(tmp_path),
        }
    }

    with pytest.raises(ValueError, match="official Qwen3-30B-A3B"):
        build_model_registration(model_config)


@pytest.mark.parametrize(
    ("vllm_overrides", "trainer_tp", "message"),
    [
        ({"tensor_parallel_size": 2}, 1, "requires rollout TP1"),
        ({"enable_expert_parallel": True, "deployment": "disjoint"}, 1, "requires colocated"),
        ({"enable_expert_parallel": True, "enforce_eager": False}, 1, "requires enforce_eager"),
        ({"enable_expert_parallel": True, "data_parallel_size": 3}, 1, "supports EP1/EP2/EP4"),
        ({"enable_eplb": True}, 1, "does not support EPLB"),
        ({}, 2, "requires Trainer TP1"),
    ],
)
def test_hyper_qwen3_moe_ep1_rejects_unimplemented_parallelism(
    vllm_overrides: dict[str, object],
    trainer_tp: int,
    message: str,
) -> None:
    """Unsupported TP, EP topology, graph capture, and EPLB fail before launch."""
    vllm = {"tensor_parallel_size": 1, **vllm_overrides}
    rollout_model = SimpleNamespace(family="qwen3_moe", is_hyper=True)
    accelerator = {
        "dp_replicate": 1,
        "dp_shard": 2,
        "tp": trainer_tp,
        "cp": 1,
        "pp": 1,
    }

    with pytest.raises(ValueError, match=message):
        _validate_moe_ep1_topology(vllm, rollout_model, accelerator)
