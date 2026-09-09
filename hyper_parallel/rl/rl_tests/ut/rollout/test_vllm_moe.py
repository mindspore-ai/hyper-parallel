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
"""CPU unit tests for shared Qwen3-MoE and DeepSeek-v3 rollout contracts."""
# pylint: disable=forbidden-backend-import,missing-public-docstring,protected-access
# pylint: disable=non-parent-init-called,wrong-import-order

import math
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen3MoeConfig
from vllm.config.compilation import CompilationMode

from hyper_parallel import Replicate, Shard
from hyper_parallel.auto_models.components.distributed import ShardingPlanner, ep_compute
from hyper_parallel.auto_models.components.distributed.ep_utils import qwen3_moe_combine
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.platform import get_platform
from rl.roles.model import QWEN3_30B_A3B_CONFIG
from rl.roles.rollout import (
    vllm_deepseek_v3,
    vllm_moe,
    vllm_qwen3,
    vllm_qwen3_moe,
)
from rl.roles.rollout.vllm_deepseek_v3 import (
    _VLLMDeepseekV3MLA,
    _VLLMDeepseekV3RoutedExperts,
    _validate_public_mla_layout,
    HyperDeepseekV3ForCausalLM,
)
from rl.roles.rollout.vllm_moe import (
    HyperExpertPlacement,
    HyperLocalFusedExperts,
    local_routed_forward,
    qwen3_combine,
)
from rl.roles.rollout.vllm_moe_parallel import build_moe_tp_plan, pad_moe_tp_tokens


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


def test_moe_adapter_configs_accept_supported_four_card_contracts() -> None:
    """Qwen3-MoE and Moonlight accept their supported DP2/TP2/EP4 settings."""
    assert vllm_qwen3.Qwen3PagedAttention is vllm_qwen3_moe.Qwen3PagedAttention
    common_parallel = SimpleNamespace(
        tensor_parallel_size=2,
        data_parallel_size=2,
        enable_expert_parallel=True,
        enable_eplb=False,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
    )
    qwen_fields = dict(QWEN3_30B_A3B_CONFIG)
    qwen_fields.update(
        model_type="qwen3_moe",
        tie_word_embeddings=False,
        attention_dropout=0.0,
    )
    qwen_runtime = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(**qwen_fields),
            dtype=torch.bfloat16,
        ),
        parallel_config=common_parallel,
        quant_config=None,
    )
    deepseek_runtime = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="deepseek_v3",
                q_lora_rank=None,
                attention_dropout=0.0,
            ),
            dtype=torch.bfloat16,
            use_mla=True,
        ),
        parallel_config=common_parallel,
        quant_config=None,
    )

    vllm_qwen3_moe._validate_adapter_config(qwen_runtime)
    vllm_deepseek_v3._validate_adapter_config(deepseek_runtime)


def test_qwen3_moe_constructor_builds_shared_runtime_leaves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Qwen3-MoE constructor replaces one sparse layer without duplicating experts."""

    class Rotary(torch.nn.Module):
        def __init__(self, _config: Any) -> None:
            super().__init__()
            self.register_buffer("inv_freq", torch.ones(2))
            self.register_buffer("original_inv_freq", torch.ones(2))

    class Attention(torch.nn.Module):
        pass

    class Layer(torch.nn.Module):
        def __init__(self, sparse_moe: torch.nn.Module) -> None:
            super().__init__()
            self.self_attn = Attention()
            self.mlp = sparse_moe

    class Backbone(torch.nn.Module):
        def __init__(self, config: Any, sparse_moe: torch.nn.Module) -> None:
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(8, 4)
            self.layers = torch.nn.ModuleList([Layer(sparse_moe)])
            self.rotary_emb = Rotary(config)
            self.norm = torch.nn.Identity()

    class PagedAttention(torch.nn.Module):
        def __init__(self, _attention: Any, **kwargs: Any) -> None:
            super().__init__()
            self.family = kwargs["family"]

    class Experts(torch.nn.Module):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__()
            self.local_expert_count = kwargs["local_expert_count"]
            self.global_expert_start = kwargs["global_expert_start"]
            self.w13_weight = torch.nn.Parameter(torch.zeros(2, 6, 4))
            self.w2_weight = torch.nn.Parameter(torch.zeros(2, 4, 3))

        def load_gate_up(self, _weight: torch.Tensor) -> None:
            return None

        def load_down(self, _weight: torch.Tensor) -> None:
            return None

    sparse_moe = object.__new__(vllm_qwen3_moe.Qwen3MoeSparseMoeBlock)
    torch.nn.Module.__init__(sparse_moe)
    sparse_moe.experts = torch.nn.Identity()
    fields = dict(QWEN3_30B_A3B_CONFIG)
    fields.update(
        model_type="qwen3_moe",
        tie_word_embeddings=False,
        attention_dropout=0.0,
    )
    config = SimpleNamespace(**fields)

    def base_init(model: torch.nn.Module, used_config: Any) -> None:
        torch.nn.Module.__init__(model)
        model.config = used_config
        model.model = Backbone(used_config, sparse_moe)
        model.lm_head = torch.nn.Linear(4, 8, bias=False)

    monkeypatch.setattr(vllm_qwen3_moe.Qwen3MoeForCausalLM, "__init__", base_init)
    monkeypatch.setattr(vllm_qwen3_moe, "Qwen3PagedAttention", PagedAttention)
    monkeypatch.setattr(vllm_qwen3_moe, "HyperLocalFusedExperts", Experts)
    monkeypatch.setattr(
        vllm_qwen3_moe.HyperExpertPlacement,
        "from_config",
        lambda _runtime, _count: HyperExpertPlacement(2, 0, None),
    )
    runtime = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config, dtype=torch.bfloat16),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            data_parallel_size=1,
            enable_expert_parallel=False,
            enable_eplb=False,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        device_config=SimpleNamespace(device=torch.device("cpu")),
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
        quant_config=None,
    )

    model = vllm_qwen3_moe.HyperQwen3MoeForCausalLM(vllm_config=runtime)

    assert model.num_moe_layers == 1
    assert model.num_logical_experts == 128
    assert model.num_local_physical_experts == 2
    assert isinstance(model.model.layers[0].self_attn, PagedAttention)
    assert isinstance(model.model.layers[0].mlp.experts, Experts)
    assert set(model._expert_loaders) == {
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
    }


def test_deepseek_constructor_builds_mla_and_local_expert_leaves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The DeepSeek constructor retains its outer model and replaces only MLA/experts."""

    class Gate(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2, 4))
            self.register_buffer("e_score_correction_bias", torch.zeros(2))

    class SparseMoe(vllm_deepseek_v3.DeepseekV3MoE):
        def __init__(self, config: Any) -> None:
            torch.nn.Module.__init__(self)
            self.config = config
            self.gate = Gate()
            self.shared_experts = torch.nn.Identity()
            self.experts = torch.nn.Identity()

    class Layer(torch.nn.Module):
        def __init__(self, config: Any) -> None:
            super().__init__()
            self.self_attn = torch.nn.Identity()
            self.mlp = SparseMoe(config)

    class Backbone(torch.nn.Module):
        def __init__(self, config: Any) -> None:
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(8, 4)
            self.layers = torch.nn.ModuleList([Layer(config)])
            self.norm = torch.nn.Identity()

    class MLALeaf(torch.nn.Module):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__()
            self.prefix = kwargs["prefix"]

    class RoutedLeaf(torch.nn.Module):
        def __init__(self, moe: Any, **_kwargs: Any) -> None:
            super().__init__()
            self.gate = moe.gate
            self.shared_experts = moe.shared_experts
            self.experts = torch.nn.Identity()

    config = SimpleNamespace(
        architectures=["HyperDeepseekV3ForCausalLM"],
        model_type="deepseek_v3",
        q_lora_rank=None,
        attention_dropout=0.0,
        n_routed_experts=2,
        hidden_size=4,
        moe_intermediate_size=3,
    )

    def base_init(model: torch.nn.Module, used_config: Any) -> None:
        torch.nn.Module.__init__(model)
        model.config = used_config
        model.model = Backbone(used_config)
        model.lm_head = torch.nn.Linear(4, 8, bias=False)

    monkeypatch.setattr(vllm_deepseek_v3.DeepseekV3ForCausalLM, "__init__", base_init)
    monkeypatch.setattr(vllm_deepseek_v3, "_VLLMDeepseekV3MLA", MLALeaf)
    monkeypatch.setattr(vllm_deepseek_v3, "_VLLMDeepseekV3RoutedExperts", RoutedLeaf)
    runtime = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=config,
            dtype=torch.bfloat16,
            use_mla=True,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            data_parallel_size=1,
            enable_expert_parallel=False,
            enable_eplb=False,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        device_config=SimpleNamespace(device=torch.device("cpu")),
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
        quant_config=None,
    )

    model = HyperDeepseekV3ForCausalLM(vllm_config=runtime)

    assert model.config.architectures == ["DeepseekV3ForCausalLM"]
    assert isinstance(model.model.layers[0].self_attn, MLALeaf)
    assert isinstance(model.model.layers[0].mlp, RoutedLeaf)
    assert set(model._expert_loaders) == {"model.layers.0.mlp.experts"}


@pytest.mark.parametrize("ep_size", [1, 2, 4])
def test_expert_placement_uses_public_layout(
    monkeypatch: pytest.MonkeyPatch,
    ep_size: int,
) -> None:
    """Static EP ownership follows the public expert-axis layout."""
    monkeypatch.setattr(get_platform(), "get_rank", lambda: 0)
    mesh = Layout((ep_size,), ("ep",), init_backend=False).mesh
    group = SimpleNamespace(
        world_size=ep_size,
        rank_in_group=0,
        device_group=object(),
    )
    monkeypatch.setattr(vllm_moe, "get_ep_group", lambda: group)
    monkeypatch.setattr(vllm_moe.DeviceMesh, "from_group", lambda *_args, **_kwargs: mesh)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_expert_parallel=True),
        model_config=SimpleNamespace(enforce_eager=True),
        device_config=SimpleNamespace(device=torch.device("cpu")),
    )

    placements = []
    for rank in range(ep_size):
        group.rank_in_group = rank
        placements.append(HyperExpertPlacement.from_config(config, 8))

    assert [placement.local_count for placement in placements] == [8 // ep_size] * ep_size
    assert [placement.global_start for placement in placements] == [
        rank * (8 // ep_size) for rank in range(ep_size)
    ]
    assert all(placement.mesh is mesh for placement in placements)


@pytest.mark.parametrize("physical", [False, True])
def test_packed_loader_targets_owned_expert_range(physical: bool) -> None:
    """Canonical packed tensors load into canonical or physical vLLM storage."""
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


@pytest.mark.parametrize("checkpoint_format", ["packed", "per_expert"])
@pytest.mark.parametrize("family", ["qwen3_moe", "deepseek_v3"])
def test_moe_loaders_accept_supported_checkpoint_formats(
    checkpoint_format: str,
    family: str,
) -> None:
    """Packed and per-expert inputs produce one canonical loaded-name set."""
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
                    (f"{base}.{expert_index}.gate_proj.weight", gate_up[expert_index, :3]),
                    (f"{base}.{expert_index}.up_proj.weight", gate_up[expert_index, 3:]),
                    (f"{base}.{expert_index}.down_proj.weight", down[expert_index]),
                )
            )

    loader = vllm_qwen3_moe.HyperQwen3MoeForCausalLM.load_weights
    if family == "deepseek_v3":
        model.config = SimpleNamespace(n_routed_experts=2, tie_word_embeddings=False)
        model._expert_loaders = {base: experts}
        model._tp_mesh = None
        model._tp_placements = {}
        model.named_buffers = lambda: ()
        loader = HyperDeepseekV3ForCausalLM.load_weights

    loaded = loader(model, weights)

    assert loaded == model._expected_weight_names
    torch.testing.assert_close(dense_weight, dense)
    torch.testing.assert_close(experts.w13_weight, gate_up)
    torch.testing.assert_close(experts.w2_weight, down)


def test_ep1_routing_preserves_qwen_and_deepseek_family_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local experts serve Qwen Top-K and DeepSeek shared-expert composition."""
    experts = _loader_only_experts()
    experts.global_expert_start = 0

    def fake_assignments(
        _experts: HyperLocalFusedExperts,
        hidden_states: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        return hidden_states * (indices.reshape(-1, 1).to(hidden_states.dtype) + 1)

    monkeypatch.setattr(HyperLocalFusedExperts, "_forward_assignments", fake_assignments)
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    indices = torch.tensor([[0, 1], [1, 0]])
    weights = torch.tensor([[0.25, 0.75], [0.5, 0.5]])
    module = torch.nn.Module()
    module.experts = experts
    qwen_output = local_routed_forward(
        module,
        hidden,
        router_fn=lambda _module, _hidden: (indices, weights),
    )

    deepseek = object.__new__(_VLLMDeepseekV3RoutedExperts)
    torch.nn.Module.__init__(deepseek)
    deepseek.config = SimpleNamespace(
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        routed_scaling_factor=2.0,
    )
    deepseek.gate = torch.nn.Linear(2, 2, bias=False)
    deepseek.gate.weight.data.copy_(torch.eye(2))
    deepseek.gate.register_buffer("e_score_correction_bias", torch.tensor([-0.2, 0.3]))
    deepseek.experts = experts
    deepseek.shared_experts = torch.nn.Identity()
    deepseek_output = deepseek(hidden)

    expected_qwen = torch.stack((hidden[0, 0] * 1.75, hidden[0, 1] * 1.5)).unsqueeze(0)
    torch.testing.assert_close(qwen_output, expected_qwen)
    # Router correction bias selects experts; uncorrected sigmoid scores weight them.
    expected_rows = []
    for row in hidden[0]:
        score0 = 1 / (1 + math.exp(-float(row[0])))
        score1 = 1 / (1 + math.exp(-float(row[1])))
        factor = 1 + 2 * (score0 + 2 * score1) / (score0 + score1)
        expected_rows.append(row * factor)
    torch.testing.assert_close(deepseek_output, torch.stack(expected_rows).unsqueeze(0))
    # With Top-1, the bias deliberately reverses the raw-score choice.
    deepseek.config.num_experts_per_tok = 1
    deepseek.gate.e_score_correction_bias.copy_(torch.tensor([2.0, -2.0]))
    torch.testing.assert_close(deepseek(hidden), hidden * 3)


def test_local_fused_experts_restore_layout_and_execute_assignments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The local expert leaf restores physical storage before direct and Top-K execution."""
    experts = _loader_only_experts()
    experts._local_comm_method = object()
    experts.ensure_moe_quant_config_init = MagicMock()
    context = SimpleNamespace(
        moe_comm_method="outer-method",
        moe_comm_type="outer-type",
        flash_comm_v1_enabled=True,
        in_profile_run=True,
    )
    monkeypatch.setattr(vllm_moe, "_EXTRA_CTX", context)
    original_context = (
        context.moe_comm_method,
        context.moe_comm_type,
        context.flash_comm_v1_enabled,
        context.in_profile_run,
    )

    def forward_impl(
        _experts: HyperLocalFusedExperts,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        return hidden_states + topk_weights

    monkeypatch.setattr(vllm_moe.AscendFusedMoE, "forward_impl", forward_impl)
    experts.ensure_physical_weight_layout()
    hidden = torch.arange(8, dtype=torch.float32).view(2, 4)

    direct = experts(hidden, torch.tensor([0, 1]))
    topk = experts(
        hidden,
        torch.tensor([[0, 1], [1, 0]]),
        torch.tensor([[0.25, 0.75], [0.5, 0.5]]),
    )

    assert tuple(experts.w13_weight.shape) == (2, 4, 6)
    assert tuple(experts.w2_weight.shape) == (2, 3, 4)
    torch.testing.assert_close(direct, hidden + 1)
    torch.testing.assert_close(topk, hidden + 1)
    assert experts.ensure_moe_quant_config_init.call_count == 2
    assert (
        context.moe_comm_method,
        context.moe_comm_type,
        context.flash_comm_v1_enabled,
        context.in_profile_run,
    ) == original_context


def test_routed_forward_preserves_shape_for_local_family_combine() -> None:
    """The shared local routing path preserves packed token order and shape."""
    hidden = torch.arange(8, dtype=torch.float32).view(1, 2, 4)
    indices = torch.tensor([[0, 1], [1, 0]])
    weights = torch.tensor([[0.25, 0.75], [0.5, 0.5]])
    experts = MagicMock(return_value=hidden.reshape(-1, 4) + 1)
    module = SimpleNamespace(hyper_ep_mesh=None, experts=experts)

    output = vllm_moe.routed_forward(
        module,
        hidden,
        router_fn=lambda _module, _hidden: (indices, weights),
        combine_fn=lambda *_args: None,
    )

    torch.testing.assert_close(output, hidden + 1)
    assert experts.call_count == 1
    flattened, routed_indices, routed_weights = experts.call_args.args
    torch.testing.assert_close(flattened, hidden.reshape(-1, 4))
    torch.testing.assert_close(routed_indices, indices)
    torch.testing.assert_close(routed_weights, weights)


def test_moe_model_forwards_preserve_packed_token_shape() -> None:
    """Qwen3-MoE and DeepSeek outer models preserve packed token order and shape."""

    class Layer(torch.nn.Module):
        def forward(self, hidden_states: torch.Tensor, **_kwargs: Any) -> torch.Tensor:
            return hidden_states + 1

    class QwenBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(8, 4)
            self.layers = torch.nn.ModuleList([Layer()])
            self.norm = torch.nn.Identity()

        @staticmethod
        def rotary_emb(
            hidden_states: torch.Tensor,
            _positions: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.ones_like(hidden_states), torch.zeros_like(hidden_states)

    qwen = object.__new__(vllm_qwen3_moe.HyperQwen3MoeForCausalLM)
    torch.nn.Module.__init__(qwen)
    qwen.model = QwenBackbone()
    qwen.lm_head = torch.nn.Linear(4, 8, bias=False)
    values = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    qwen_output = qwen.forward(
        None,
        torch.tensor([0, 1]),
        inputs_embeds=values,
    )
    qwen_logits = qwen.compute_logits(qwen_output)

    deepseek = object.__new__(HyperDeepseekV3ForCausalLM)
    torch.nn.Module.__init__(deepseek)
    deepseek.model = QwenBackbone()
    deepseek.lm_head = torch.nn.Linear(4, 8, bias=False)
    deepseek_output = deepseek.forward(
        None,
        torch.tensor([0, 1]),
        inputs_embeds=values,
    )

    torch.testing.assert_close(qwen_output, values + 1)
    torch.testing.assert_close(deepseek_output, values + 1)
    assert qwen_logits.shape == (2, 8)
    assert qwen.get_input_embeddings() is qwen.model.embed_tokens
    assert deepseek.get_input_embeddings() is deepseek.model.embed_tokens


def test_deepseek_mla_and_checkpoint_loading_keep_runtime_owned_leaves() -> None:
    """MLA consumes packed positions and dense checkpoint tensors use the shared loader."""
    leaf = object.__new__(_VLLMDeepseekV3MLA)
    torch.nn.Module.__init__(leaf)
    hidden = torch.ones(1, 2, 4)
    with patch.object(
        vllm_deepseek_v3.DeepseekV2MLAAttention,
        "forward",
        return_value=torch.full((2, 4), 3.0),
    ) as mla_forward:
        output, cache = leaf(
            hidden,
            position_embeddings=None,
            attention_mask=None,
            position_ids=torch.tensor([[0, 1]]),
            use_cache=False,
        )

    parameter = torch.nn.Parameter(torch.zeros(4))
    model = SimpleNamespace(
        config=SimpleNamespace(tie_word_embeddings=False, n_routed_experts=2),
        _expert_loaders={},
        _tp_mesh=None,
        _tp_placements={},
        named_parameters=lambda: (("model.norm.weight", parameter),),
        named_buffers=lambda: (),
    )
    loaded = HyperDeepseekV3ForCausalLM.load_weights(
        model,
        (("model.norm.weight", torch.arange(4, dtype=torch.float32)),),
    )

    torch.testing.assert_close(output, torch.full((1, 2, 4), 3.0))
    assert cache is None
    assert mla_forward.call_args.kwargs["positions"].shape == (2,)
    assert loaded == {"model.norm.weight"}
    torch.testing.assert_close(parameter, torch.arange(4, dtype=torch.float32))


def test_deepseek_public_layout_routed_leaf_and_router_bias_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MLA slices, local experts, and persistent router bias share one public contract."""
    monkeypatch.setattr(get_platform(), "get_rank", lambda: 0)
    mesh = Layout((2,), ("tp",), init_backend=False).mesh
    parameter_name = "model.layers.0.self_attn.q_proj.weight"
    layout_model = SimpleNamespace(
        named_parameters=lambda: ((parameter_name, torch.empty(4, 4)),)
    )
    _validate_public_mla_layout(
        layout_model,
        {parameter_name: (Shard(0),)},
        {parameter_name: (8, 4)},
        mesh,
    )

    class Experts(torch.nn.Module):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__()
            self.local_expert_count = kwargs["local_expert_count"]

    monkeypatch.setattr(vllm_deepseek_v3, "HyperLocalFusedExperts", Experts)
    monkeypatch.setattr(
        vllm_deepseek_v3.HyperExpertPlacement,
        "from_config",
        lambda _runtime, _count: HyperExpertPlacement(2, 4, mesh),
    )
    config = SimpleNamespace(
        n_routed_experts=8,
        hidden_size=4,
        moe_intermediate_size=3,
    )
    moe = SimpleNamespace(
        config=config,
        gate=torch.nn.Linear(4, 8, bias=False),
        shared_experts=torch.nn.Identity(),
    )
    runtime = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
    )
    routed = _VLLMDeepseekV3RoutedExperts(
        moe,
        vllm_config=runtime,
        prefix="model.layers.0.mlp",
    )

    class Gate(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("e_score_correction_bias", torch.ones(8))

    checkpoint_model = torch.nn.Module()
    checkpoint_model.register_parameter("weight", torch.nn.Parameter(torch.ones(1)))
    checkpoint_model.model = torch.nn.Module()
    checkpoint_model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    checkpoint_model.model.layers[0].mlp = torch.nn.Module()
    checkpoint_model.model.layers[0].mlp.gate = Gate()
    tensors = vllm_deepseek_v3._checkpoint_tensors(checkpoint_model)

    assert routed.gate is moe.gate
    assert routed.shared_experts is moe.shared_experts
    assert routed.hyper_ep_mesh is mesh
    assert routed.experts.local_expert_count == 2
    assert set(tensors) == {
        "weight",
        "model.layers.0.mlp.gate.e_score_correction_bias",
    }


def test_moe_tp_plan_preserves_public_dense_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The MoE plan adapts local experts while retaining public dense TP rules."""
    monkeypatch.setattr(get_platform(), "get_rank", lambda: 0)
    mesh = Layout((2, 2), ("dp", "tp"), init_backend=False).mesh
    config = Qwen3MoeConfig.from_dict(
        {
            "hidden_size": 32,
            "intermediate_size": 64,
            "moe_intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "vocab_size": 64,
        }
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    reference = ShardingPlanner().plan(
        model,
        mesh,
        tp_size=2,
        ep_size=4,
        sequence_parallel=False,
    )
    adapted = build_moe_tp_plan(model, mesh, tp_size=2, ep_size=4)

    assert adapted.mesh_dim_names == reference.mesh_dim_names == ("tp",)
    for name, expected in reference.modules.items():
        actual = adapted.modules[name]
        if not name.endswith(".mlp"):
            assert actual.params == expected.params
            continue
        assert actual.params == {
            key: axes
            for key, axes in expected.params.items()
            if not key.startswith("experts.")
        }
        assert next(iter(actual.in_src.values()))["tp"] == Replicate()
        assert next(iter(actual.in_dst.values()))["tp"] == Shard(1)
        assert actual.region_dispatch is False


@pytest.mark.parametrize("tokens", [0, 3, 7])
def test_moe_padding_preserves_real_tokens_and_gradients(tokens: int) -> None:
    """Padding remains inside the pointwise MoE region and preserves gradients."""
    seen = []

    class Region(torch.nn.Module):
        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            seen.append(hidden_states.shape[1])
            return hidden_states * 3

    module = Region()
    pad_moe_tp_tokens(module, 2)
    values = torch.randn(1, tokens, 4, requires_grad=True)
    output = module(values)
    output.sum().backward()

    torch.testing.assert_close(output, values * 3, rtol=0, atol=0)
    torch.testing.assert_close(values.grad, torch.full_like(values, 3), rtol=0, atol=0)
    assert seen[0] > 0 and seen[0] % 2 == 0


@pytest.mark.parametrize("hf_combine", [False, True])
def test_public_qwen_factory_preserves_default_and_shared_combiner(
    hf_combine: bool,
) -> None:
    """Existing consumers keep the default while RL opts into the public combiner."""
    module = SimpleNamespace(gate=object(), experts=object())
    mesh = MagicMock()
    mesh.__getitem__.return_value.size.return_value = 4
    hidden = torch.randn(1, 2, 4)
    with patch.object(ep_compute, "bind_local_expert_forward") as bind, patch.object(
        ep_compute,
        "ep_routed_forward",
        return_value=hidden,
    ) as routed:
        factory = ep_compute.qwen3moe_ep_compute_fn(
            module=module,
            mesh=None,
            tp_mesh=None,
            cp_mesh=None,
            ep_mesh=mesh,
            **({"hf_combine": True} if hf_combine else {}),
        )
        assert factory(module, hidden) is hidden
        bind.assert_called_once_with(module, 4)
        if hf_combine:
            assert routed.call_args.kwargs["combine_fn"] is qwen3_moe_combine
        else:
            assert "combine_fn" not in routed.call_args.kwargs
    assert qwen3_combine is qwen3_moe_combine
