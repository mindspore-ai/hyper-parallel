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
"""Unit tests for LlamaFactory context-parallel integration."""
# pylint: disable=wrong-import-position,protected-access
import sys
import types
from types import ModuleType

import pytest
import torch

import hyper_parallel.integration.llamafactory.context_parallel.context_parallel_prepare as cp_prepare_mod
import hyper_parallel.integration.llamafactory.context_parallel.loss as loss_mod
from hyper_parallel.integration.llamafactory.context_parallel import (
    _build_cp_shift_labels,
    _enable_context_parallel_loss_patch,
    _get_cp_dp_ranks,
    get_cp_group_ranks,
    get_cp_rank,
    get_dp_rank,
    shard_inputs_for_cp,
)
from hyper_parallel.integration.llamafactory.context_parallel.loss import _wrap_loss_function
from hyper_parallel.integration.llamafactory.context_parallel.models.qwen3_vl import qwen3vl_forward as qwen3vl_cp_mod
from hyper_parallel.integration.llamafactory.utils import (
    HyperParallelArguments,
    _build_device_mesh,
    _build_fsdp2_kwargs,
)

_apply_qwen3vl_moe_attention_patch = qwen3vl_cp_mod._apply_qwen3vl_moe_attention_patch
_enable_qwen3vl_moe_attention_patch = qwen3vl_cp_mod._enable_qwen3vl_moe_attention_patch

# ---- Context Parallel tests ----


@pytest.mark.parametrize("embeds_arg", ["input_embeds", "inputs_embeds"])
def test_qwen3vl_global_attention_mask_supports_transformers_embed_argument(monkeypatch, embeds_arg):
    """Qwen3VL CP should support both historical causal-mask embedding argument names."""
    recorded = {}

    def old_mask(config, input_embeds, attention_mask, cache_position, past_key_values, position_ids):
        del config, past_key_values, position_ids
        recorded.update(embed=input_embeds, cache_position=cache_position)
        return attention_mask

    def new_mask(config, inputs_embeds, attention_mask, cache_position, past_key_values, position_ids):
        del config, past_key_values, position_ids
        recorded.update(embed=inputs_embeds, cache_position=cache_position)
        return attention_mask

    monkeypatch.setattr(qwen3vl_cp_mod, "create_causal_mask", new_mask if embeds_arg.endswith("s") else old_mask)
    model = types.SimpleNamespace(language_model=types.SimpleNamespace(config=object()))
    embeds = torch.randn(1, 4, 8)
    attention_mask = torch.ones(1, 4, dtype=torch.bool)

    result = qwen3vl_cp_mod._build_qwen3vl_global_attention_mask(
        model, attention_mask, torch.arange(4).view(1, 4), embeds, cache_position=None, past_key_values=None
    )

    assert result is attention_mask
    assert recorded["embed"] is embeds
    assert torch.equal(recorded["cache_position"], torch.arange(4))


def test_hp_args_cp_size_defaults_to_one():
    """
    Feature: Context parallel config default
    Description: cp_size defaults to 1, meaning no context parallel split.
    Expectation: Default HyperParallelArguments has cp_size=1.
    """
    args = HyperParallelArguments()
    assert args.cp_size == 1


def test_hp_args_cp_size_validation_rejects_non_positive():
    """
    Feature: Context parallel config validation
    Description: cp_size must be a positive integer.
    Expectation: Validation raises ValueError for cp_size <= 0.
    """
    with pytest.raises(ValueError, match="cp_size must be a positive integer"):
        HyperParallelArguments(cp_size=0).validate()

    with pytest.raises(ValueError, match="cp_size must be a positive integer"):
        HyperParallelArguments(cp_size=-1).validate()


def test_cp_prepare_model_skips_model_specific_patches_for_generic_model(monkeypatch):
    """
    Feature: CP preparation model dispatch
    Description: Model-specific patches should opt in through the CP model registry.
    Expectation: Generic models only receive the common loss patch, and no CP mesh is built.
    """
    model = torch.nn.Module()
    hp_args = types.SimpleNamespace(cp_size=2)
    calls = []

    fake_patch = types.SimpleNamespace(
        supports=lambda target_model: False,
        prepare=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("model-specific patch should be skipped")),
    )
    monkeypatch.setattr(cp_prepare_mod, "get_context_parallel_model_patches", lambda: (fake_patch,))
    monkeypatch.setattr(
        cp_prepare_mod,
        "_enable_context_parallel_loss_patch",
        lambda target_model, target_args: calls.append(("loss", target_model, target_args)),
    )
    monkeypatch.setattr(
        "hyper_parallel.integration.llamafactory.utils._build_device_mesh",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("mesh should be skipped")),
    )

    result = cp_prepare_mod.cp_prepare_model(model, accelerator=object(), hp_args=hp_args)

    assert result is model
    assert calls == [("loss", model, hp_args)]


def test_cp_rank_and_dp_rank_follow_cp_dp_mesh_layout(monkeypatch):
    """
    Feature: CP/DP rank mapping
    Description: Rank mapping should match the flattened (dp, cp) mesh layout used by the trainer integration.
    Expectation: dp_rank indexes rows and cp_rank indexes columns of the logical mesh.
    """
    hp_args = types.SimpleNamespace(cp_size=2)
    monkeypatch.setattr(
        "hyper_parallel.integration.llamafactory.context_parallel.inputs.get_platform",
        lambda: types.SimpleNamespace(get_rank=lambda: 5, get_world_size=lambda: 8),
    )

    assert _get_cp_dp_ranks(hp_args) == (1, 2)
    assert get_cp_rank(hp_args) == 1
    assert get_dp_rank(hp_args) == 2


def test_shard_inputs_for_cp_disabled():
    """
    Feature: CP input sharding passthrough
    Description: When cp_size=1, inputs should be returned unchanged.
    Expectation: Output is the same dict object.
    """
    inputs = {"input_ids": torch.arange(8).reshape(2, 4)}
    result = shard_inputs_for_cp(inputs, cp_rank=0, cp_size=1)
    assert result is inputs


def test_shard_inputs_for_cp_splits_sequence():
    """
    Feature: CP input sharding
    Description: With cp_size=2, each rank should get half the sequence.
    Expectation: Rank 0 gets the first half, rank 1 gets the second half.
    """
    batch, seq = 2, 8
    input_ids = torch.arange(batch * seq).reshape(batch, seq)
    attention_mask = torch.ones(batch, seq, dtype=torch.long)
    labels = torch.arange(batch * seq).reshape(batch, seq) + 100

    inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    r0 = shard_inputs_for_cp(inputs, cp_rank=0, cp_size=2)
    r1 = shard_inputs_for_cp(inputs, cp_rank=1, cp_size=2)

    assert r0["input_ids"].shape == (batch, seq // 2)
    assert r1["input_ids"].shape == (batch, seq // 2)
    assert torch.equal(r0["input_ids"], input_ids[:, :4])
    assert torch.equal(r1["input_ids"], input_ids[:, 4:])
    assert torch.equal(r0["attention_mask"], attention_mask)
    assert torch.equal(r1["attention_mask"], attention_mask)
    assert torch.equal(r0["labels"], labels)
    assert torch.equal(r1["labels"], labels)


def test_shard_inputs_for_cp_slices_4d_attention_mask_on_query_dim():
    """
    Feature: CP 4D attention mask handling
    Description: Prepared 4D masks should keep the full key dimension but shard the query dimension.
    Expectation: The query axis is sliced while the last axis remains the full sequence length.
    """
    mask = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
    inputs = {
        "input_ids": torch.zeros(1, 8, dtype=torch.long),
        "attention_mask": mask,
    }

    result = shard_inputs_for_cp(inputs, cp_rank=1, cp_size=2)

    assert result["attention_mask"].shape == (1, 1, 4, 8)
    assert torch.equal(result["attention_mask"], mask[:, :, 4:, :])


def test_shard_inputs_for_cp_generates_position_ids():
    """
    Feature: CP position_ids generation
    Description: When position_ids is not in the input, it should be generated with the correct offset.
    Expectation: Rank 1 gets position_ids starting at seq_len // cp_size.
    """
    batch, seq = 2, 8
    inputs = {"input_ids": torch.zeros(batch, seq, dtype=torch.long)}

    r0 = shard_inputs_for_cp(inputs, cp_rank=0, cp_size=2)
    r1 = shard_inputs_for_cp(inputs, cp_rank=1, cp_size=2)

    assert "position_ids" in r0
    assert "position_ids" in r1
    assert r0["position_ids"].shape == (batch, 4)
    assert r1["position_ids"].shape == (batch, 4)
    assert torch.equal(r0["position_ids"][0], torch.arange(0, 4))
    assert torch.equal(r1["position_ids"][0], torch.arange(4, 8))


def test_shard_inputs_for_cp_preserves_existing_position_ids():
    """
    Feature: CP position_ids preservation
    Description: When position_ids is already in the input, it should be sliced, not regenerated.
    Expectation: Existing position_ids are sliced along the sequence dimension.
    """
    batch, seq = 1, 8
    pos_ids = torch.arange(seq).unsqueeze(0) * 10
    inputs = {"input_ids": torch.zeros(batch, seq, dtype=torch.long), "position_ids": pos_ids}

    r1 = shard_inputs_for_cp(inputs, cp_rank=1, cp_size=2)

    assert torch.equal(r1["position_ids"], pos_ids[:, 4:])


def test_shard_inputs_for_cp_slices_mrope_position_ids_on_last_dim():
    """
    Feature: CP MRoPE position_ids sharding
    Description: Qwen3VL-style position ids use shape [3, batch, seq] and must shard the last dimension.
    Expectation: Rotary rank dimensions are preserved while only the sequence axis is sliced.
    """
    position_ids = torch.arange(24, dtype=torch.long).reshape(3, 1, 8)
    inputs = {
        "input_ids": torch.zeros(1, 8, dtype=torch.long),
        "position_ids": position_ids,
    }

    result = shard_inputs_for_cp(inputs, cp_rank=1, cp_size=2)

    assert torch.equal(result["position_ids"], position_ids[..., 4:8])


def test_shard_inputs_for_cp_skips_non_tensor_entries():
    """
    Feature: CP input sharding non-tensor handling
    Description: Non-tensor entries in inputs should be passed through unchanged.
    Expectation: Scalar and string values remain as-is.
    """
    inputs = {
        "input_ids": torch.zeros(2, 8, dtype=torch.long),
        "num_items_in_batch": 4,
        "task_name": "sft",
    }

    result = shard_inputs_for_cp(inputs, cp_rank=0, cp_size=2)

    assert result["num_items_in_batch"] == 4
    assert result["task_name"] == "sft"
    assert result["input_ids"].shape == (2, 4)


def test_shard_inputs_for_cp_preserves_multimodal_metadata():
    """
    Feature: CP multimodal metadata propagation
    Description: Vision batches still need text sharding, but the runtime patch also needs the global token layout.
    Expectation: Text tensors are sharded while vision tensors and global CP metadata are preserved.
    """
    pixel_values = torch.randn(3, 4, 8)
    image_grid_thw = torch.tensor([[1, 2, 2], [1, 1, 4]], dtype=torch.long)
    mm_token_type_ids = torch.tensor([[0, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.int32)
    inputs = {
        "input_ids": torch.arange(8, dtype=torch.long).reshape(1, 8),
        "position_ids": torch.arange(8, dtype=torch.long).reshape(1, 8),
        "labels": torch.arange(8, dtype=torch.long).reshape(1, 8),
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "mm_token_type_ids": mm_token_type_ids,
    }

    result = shard_inputs_for_cp(inputs, cp_rank=1, cp_size=2)

    assert torch.equal(result["input_ids"], torch.arange(4, 8, dtype=torch.long).reshape(1, 4))
    assert torch.equal(result["labels"], inputs["labels"])
    assert result["pixel_values"] is pixel_values
    assert result["image_grid_thw"] is image_grid_thw
    assert result["mm_token_type_ids"] is mm_token_type_ids
    assert result["_hp_cp_global_input_ids"] is inputs["input_ids"]
    assert result["_hp_cp_global_position_ids"] is inputs["position_ids"]
    assert result["_hp_cp_local_seq_start"] == 4
    assert result["_hp_cp_local_seq_end"] == 8
    assert torch.equal(result["position_ids"], torch.arange(4, 8, dtype=torch.long).reshape(1, 4))


def test_shard_inputs_for_cp_allows_multimodal_without_position_ids():
    """
    Feature: CP multimodal metadata propagation
    Description: Transformers Trainer inputs do not always include position_ids before the model forward.
    Expectation: Generic CP input sharding keeps global token metadata and lets model-specific patches build positions.
    """
    inputs = {
        "input_ids": torch.arange(8, dtype=torch.long).reshape(1, 8),
        "pixel_values": torch.randn(3, 4, 8),
        "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
    }

    result = shard_inputs_for_cp(inputs, cp_rank=1, cp_size=2)

    assert torch.equal(result["input_ids"], torch.arange(4, 8, dtype=torch.long).reshape(1, 4))
    assert result["pixel_values"] is inputs["pixel_values"]
    assert result["image_grid_thw"] is inputs["image_grid_thw"]
    assert result["_hp_cp_global_input_ids"] is inputs["input_ids"]
    assert "_hp_cp_global_position_ids" not in result
    assert result["_hp_cp_local_seq_start"] == 4
    assert result["_hp_cp_local_seq_end"] == 8


def test_shard_inputs_for_cp_handles_indivisible_seq_len():
    """
    Feature: CP input sharding seq_len not divisible by cp_size
    Description: Tensors whose sequence dimension is not divisible by cp_size are passed through.
    Expectation: The tensor is not sliced.
    """
    inputs = {"input_ids": torch.zeros(2, 7, dtype=torch.long)}
    result = shard_inputs_for_cp(inputs, cp_rank=0, cp_size=2)
    assert result["input_ids"].shape == (2, 7)


def test_cp_mesh_construction_with_cp_disabled(monkeypatch):
    """
    Feature: Mesh construction without CP
    Description: When cp_size=1, a standard 1D DP mesh should be built.
    Expectation: The mesh is 1D with a single "dp" dimension.
    """
    calls = []

    def _fake_init_device_mesh(device_type, mesh_shape, *, mesh_dim_names=None):
        calls.append({"device_type": device_type, "mesh_shape": mesh_shape, "mesh_dim_names": mesh_dim_names})
        return f"mesh_{mesh_shape}"

    monkeypatch.setattr(
        "hyper_parallel.integration.llamafactory.utils.init_device_mesh",
        _fake_init_device_mesh,
    )
    monkeypatch.setattr(
        "hyper_parallel.integration.llamafactory.utils.get_platform",
        lambda: types.SimpleNamespace(get_world_size=lambda: 8),
    )

    accelerator = types.SimpleNamespace(torch_device_mesh=None, parallelism_config=None)
    hp_args = HyperParallelArguments(device_type="cuda", cp_size=1)
    mesh = _build_device_mesh(accelerator, hp_args)

    assert mesh == "mesh_(8,)"
    assert calls[-1]["mesh_shape"] == (8,)
    assert calls[-1]["mesh_dim_names"] == ("dp",)


def test_cp_mesh_construction_with_cp_enabled(monkeypatch):
    """
    Feature: Mesh construction with CP
    Description: When cp_size=2, a 2D (dp, cp) mesh should be built.
    Expectation: The mesh is 2D with shape (cp_size, dp_size).
    """
    calls = []

    def _fake_init_device_mesh(device_type, mesh_shape, *, mesh_dim_names=None):
        calls.append({"device_type": device_type, "mesh_shape": mesh_shape, "mesh_dim_names": mesh_dim_names})
        return f"mesh_{mesh_shape}"

    monkeypatch.setattr(
        "hyper_parallel.integration.llamafactory.utils.init_device_mesh",
        _fake_init_device_mesh,
    )
    monkeypatch.setattr(
        "hyper_parallel.integration.llamafactory.utils.get_platform",
        lambda: types.SimpleNamespace(get_world_size=lambda: 8),
    )

    accelerator = types.SimpleNamespace(torch_device_mesh=None, parallelism_config=None)
    hp_args = HyperParallelArguments(device_type="cuda", cp_size=2)
    mesh = _build_device_mesh(accelerator, hp_args)

    assert mesh == "mesh_(4, 2)"
    assert calls[-1]["mesh_shape"] == (4, 2)
    assert calls[-1]["mesh_dim_names"] == ("dp", "cp")


def test_cp_mesh_construction_reuses_cached_mesh(monkeypatch):
    """
    Feature: Mesh construction cache
    Description: Trainer CP enablement and FSDP wrapping should reuse the same mesh object on the accelerator.
    Expectation: init_device_mesh is called only once and later requests return the cached mesh.
    """
    calls = []

    def _fake_init_device_mesh(device_type, mesh_shape, *, mesh_dim_names=None):
        calls.append({"device_type": device_type, "mesh_shape": mesh_shape, "mesh_dim_names": mesh_dim_names})
        return {"shape": mesh_shape, "names": mesh_dim_names, "id": len(calls)}

    monkeypatch.setattr(
        "hyper_parallel.integration.llamafactory.utils.init_device_mesh",
        _fake_init_device_mesh,
    )
    monkeypatch.setattr(
        "hyper_parallel.integration.llamafactory.utils.get_platform",
        lambda: types.SimpleNamespace(get_world_size=lambda: 8),
    )

    accelerator = types.SimpleNamespace(torch_device_mesh=None, parallelism_config=None)
    hp_args = HyperParallelArguments(device_type="cuda", cp_size=2)

    mesh0 = _build_device_mesh(accelerator, hp_args)
    mesh1 = _build_device_mesh(accelerator, hp_args)

    assert mesh0 is mesh1
    assert len(calls) == 1
    assert getattr(accelerator, "_hp_device_mesh") is mesh0


def test_build_fsdp2_kwargs_flattens_dp_cp_mesh(monkeypatch):
    """
    Feature: FSDP mesh selection
    Description: CP should keep the parent mesh for runtime hooks, while fully_shard should shard over DP*CP.
    Expectation: _build_fsdp2_kwargs passes the flattened ("dp", "cp") mesh into fully_shard kwargs.
    """
    class _FakeFlatMesh:
        mesh_dim_names = ("fsdp",)
        mesh_shape = (8,)

    class _FakeMesh:
        """Fake DP/CP mesh that records flatten calls."""

        mesh_dim_names = ("dp", "cp")

        def __getitem__(self, key):
            if key == ("dp", "cp"):
                return self
            raise KeyError(key)

        def flatten(self, mesh_dim_name=None):
            self.flatten_name = mesh_dim_name
            return _FakeFlatMesh()

    monkeypatch.setattr(
        "hyper_parallel.integration.llamafactory.utils._build_device_mesh",
        lambda accelerator, hp_args: _FakeMesh(),
    )
    monkeypatch.setattr(
        "hyper_parallel.integration.llamafactory.utils._resolve_offload_policy",
        lambda plugin: "offload",
    )
    monkeypatch.setattr(
        "hyper_parallel.integration.llamafactory.utils._resolve_mp_policy",
        lambda plugin, hp_args: "mp",
    )
    monkeypatch.setattr(
        "hyper_parallel.integration.llamafactory.utils.get_parameters_from_modules",
        lambda modules, model, device: {"ignored"},
    )

    accelerator = types.SimpleNamespace(device="npu:0")
    plugin = types.SimpleNamespace(reshard_after_forward=True, ignored_modules=())

    hp_args = HyperParallelArguments(reshard_after_forward=None)
    kwargs = _build_fsdp2_kwargs(accelerator, torch.nn.Linear(2, 2), hp_args, plugin)

    assert kwargs["mesh"].mesh_dim_names == ("fsdp",)


def test_get_cp_group_ranks_matches_dp_column(monkeypatch):
    """
    Feature: CP group rank discovery
    Description: CP peers should resolve to the same DP column ranks used for input synchronization.
    Expectation: The computed CP group ranks match the current dp_rank column.
    """
    monkeypatch.setattr(
        "hyper_parallel.integration.llamafactory.context_parallel.inputs.get_platform",
        lambda: types.SimpleNamespace(get_rank=lambda: 3, get_world_size=lambda: 8),
    )

    ranks = get_cp_group_ranks(types.SimpleNamespace(cp_size=2))

    assert ranks == (2, 3)


class _FakeQwenFamilyTextAttention(torch.nn.Module):
    """Fake Qwen-family text attention module for runtime patch tests."""

    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(_attn_implementation="sdpa")
        self.layer_idx = 0
        self.num_attention_heads = 2
        self.num_key_value_heads = 1
        self.head_dim = 4
        self.attention_dropout = 0.0
        self.is_causal = True
        self.q_proj = torch.nn.Linear(8, 8, bias=False)
        self.k_proj = torch.nn.Linear(8, 4, bias=False)
        self.v_proj = torch.nn.Linear(8, 4, bias=False)
        self.o_proj = torch.nn.Linear(8, 8, bias=False)
        self.q_norm = torch.nn.Identity()
        self.k_norm = torch.nn.Identity()
        self.rotary_emb = lambda value_states, position_ids: (
            torch.ones(1, 1, value_states.shape[-2], self.head_dim, device=value_states.device),
            torch.zeros(1, 1, value_states.shape[-2], self.head_dim, device=value_states.device),
        )

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
        del hidden_states, position_embeddings, attention_mask, kwargs
        return ("original", None)


def _fake_apply_rotary_pos_emb(query_states, key_states, cos, sin):
    del cos, sin
    return query_states, key_states


_FakeQwenFamilyTextAttention.forward.__globals__["apply_rotary_pos_emb"] = _fake_apply_rotary_pos_emb


class _FakeQwen3VLMoeTextAttention(_FakeQwenFamilyTextAttention):
    def __init__(self):
        super().__init__()
        self.config.num_attention_heads = self.num_attention_heads
        self.config.num_key_value_heads = self.num_key_value_heads
        del self.num_attention_heads
        del self.num_key_value_heads


_FakeQwen3VLMoeTextAttention.__name__ = "Qwen3VLMoeTextAttention"
_FakeQwenFamilyTextAttention.__name__ = "Qwen3VLTextAttention"


_fake_rotary_module = ModuleType("fake_qwen_rotary_module")
_fake_rotary_module.apply_rotary_pos_emb = _fake_apply_rotary_pos_emb
sys.modules["fake_qwen_rotary_module"] = _fake_rotary_module


class _FakeQwenConfigRotaryTextAttention(_FakeQwen3VLMoeTextAttention):
    __module__ = "fake_qwen_rotary_module"

    def __init__(self):
        super().__init__()
        del self.rotary_emb


_FakeQwenConfigRotaryTextAttention.__name__ = "Qwen3VLMoeTextAttention"


class _FakeQwen3VLMoeVisionAttention(torch.nn.Module):
    """Fake Qwen3VL-MoE vision attention that should not receive text CP."""

    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(_attn_implementation="sdpa")
        self.num_heads = 2
        self.head_dim = 4
        self.attention_dropout = 0.0
        self.qkv = torch.nn.Linear(8, 24, bias=False)
        self.proj = torch.nn.Linear(8, 8, bias=False)

    def forward(self, hidden_states, cu_seqlens=None, position_embeddings=None, **kwargs):
        del hidden_states, cu_seqlens, position_embeddings, kwargs
        return hidden_states


def _fake_apply_rotary_pos_emb_vision(query_states, key_states, cos, sin):
    del cos, sin
    return query_states, key_states


_FakeQwen3VLMoeVisionAttention.forward.__globals__["apply_rotary_pos_emb_vision"] = _fake_apply_rotary_pos_emb_vision


class _FakeQwen3VLMoeModelOutputWithPast:
    def __init__(self, last_hidden_state, past_key_values, rope_deltas):
        self.last_hidden_state = last_hidden_state
        self.past_key_values = past_key_values
        self.rope_deltas = rope_deltas


class _FakeQwen3VLMoeLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_call = None

    def forward(self, **kwargs):
        self.last_call = kwargs
        return types.SimpleNamespace(last_hidden_state=kwargs["inputs_embeds"], past_key_values=None)


class _FakeQwen3VLMoeModel(torch.nn.Module):
    """Fake Qwen3VL-MoE model shell for visual feature patch tests."""

    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(image_token_id=99, video_token_id=98)
        self.visual = types.SimpleNamespace(dtype=torch.float32)
        self.language_model = _FakeQwen3VLMoeLanguageModel()
        self.embed_tokens = torch.nn.Embedding(128, 2)
        self.rope_deltas = None

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_image_features(self, pixel_values, image_grid_thw=None):
        del pixel_values, image_grid_thw
        return [torch.tensor([[10.0, 10.0], [11.0, 11.0], [12.0, 12.0], [13.0, 13.0]])], [
            torch.tensor([[100.0, 100.0], [101.0, 101.0], [102.0, 102.0], [103.0, 103.0]])
        ]

    def get_video_features(self, pixel_values_videos, video_grid_thw=None):
        del pixel_values_videos, video_grid_thw
        return [torch.empty((0, 2))], [torch.empty((0, 2))]

    def get_placeholder_mask(self, input_ids, inputs_embeds, image_features=None, video_features=None):
        del inputs_embeds
        image_mask = input_ids == self.config.image_token_id
        video_mask = input_ids == self.config.video_token_id
        if image_features is not None and int(image_mask.sum()) != image_features.shape[0]:
            raise ValueError("image placeholder mismatch in test double")
        if video_features is not None and int(video_mask.sum()) != video_features.shape[0]:
            raise ValueError("video placeholder mismatch in test double")
        return image_mask.unsqueeze(-1).expand(-1, -1, 2), video_mask.unsqueeze(-1).expand(-1, -1, 2)

    def get_rope_index(self, input_ids, image_grid_thw, video_grid_thw, attention_mask=None):
        del input_ids, image_grid_thw, video_grid_thw, attention_mask
        position_ids = torch.arange(8, dtype=torch.long).view(1, 1, 8).expand(3, 1, 8)
        rope_deltas = torch.tensor([[3]], dtype=torch.long)
        return position_ids, rope_deltas

    def forward(self, **kwargs):
        del kwargs
        raise AssertionError("patched forward should intercept multimodal CP path")


_FakeQwen3VLMoeModel.__name__ = "Qwen3VLMoeModel"
_FakeQwen3VLMoeModel.forward.__globals__["Qwen3VLMoeModelOutputWithPast"] = _FakeQwen3VLMoeModelOutputWithPast
_FakeQwen3VLMoeModel.forward.__globals__["is_torchdynamo_compiling"] = lambda: False


def test_qwen3vl_rope_index_supports_old_and_new_signatures():
    """Qwen3VL CP should pass the arguments required by both mRoPE APIs."""
    recorded = []

    class _OldRopeModel:
        def get_rope_index(self, input_ids, image_grid_thw, video_grid_thw, attention_mask=None):
            recorded.append((input_ids, None, image_grid_thw, video_grid_thw, attention_mask))
            return "positions", "deltas"

    class _NewRopeModel:
        def get_rope_index(self, input_ids, mm_token_type_ids, **kwargs):
            recorded.append((input_ids, mm_token_type_ids, *kwargs.values()))
            return "positions", "deltas"

    input_ids = torch.ones(1, 4, dtype=torch.long)
    token_types = torch.ones_like(input_ids, dtype=torch.int32)
    for model, types_arg in ((_OldRopeModel(), None), (_NewRopeModel(), token_types)):
        assert qwen3vl_cp_mod._get_qwen3vl_rope_index(
            model, input_ids, types_arg, "image-grid", "video-grid", "mask"
        ) == ("positions", "deltas")
    assert recorded == [
        (input_ids, None, "image-grid", "video-grid", "mask"),
        (input_ids, token_types, "image-grid", "video-grid", "mask"),
    ]


def test_qwen3vl_visual_feature_api_compatibility():
    """Qwen3VL CP should normalize the old tuple and new model-output APIs."""
    recorded = []

    def old_features(pixel_values, grid_thw=None):
        recorded.append((pixel_values, grid_thw, {}))
        return outputs

    def new_features(pixel_values, grid_thw=None, **kwargs):
        recorded.append((pixel_values, grid_thw, kwargs))
        return types.SimpleNamespace(pooler_output=outputs[0], deepstack_features=outputs[1])

    pixels = torch.ones(2, 4)
    outputs = ([pixels], [pixels + 1])
    assert qwen3vl_cp_mod._get_qwen3vl_visual_features(old_features, pixels, "grid") == outputs
    assert qwen3vl_cp_mod._get_qwen3vl_visual_features(new_features, pixels, "grid") == outputs
    assert recorded == [(pixels, "grid", {}), (pixels, "grid", {"return_dict": True})]


def test_apply_qwen3vl_moe_attention_patch_wraps_text_attention(monkeypatch):
    """
    Feature: Qwen3VL-MoE runtime CP attention enablement
    Description: Supported Qwen3VL-MoE text attention modules should be patched and wrapped via ContextParallel.
    Expectation: A CP core submodule is added and parallelize_module receives the CP plan.
    """
    module = _FakeQwen3VLMoeTextAttention()
    module.config._attn_implementation = "flash_attention_2"
    calls = []

    monkeypatch.setattr(
        qwen3vl_cp_mod,
        "parallelize_module",
        lambda mod, mesh, plan: calls.append((mod, mesh, plan)) or mod,
    )
    monkeypatch.setattr(qwen3vl_cp_mod, "_Qwen3VLMoeTextAttention", _FakeQwen3VLMoeTextAttention)
    monkeypatch.setattr(
        qwen3vl_cp_mod,
        "_resolve_qwen3vl_moe_attention_interface",
        lambda mod: (lambda owner, query, key, value, attention_mask, **kwargs: (query.transpose(1, 2), None)),
    )

    enabled = _apply_qwen3vl_moe_attention_patch(module, cp_mesh="cp-mesh", cp_rank=1, cp_size=2)

    assert enabled is True
    assert hasattr(module, "_hp_context_parallel_core_attn")
    assert getattr(module, "_hp_cp_attention_enabled") is True
    assert calls and calls[0][0] is module
    assert calls[0][1] == "cp-mesh"
    assert "_hp_context_parallel_core_attn" in calls[0][2]
    assert calls[0][2]["_hp_context_parallel_core_attn"].ulysses_degree is None


def test_qwen3vl_moe_cp_attention_forward_preserves_packed_kwargs(monkeypatch):
    """
    Feature: Packed attention metadata preservation
    Description: Qwen3VL-MoE CP attention should forward FlashAttention varlen metadata into the CP core.
    Expectation: Packed kwargs reach the core unchanged instead of being dropped at the patch boundary.
    """
    module = _FakeQwen3VLMoeTextAttention()
    module.config._attn_implementation = "flash_attention_2"

    monkeypatch.setattr(
        qwen3vl_cp_mod,
        "parallelize_module",
        lambda mod, mesh, plan: mod,
    )
    monkeypatch.setattr(qwen3vl_cp_mod, "_Qwen3VLMoeTextAttention", _FakeQwen3VLMoeTextAttention)
    monkeypatch.setattr(
        qwen3vl_cp_mod,
        "_resolve_qwen3vl_moe_attention_interface",
        lambda mod: (lambda owner, query, key, value, attention_mask, **kwargs: (query.transpose(1, 2), None)),
    )

    enabled = _apply_qwen3vl_moe_attention_patch(module, cp_mesh="cp-mesh", cp_rank=0, cp_size=2)
    assert enabled is True

    recorded = {}

    class _RecordingCore(torch.nn.Module):
        def forward(self, query, key, value, **kwargs):
            recorded["query_shape"] = query.shape
            recorded["key_shape"] = key.shape
            recorded["value_shape"] = value.shape
            recorded["kwargs"] = kwargs
            return query

    module._modules["_hp_context_parallel_core_attn"] = _RecordingCore()

    hidden_states = torch.randn(1, 4, 8)
    attention_mask = torch.ones(1, 4, dtype=torch.bool)
    global_position_embeddings = (
        torch.ones(1, 1, 4, module.head_dim),
        torch.zeros(1, 1, 4, module.head_dim),
    )
    packed_kwargs = {
        "cu_seq_lens_q": torch.tensor([0, 4], dtype=torch.int32),
        "cu_seq_lens_k": torch.tensor([0, 4], dtype=torch.int32),
        "max_length_q": 4,
        "max_length_k": 4,
        "_hp_cp_global_position_ids": torch.arange(4).view(1, 4),
        "_hp_cp_global_position_embeddings": global_position_embeddings,
        "_hp_cp_global_attention_mask": attention_mask,
    }

    output, attn_weights = module(hidden_states, attention_mask=attention_mask, **packed_kwargs)

    assert output.shape == (1, 4, 8)
    assert attn_weights is None
    assert recorded["query_shape"] == (1, module.config.num_attention_heads, 4, module.head_dim)
    assert recorded["kwargs"]["attention_mask"] is attention_mask
    assert recorded["kwargs"]["cu_seq_lens_q"] is packed_kwargs["cu_seq_lens_q"]
    assert recorded["kwargs"]["cu_seq_lens_k"] is packed_kwargs["cu_seq_lens_k"]
    assert recorded["kwargs"]["max_length_q"] == 4
    assert recorded["kwargs"]["max_length_k"] == 4


def test_apply_qwen3vl_moe_attention_patch_skips_vision_attention(monkeypatch):
    """
    Feature: Text-only runtime CP attention enablement
    Description: Vision attention should remain untouched while trainer CP only covers text attention.
    Expectation: Qwen vision attention modules are detected but not wrapped by the runtime CP adapter.
    """
    module = _FakeQwen3VLMoeVisionAttention()
    calls = []

    monkeypatch.setattr(
        qwen3vl_cp_mod,
        "parallelize_module",
        lambda mod, mesh, plan: calls.append((mod, mesh, plan)) or mod,
    )
    monkeypatch.setattr(qwen3vl_cp_mod, "_Qwen3VLMoeTextAttention", _FakeQwen3VLMoeTextAttention)

    enabled = _apply_qwen3vl_moe_attention_patch(module, cp_mesh="cp-mesh", cp_rank=0, cp_size=2)

    assert enabled is False
    assert not calls
    assert not hasattr(module, "_hp_context_parallel_core_attn")


def test_apply_qwen3vl_moe_attention_patch_skips_non_qwen_attention():
    """
    Feature: Qwen3VL-MoE runtime CP attention enablement
    Description: Runtime CP attention wrapping should not trigger on modules outside Qwen3VL-MoE text attention.
    Expectation: Non-Qwen3VL attention modules are left untouched.
    """
    module = torch.nn.Linear(8, 8)

    enabled = _apply_qwen3vl_moe_attention_patch(module, cp_mesh="cp-mesh", cp_rank=1, cp_size=2)

    assert enabled is False
    assert not hasattr(module, "_hp_context_parallel_core_attn")


def test_qwen3vl_moe_visual_feature_stream_selects_local_placeholders():
    """
    Feature: Qwen3VL-MoE visual feature sharding
    Description: Dense visual features are stored in global placeholder order while text inputs are CP-sharded.
    Expectation: Only features belonging to the local CP text shard are selected, including deepstack features.
    """
    dense_embeds = [torch.tensor([[10.0, 10.0], [11.0, 11.0], [12.0, 12.0], [13.0, 13.0]])]
    deepstack_embeds = [torch.tensor([[100.0, 100.0], [101.0, 101.0], [102.0, 102.0], [103.0, 103.0]])]
    global_mask = torch.tensor([[False, True, True, False, False, True, True, False]])

    local_dense, local_deepstack = qwen3vl_cp_mod._select_local_visual_feature_stream(
        dense_embeds=dense_embeds,
        deepstack_embeds=deepstack_embeds,
        global_mask=global_mask,
        seq_start=4,
        seq_end=8,
        feature_name="Image",
    )

    assert torch.equal(local_dense[0], torch.tensor([[12.0, 12.0], [13.0, 13.0]]))
    assert torch.equal(local_deepstack[0], torch.tensor([[102.0, 102.0], [103.0, 103.0]]))


def test_enable_qwen3vl_moe_attention_patch_skips_vision_only_models(monkeypatch):
    """
    Feature: Text-only runtime CP attention enablement
    Description: Models that only expose vision attention should be ignored instead of failing CP setup.
    Expectation: Text-only CP enablement returns cleanly when no text attention candidates are present.
    """
    model = torch.nn.Module()
    model.visual_attn = _FakeQwen3VLMoeVisionAttention()
    hp_args = HyperParallelArguments(cp_size=2)

    class _FakeMesh:
        mesh_dim_names = ("dp", "cp")

        def __getitem__(self, key):
            if key != "cp":
                raise KeyError(key)
            return "cp-mesh"

    monkeypatch.setattr(qwen3vl_cp_mod, "_Qwen3VLMoeTextAttention", _FakeQwen3VLMoeTextAttention)
    monkeypatch.setattr(qwen3vl_cp_mod, "get_cp_rank", lambda _: 0)
    monkeypatch.setattr(qwen3vl_cp_mod, "get_cp_group", lambda _: None)
    monkeypatch.setattr(qwen3vl_cp_mod, "get_cp_group_ranks", lambda _: None)
    monkeypatch.setattr(
        qwen3vl_cp_mod,
        "parallelize_module",
        lambda mod, mesh, plan: (_ for _ in ()).throw(AssertionError("vision attention should not be parallelized")),
    )

    _enable_qwen3vl_moe_attention_patch(model, mesh=_FakeMesh(), hp_args=hp_args)


def test_cp_shift_labels_include_cross_shard_target():
    """
    Feature: CP shifted labels
    Description: Local logits need targets from the next global position, including CP shard boundaries.
    Expectation: Rank 0 keeps the first boundary target and the last rank pads the final target.
    """
    labels = torch.tensor([[10, 11, 12, 13]])

    rank0 = _build_cp_shift_labels(labels, local_seq_len=2, cp_rank=0, ignore_index=-100)
    rank1 = _build_cp_shift_labels(labels, local_seq_len=2, cp_rank=1, ignore_index=-100)

    assert torch.equal(rank0, torch.tensor([[11, 12]]))
    assert torch.equal(rank1, torch.tensor([[13, -100]]))


def test_cp_loss_wrapper_uses_per_token_loss_and_shift_labels(monkeypatch):
    """
    Feature: CP Causal LM loss patch
    Description: The wrapper should align shifted labels and compute unreduced local per-token CE.
    Expectation: The returned scalar reports the CP-group token mean without delegating to scalar HF loss.
    """
    calls = []

    def _original_loss(**kwargs):
        calls.append(kwargs)
        raise AssertionError("CP loss should use local per-token CE instead of scalar HF loss.")

    monkeypatch.setattr(loss_mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(loss_mod.dist, "is_initialized", lambda: True)
    def _fake_all_reduce(tensor, group=None):
        tensor.mul_(2 if group == "cp-group" else 4)

    monkeypatch.setattr(loss_mod.dist, "all_reduce", _fake_all_reduce)

    wrapped = _wrap_loss_function(_original_loss, cp_rank=1, cp_size=2, cp_group="cp-group")
    logits = torch.zeros(1, 2, 8)
    labels = torch.tensor([[1, 2, 3, 4]])

    loss = wrapped(logits=logits, labels=labels, vocab_size=8)

    assert torch.allclose(loss, torch.log(torch.tensor(8.0)))
    assert not calls


def test_cp_loss_wrapper_uses_cp_tokens_for_backward(monkeypatch):
    """
    Feature: CP Causal LM loss gradient scale
    Description: Fully_shard averages over DP*CP, so local token-loss gradients are scaled by cp_size.
    Expectation: Each rank backprops scaled local loss while the visible loss stays the token mean.
    """
    monkeypatch.setattr(loss_mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(loss_mod.dist, "is_initialized", lambda: True)
    def _fake_all_reduce(tensor, group=None):
        tensor.mul_(4 if group == "cp-group" else 8)

    monkeypatch.setattr(loss_mod.dist, "all_reduce", _fake_all_reduce)

    def _original_loss(**kwargs):
        del kwargs
        raise AssertionError("CP loss should use local per-token CE instead of scalar HF loss.")

    wrapped = _wrap_loss_function(_original_loss, cp_rank=0, cp_size=4, cp_group="cp-group")
    logits = torch.zeros(1, 2, 8, requires_grad=True)
    loss = wrapped(logits=logits, labels=torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]), vocab_size=8)

    assert torch.allclose(loss.detach(), torch.log(torch.tensor(8.0)))
    loss.backward()
    assert logits.grad is not None
    assert torch.allclose(logits.grad.sum(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(logits.grad.abs().sum(), torch.tensor(1.75), atol=1e-6)


def test_enable_context_parallel_loss_patch_wraps_model_loss(monkeypatch):
    """
    Feature: CP loss patch installation
    Description: cp_prepare_model should patch modules exposing a HuggingFace-style loss_function.
    Expectation: The patched loss receives CP-correct shifted labels when logits are local and labels are global.
    """
    module = torch.nn.Module()
    calls = []

    def _original_loss(**kwargs):
        calls.append(kwargs)
        raise AssertionError("CP loss should use local per-token CE instead of scalar HF loss.")

    module.loss_function = _original_loss
    monkeypatch.setattr(
        "hyper_parallel.integration.llamafactory.context_parallel.loss.get_cp_rank",
        lambda hp_args: 0,
    )
    monkeypatch.setattr(
        "hyper_parallel.integration.llamafactory.context_parallel.loss.get_cp_group",
        lambda hp_args: None,
    )
    monkeypatch.setattr(
        "hyper_parallel.integration.llamafactory.context_parallel.loss.dist.is_available",
        lambda: False,
    )

    _enable_context_parallel_loss_patch(module, types.SimpleNamespace(cp_size=2))
    result = module.loss_function(logits=torch.zeros(1, 2, 8), labels=torch.tensor([[1, 2, 3, 4]]), vocab_size=8)

    assert torch.allclose(result, torch.log(torch.tensor(8.0)))
    assert getattr(module, "_hp_cp_loss_enabled")
    assert not calls
