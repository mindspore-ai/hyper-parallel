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
"""Tests for loading and converting pretrained safetensors weights."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import List
from unittest.mock import Mock

import pytest
import torch
import yaml
from safetensors.torch import save_file
from torch import nn  # pylint: disable=forbidden-backend-import
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.conversion_mapping import get_model_conversion_mapping
from transformers.core_model_loading import Chunk, WeightConverter, WeightRenaming

import hyper_models._transformers.checkpoint_loader as checkpoint_loader
import hyper_models._transformers.infrastructure as infrastructure_module
from hyper_models.config.resolver import resolve_component
from hyper_models.trainer.config import PlanOverride, entries_to_module_replacements
from tests.common.mark_utils import arg_mark


class _SingleWeightModel(nn.Module):
    """A minimal model with one checkpoint target."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False)


class _QKModel(nn.Module):
    """A minimal model whose targets are produced by a weight converter."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Parameter(torch.zeros(2, 2))
        self.k_proj = nn.Parameter(torch.zeros(2, 2))


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_load_pretrained_weights_preserves_parameter_identity(tmp_path, monkeypatch) -> None:
    """
    Feature: Identity pretrained weight loading.
    Description: Load an unconverted safetensors tensor into an existing parameter.
    Expectation: Values are copied in place and the parameter object is not replaced.
    """
    expected = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    save_file({"proj.weight": expected}, str(tmp_path / "model.safetensors"))
    model = _SingleWeightModel()
    parameter = model.proj.weight
    monkeypatch.setattr(checkpoint_loader, "get_model_conversion_mapping", lambda *_, **__: [])

    report = checkpoint_loader.load_pretrained_weights(model, str(tmp_path))

    assert model.proj.weight is parameter
    assert torch.equal(model.proj.weight, expected)
    assert report.loaded_keys == ("proj.weight",)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_load_pretrained_weights_applies_renaming(tmp_path, monkeypatch) -> None:
    """
    Feature: Transformers weight renaming.
    Description: Rename a legacy checkpoint key before loading.
    Expectation: The renamed tensor is copied into the matching model parameter.
    """
    expected = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    save_file({"legacy.weight": expected}, str(tmp_path / "model.safetensors"))
    model = _SingleWeightModel()
    mapping = [WeightRenaming("legacy.weight", "proj.weight")]
    monkeypatch.setattr(
        checkpoint_loader,
        "get_model_conversion_mapping",
        lambda *_, **__: mapping,
    )

    checkpoint_loader.load_pretrained_weights(model, str(tmp_path))

    assert torch.equal(model.proj.weight, expected)


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_yaml_replacement_weight_mapping_is_consumed_by_checkpoint_loader(
    tmp_path,
    monkeypatch,
) -> None:
    """
    Feature: YAML replacement checkpoint mapping.
    Description: Resolve the example replacement and load its legacy checkpoint key.
    Expectation: The YAML-derived mapping reaches checkpoint loading without a second lookup.
    """
    examples_dir = Path(__file__).resolve().parents[6] / "examples" / "distributed"
    monkeypatch.syspath_prepend(str(examples_dir))
    perf_kernels = importlib.import_module("perf_kernels")
    perf_replacement = importlib.import_module("perf_replacement")
    with (examples_dir / "perf_replacement.yaml").open(encoding="utf-8") as yaml_file:
        raw = yaml.safe_load(yaml_file)
    entries = resolve_component(
        raw["plan_overrides"],
        expected_type=List[PlanOverride],
        path="plan_overrides",
    )
    specs = entries_to_module_replacements(entries)
    assert len(specs) == 1

    model = perf_replacement.TinyModel(vocab=8, h=4, n_heads=1, n_layers=1)
    model.config = SimpleNamespace()
    target = model.model.layers[0].mlp.down_proj.weight
    expected = torch.arange(target.numel(), dtype=target.dtype).reshape_as(target)
    save_file(
        {"model.layers.0.mlp.down_proj.weight": expected},
        str(tmp_path / "model.safetensors"),
    )
    weights_mapping = []

    model, weights_mapping = infrastructure_module._apply_module_replacement_actions(
        model,
        SimpleNamespace(plan_overrides=entries),
        weights_mapping,
    )
    mapping_lookup = Mock(side_effect=AssertionError("mapping must not be recomputed"))
    monkeypatch.setattr(checkpoint_loader, "get_model_conversion_mapping", mapping_lookup)

    report = checkpoint_loader.CheckpointManager(model).load_checkpoint(
        str(tmp_path),
        strict=False,
        weights_mapping=weights_mapping,
    )

    assert isinstance(
        model.model.layers[0].mlp.down_proj,
        perf_kernels.CheckpointMappedLinear,
    )
    assert len(weights_mapping) == 1
    assert weights_mapping[0].scope_prefix == "model.layers.0.mlp.down_proj"
    assert torch.equal(model.model.layers[0].mlp.down_proj.packed_weight, expected)
    inputs = torch.arange(32, dtype=expected.dtype).reshape(2, 16)
    torch.testing.assert_close(
        model.model.layers[0].mlp.down_proj(inputs),
        torch.nn.functional.linear(inputs, expected),
    )
    assert "weight" not in model.model.layers[0].mlp.down_proj._parameters
    assert "model.layers.0.mlp.down_proj.packed_weight" in report.loaded_keys
    mapping_lookup.assert_not_called()


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_qwen3_moe_replacement_converts_raw_expert_checkpoint(tmp_path) -> None:
    """Qwen's generic expert merge must feed the replacement's layout transpose."""
    config = Qwen3MoeConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_experts=2,
        num_experts_per_tok=1,
    )
    model = Qwen3MoeForCausalLM(config)
    raw_entry = {
        "match": "*.mlp",
        "module_type": (
            "transformers.models.qwen3_moe.modeling_qwen3_moe."
            "Qwen3MoeSparseMoeBlock"
        ),
        "replace_module": {
            "_target_": "examples.distributed.perf_kernels.NpuGroupedMoe",
        },
    }
    entries = resolve_component(
        [raw_entry],
        expected_type=List[PlanOverride],
        path="plan_overrides",
    )
    weights_mapping = get_model_conversion_mapping(model)
    model, weights_mapping = infrastructure_module._apply_module_replacement_actions(
        model,
        SimpleNamespace(plan_overrides=entries),
        weights_mapping,
    )
    gate_up_mapping = next(
        transform
        for transform in weights_mapping
        if isinstance(transform, WeightConverter)
        and transform.scope_prefix == "model.layers.0.mlp"
        and transform.target_patterns == ["experts.gate_up_proj"]
    )
    down_mapping = next(
        transform
        for transform in weights_mapping
        if isinstance(transform, WeightConverter)
        and transform.scope_prefix == "model.layers.0.mlp"
        and transform.target_patterns == ["experts.down_proj"]
    )
    assert gate_up_mapping.source_patterns == [
        "experts.*.gate_proj.weight",
        "experts.*.up_proj.weight",
    ]
    assert down_mapping.source_patterns == ["experts.*.down_proj.weight"]
    assert [type(operation).__name__ for operation in gate_up_mapping.operations] == [
        "MergeModulelist",
        "Concatenate",
        "Transpose",
    ]
    assert [type(operation).__name__ for operation in down_mapping.operations] == [
        "MergeModulelist",
        "Transpose",
    ]

    gate_weights = []
    up_weights = []
    down_weights = []
    checkpoint = {}
    for expert in range(config.num_experts):
        gate = torch.arange(32, dtype=torch.float32).reshape(4, 8) + expert * 100
        up = gate + 1000
        down = torch.arange(32, dtype=torch.float32).reshape(8, 4) + expert * 2000
        gate_weights.append(gate)
        up_weights.append(up)
        down_weights.append(down)
        prefix = f"model.layers.0.mlp.experts.{expert}"
        checkpoint[f"{prefix}.gate_proj.weight"] = gate
        checkpoint[f"{prefix}.up_proj.weight"] = up
        checkpoint[f"{prefix}.down_proj.weight"] = down
    save_file(checkpoint, str(tmp_path / "model.safetensors"))

    report = checkpoint_loader.CheckpointManager(model).load_checkpoint(
        str(tmp_path),
        strict=False,
        weights_mapping=weights_mapping,
    )

    expected_gate_up = torch.cat(
        [torch.stack(gate_weights), torch.stack(up_weights)],
        dim=1,
    ).transpose(1, 2).contiguous()
    expected_down = torch.stack(down_weights).transpose(1, 2).contiguous()
    experts = model.model.layers[0].mlp.experts
    torch.testing.assert_close(experts.gate_up_proj, expected_gate_up)
    torch.testing.assert_close(experts.down_proj, expected_down)
    assert "model.layers.0.mlp.experts.gate_up_proj" in report.loaded_keys
    assert "model.layers.0.mlp.experts.down_proj" in report.loaded_keys


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_load_pretrained_weights_applies_converter(tmp_path, monkeypatch) -> None:
    """
    Feature: Transformers weight conversion.
    Description: Split one fused checkpoint tensor into two model parameters.
    Expectation: Both converted tensors are loaded into their matching targets.
    """
    fused = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    save_file({"qk.weight": fused}, str(tmp_path / "model.safetensors"))
    model = _QKModel()
    mapping = [
        WeightConverter(
            source_patterns="qk.weight",
            target_patterns=["q_proj", "k_proj"],
            operations=[Chunk(dim=0)],
        )
    ]
    monkeypatch.setattr(
        checkpoint_loader,
        "get_model_conversion_mapping",
        lambda *_, **__: mapping,
    )

    checkpoint_loader.load_pretrained_weights(model, str(tmp_path))

    assert torch.equal(model.q_proj, fused[:2])
    assert torch.equal(model.k_proj, fused[2:])


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_load_pretrained_weights_rejects_missing_keys(tmp_path, monkeypatch) -> None:
    """
    Feature: Strict checkpoint validation.
    Description: Load an empty safetensors checkpoint into a model with a parameter.
    Expectation: The loader reports the missing model key.
    """
    save_file({}, str(tmp_path / "model.safetensors"))
    model = _SingleWeightModel()
    monkeypatch.setattr(checkpoint_loader, "get_model_conversion_mapping", lambda *_, **__: [])

    with pytest.raises(RuntimeError, match="proj.weight"):
        checkpoint_loader.load_pretrained_weights(model, str(tmp_path))


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_checkpoint_manager_save_pretrained_passes_full_cpu_state(tmp_path) -> None:
    """
    Feature: Transformers-compatible checkpoint export.
    Description: Gather a finalized model state and delegate file writing.
    Expectation: save_pretrained receives full CPU weights and save options.
    """
    model = _SingleWeightModel()
    model.save_pretrained = Mock()
    manager = checkpoint_loader.CheckpointManager(model)

    saved = manager.save_pretrained(tmp_path, max_shard_size="1GB")

    assert saved
    _, kwargs = model.save_pretrained.call_args
    assert torch.equal(kwargs["state_dict"]["proj.weight"], model.proj.weight.cpu())
    assert kwargs["state_dict"]["proj.weight"].device.type == "cpu"
    assert kwargs["is_main_process"]
    assert kwargs["max_shard_size"] == "1GB"
    assert kwargs["save_original_format"]


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_checkpoint_manager_delegates_dcp_save_and_load(tmp_path) -> None:
    """
    Feature: Pluggable DCP backend contract.
    Description: Delegate sharded save/load through an injected backend.
    Expectation: The model namespace and checkpoint ID match the DCP API.
    """
    model = _SingleWeightModel()
    expected = torch.full_like(model.proj.weight, 3)
    backend = Mock()
    backend.save.return_value = "saved"

    def _load(state_dict, *, checkpoint_id, **kwargs):
        assert checkpoint_id == tmp_path
        assert kwargs == {"use_collectives": False}
        state_dict["model"]["proj.weight"] = expected
        return "loaded"

    backend.load.side_effect = _load
    manager = checkpoint_loader.CheckpointManager(model, dcp_backend=backend)

    save_result = manager.save_dcp(tmp_path, use_collectives=False)
    load_result = manager.load_dcp(tmp_path, use_collectives=False)

    assert save_result == "saved"
    assert load_result == "loaded"
    assert torch.equal(model.proj.weight, expected)
    saved_state = backend.save.call_args.args[0]
    assert set(saved_state) == {"model"}
    assert "proj.weight" in saved_state["model"]


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_checkpoint_manager_requires_dcp_backend(tmp_path) -> None:
    """
    Feature: Explicit DCP integration boundary.
    Description: Call a DCP method before a backend has been connected.
    Expectation: The manager identifies the missing integration contract.
    """
    manager = checkpoint_loader.CheckpointManager(_SingleWeightModel())

    with pytest.raises(NotImplementedError, match="DCP backend"):
        manager.save_dcp(tmp_path)
