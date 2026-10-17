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
"""Contracts for native Qwen3-MoE configuration and recipe integration."""

import json
from pathlib import Path

import pytest
import torch
import yaml

import rl

from rl.algorithm import build_algorithm
from rl.config import (
    build_model_registration,
    build_runtime_config,
    resolve_router_replay_config,
    resolve_vllm_automatic_limits,
    validate_config,
)
from rl.roles.model_setup import resolve_vllm_model


@pytest.fixture(name="moe_config")
def make_moe_config(tmp_path: Path) -> dict:
    """Resolve the production example against a tiny local checkpoint identity."""
    recipe = Path(rl.__file__).resolve().parent.parent / "examples/gsm8k/configs/qwen3_30b_a3b_gsm8k_vllm.yaml"
    config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    (tmp_path / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe",
        "num_experts": 8, "hidden_size": 16, "num_attention_heads": 4,
        "num_key_value_heads": 4, "num_hidden_layers": 2, "head_dim": 4,
    }), encoding="utf-8")
    config["model"].update(weights_path=str(tmp_path), tokenizer_path=str(tmp_path))
    return config


def test_moe_runtime_uses_public_recipe_and_native_rollout(moe_config: dict) -> None:
    """EP/EDP and model replacement rules survive RL configuration conversion."""
    registration = build_model_registration(moe_config)
    assert resolve_vllm_model(registration, "native").architecture == "Qwen3MoeForCausalLM"
    runtime = build_runtime_config(moe_config)
    target = runtime.model.to_dict()
    assert target["_target_"] == "hyper_parallel.models.HyperAutoModelForCausalLM.from_pretrained"
    assert "fused" not in target
    assert runtime.accelerator.ep_size == 2
    assert runtime.fsdp_config.edp_shard_size == 2
    assert any(entry.when == "ep" for entry in runtime.plan_overrides)
    assert any(entry.replace_module is not None for entry in runtime.plan_overrides)
    resolved = resolve_vllm_automatic_limits(moe_config)
    assert resolved["rollout"]["vllm"]["max_num_seqs"] > 0


@pytest.mark.parametrize(("path", "value", "error"), [
    (("algorithm", "name"), "ppo", "GRPO only"),
    (("rollout", "vllm", "deployment"), "disjoint", "colocated native"),
    (("rollout", "vllm", "model_implementation"), "hyper", "colocated native"),
    (("rollout", "vllm", "enable_eplb"), True, "enable_eplb"),
    (("rollout", "vllm", "enable_expert_parallel"), "true", "boolean"),
    (("train", "accelerator", "ep"), True, "positive integer"),
    (("train", "accelerator", "ep"), 3, "divide training world_size"),
    (("train", "accelerator", "ep"), 1, "edp_shard requires ep"),
    (("train", "accelerator", "edp_shard"), 0, "positive integer"),
    (("consistency", "enabled"), True, "consistency off"),
])
def test_moe_rejects_unsupported_runtime(moe_config: dict, path: tuple, value: object, error: str) -> None:
    """Unsupported combinations fail before constructing models or process groups."""
    section = moe_config
    for key in path[:-1]:
        section = section[key]
    section[path[-1]] = value
    with pytest.raises(ValueError, match=error):
        build_runtime_config(moe_config)


def test_moe_rejects_nondivisible_expert_count(moe_config: dict) -> None:
    """An otherwise valid mesh cannot split an uneven expert count."""
    path = Path(moe_config["model"]["weights_path"]) / "config.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["num_experts"] = 7
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="divide num_experts"):
        build_runtime_config(moe_config)


def test_r3_derives_capture_without_changing_disabled_recipe(moe_config: dict) -> None:
    """One switch enables capture and full-sequence replay while preserving disabled defaults."""
    assert resolve_router_replay_config(moe_config) == moe_config
    moe_config["train"]["router_replay"] = {"mode": "R3"}
    moe_config["rollout"]["vllm"].update(tensor_parallel_size=4, data_parallel_size=1)
    resolved = resolve_router_replay_config(moe_config)
    assert resolved["rollout"]["vllm"]["enable_return_routed_experts"] is True
    assert "enable_return_routed_experts" not in moe_config["rollout"]["vllm"]
    assert build_runtime_config(resolved).accelerator.sequence_parallel is False
    resolved["rollout"]["vllm"]["enable_return_routed_experts"] = False
    with pytest.raises(ValueError, match="must match"):
        resolve_router_replay_config(resolved)


def test_r3_rejects_rollout_dp_padding(moe_config: dict) -> None:
    """Captured routes require one inference DP engine before model allocation."""
    moe_config["train"]["router_replay"] = {"mode": "R3"}
    with pytest.raises(ValueError, match="data_parallel_size=1.*DP padding"):
        build_runtime_config(moe_config)


def test_r3_rejects_enabled_distillation(moe_config: dict) -> None:
    """Replay remains separate from the independently supported OPD objective."""
    moe_config["train"]["router_replay"] = {"mode": "R3"}
    moe_config["rollout"]["vllm"].update(tensor_parallel_size=4, data_parallel_size=1)
    moe_config["distillation"] = {"enabled": True, "teachers": {"teacher": {
        "model_path": moe_config["model"]["weights_path"], "endpoint": "http://localhost:8000",
        "max_model_len": 8192,
    }}}
    with pytest.raises(ValueError, match="R3 does not support enabled distillation"):
        build_runtime_config(moe_config)


def test_r3_supports_internal_and_codex_multiturn(
    moe_config: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-turn replay keeps native MoE constraints and excludes unadapted harnesses."""
    train_path = tmp_path / "train.parquet"
    train_path.touch()
    moe_config["data"]["train_path"] = str(train_path)
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: torch.device("npu"))
    moe_config["train"].update(router_replay={"mode": "R3"}, model_init_cpu_threads=8)
    moe_config["rollout"]["vllm"].update(tensor_parallel_size=4, data_parallel_size=1, max_model_len=8192)
    moe_config["agentic"].update(max_turns=3, interaction_mode="multi_turn")
    validate_config(resolve_vllm_automatic_limits(moe_config), build_algorithm(moe_config["algorithm"]))

    recipe = Path(rl.__file__).resolve().parent.parent / "examples/gsm8k/configs/codex_multi_turn.yaml"
    moe_config["agentic"] = yaml.safe_load(recipe.read_text(encoding="utf-8"))["agentic"]
    moe_config["rollout"]["vllm"].update(enable_auto_tool_choice=True, tool_call_parser="hermes")
    resolved = resolve_vllm_automatic_limits(resolve_router_replay_config(moe_config))
    validate_config(resolved, build_algorithm(resolved["algorithm"]))
    assert resolved["rollout"]["vllm"]["enable_return_routed_experts"] is True
    assert resolved["train"]["model_init_cpu_threads"] == 8
    assert resolved["agentic"] == moe_config["agentic"], "R3 must not change the harness or its task budget"
    assert resolved["rollout"]["max_new_tokens"] == moe_config["rollout"]["max_new_tokens"]

    resolved["agentic"]["runner"] = "deepseek"
    with pytest.raises(ValueError, match="only internal and Codex"):
        build_runtime_config(resolved)


def test_ppo_rejects_moe_critic_override(moe_config: dict, tmp_path: Path) -> None:
    """Registering MoE actors must not accidentally permit an MoE PPO Critic."""
    moe_path = moe_config["model"]["weights_path"]
    dense_path = tmp_path / "dense"
    dense_path.mkdir()
    (dense_path / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
    }), encoding="utf-8")
    moe_config["model"]["weights_path"] = str(dense_path)
    moe_config["train"]["critic"] = {"weights_path": moe_path}
    moe_config["algorithm"]["name"] = "ppo"
    with pytest.raises(ValueError, match="Critic requires a dense Qwen3 checkpoint"):
        validate_config(moe_config, build_algorithm(moe_config["algorithm"]))
