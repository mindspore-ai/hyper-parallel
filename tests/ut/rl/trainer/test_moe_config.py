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
from types import SimpleNamespace

import pytest
import torch
import yaml

import rl

from rl.algorithm import build_algorithm
from rl.config.schema import build_model_registration
from rl.config.resolve import build_runtime_config, resolve_router_replay_config, resolve_unified_execution_config
from rl.config.limits import resolve_vllm_automatic_limits
from rl.config.validate import validate_config
from rl.rollout.vllm import VLLMGenerationEngine
from rl.workers.engine.model import resolve_vllm_model


@pytest.fixture(name="moe_config")
def make_moe_config(tmp_path: Path) -> dict:
    """Resolve the production example against a tiny local checkpoint identity."""
    recipe = Path(rl.__file__).resolve().parent.parent / "examples/gsm8k/configs/qwen3_30b_a3b_gsm8k_vllm.yaml"
    config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    (tmp_path / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe",
        "num_experts": 8, "num_experts_per_tok": 2, "hidden_size": 16, "num_attention_heads": 4,
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
    experts, = (entry for entry in runtime.plan_overrides if entry.match == "*.mlp.experts")
    assert experts.when is None
    assert experts.replace.kind == "module"
    assert experts.replace.implementation.to_dict()["_target_"] == "hyper_parallel.components.modules.GroupedExperts"
    resolved = resolve_vllm_automatic_limits(moe_config)
    assert resolved["rollout"]["vllm"]["max_num_seqs"] > 0


@pytest.mark.parametrize("implementation", ["native", "unified"])
def test_16npu_moe_recipe_separates_update_and_rollout_residency(moe_config, tmp_path, monkeypatch, implementation):
    """Exercise the colocated validation gate and both roles' resolved FSDP policies."""
    recipes = Path(rl.__file__).resolve().parent.parent / "examples/gsm8k/configs"
    recipe = yaml.safe_load((recipes / f"qwen3_30b_a3b_gsm8k_{implementation}_16npu.yaml").read_text())
    recipe["model"].update(weights_path=str(tmp_path), tokenizer_path=str(tmp_path))
    for split, field in (("train", "train_path"), ("test", "test_path")):
        path = tmp_path / f"{split}.parquet"
        path.touch()
        recipe["data"][field] = str(path)
    # Use the fixture checkpoint schema with enough experts for the production EP group.
    checkpoint = Path(moe_config["model"]["weights_path"]) / "config.json"
    metadata = json.loads(checkpoint.read_text())
    metadata["num_experts"] = 128
    checkpoint.write_text(json.dumps(metadata))
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: torch.device("npu"))

    config = resolve_vllm_automatic_limits(resolve_unified_execution_config(recipe))
    validate_config(config, build_algorithm(config["algorithm"]))
    assert config["train"]["accelerator"]["offload_during_rollout"] is True
    assert config["rollout"]["vllm"]["gpu_memory_utilization"] == 0.8
    assert config["rollout"]["vllm"].get("kv_cache_memory_bytes") is None
    assert config["rollout"]["vllm"]["max_num_seqs"] == 128
    for reference in (False, True):
        runtime = build_runtime_config(config, reference=reference)
        assert not runtime.fsdp_config.enable_offload
        assert runtime.optimizer.fp32_main_params
        assert runtime.accelerator.ep_size == 8


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
    (("consistency", "enabled"), True, "consistency requires schema_version"),
])
def test_moe_rejects_unsupported_runtime(moe_config: dict, path: tuple, value: object, error: str) -> None:
    """Unsupported combinations fail before constructing models or process groups."""
    section = moe_config
    for key in path[:-1]:
        section = section.setdefault(key, {})
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


@pytest.mark.parametrize("ascend_version", ["0.23.0.post1", "0.30.0"])
def test_r3_rollout_dp_and_graphs(moe_config: dict, monkeypatch, ascend_version: str) -> None:
    """Only the V2 capture path supports this graph-mode, multi-DP recipe."""
    monkeypatch.setattr("rl.rollout.router_capture.version", lambda _: ascend_version)
    moe_config["train"]["router_replay"] = {"mode": "R3"}
    moe_config["rollout"]["vllm"]["enforce_eager"] = False
    if ascend_version == "0.23.0.post1":
        with pytest.raises(ValueError, match="data_parallel_size=1.*DP padding"):
            build_runtime_config(moe_config)
    else:
        resolved = resolve_router_replay_config(moe_config)
        assert resolved["rollout"]["vllm"]["enable_return_routed_experts"]
        assert build_runtime_config(resolved).accelerator.ep_size == 2
        engine = VLLMGenerationEngine(model=build_model_registration(resolved), config=resolved["rollout"])
        command = engine._server_command("127.0.0.1", 8100)
        assert "--enable-return-routed-experts" in command
        assert "--prefill-context-parallel-size" not in command
        assert engine._server_environment("0,1,2,3")["VLLM_USE_V2_MODEL_RUNNER"] == "1"


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
    monkeypatch.setattr(
        torch.accelerator,
        "current_accelerator",
        lambda: SimpleNamespace(type="npu"),
    )
    moe_config["train"].update(router_replay={"mode": "R3"}, model_init_cpu_threads=8)
    moe_config["rollout"]["vllm"].update(tensor_parallel_size=4, data_parallel_size=1, max_model_len=8192)
    moe_config["agentic"].update(max_turns=3, interaction_mode="multi_turn")
    validate_config(resolve_vllm_automatic_limits(moe_config), build_algorithm(moe_config["algorithm"]))

    recipe = Path(rl.__file__).resolve().parent.parent / "examples/gsm8k/configs/qwen3_30b_a3b_codex_multi_turn.yaml"
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
