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
"""Unit tests for HyperParallel LlamaFactory integration utilities.

Tests cover: args, checkpoint_io, optimizer, and utils modules.
Trainer tests live on the LlamaFactory side.
"""

# pylint: disable=wrong-import-position,protected-access
from types import SimpleNamespace

import pytest
import torch

import hyper_parallel.integration.llamafactory.utils as lf_utils
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer
from hyper_parallel.distributed._builder.fsdp_adapter import FSDP2Manager
from hyper_parallel.integration.llamafactory.utils import (
    HyperParallelArguments,
    export_to_hf_format,
    wrap_optimizer_with_skip_dtensor_dispatch,
)
from hyper_parallel.models._transformers import model_builder


class _FakeOptimizer:
    def __init__(self):
        self.calls = []

    def step(self, closure=None):
        self.calls.append(closure)
        return "stepped"


def test_fsdp_block_discovery_preserves_registered_child_names():
    """FSDP source metadata lookup must use the module's real FQN."""

    class _Owner(torch.nn.Module):
        gradient_checkpointing = False

        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleDict(
                {"2": torch.nn.Linear(2, 2), "7": torch.nn.Linear(2, 2)}
            )

    blocks, _ = FSDP2Manager._find_transformer_block_modules(_Owner())

    assert [block.fqn for block in blocks] == ["layers.2", "layers.7"]


def test_parallelize_model_delegates_to_existing_model_infrastructure(monkeypatch):
    """The integration entry should compose existing Hyper model-building APIs."""
    model = torch.nn.Linear(2, 2, device="meta")
    setup = SimpleNamespace(mesh_context=object())
    calls = {}
    monkeypatch.setattr(
        model_builder,
        "instantiate_infrastructure",
        lambda **kwargs: ("planner", "manager"),
    )

    def _apply(input_model, **kwargs):
        calls.update(kwargs)
        return input_model

    monkeypatch.setattr(model_builder, "apply_model_infrastructure", _apply)

    result = lf_utils.parallelize_model(
        model,
        setup,
        pretrained_path="checkpoint",
        device=torch.device("cpu"),
        activation_checkpoint="full",
    )

    assert result is model
    assert calls["mesh"] is setup.mesh_context
    assert calls["sharding_planner"] == "planner"
    assert calls["fsdp2_manager"] == "manager"
    assert calls["pretrained_path"] == "checkpoint"
    assert calls["activation_checkpoint"] == "full"


def test_parallel_args_build_new_distributed_setup_for_cp_ep(monkeypatch):
    """LlamaFactory arguments should feed Hyper's unified CP/EP/FSDP topology."""
    build_calls = []
    monkeypatch.setattr(lf_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(lf_utils.dist, "get_world_size", lambda: 8)
    monkeypatch.setattr(
        lf_utils.MeshContext,
        "build_meshs",
        lambda self, device_type, world_size: build_calls.append((self, device_type, world_size)),
    )
    parallel_args = HyperParallelArguments(
        cp_size=2,
        ep_size=4,
        fsdp_size=4,
        efsdp_size=2,
        device_type="npu",
        param_dtype="bf16",
        reduce_dtype="fp32",
        reshard_after_forward=False,
        plan_overrides=[
            {
                "match": "*.mlp.forward",
                "when": "ep",
                "replace": {
                    "kind": "method",
                    "implementation": (
                        "hyper_parallel.distributed.expert_parallel.recipes."
                        "routed_only_ep_compute_fn"
                    ),
                },
            }
        ],
    )

    distributed_setup = parallel_args.build_distributed_setup()

    mesh = distributed_setup.mesh_context
    assert (mesh.dp_size, mesh.cp_size, mesh.ep_size) == (4, 2, 4)
    assert (mesh.dp_replicate_size, mesh.dp_shard_size, mesh.edp_shard_size) == (2, 4, 2)
    assert build_calls == [(mesh, "npu", 8)]
    cp_spec = distributed_setup.plan_overrides["*.self_attn"]
    assert cp_spec.inner_target == "self"
    assert cp_spec.inner_wrapper.callable.__name__ == "sdpa_hf_cp_wrapper"
    assert cp_spec.region_dispatch is False
    ep_spec = distributed_setup.plan_overrides["*.mlp"]
    assert ep_spec.local_compute_fn.callable.__name__ == "routed_only_ep_compute_fn"
    assert ep_spec.region_dispatch is False
    strategy = distributed_setup.strategy_config
    assert strategy.dp_shard_size == 4
    assert strategy.edp_shard_size == 2
    assert strategy.mix_precision.param_dtype == "bfloat16"
    assert strategy.mix_precision.reduce_dtype == "float32"
    assert strategy.mix_precision.output_dtype == "bfloat16"
    assert strategy.reshard_after_forward is False


def test_parallel_args_do_not_enable_model_specific_replacements_by_default(monkeypatch):
    """LlamaFactory defaults should not silently enable performance modules."""
    monkeypatch.setattr(lf_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(lf_utils.dist, "get_world_size", lambda: 8)
    monkeypatch.setattr(lf_utils.MeshContext, "build_meshs", lambda *args, **kwargs: None)

    setup = HyperParallelArguments(cp_size=2, device_type="npu").build_distributed_setup()

    cp_spec = setup.plan_overrides["*.self_attn"]
    assert cp_spec.inner_wrapper.callable.__name__ == "sdpa_hf_cp_wrapper"
    assert not setup.module_replacements


def test_wrap_optimizer_step_uses_skip_dtensor_dispatch(monkeypatch):
    """
    Feature: Optimizer step wrapper
    Description: Optimizer.step should run under SkipDTensorDispatch and remain method-shaped.
    Expectation: Wrapped step calls the original step and still exposes __func__.
    """
    enter_exit = []

    class _FakeSkip:
        def __enter__(self):
            enter_exit.append("enter")

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            enter_exit.append("exit")

    optimizer = _FakeOptimizer()
    monkeypatch.setattr(lf_utils, "SkipDTensorDispatch", _FakeSkip)

    wrap_optimizer_with_skip_dtensor_dispatch(optimizer)

    assert hasattr(optimizer.step, "__func__")
    result = optimizer.step("closure")

    assert result == "stepped"
    assert optimizer.calls == ["closure"]
    assert enter_exit == ["enter", "exit"]


def test_wrap_chained_optimizer_wraps_leaf_steps(monkeypatch):
    """DCP may call a leaf optimizer directly while initializing its state dict."""
    enter_exit = []

    class _FakeSkip:
        def __enter__(self):
            enter_exit.append("enter")

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            enter_exit.append("exit")

    leaf_optimizer = _FakeOptimizer()
    chained_optimizer = _FakeOptimizer()
    chained_optimizer.optimizers_dict = {"leaf": leaf_optimizer}
    monkeypatch.setattr(lf_utils, "SkipDTensorDispatch", _FakeSkip)

    wrap_optimizer_with_skip_dtensor_dispatch(chained_optimizer)
    leaf_optimizer.step()
    chained_optimizer.step()

    assert leaf_optimizer.calls == [None]
    assert chained_optimizer.calls == [None]
    assert enter_exit == ["enter", "exit", "enter", "exit"]


def test_load_hsdp_checkpoint_delegates_chained_optimizer_state(monkeypatch, tmp_path):
    """Chained optimizer checkpoints should use their FQN-aware load_state_dict implementation."""
    model = torch.nn.Linear(2, 2)
    chained_optimizer = ChainedOptimizer(
        model,
        {"adamw": torch.optim.AdamW(model.parameters(), lr=1.0e-3)},
    )
    saved_state = {"state.weight.exp_avg": torch.ones_like(model.weight)}
    torch.save(saved_state, tmp_path / "optimizer_rank0.pt")
    loaded_states = []
    monkeypatch.setattr(lf_utils.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(chained_optimizer, "load_state_dict", loaded_states.append)

    lf_utils.load_hsdp_optimizer_and_scheduler(chained_optimizer, None, str(tmp_path))

    assert len(loaded_states) == 1
    assert torch.equal(loaded_states[0]["state.weight.exp_avg"], saved_state["state.weight.exp_avg"])


def test_export_to_hf_format_uses_hf_default_shard_size(monkeypatch, tmp_path):
    """
    Feature: Final HF export
    Description: Export should delegate shard sizing to HuggingFace defaults instead of forcing a custom limit.
    Expectation: save_pretrained and tokenizer.save_pretrained are called without max_shard_size.
    """
    captured = {}

    class _FakeModel:
        def save_pretrained(self, save_dir, state_dict=None, max_shard_size=None):
            captured["save_dir"] = save_dir
            captured["state_dict"] = state_dict
            captured["model_max_shard_size"] = max_shard_size

    class _FakeTokenizer:
        def save_pretrained(self, save_dir, max_shard_size=None):
            captured["tokenizer_dir"] = save_dir
            captured["tokenizer_max_shard_size"] = max_shard_size

    monkeypatch.setattr(lf_utils.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(lf_utils.dist, "get_world_size", lambda: 1)

    def _fake_get_model_state_dict(model, options=None):
        del model, options
        return {"weight": torch.ones(4, dtype=torch.bfloat16)}

    monkeypatch.setattr(
        "hyper_parallel.core.fully_shard.api.get_model_state_dict",
        _fake_get_model_state_dict,
    )

    export_to_hf_format(_FakeModel(), _FakeTokenizer(), str(tmp_path))

    assert captured["save_dir"] == str(tmp_path)
    assert captured["tokenizer_dir"] == str(tmp_path)
    # Export preserves the gathered state-dict dtype; no fp32 cast at save time.
    assert captured["state_dict"]["weight"].dtype == torch.bfloat16
    assert captured["model_max_shard_size"] is None
    assert captured["tokenizer_max_shard_size"] is None


# ---------------------------------------------------------------------------
# Tests: fsdp_size validation
# ---------------------------------------------------------------------------


def test_parallel_args_fsdp_size_defaults_to_none():
    """
    Feature: fsdp_size default
    Description: fsdp_size is optional; when not provided it stays None and validate passes.
    Expectation: HyperParallelArguments() has fsdp_size=None and validates cleanly.
    """
    args = HyperParallelArguments()
    args.validate()
    assert args.fsdp_size is None, f"Expected fsdp_size=None, got {args.fsdp_size!r}"


def test_parallel_args_fsdp_size_accepts_positive_int():
    """
    Feature: fsdp_size validation accepts positive int
    Description: A positive integer fsdp_size must pass validation.
    Expectation: validate() does not raise for fsdp_size=4.
    """
    args = HyperParallelArguments(fsdp_size=4)
    args.validate()
    assert args.fsdp_size == 4, f"Expected fsdp_size=4, got {args.fsdp_size!r}"


def test_parallel_args_fsdp_size_rejects_zero_and_negative():
    """
    Feature: fsdp_size validation rejects non-positive
    Description: 0 and negative values must raise during validation.
    Expectation: ValueError mentioning fsdp_size for both 0 and -1.
    """
    for bad in (0, -1, -8):
        args = HyperParallelArguments(fsdp_size=bad)
        with pytest.raises(ValueError, match="fsdp_size"):
            args.validate()


def test_parallel_args_fsdp_size_rejects_non_int():
    """
    Feature: fsdp_size validation rejects non-int
    Description: Non-int values (including bool, which Python treats as int subtype) must raise.
    Expectation: ValueError mentioning fsdp_size for string, float, bool inputs.
    """
    for bad in ("4", 4.0, True):
        args = HyperParallelArguments(fsdp_size=bad)
        with pytest.raises(ValueError, match="fsdp_size"):
            args.validate()
