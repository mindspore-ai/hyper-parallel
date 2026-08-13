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
"""Contract tests for the shared Qwen3.5 tensor-parallel policy."""

import os
from types import SimpleNamespace

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import pytest
import torch

from hyper_parallel import ColwiseParallel, PrepareModuleInput, RowwiseParallel, SequenceParallel
from hyper_parallel.models.qwen3_5 import parallelize as qwen3_5_parallelize
from hyper_parallel.models.qwen3_5.parallelize import (
    _TP_PROFILE_INFERENCE_REPLICATED,
    _TP_PROFILE_TRAINING_SP,
    _build_qwen3_5_tp_plans,
    qwen3_5_inference_tp_load_transforms,
    qwen3_5_tp_load_transforms,
)


_COLWISE_PATHS = (
    ("norm_and_mlp", "mlp.gate_proj"),
    ("norm_and_mlp", "mlp.up_proj"),
    ("full_attention", "self_attn.q_proj"),
    ("full_attention", "self_attn.k_proj"),
    ("full_attention", "self_attn.v_proj"),
    ("linear_attention", "linear_attn.in_proj_z"),
    ("linear_attention", "linear_attn.in_proj_b"),
    ("linear_attention", "linear_attn.in_proj_a"),
)
_ROWWISE_PATHS = (
    ("norm_and_mlp", "mlp.down_proj"),
    ("full_attention", "self_attn.o_proj"),
    ("linear_attention", "linear_attn.out_proj"),
)


def _style(plans: SimpleNamespace, group: str, path: str):
    """Return one style from a grouped TP plan."""
    return getattr(plans, group)[path]


def test_qwen3_5_tp_profiles_share_parameter_sharding_policy() -> None:
    """Training and inference must use the same shard direction for every weight."""
    training = _build_qwen3_5_tp_plans(_TP_PROFILE_TRAINING_SP)
    inference = _build_qwen3_5_tp_plans(_TP_PROFILE_INFERENCE_REPLICATED)

    assert isinstance(training.root["model.embed_tokens"], RowwiseParallel)
    assert isinstance(inference.root["model.embed_tokens"], RowwiseParallel)
    assert isinstance(training.root["lm_head"], ColwiseParallel)
    assert isinstance(inference.root["lm_head"], ColwiseParallel)
    for group, path in _COLWISE_PATHS:
        assert isinstance(_style(training, group, path), ColwiseParallel)
        assert isinstance(_style(inference, group, path), ColwiseParallel)
    for group, path in _ROWWISE_PATHS:
        assert isinstance(_style(training, group, path), RowwiseParallel)
        assert isinstance(_style(inference, group, path), RowwiseParallel)


def test_qwen3_5_tp_profiles_keep_activation_contracts_separate() -> None:
    """SP-only hooks and layouts must not leak into packed inference."""
    training = _build_qwen3_5_tp_plans(_TP_PROFILE_TRAINING_SP)
    inference = _build_qwen3_5_tp_plans(_TP_PROFILE_INFERENCE_REPLICATED)

    assert training.root["model.embed_tokens"].output_layouts[0].is_shard(1)
    assert inference.root["model.embed_tokens"].output_layouts[0].is_replicate()
    assert isinstance(training.root["model.norm"], SequenceParallel)
    assert "model.norm" not in inference.root
    assert isinstance(training.norm_and_mlp["mlp"], PrepareModuleInput)
    assert "mlp" not in inference.norm_and_mlp
    assert isinstance(training.full_attention["self_attn.q_norm"], SequenceParallel)
    assert "self_attn.q_norm" not in inference.full_attention
    assert training.full_attention["self_attn.o_proj"].reduce_dtype is torch.float32
    assert inference.full_attention["self_attn.o_proj"].reduce_dtype is None


def test_qwen3_5_training_tp_plan_preserves_application_order() -> None:
    """Refactors must retain the established training hook-registration order."""
    plans = _build_qwen3_5_tp_plans(_TP_PROFILE_TRAINING_SP)

    assert list(plans.root) == ["model.embed_tokens", "model.norm", "lm_head"]
    assert list(plans.norm_and_mlp) == [
        "input_layernorm",
        "post_attention_layernorm",
        "mlp",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]
    assert list(plans.full_attention) == [
        "self_attn",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.q_norm",
        "self_attn.k_norm",
        "self_attn.o_proj",
    ]
    assert list(plans.linear_attention) == [
        "linear_attn",
        "linear_attn.in_proj_z",
        "linear_attn.in_proj_b",
        "linear_attn.in_proj_a",
        "linear_attn.out_proj",
    ]


def test_qwen3_5_tp_plan_rejects_unknown_activation_profile() -> None:
    """An unknown activation profile must fail before any module mutation."""
    with pytest.raises(ValueError, match="Unknown Qwen3.5 TP activation profile"):
        _build_qwen3_5_tp_plans("unknown")


def test_qwen3_5_tp_wrappers_select_only_their_activation_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public training and inference contracts must remain profile-specific."""
    calls = []

    def record_apply(model, tp_mesh, **kwargs):
        calls.append((model, tp_mesh, kwargs))
        return model

    monkeypatch.setattr(qwen3_5_parallelize, "_apply_qwen3_5_tp", record_apply)
    training_model = object()
    inference_model = object()
    mesh = object()

    assert qwen3_5_parallelize.parallelize_qwen3_5_tp(
        training_model,
        mesh,
        enable_loss_parallel=True,
        register_grad_hooks=False,
    ) is training_model
    assert qwen3_5_parallelize.parallelize_qwen3_5_inference_tp(inference_model, mesh) is inference_model
    assert calls == [
        (
            training_model,
            mesh,
            {
                "activation_profile": _TP_PROFILE_TRAINING_SP,
                "enable_loss_parallel": True,
                "register_grad_hooks": False,
            },
        ),
        (
            inference_model,
            mesh,
            {
                "activation_profile": _TP_PROFILE_INFERENCE_REPLICATED,
                "enable_loss_parallel": False,
                "register_grad_hooks": False,
            },
        ),
    ]


@pytest.mark.parametrize(
    "profile,expected_src_data_rank,expected_common,expected_linear",
    [
        (_TP_PROFILE_TRAINING_SP, "default", "common-reduce", "linear-reduce"),
        (_TP_PROFILE_INFERENCE_REPLICATED, None, "common", "linear"),
    ],
)
def test_apply_qwen3_5_tp_preserves_profile_specific_runtime_contract(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    expected_src_data_rank: str | None,
    expected_common: str,
    expected_linear: str,
) -> None:
    """The shared apply path must preserve source ranks, reductions, and training state."""
    plans = SimpleNamespace(
        root={"root": "root"},
        norm_and_mlp={"common": "common"},
        norm_and_mlp_reduce={"common": "common-reduce"},
        full_attention={"attention": "full"},
        linear_attention={"linear": "linear"},
        linear_attention_reduce={"linear": "linear-reduce"},
    )
    full_attention = SimpleNamespace(layer_type="full_attention")
    linear_attention = SimpleNamespace(layer_type="linear_attention", linear_attn=object())
    model = SimpleNamespace(
        config=SimpleNamespace(tie_word_embeddings=False),
        layers=[full_attention, linear_attention],
    )
    mesh = SimpleNamespace(size=lambda: 2)
    parallelize_calls = []
    shard_calls = []
    grad_hook_calls = []

    def record_parallelize(module, tp_mesh, plan, **kwargs):
        parallelize_calls.append((module, tp_mesh, plan, kwargs))

    monkeypatch.setattr(qwen3_5_parallelize, "_validate_qwen3_5_tp_config", lambda *_: None)
    monkeypatch.setattr(qwen3_5_parallelize, "_validate_qwen3_5_inference_tp_config", lambda *_: None)
    monkeypatch.setattr(qwen3_5_parallelize, "_build_qwen3_5_tp_plans", lambda *_, **__: plans)
    monkeypatch.setattr(qwen3_5_parallelize, "parallelize_module", record_parallelize)
    monkeypatch.setattr(
        qwen3_5_parallelize,
        "_shard_gated_delta_local_params",
        lambda *args: shard_calls.append(args),
    )
    monkeypatch.setattr(
        qwen3_5_parallelize,
        "_register_tp_replicated_param_grad_sum",
        lambda *args: grad_hook_calls.append(args),
    )

    result = qwen3_5_parallelize._apply_qwen3_5_tp(
        model,
        mesh,
        activation_profile=profile,
        enable_loss_parallel=False,
        register_grad_hooks=True,
    )

    expected_kwargs = {} if expected_src_data_rank == "default" else {"src_data_rank": None}
    assert result is model
    assert parallelize_calls == [
        (model, mesh, plans.root, expected_kwargs),
        (full_attention, mesh, {"common": expected_common, "attention": "full"}, expected_kwargs),
        (linear_attention, mesh, {"common": "common", "linear": expected_linear}, expected_kwargs),
    ]
    assert shard_calls == [(linear_attention.linear_attn, mesh)]
    if profile == _TP_PROFILE_TRAINING_SP:
        assert grad_hook_calls == [(model, mesh)]
        assert model.hp_loss_tp_scale_size == 2
    else:
        assert grad_hook_calls == []
        assert not hasattr(model, "hp_loss_tp_scale_size")


def test_apply_qwen3_5_inference_tp_one_is_mutation_free(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inference TP1 must return before building or applying a parallel plan."""
    model = SimpleNamespace()
    mesh = SimpleNamespace(size=lambda: 1)
    monkeypatch.setattr(qwen3_5_parallelize, "_validate_qwen3_5_inference_tp_config", lambda *_: None)

    def fail(*args, **kwargs):
        del args, kwargs
        raise AssertionError("TP1 inference must not build or apply a TP plan")

    monkeypatch.setattr(qwen3_5_parallelize, "_build_qwen3_5_tp_plans", fail)
    monkeypatch.setattr(qwen3_5_parallelize, "parallelize_module", fail)

    assert qwen3_5_parallelize._apply_qwen3_5_tp(
        model,
        mesh,
        activation_profile=_TP_PROFILE_INFERENCE_REPLICATED,
        enable_loss_parallel=False,
        register_grad_hooks=False,
    ) is model


def test_qwen3_5_linear_attention_cp_keeps_registered_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """The vLLM TP merge must retain the MR's dedicated linear-attention CP wrapper."""
    calls = []

    class LinearAttentionContextParallel:
        """Record the selected MR-side CP execution mode."""

        def __init__(self, mode: str) -> None:
            self.mode = mode

        def apply(self, module: object, mesh: object) -> None:
            calls.append((self.mode, module, mesh))

    monkeypatch.setattr(
        qwen3_5_parallelize,
        "LinearAttentionContextParallel",
        LinearAttentionContextParallel,
    )
    module = object()
    mesh = object()

    qwen3_5_parallelize._apply_linear_attention_cp(  # pylint: disable=protected-access
        module,
        mesh,
        "p2p",
    )

    assert calls == [("p2p", module, mesh)]


def test_inference_tp_rebinds_gdn_runtime_after_parameter_slicing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """vLLM state aliases must follow the TP-local dt_bias and A_log parameters."""
    calls = []
    linear_attn = SimpleNamespace(bind_state_runtime_parameters=lambda: calls.append("bind"))
    block = SimpleNamespace(layer_type="linear_attention", linear_attn=linear_attn)
    plans = SimpleNamespace(
        norm_and_mlp={},
        norm_and_mlp_reduce={},
        full_attention={},
        linear_attention={},
        linear_attention_reduce={},
    )
    monkeypatch.setattr(qwen3_5_parallelize, "parallelize_module", lambda *_, **__: None)
    monkeypatch.setattr(
        qwen3_5_parallelize,
        "_shard_gated_delta_local_params",
        lambda *args: calls.append(("shard", args)),
    )

    qwen3_5_parallelize._apply_qwen3_5_tp_layer_plan(  # pylint: disable=protected-access
        block,
        0,
        object(),
        plans,
        set(),
        set(),
        src_data_rank=None,
    )

    assert calls[0][0] == "shard"
    assert calls[1] == "bind"


def test_qwen3_5_tp_profiles_share_gdn_load_slices() -> None:
    """Training and inference loaders must select identical local GDN blocks."""
    linear_attn = SimpleNamespace(
        head_k_dim=2,
        head_v_dim=3,
        key_dim=8,
        num_k_heads=4,
        num_v_heads=4,
        conv1d=SimpleNamespace(weight=torch.empty(14, 1, 4)),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(layer_types=["linear_attention"]),
        layers=[SimpleNamespace(layer_type="linear_attention", linear_attn=linear_attn)],
    )
    tp_mesh = SimpleNamespace(size=lambda: 2, get_local_rank=lambda: 1)
    mesh = {"tp": tp_mesh}
    cfg = SimpleNamespace(train=SimpleNamespace(accelerator=SimpleNamespace(tp=2)))

    training = qwen3_5_tp_load_transforms(model, mesh, cfg)
    inference = qwen3_5_inference_tp_load_transforms(model, tp_mesh)

    assert training.keys() == inference.keys()
    prefix = "model.layers.0.linear_attn"
    qkv = torch.arange(28 * 2).reshape(28, 2)
    conv = torch.arange(28 * 4).reshape(28, 1, 4)
    state = torch.arange(4)
    for suffix, tensor, expected in (
        ("in_proj_qkv.weight", qkv, torch.cat([qkv[4:8], qkv[12:16], qkv[22:28]])),
        ("conv1d.weight", conv, torch.cat([conv[4:8], conv[12:16], conv[22:28]])),
        ("dt_bias", state, state[2:4]),
        ("A_log", state, state[2:4]),
    ):
        key = f"{prefix}.{suffix}"
        assert torch.equal(training[key](tensor), expected)
        assert torch.equal(inference[key](tensor), expected)
