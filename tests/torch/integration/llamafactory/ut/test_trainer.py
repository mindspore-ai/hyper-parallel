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
import sys
import types
from types import ModuleType

import pytest
import torch

# Stub transformers and accelerate if not installed (needed by utils.py imports)
try:
    from transformers import Seq2SeqTrainer  # noqa: F401
except ImportError:
    transformers_stub = ModuleType("transformers")

    class Seq2SeqTrainer:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            del args, kwargs

    transformers_stub.Seq2SeqTrainer = Seq2SeqTrainer
    sys.modules["transformers"] = transformers_stub

if "accelerate.accelerator" not in sys.modules:
    accelerate_stub = sys.modules.setdefault("accelerate", ModuleType("accelerate"))
    accelerator_stub = ModuleType("accelerate.accelerator")
    accelerator_stub.fsdp2_prepare_model = lambda accelerator, model: model
    accelerate_stub.accelerator = accelerator_stub
    sys.modules["accelerate.accelerator"] = accelerator_stub

import hyper_parallel.integration.llamafactory.utils as lf_utils
from hyper_parallel.integration.llamafactory.utils import (
    HyperParallelArguments,
    export_to_hf_format,
    _build_fsdp2_kwargs,
    _resolve_mp_policy,
    wrap_optimizer_with_skip_dtensor_dispatch,
)


class _FakeOptimizer:
    def __init__(self):
        self.calls = []

    def step(self, closure=None):
        self.calls.append(closure)
        return "stepped"


def test_hp_args_reshard_after_forward_defaults_to_accelerate_plugin(monkeypatch):
    """
    Feature: reshard_after_forward config source
    Description: When HyperParallel args do not override the setting, use Accelerate's FSDP2 plugin value.
    Expectation: fully_shard kwargs inherit reshard_after_forward from the plugin by default.
    """
    monkeypatch.setattr(
        "hyper_parallel.integration.llamafactory.utils._build_device_mesh",
        lambda accelerator, hp_args: None,
    )
    monkeypatch.setattr(
        "hyper_parallel.integration.llamafactory.utils.get_parameters_from_modules",
        lambda modules, model, device: set(),
    )
    monkeypatch.setattr(
        "hyper_parallel.integration.llamafactory.utils._resolve_shard_size",
        lambda mesh: 1,
    )

    accelerator = types.SimpleNamespace(device=torch.device("cpu"))
    plugin = types.SimpleNamespace(
        reshard_after_forward=False,
        cpu_offload=False,
        mixed_precision_policy=None,
        ignored_modules=None,
    )

    kwargs = _build_fsdp2_kwargs(
        accelerator, torch.nn.Linear(2, 2), HyperParallelArguments(), plugin
    )

    assert kwargs["reshard_after_forward"] is False


def test_hp_args_reshard_after_forward_overrides_accelerate_plugin(monkeypatch):
    """
    Feature: reshard_after_forward config override
    Description: Explicit HyperParallel args should override Accelerate's FSDP2 plugin value.
    Expectation: fully_shard kwargs use the HyperParallel override when provided.
    """
    monkeypatch.setattr(
        "hyper_parallel.integration.llamafactory.utils._build_device_mesh",
        lambda accelerator, hp_args: None,
    )
    monkeypatch.setattr(
        "hyper_parallel.integration.llamafactory.utils.get_parameters_from_modules",
        lambda modules, model, device: set(),
    )
    monkeypatch.setattr(
        "hyper_parallel.integration.llamafactory.utils._resolve_shard_size",
        lambda mesh: 1,
    )

    accelerator = types.SimpleNamespace(device=torch.device("cpu"))
    plugin = types.SimpleNamespace(
        reshard_after_forward=False,
        cpu_offload=False,
        mixed_precision_policy=None,
        ignored_modules=None,
    )

    hp_args = HyperParallelArguments(reshard_after_forward=True)
    kwargs = _build_fsdp2_kwargs(accelerator, torch.nn.Linear(2, 2), hp_args, plugin)

    assert kwargs["reshard_after_forward"] is True


def test_hp_args_mp_defaults_to_accelerate_policy():
    """
    Feature: mixed precision config source
    Description: When HyperParallel args do not override mp settings, inherit Accelerate's normalized policy.
    Expectation: HyperParallel mp policy matches the plugin-provided policy values.
    """
    plugin = types.SimpleNamespace(
        mixed_precision_policy=types.SimpleNamespace(
            param_dtype=torch.float16,
            reduce_dtype=torch.bfloat16,
            output_dtype=torch.float16,
            cast_forward_inputs=False,
        )
    )

    policy = _resolve_mp_policy(plugin, HyperParallelArguments())

    assert policy.param_dtype == torch.float16
    assert policy.reduce_dtype == torch.bfloat16
    assert policy.output_dtype == torch.float16
    assert policy.cast_forward_inputs is False


def test_hp_args_mp_overrides_accelerate_policy():
    """
    Feature: mixed precision override
    Description: Explicit HyperParallel dtype overrides should replace the inherited Accelerate policy fields.
    Expectation: Only the configured HyperParallel fields override the plugin policy.
    """
    plugin = types.SimpleNamespace(
        mixed_precision_policy=types.SimpleNamespace(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            output_dtype=torch.float16,
            cast_forward_inputs=False,
        )
    )

    hp_args = HyperParallelArguments(param_dtype="bfloat16", reduce_dtype="float32")
    policy = _resolve_mp_policy(plugin, hp_args)

    assert policy.param_dtype == torch.bfloat16
    assert policy.reduce_dtype == torch.float32
    assert policy.output_dtype == torch.bfloat16
    assert policy.cast_forward_inputs is False


def test_hp_args_mp_without_accelerate_policy_stays_empty_by_default():
    """
    Feature: mixed precision default inheritance
    Description: Without an Accelerate policy and without HyperParallel overrides,
    do not force a backend default mp policy.
    Expectation: The resulting HyperParallel mixed precision policy remains empty.
    """
    plugin = types.SimpleNamespace(mixed_precision_policy=None)

    policy = _resolve_mp_policy(plugin, HyperParallelArguments())

    assert policy.param_dtype is None
    assert policy.reduce_dtype is None
    assert policy.output_dtype is None


def test_hp_args_mp_accepts_accelerate_style_dtype_aliases():
    """
    Feature: dtype alias compatibility
    Description: HyperParallel dtype overrides should accept Accelerate-style fpXX / bf16 aliases.
    Expectation: Aliases are normalized to the matching torch dtypes.
    """
    plugin = types.SimpleNamespace(mixed_precision_policy=None)

    hp_args = HyperParallelArguments(param_dtype="bf16", reduce_dtype="fp32")
    policy = _resolve_mp_policy(plugin, hp_args)

    assert policy.param_dtype == torch.bfloat16
    assert policy.reduce_dtype == torch.float32
    assert policy.output_dtype == torch.bfloat16


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

    class _FakePlatform:
        @staticmethod
        def get_rank():
            return 0

        @staticmethod
        def get_world_size():
            return 1

    def _fake_get_platform():
        return _FakePlatform()

    monkeypatch.setattr(lf_utils, "get_platform", _fake_get_platform)

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
# Tests: fsdp_size validation and device mesh construction
# ---------------------------------------------------------------------------


def test_hp_args_fsdp_size_defaults_to_none():
    """
    Feature: fsdp_size default
    Description: fsdp_size is optional; when not provided it stays None and validate passes.
    Expectation: HyperParallelArguments() has fsdp_size=None and validates cleanly.
    """
    args = HyperParallelArguments()
    args.validate()
    assert args.fsdp_size is None, f"Expected fsdp_size=None, got {args.fsdp_size!r}"


def test_hp_args_fsdp_size_accepts_positive_int():
    """
    Feature: fsdp_size validation accepts positive int
    Description: A positive integer fsdp_size must pass validation.
    Expectation: validate() does not raise for fsdp_size=4.
    """
    args = HyperParallelArguments(fsdp_size=4)
    args.validate()
    assert args.fsdp_size == 4, f"Expected fsdp_size=4, got {args.fsdp_size!r}"


def test_hp_args_fsdp_size_rejects_zero_and_negative():
    """
    Feature: fsdp_size validation rejects non-positive
    Description: 0 and negative values must raise during validation.
    Expectation: ValueError mentioning fsdp_size for both 0 and -1.
    """
    for bad in (0, -1, -8):
        args = HyperParallelArguments(fsdp_size=bad)
        with pytest.raises(ValueError, match="fsdp_size"):
            args.validate()


def test_hp_args_fsdp_size_rejects_non_int():
    """
    Feature: fsdp_size validation rejects non-int
    Description: Non-int values (including bool, which Python treats as int subtype) must raise.
    Expectation: ValueError mentioning fsdp_size for string, float, bool inputs.
    """
    for bad in ("4", 4.0, True):
        args = HyperParallelArguments(fsdp_size=bad)
        with pytest.raises(ValueError, match="fsdp_size"):
            args.validate()


def _patch_mesh_environment(monkeypatch, world_size):
    """Install fake init_device_mesh + get_platform that records call args."""
    calls = {}

    def fake_init(device_type, shape, mesh_dim_names=None):
        calls["device_type"] = device_type
        calls["shape"] = shape
        calls["mesh_dim_names"] = mesh_dim_names
        return f"mesh<{shape}, {mesh_dim_names}>"

    class FakePlatform:
        @staticmethod
        def get_world_size():
            return world_size

    monkeypatch.setattr(lf_utils, "init_device_mesh", fake_init)
    monkeypatch.setattr(lf_utils, "get_platform", FakePlatform)
    return calls


def test_build_device_mesh_2d_when_fsdp_size_lt_world_size(monkeypatch):
    """
    Feature: 2D HSDP mesh construction
    Description: fsdp_size < world_size and world_size % fsdp_size == 0 builds a 2D (dp, fsdp) mesh.
    Expectation: init_device_mesh receives shape (dp_size, fsdp_size) with dim names ("dp", "fsdp").
    """
    calls = _patch_mesh_environment(monkeypatch, world_size=8)
    accelerator = types.SimpleNamespace(device=torch.device("cpu"))
    hp_args = HyperParallelArguments(fsdp_size=4, device_type="cpu")

    mesh = lf_utils._build_device_mesh(accelerator, hp_args)

    assert calls["shape"] == (2, 4), f"Expected shape (2, 4), got {calls['shape']}"
    assert calls["mesh_dim_names"] == ("dp", "fsdp"), (
        f"Expected dim names ('dp', 'fsdp'), got {calls['mesh_dim_names']}"
    )
    assert mesh == "mesh<(2, 4), ('dp', 'fsdp')>", f"Unexpected mesh: {mesh}"


def test_build_device_mesh_1d_when_fsdp_size_ge_world_size(monkeypatch):
    """
    Feature: 1D fallback when fsdp_size covers the whole world
    Description: fsdp_size >= world_size collapses to a 1D ("dp",) mesh.
    Expectation: init_device_mesh receives shape (world_size,) with dim name ("dp",).
    """
    calls = _patch_mesh_environment(monkeypatch, world_size=4)
    accelerator = types.SimpleNamespace(device=torch.device("cpu"))
    hp_args = HyperParallelArguments(fsdp_size=4, device_type="cpu")

    lf_utils._build_device_mesh(accelerator, hp_args)

    assert calls["shape"] == (4,), f"Expected shape (4,), got {calls['shape']}"
    assert calls["mesh_dim_names"] == ("dp",), (
        f"Expected dim names ('dp',), got {calls['mesh_dim_names']}"
    )


def test_build_device_mesh_raises_when_world_size_not_divisible(monkeypatch):
    """
    Feature: Divisibility check
    Description: world_size not divisible by fsdp_size must raise.
    Expectation: ValueError naming both world_size and fsdp_size.
    """
    _patch_mesh_environment(monkeypatch, world_size=8)
    accelerator = types.SimpleNamespace(device=torch.device("cpu"))
    hp_args = HyperParallelArguments(fsdp_size=3, device_type="cpu")

    with pytest.raises(
        ValueError, match="world_size=8 must be divisible by fsdp_size=3"
    ):
        lf_utils._build_device_mesh(accelerator, hp_args)


def test_build_device_mesh_ignores_accelerate_mesh_when_fsdp_size_set(monkeypatch):
    """
    Feature: fsdp_size bypasses accelerate-cached mesh
    Description: When fsdp_size is set, accelerator.torch_device_mesh is ignored
    so we get a fresh 2D HSDP mesh instead of a cached 1D FSDP mesh.
    Expectation: init_device_mesh is called and the cached mesh is not returned.
    """
    calls = _patch_mesh_environment(monkeypatch, world_size=8)
    # Accelerator carries a cached mesh that *would* be returned by the
    # default branch — fsdp_size must override it.
    accelerator = types.SimpleNamespace(
        device=torch.device("cpu"),
        torch_device_mesh="CACHED_1D_MESH",
        parallelism_config=None,
    )
    hp_args = HyperParallelArguments(fsdp_size=4, device_type="cpu")

    mesh = lf_utils._build_device_mesh(accelerator, hp_args)

    assert mesh != "CACHED_1D_MESH", (
        "fsdp_size must override the cached accelerator mesh"
    )
    assert calls["shape"] == (2, 4), (
        f"Expected init_device_mesh called with shape (2, 4), got {calls['shape']}"
    )


def test_build_device_mesh_uses_accelerate_mesh_when_fsdp_size_none(monkeypatch):
    """
    Feature: backward compatibility when fsdp_size is None
    Description: With fsdp_size=None, behavior is unchanged — the accelerate-cached mesh is returned.
    Expectation: init_device_mesh is NOT called; the cached mesh is returned verbatim.
    """
    calls = _patch_mesh_environment(monkeypatch, world_size=8)
    accelerator = types.SimpleNamespace(
        device=torch.device("cpu"),
        torch_device_mesh="CACHED_1D_MESH",
        parallelism_config=None,
    )
    hp_args = HyperParallelArguments()  # fsdp_size=None

    mesh = lf_utils._build_device_mesh(accelerator, hp_args)

    assert mesh == "CACHED_1D_MESH", (
        f"Expected cached mesh to be returned, got {mesh!r}"
    )
    assert "shape" not in calls, (
        "init_device_mesh should not be called when fsdp_size is None and cached mesh exists"
    )
