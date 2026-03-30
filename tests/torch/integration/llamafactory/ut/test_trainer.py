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
"""Minimal unit tests for the HyperParallel LlamaFactory trainer integration."""
# pylint: disable=wrong-import-position,protected-access
import sys
import types
from types import ModuleType

import torch

try:
    from transformers import Seq2SeqTrainer
except ImportError:
    transformers_stub = ModuleType("transformers")

    class Seq2SeqTrainer:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def train(self, *args, **kwargs):
            del args, kwargs
            raise NotImplementedError

        def create_optimizer(self):
            raise NotImplementedError

    transformers_stub.Seq2SeqTrainer = Seq2SeqTrainer
    sys.modules["transformers"] = transformers_stub

if "accelerate.accelerator" not in sys.modules:
    accelerate_stub = sys.modules.setdefault("accelerate", ModuleType("accelerate"))
    accelerator_stub = ModuleType("accelerate.accelerator")
    accelerator_stub.fsdp2_prepare_model = lambda accelerator, model: model
    accelerate_stub.accelerator = accelerator_stub
    sys.modules["accelerate.accelerator"] = accelerator_stub

import hyper_parallel.integration.llamafactory.trainer as trainer_mod
from hyper_parallel.integration.llamafactory.trainer import (
    HyperParallelArguments,
    HyperParallelTrainer,
    _export_to_hf_format,
    _normalize_hf_export_state_dict,
    _wrap_optimizer_step_with_skip_dtensor_dispatch,
)
from hyper_parallel.integration.llamafactory.utils import (
    _build_fsdp2_kwargs,
    _is_cpu_offload_enabled,
    _resolve_mp_policy,
)
from hyper_parallel.core.fully_shard.utils import CPUOffloadPolicy, OffloadPolicy


class _FakeOptimizer:
    def __init__(self):
        self.calls = []

    def step(self, closure=None):
        self.calls.append(closure)
        return "stepped"


class _FakeAccelerator:
    """Minimal accelerator double for patch lifecycle tests."""

    def __init__(self):
        self.is_fsdp2 = True
        self._models = []
        self.unscale_calls = 0

    def clip_grad_norm_(self, parameters, max_norm, norm_type=2):
        return ("orig", list(parameters), max_norm, norm_type)

    def unscale_gradients(self):
        self.unscale_calls += 1


class _FakeHSDPModule:
    def parameters(self):
        return []


def _build_trainer_for_patch_tests():
    trainer = object.__new__(HyperParallelTrainer)
    trainer._hp_args = HyperParallelArguments()
    trainer.accelerator = _FakeAccelerator()
    trainer._orig_accelerator_clip_grad_norm = trainer.accelerator.clip_grad_norm_
    trainer._orig_fsdp2_prepare_model = None
    trainer._accelerator_patches_active = False
    return trainer


def test_cpu_offload_enabled_detection():
    """
    Feature: CPU offload enablement detection
    Description: Only real CPU offload config should be treated as enabled.
    Expectation: Falsey/default policies stay disabled, CPU offload policies enable the path.
    """
    assert _is_cpu_offload_enabled(False) is False
    assert _is_cpu_offload_enabled(None) is False
    assert _is_cpu_offload_enabled(OffloadPolicy()) is False
    assert _is_cpu_offload_enabled(True) is True
    assert _is_cpu_offload_enabled(CPUOffloadPolicy()) is True


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

    accelerator = types.SimpleNamespace(device=torch.device("cpu"))
    plugin = types.SimpleNamespace(
        reshard_after_forward=False,
        cpu_offload=False,
        mixed_precision_policy=None,
        ignored_modules=None,
    )

    kwargs = _build_fsdp2_kwargs(accelerator, torch.nn.Linear(2, 2), HyperParallelArguments(), plugin)

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
    monkeypatch.setattr(trainer_mod, "SkipDTensorDispatch", _FakeSkip)

    _wrap_optimizer_step_with_skip_dtensor_dispatch(optimizer)

    assert hasattr(optimizer.step, "__func__")
    result = optimizer.step("closure")

    assert result == "stepped"
    assert optimizer.calls == ["closure"]
    assert enter_exit == ["enter", "exit"]


def test_train_activates_and_restores_accelerator_patches(monkeypatch):
    """
    Feature: Patch lifecycle isolation
    Description: Trainer should patch Accelerate only during train() and restore afterwards.
    Expectation: fsdp2_prepare_model and clip_grad_norm_ are restored after training.
    """
    trainer = _build_trainer_for_patch_tests()

    import accelerate.accelerator as acc_module  # pylint: disable=C0415

    original_fsdp2_prepare_model = acc_module.fsdp2_prepare_model
    original_clip_grad_norm = trainer.accelerator.clip_grad_norm_

    super_calls = []

    def _fake_super_train(self, *args, **kwargs):
        super_calls.append((args, kwargs))
        assert acc_module.fsdp2_prepare_model is not original_fsdp2_prepare_model
        assert self.accelerator.clip_grad_norm_ is not original_clip_grad_norm
        return "trained"

    monkeypatch.setattr(Seq2SeqTrainer, "train", _fake_super_train)

    result = trainer.train("arg", named=True)

    assert result == "trained"
    assert super_calls == [(("arg",), {"named": True})]
    assert acc_module.fsdp2_prepare_model is original_fsdp2_prepare_model
    assert trainer.accelerator.clip_grad_norm_.__func__ is original_clip_grad_norm.__func__
    assert trainer.accelerator.clip_grad_norm_.__self__ is original_clip_grad_norm.__self__
    assert trainer._accelerator_patches_active is False


def test_clip_grad_norm_dispatches_to_local_impl(monkeypatch):
    """
    Feature: Gradient clipping backend replacement
    Description: In FSDP2 mode with an HSDP model, Accelerate clip_grad_norm_
    should dispatch to HyperParallel clip impl.
    Expectation: HyperParallel clip_grad_norm_ is called by default instead of the original accelerator path.
    """
    trainer = _build_trainer_for_patch_tests()
    model = _FakeHSDPModule()
    params = [torch.nn.Parameter(torch.tensor([1.0]))]
    model.parameters = lambda: params
    trainer.accelerator._models = [model]

    local_calls = []
    monkeypatch.setattr(trainer_mod, "HSDPModule", _FakeHSDPModule)
    monkeypatch.setattr(
        trainer_mod,
        "hp_clip_grad_norm_",
        lambda parameters, max_norm, norm_type=2: local_calls.append((list(parameters), max_norm, norm_type))
        or "local",
    )

    trainer._activate_accelerator_patches()
    try:
        result = trainer.accelerator.clip_grad_norm_(params, 1.5, norm_type=1.0)
    finally:
        trainer._restore_accelerator_patches()

    assert result == "local"
    assert trainer.accelerator.unscale_calls == 1
    assert local_calls == [(params, 1.5, 1.0)]


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

    monkeypatch.setattr(trainer_mod, "get_platform", _fake_get_platform)

    def _fake_get_model_state_dict(model, options=None):
        del model, options
        return {"weight": torch.ones(4, dtype=torch.bfloat16)}

    monkeypatch.setattr(
        "hyper_parallel.core.fully_shard.api.get_model_state_dict",
        _fake_get_model_state_dict,
    )

    _export_to_hf_format(_FakeModel(), _FakeTokenizer(), str(tmp_path))

    assert captured["save_dir"] == str(tmp_path)
    assert captured["tokenizer_dir"] == str(tmp_path)
    assert captured["state_dict"]["weight"].dtype == torch.float32
    assert captured["model_max_shard_size"] is None
    assert captured["tokenizer_max_shard_size"] is None


def test_normalize_hf_export_state_dict_upcasts_floating_tensors_and_keeps_aliases():
    """
    Feature: HF export state dict normalization
    Description: Floating tensors should be exported in fp32 without breaking tied/shared tensors.
    Expectation: Lower-precision floating tensors are upcast to fp32 and aliases remain shared.
    """
    base = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    aliased = base.view(2, 3)
    int_tensor = torch.tensor([1, 2, 3], dtype=torch.int64)

    normalized = _normalize_hf_export_state_dict(
        {
            "bf16": base,
            "alias": aliased,
            "fp32": torch.ones(2, dtype=torch.float32),
            "int": int_tensor,
        }
    )

    assert normalized["bf16"].dtype == torch.float32
    assert normalized["alias"].dtype == torch.float32
    assert normalized["fp32"].dtype == torch.float32
    assert normalized["int"] is int_tensor
    assert normalized["bf16"] is normalized["alias"]


def test_save_model_internal_call_exports_hf_weights(monkeypatch, tmp_path):
    """
    Feature: Intermediate checkpoint export
    Description: save_model(_internal_call=True) should still export HF-format
    weights into the checkpoint directory.
    Expectation: The internal save path calls the HF export helper with the checkpoint directory.
    """
    trainer = object.__new__(HyperParallelTrainer)
    trainer.args = types.SimpleNamespace(output_dir=str(tmp_path / "default"))
    trainer.model = object()
    trainer.processing_class = object()

    calls = []
    monkeypatch.setattr(
        trainer_mod,
        "_export_to_hf_format",
        lambda model, tokenizer, save_dir: calls.append((model, tokenizer, save_dir)),
    )

    checkpoint_dir = tmp_path / "checkpoint-3"
    trainer.save_model(str(checkpoint_dir), _internal_call=True)

    assert checkpoint_dir.is_dir()
    assert calls == [(trainer.model, trainer.processing_class, str(checkpoint_dir))]
