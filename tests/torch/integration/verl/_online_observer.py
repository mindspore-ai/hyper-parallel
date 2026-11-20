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
"""Worker-only assertions for the real online Engine/HyperParallel training path."""

import json
import math
import os
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from weakref import WeakKeyDictionary

import torch
import torch.distributed as dist

from hyper_parallel.integration.verl.engine.transformer_impl import HyperParallelEngineWithLMHead


def _emit(engine, event, **details):
    path = Path(os.environ["VERL_ST_OUTPUT_DIR"]) / f"rank_{dist.get_rank()}.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": event, "step": _observers[engine].optimizer_steps, **details}) + "\n")


def _epochs(engine):
    container = engine.trainer.lr_scheduler
    return [scheduler.last_epoch for scheduler in container.schedulers] if container is not None else []


_observers = WeakKeyDictionary()
_original_initialize = HyperParallelEngineWithLMHead.initialize


@wraps(_original_initialize)
def _initialize(engine, *args, **kwargs):
    """Install the runtime checks on the training engine."""
    _original_initialize(engine, *args, **kwargs)
    if engine.engine_config.forward_only:
        return
    if not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled():
        raise AssertionError("Expected strict runtime determinism")
    expected = int(os.environ["DEBUG_LAYERS"])
    if engine.module.config.num_hidden_layers != expected:
        raise AssertionError(f"layers={engine.module.config.num_hidden_layers}, expected={expected}")
    observer = SimpleNamespace(phase=None, optimizer_steps=0)
    _observers[engine] = observer

    def check_forward(module, inputs):
        """Check the module mode and autograd state for the active phase."""
        del inputs
        training = observer.phase == "train"
        if module.training != training or torch.is_grad_enabled() != training:
            raise AssertionError(
                f"phase={observer.phase}, training={module.training}, grad={torch.is_grad_enabled()}"
            )

    engine.module.register_forward_pre_hook(check_forward)
    _emit(engine, "initialized", world=dist.get_world_size(), layers=expected)


_original_infer = HyperParallelEngineWithLMHead.infer_batch


@wraps(_original_infer)
def _infer(engine, *args, **kwargs):
    """Verify that inference leaves training state unchanged."""
    if engine.engine_config.forward_only:
        return _original_infer(engine, *args, **kwargs)
    observer = _observers[engine]
    observer.phase = "infer"
    step = observer.optimizer_steps
    gradients = sum(parameter.grad is not None for parameter in engine.module.parameters())
    result = _original_infer(engine, *args, **kwargs)
    if step != observer.optimizer_steps:
        raise AssertionError("Inference changed the optimizer step")
    if gradients != sum(parameter.grad is not None for parameter in engine.module.parameters()):
        raise AssertionError("Inference created gradients")
    _emit(engine, "infer")
    return result


_original_train = HyperParallelEngineWithLMHead.train_batch


@wraps(_original_train)
def _train(engine, *args, **kwargs):
    """Verify the optimizer and scheduler boundaries."""
    observer = _observers[engine]
    observer.phase = "train"
    step, epochs = observer.optimizer_steps, _epochs(engine)
    result = _original_train(engine, *args, **kwargs)
    if observer.optimizer_steps != step + 1 or _epochs(engine) != epochs:
        raise AssertionError("Expected one optimizer update and no premature scheduler update")
    if engine.is_mp_src_rank_with_outputs():
        grad_norm = result["metrics"]["grad_norm"]
        if not math.isfinite(grad_norm):
            raise AssertionError(f"Non-finite gradient norm: {grad_norm}")
        _emit(engine, "train", grad_norm=grad_norm)
    else:
        _emit(engine, "train")
    return result


_original_optimizer_step = HyperParallelEngineWithLMHead.optimizer_step


@wraps(_original_optimizer_step)
def _optimizer_step(engine):
    """Count successful optimizer updates for the observer."""
    grad_norm = _original_optimizer_step(engine)
    _observers[engine].optimizer_steps += 1
    return grad_norm


_original_scheduler = HyperParallelEngineWithLMHead.lr_scheduler_step


@wraps(_original_scheduler)
def _scheduler(engine, *args, **kwargs):
    epochs = _epochs(engine)
    result = _original_scheduler(engine, *args, **kwargs)
    if _epochs(engine) != [epoch + 1 for epoch in epochs]:
        raise AssertionError(f"scheduler epochs={_epochs(engine)}, expected={[epoch + 1 for epoch in epochs]}")
    _emit(engine, "scheduler", lr=result)
    return result


HyperParallelEngineWithLMHead.initialize = _initialize
HyperParallelEngineWithLMHead.infer_batch = _infer
HyperParallelEngineWithLMHead.train_batch = _train
HyperParallelEngineWithLMHead.optimizer_step = _optimizer_step
HyperParallelEngineWithLMHead.lr_scheduler_step = _scheduler
