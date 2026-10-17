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
"""CPU unit tests for trainable and frozen Actor roles."""
# Hyper-RL CPU tests intentionally exercise the verified Torch runtime.
# pylint: disable=forbidden-backend-import,missing-public-docstring

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from rl.algorithm import build_algorithm
from rl.roles.model_setup import build_role_model, build_role_optimizer
from rl.roles.policy.actor import Actor


def test_actor_runtime_distinguishes_trainable_and_reference_roles() -> None:
    """A trainable Actor and its independent Reference keep distinct runtime state."""
    algorithm = build_algorithm({"name": "grpo", "loss_aggregation": "token-mean"})
    actor_model = torch.nn.Linear(3, 3)
    reference_model = torch.nn.Linear(3, 3)
    optimizer = torch.optim.SGD(actor_model.parameters(), lr=0.1)

    actor = Actor(
        actor_model,
        algorithm,
        micro_batch_size=1,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )
    reference = Actor(reference_model, algorithm, micro_batch_size=1)

    assert actor.actor_model is actor_model
    assert actor.optimizer is optimizer
    assert actor.actor_model is not reference.actor_model
    assert all(parameter.requires_grad for parameter in actor.parameters())
    assert reference.optimizer is None
    assert not reference.training
    assert all(not parameter.requires_grad for parameter in reference.parameters())

    class Builder:
        def __init__(self, result: object) -> None:
            self.result = result
            self.kwargs = None

        def build(self, **kwargs: object) -> object:
            self.kwargs = kwargs
            return self.result

    class OptimizerProduct:
        @staticmethod
        def get_optimizer() -> object:
            return optimizer

    class SchedulerProduct:
        @staticmethod
        def get_lr_scheduler() -> str:
            return "scheduler"

    model_builder = Builder(reference_model)
    optimizer_builder = Builder(OptimizerProduct())
    scheduler_builder = Builder(SchedulerProduct())
    runtime = type(
        "Runtime",
        (),
        {
            "activation_checkpoint": type("Activation", (), {"mode": "full"})(),
            "model": model_builder,
            "peft": "peft",
            "optimizer": type("OptimizerConfig", (), {"target": optimizer_builder})(),
            "lr_scheduler": scheduler_builder,
            "training": type("Training", (), {"train_iters": 5})(),
        },
    )()
    built_reference = build_role_model(runtime, "distributed", frozen=True)
    built_optimizer, built_scheduler = build_role_optimizer(runtime, actor_model)

    assert built_reference is reference_model
    assert model_builder.kwargs == {
        "distributed_setup": "distributed",
        "activation_checkpoint": "full",
        "peft_config": "peft",
    }
    assert built_optimizer is optimizer
    assert built_scheduler == "scheduler"
    assert optimizer_builder.kwargs == {"model": actor_model}
    assert scheduler_builder.kwargs == {"optimizer": optimizer, "train_iters": 5}


@pytest.mark.parametrize("threads", [None, 8])
def test_model_init_threads_are_scoped_before_optimizer(
    monkeypatch: pytest.MonkeyPatch, threads: int | None,
) -> None:
    """The default leaves threads alone; an override ends before optimizer construction."""
    current = [3]
    observed = []
    model = torch.nn.Linear(2, 2, device="cpu")
    optimizer = object()
    getter = Mock(side_effect=lambda: current[0])
    setter = Mock(side_effect=lambda value: current.__setitem__(0, value))
    interop_setter = Mock()
    model_builder = Mock(side_effect=lambda **_kwargs: observed.append(("model", current[0])) or model)
    product = SimpleNamespace(get_optimizer=lambda: optimizer)
    optimizer_builder = Mock(side_effect=lambda **_kwargs: observed.append(("optimizer", current[0])) or product)
    runtime = SimpleNamespace(
        model=SimpleNamespace(build=model_builder), activation_checkpoint=SimpleNamespace(mode="off"), peft=None,
        optimizer=SimpleNamespace(target=SimpleNamespace(build=optimizer_builder)), lr_scheduler=None,
    )
    with monkeypatch.context() as cpu:
        cpu.setattr(torch, "get_num_threads", getter)
        cpu.setattr(torch, "set_num_threads", setter)
        cpu.setattr(torch, "set_num_interop_threads", interop_setter)
        options = {} if threads is None else {"cpu_threads": threads}
        assert build_role_model(runtime, object(), frozen=False, **options) is model
        assert build_role_optimizer(runtime, model) == (optimizer, None)
    expected = [("model", 3 if threads is None else threads), ("optimizer", 3)]
    assert observed == expected, f"Expected scoped threads {expected}, got {observed}"
    interop_setter.assert_not_called()
    if threads is None:
        getter.assert_not_called()
        setter.assert_not_called()
    else:
        assert [call.args[0] for call in setter.call_args_list] == [8, 3], (
            f"Expected temporary threads 8 then restored 3, got {setter.call_args_list}")


def test_model_init_threads_restore_after_builder_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model construction exception still restores the previous intra-op setting."""
    current = [3]
    setter = Mock(side_effect=lambda value: current.__setitem__(0, value))
    runtime = SimpleNamespace(
        model=SimpleNamespace(build=Mock(side_effect=RuntimeError("builder failed"))),
        activation_checkpoint=SimpleNamespace(mode="off"), peft=None,
    )
    monkeypatch.setattr(torch, "get_num_threads", lambda: current[0])
    monkeypatch.setattr(torch, "set_num_threads", setter)

    with pytest.raises(RuntimeError, match="builder failed"):
        build_role_model(runtime, object(), frozen=False, cpu_threads=8)

    assert current == [3], f"Expected restored threads 3, got {current}"
    assert [call.args[0] for call in setter.call_args_list] == [8, 3], (
        f"Expected temporary threads 8 then restored 3, got {setter.call_args_list}")
