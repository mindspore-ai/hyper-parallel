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

import torch

from rl.algorithm import build_algorithm
from rl.roles.model import build_role_model, build_role_optimizer
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
            "optimizer": optimizer_builder,
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
