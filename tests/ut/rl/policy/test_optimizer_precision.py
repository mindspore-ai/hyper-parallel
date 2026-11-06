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
"""Small learning-rate accumulation, clipping and exact optimizer resume in RL roles."""
# pylint: disable=protected-access

from types import SimpleNamespace

import pytest
import rl.workers.engine.optimizer as optimizer_module
import torch

from rl.algorithm import build_algorithm
from rl.workers.engine.optimizer import build_role_optimizer
from rl.workers.actor import Actor
from rl.workers.critic import Critic
import rl.workers.actor as actor_module
import rl.workers.critic as critic_module

from hyper_parallel.core.optimizer.adamw import AdamW
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer


def _role(kind: str) -> tuple[torch.nn.Module, object, torch.nn.Module]:
    """Build real core AdamW through the RL optimizer boundary."""
    model = torch.nn.Linear(1, 1, bias=False).to(dtype=torch.bfloat16)
    with torch.no_grad():
        model.weight.fill_(1)
    chain = ChainedOptimizer(model, {"adamw": AdamW(model.parameters(), lr=1e-4, betas=(0, 0), weight_decay=0)})
    runtime = SimpleNamespace(
        optimizer=SimpleNamespace(fp32_main_params=True, target=SimpleNamespace(
            build=lambda **_kwargs: SimpleNamespace(get_optimizer=lambda: chain))),
        lr_scheduler=None,
    )
    optimizer, _ = build_role_optimizer(runtime, model)
    if kind == "actor":
        role = Actor(model, build_algorithm({"name": "grpo", "loss_aggregation": "token-mean"}),
                     1, optimizer=optimizer, max_grad_norm=1, device=torch.device("cpu"))
    else:
        role = Critic(model, build_algorithm({"name": "ppo", "loss_aggregation": "token-mean"}),
                      optimizer, None, torch.device("cpu"),
                      None, 1, 1, 1, 1, 1)
    return model, optimizer, role


@pytest.mark.parametrize("kind", ["actor", "critic"])
def test_main_parameters_accumulate_sub_bf16_updates_and_resume(kind, tmp_path, monkeypatch) -> None:
    """Hidden small updates survive a checkpoint and clipping acts on the actual FP32 gradient."""
    monkeypatch.setattr(optimizer_module, "hsdp_sync_stream", lambda: None)
    monkeypatch.setattr(optimizer_module, "hsdp_sync_stream", lambda: None)
    model, optimizer, role = _role(kind)
    model.weight.main_grad = torch.full((1, 1), float("nan"), dtype=torch.float32)
    with pytest.raises(RuntimeError):
        role._optimizer_step()
    assert optimizer.param_groups[0]["params"][0].item() == 1.0
    optimizer.zero_grad(set_to_none=True)

    def advance(count: int) -> torch.Tensor:
        """Accumulate updates smaller than BF16 spacing using the production role step."""
        for _ in range(count):
            model.weight.main_grad = torch.full((1, 1), 100.0, dtype=torch.float32)
            norm = role._optimizer_step()
            assert norm == pytest.approx(100.0)
            assert model.weight.main_grad is None
        main = optimizer.param_groups[0]["params"][0]
        assert main.dtype == torch.float32
        assert optimizer.chained_optimizers[0].state[main]["exp_avg"].item() == pytest.approx(1.0, abs=1e-5)
        return main

    main = advance(16)
    assert model.weight.item() == 1.0
    assert main.item() == pytest.approx(0.9984, abs=1e-5)
    path = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
    checkpoint = torch.load(path, weights_only=False)
    model, optimizer, role = _role(kind)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    main = advance(16)
    assert main.item() == pytest.approx(0.9968, abs=1e-5)
    assert model.weight.item() < 1.0
