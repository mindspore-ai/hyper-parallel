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
"""CPU role doubles; real sample decoding remains exercised by trainer tests."""
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from rl.async_runtime.protocol import SamplingTask
from rl.trainer.colocate_async import ColocateAsyncTrainer
from rl.config.schema import ColocateAsyncConfig
from rl.rollout.base import GenerationSettings
from rl.utils.monitoring.metrics import ActorUpdateMetrics
from tests.ut.rl.fully_async.support import group


def trainer_fixture() -> tuple[ColocateAsyncTrainer, list, tuple[bytes, ...]]:
    """Build a trainer without devices, recording externally visible stage boundaries."""
    trainer = ColocateAsyncTrainer.__new__(ColocateAsyncTrainer)
    trainer.settings = ColocateAsyncConfig(groups_per_dp_rank=1)
    trainer.sampling = trainer.settings.sampling_config(2)
    trainer.rank, trainer.batch_size, trainer.candidates = 0, 2, 2
    trainer.optimizer_steps = 0
    trainer.timings, trainer._restored, trainer._snapshot = {}, None, None
    trainer.config = {"train": {"max_steps": 1, "checkpoint": {"save_final": False}},
                      "colocate_async": {"groups_per_dp_rank": 1}}
    events = []
    payloads = tuple(group(SamplingTask(index, 1, 0)).encode(2, trainer.settings.max_group_bytes)
                     for index in range(2))
    engine = SimpleNamespace(phase="rollout")

    def sleep() -> None:
        """Record the simulated engine sleep boundary."""
        events.append("sleep")
        engine.phase = "training"

    def publish(version: int) -> None:
        """Record publication while enforcing training-side ownership."""
        del version
        if engine.phase != "training":
            raise RuntimeError("publication without sleeping")
        events.append("publish")
        engine.phase = "rollout"

    engine.prepare_for_training = Mock(side_effect=sleep)
    engine.prepare_for_rollout = Mock(side_effect=lambda: setattr(engine, "phase", "rollout"))
    engine.start_service = Mock(side_effect=lambda: events.append("start_service"))
    update = ActorUpdateMetrics(1., 1., 0., 0., 0., 0., .1, .001, 4, 1)
    roles = SimpleNamespace(
        state=SimpleNamespace(global_step=0, max_steps=1, consumed_samples=0, consumed_tokens=0),
        device=torch.device("cpu"), parallel_dims=SimpleNamespace(dp_rank=0, dp_size=2, tp_rank=0),
        rollout_manager=SimpleNamespace(agent_runner=SimpleNamespace(
            settings=GenerationSettings(8, 1., 1., 0, True, 0, 2, collect_log_probs=True))),
        rollout_engine=engine, actor=SimpleNamespace(update=Mock(return_value=update)),
        run_synchronized=lambda _, callback: callback(), align_rollout=lambda batch: batch,
        prepare_experience=lambda batch: batch, release_training_residency=Mock(),
        publish_policy=Mock(side_effect=publish), tracker=Mock(), evaluator=None,
        close=Mock(side_effect=lambda: events.append("roles_close")), checkpoints=Mock(),
        evaluation_needs_gateway_handoff=False,
    )

    roles.checkpoints.will_save.return_value = False
    roles.checkpoints.config = {}
    trainer.__dict__.update(vars(roles))
    trainer.publication = SimpleNamespace(publish=getattr(trainer, "publish_policy", None))
    trainer.lifecycle = SimpleNamespace(close=getattr(trainer, "close", None))
    trainer.residency = SimpleNamespace(release=getattr(trainer, "release_training_residency", None))
    trainer.targets = SimpleNamespace(prepare=lambda batch, *_args: (trainer.prepare_experience(batch), {}))
    trainer._dp_group_info = None
    trainer.rollouter = Mock()
    trainer.rollouter.sample_batch.side_effect = lambda *_: (events.append("sample_return") or payloads)
    trainer.rollouter.pause.side_effect = lambda: events.append("partial_pause_done")
    trainer.rollouter.commit_batch.side_effect = lambda: events.append("buffer_commit")
    trainer.rollouter.resume.side_effect = lambda _: events.append("partial_resume_done")
    trainer.rollouter.drain_and_snapshot.return_value = {"buffer": {}, "source": {"cursor": 2}}
    trainer.trace = Mock()
    trainer.trace.emit.side_effect = lambda event, **_: events.append(event)
    trainer._restore_and_start = Mock()
    trainer._report_update = Mock()
    return trainer, events, payloads
