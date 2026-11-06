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
"""Teacher scoring at the training boundary."""
from typing import Any
import torch.distributed as dist
from rl.distillation.teacher import attach_teacher_scores
from rl.data.contracts import ExperienceBatch
from rl.trainer.distributed import run_synchronized

class TeacherScorer:
    """Score and align teacher outputs across tensor-parallel ranks."""

    def __init__(self, *, teacher_manager: Any = None, parallel_dims: Any = None) -> None:
        """Bind explicit dependencies for this lifecycle boundary."""
        self.teacher_manager = teacher_manager
        self.parallel_dims = parallel_dims

    def score(self, prompt_records: Any, rollout: ExperienceBatch) -> ExperienceBatch:
        """Only TP owners score; synchronize errors before sharing results with siblings."""
        owner = self.parallel_dims.tp_size == 1 or self.parallel_dims.tp_rank == 0
        result = run_synchronized(
            'teacher scoring',
            lambda: self.teacher_manager.score_batch(rollout, prompt_records) if owner else None,
        )
        if self.parallel_dims.tp_size > 1:
            group = self.parallel_dims.device_mesh['tp'].get_group()
            source = dist.get_global_rank(group, 0)
            shared = [result]
            dist.broadcast_object_list(shared, src=source, group=group)
            result = shared[0]
        if result is None:
            raise RuntimeError('Teacher scoring completed without a result')
        return run_synchronized('teacher score alignment', lambda: attach_teacher_scores(rollout, result))
