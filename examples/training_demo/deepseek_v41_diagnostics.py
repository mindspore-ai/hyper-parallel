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
"""Opt-in DSV4.1 MoE module timing for the full Trainer benchmark."""

import time
from typing import Any

import torch  # pylint: disable=forbidden-backend-import


class MoeTrainingDiagnostics:
    """Measure the four real MoE modules without replacing their computation."""

    def __init__(self, trainer: Any) -> None:
        """Install removable module hooks after the benchmark warmup.

        Args:
            trainer: Initialized text Trainer containing the real DSV4.1 model.
        """
        self.rows = []
        self.step = -1
        self._hooks = []
        self._starts = {}
        for name, module in trainer.base.model.named_modules():
            if name.endswith(".mlp"):
                self._time_moe(name, module)

    def _time_moe(self, name, module):
        def _begin(phase):
            torch.npu.synchronize()
            self._starts[(name, phase)] = time.perf_counter()

        def _end(phase):
            torch.npu.synchronize()
            self.rows.append({"step": self.step, "phase": f"{name}/{phase}",
                              "seconds": time.perf_counter() - self._starts.pop((name, phase))})

        self._hooks.extend([
            module.register_forward_pre_hook(lambda *_args: _begin("forward")),
            module.register_forward_hook(lambda *_args: _end("forward")),
            module.register_full_backward_pre_hook(lambda *_args: _begin("backward")),
            module.register_full_backward_hook(lambda *_args: _end("backward")),
        ])

    def close(self) -> None:
        """Remove the temporary timing hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
