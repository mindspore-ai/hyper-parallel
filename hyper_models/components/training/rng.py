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
"""StatefulRNG — stub for stateful random number generator.

Full implementation depends on distributed RNG alignment (rank-aware seeding).
Following design doc 03_training_loop.md §3 (③.2).
"""

import torch


class StatefulRNG:
    """Stateful RNG with rank-aware seeding.

    Stub — stores seed and provides state_dict/load_state_dict for checkpoint.
    """

    def __init__(self, seed: int = 42, ranked: bool = True):
        self.seed = seed
        self.ranked = ranked
        import torch.distributed as dist
        if ranked and dist.is_initialized():
            self.seed = seed + dist.get_rank()
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)

    def state_dict(self) -> dict:
        return {
            "seed": self.seed,
            "generator_state": self._generator.get_state(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.seed = state.get("seed", self.seed)
        if "generator_state" in state:
            self._generator.set_state(state["generator_state"])

    @property
    def generator(self) -> torch.Generator:
        return self._generator