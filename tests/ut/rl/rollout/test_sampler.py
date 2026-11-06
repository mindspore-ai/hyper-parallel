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
"""Native seeded sampling correctness."""
from types import SimpleNamespace
from rl.rollout.sampler import patch_partial_prefill_rng


def test_partial_prefill_rng_restores_discarded_request_offsets() -> None:
    """Discarded partial-prefill samples restore their seeded generator offsets."""

    class FakeGenerator:
        """Expose the generator offset interface patched by vLLM-Ascend."""

        def __init__(self) -> None:
            self.offset = 0

        def get_offset(self) -> int:
            return self.offset

        def set_offset(self, offset: int) -> None:
            self.offset = offset

    class FakeModelRunner:
        """Model one sampling and bookkeeping cycle with one discarded row."""

        def __init__(self) -> None:
            self.generators = {0: FakeGenerator(), 1: FakeGenerator()}
            self.input_batch = SimpleNamespace(generators=self.generators)
            self.discard_request_indices = SimpleNamespace(np=[1])
            self.num_discarded_requests = 1

        def _sample(self) -> str:
            for generator in self.generators.values():
                generator.offset += 12
            return "sampled"

        def _bookkeeping_sync(self) -> str:
            return "bookkept"

    patch_partial_prefill_rng(FakeModelRunner)
    patch_partial_prefill_rng(FakeModelRunner)
    runner = FakeModelRunner()

    assert runner._sample() == "sampled"
    assert runner._bookkeeping_sync() == "bookkept"
    assert runner.generators[0].offset == 12
    assert runner.generators[1].offset == 0
    assert getattr(FakeModelRunner, "_rl_partial_prefill_rng_patched")
