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
"""Two-process (gloo, CPU) check that DP aggregation equals a single-process pass."""

import math
import os
import tempfile
import unittest
from typing import Optional

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import
import torch.multiprocessing as mp  # pylint: disable=forbidden-backend-import

from hyper_parallel.evaluation import (
    Evaluator,
    EvalStepResult,
    LossMetric,
    LossStatistics,
    PerplexityMetric,
    ReductionContext,
)

# Per-sample (mean loss, valid tokens). Ranks hold disjoint samples and the
# token counts differ, so averaging per-rank means would give a wrong answer.
_SAMPLES = [(2.0, 10), (3.0, 4), (1.5, 6), (4.0, 20), (0.5, 8), (2.5, 12)]
_EXPECTED_LOSS = sum(m * n for m, n in _SAMPLES) / sum(n for _, n in _SAMPLES)


def _make_step(local: list) -> object:
    """Return a step that replays this rank's samples one per round."""
    state = {"i": 0}

    def step(_round: int) -> Optional[EvalStepResult]:
        """Replay the next local sample, or report exhaustion."""
        if state["i"] >= len(local):
            return None
        mean, tokens = local[state["i"]]
        state["i"] += 1
        return EvalStepResult(
            statistics={
                "foundation": LossStatistics(
                    numerator=torch.tensor(mean * tokens, dtype=torch.float64),
                    denominator=torch.tensor(float(tokens), dtype=torch.float64),
                    kind="lm_nll",
                )
            },
            sample_count=torch.tensor(1.0, dtype=torch.float64),
        )

    return step


def _worker(rank: int, world_size: int, init_file: str, result_dir: str) -> None:
    """Evaluate this rank's shard and save its report for the parent to compare."""
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        local = _SAMPLES[rank::world_size]
        evaluator = Evaluator(
            _make_step(local),
            [LossMetric(), PerplexityMetric()],
            ReductionContext(group=dist.group.WORLD, world_size=world_size),
            torch.device("cpu"),
        )
        report = evaluator.run()
        torch.save(report.to_dict(), os.path.join(result_dir, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


class TestDataParallelReduction(unittest.TestCase):
    """The global result must not depend on how samples were sharded."""

    def test_two_rank_result_equals_single_process_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            init_file = os.path.join(tmp, "init")
            mp.spawn(_worker, args=(2, init_file, tmp), nprocs=2, join=True)
            reports = [torch.load(os.path.join(tmp, f"rank{r}.pt")) for r in range(2)]

        for report in reports:
            self.assertAlmostEqual(report["metrics"]["loss"], _EXPECTED_LOSS, places=12)
            self.assertAlmostEqual(report["metrics"]["perplexity"], math.exp(_EXPECTED_LOSS), places=9)
            self.assertEqual(report["num_samples"], len(_SAMPLES))
            self.assertEqual(int(report["num_tokens"]), sum(n for _, n in _SAMPLES))
        # Every rank reports the same global numbers.
        self.assertEqual(reports[0]["metrics"], reports[1]["metrics"])

    def test_naive_per_rank_mean_would_differ(self):
        """Guard the premise: averaging rank means is not the token-weighted mean."""
        rank_means = []
        for rank in range(2):
            local = _SAMPLES[rank::2]
            rank_means.append(sum(m * n for m, n in local) / sum(n for _, n in local))
        self.assertNotAlmostEqual(sum(rank_means) / 2, _EXPECTED_LOSS, places=3)


if __name__ == "__main__":
    unittest.main()
