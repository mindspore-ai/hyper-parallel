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
"""Unit tests for ``PPRankOwnedSampler`` and ``mpipe_owned_micros``.

These pin down the disk-shard data path used by ``data.load: single`` +
``pp_schedule: mpipe``. A regression here would resurface the
"dataloader exhausted" symptom (per-rank batch count desync with the
schedule's per-step consumption) and the loss-drift symptom (per-rank
sample identity diverging across PP ranks).

Pure-Python: the sampler only manipulates integers, so a stub iterable
stands in for the base ``DistributedSampler``.
"""
# pylint: disable=protected-access
import os
import unittest

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.core.pipeline_parallel.mpipe import (  # pylint: disable=wrong-import-position
    PPRankOwnedSampler, mpipe_owned_micros)


class TestMpipeOwnedMicros(unittest.TestCase):
    """``mpipe_owned_micros`` ownership contract (round-robin):

    ``NT = min(pp_size, micro_batch_num)``; rank ``i < NT`` owns
    ``{m | m in range(M) and m % NT == i}``; rank ``i >= NT`` (the M<PP
    case) owns nothing. The schedule emits ``DATA_LOAD`` per owned micro
    and the sampler must yield exactly the samples for those slots.
    """

    def test_balanced_pp_equals_m(self):
        """PP == M: round-robin collapses to identity (each rank owns its own index)."""
        for r in range(4):
            self.assertEqual(
                mpipe_owned_micros(pp_size=4, micro_batch_num=4, pp_rank=r),
                frozenset({r}),
                f"rank {r} should own only micro {r} when PP==M",
            )

    def test_m_greater_than_pp_round_robin(self):
        """M > PP: overflow micros distribute round-robin instead of dumping on rank 0."""
        # PP=4, M=8 → each rank owns 2 micros, ⌊M/NT⌋ apart.
        for r in range(4):
            self.assertEqual(
                mpipe_owned_micros(pp_size=4, micro_batch_num=8, pp_rank=r),
                frozenset({r, r + 4}),
                f"rank {r} should own {{{r}, {r + 4}}} under round-robin",
            )

    def test_m_uneven_round_robin(self):
        """M not a multiple of NT: lower ranks get one extra micro."""
        # PP=4, M=6 → ranks 0,1 own 2 each; ranks 2,3 own 1 each.
        self.assertEqual(mpipe_owned_micros(4, 6, 0), frozenset({0, 4}))
        self.assertEqual(mpipe_owned_micros(4, 6, 1), frozenset({1, 5}))
        self.assertEqual(mpipe_owned_micros(4, 6, 2), frozenset({2}))
        self.assertEqual(mpipe_owned_micros(4, 6, 3), frozenset({3}))

    def test_min_mode_overflow_on_rank0(self):
        """mode='min': rank 0 absorbs the overflow; ranks 1..NT-1 own only themselves."""
        # PP=4, M=8 → rank 0 owns {0, 4, 5, 6, 7}; others own {i}.
        self.assertEqual(
            mpipe_owned_micros(4, 8, 0, mode="min"),
            frozenset({0, 4, 5, 6, 7}),
        )
        for r in (1, 2, 3):
            self.assertEqual(
                mpipe_owned_micros(4, 8, r, mode="min"),
                frozenset({r}),
            )

    def test_min_and_full_agree_when_m_equals_nt(self):
        """Both modes collapse to identity when M <= NT."""
        for r in range(4):
            self.assertEqual(
                mpipe_owned_micros(4, 4, r, mode="full"),
                mpipe_owned_micros(4, 4, r, mode="min"),
                f"rank {r}: 'full' and 'min' should agree at M=NT",
            )

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            mpipe_owned_micros(4, 4, 0, mode="half")

    def test_m_less_than_pp_trailing_ranks_idle(self):
        """M < PP: ranks ``>= NT`` own nothing (the trainer rejects this config)."""
        self.assertEqual(
            mpipe_owned_micros(pp_size=4, micro_batch_num=2, pp_rank=0),
            frozenset({0}),
        )
        self.assertEqual(
            mpipe_owned_micros(pp_size=4, micro_batch_num=2, pp_rank=1),
            frozenset({1}),
        )
        self.assertEqual(
            mpipe_owned_micros(pp_size=4, micro_batch_num=2, pp_rank=2),
            frozenset(),
        )
        self.assertEqual(
            mpipe_owned_micros(pp_size=4, micro_batch_num=2, pp_rank=3),
            frozenset(),
        )

    def test_coverage_disjoint_and_complete(self):
        """Across all PP ranks, owned-micro sets are disjoint and cover ``0..M-1`` in both modes."""
        for mode in ("full", "min"):
            for pp in (1, 2, 4, 8):
                for mbn in (1, 2, 4, 8):
                    union = set()
                    for r in range(pp):
                        owned = mpipe_owned_micros(pp, mbn, r, mode=mode)
                        self.assertTrue(
                            union.isdisjoint(owned),
                            f"mode={mode} PP={pp} M={mbn} rank={r} owned={owned} overlaps {union}",
                        )
                        union |= owned
                    self.assertEqual(
                        union, set(range(mbn)),
                        f"mode={mode} PP={pp} M={mbn} coverage={union} != {set(range(mbn))}",
                    )


class TestPPRankOwnedSampler(unittest.TestCase):
    """``PPRankOwnedSampler`` shards a base sampler's stream by micro slot.

    The base sampler emits a deterministic DP-sharded stream of dataset
    indices. The wrapper walks the stream in chunks of
    ``pp_micro_batch_num * micro_batch_size`` (micro-major layout) and
    re-yields only the slots in ``owned_micros``. The number of steps an
    epoch supports == ``len(base) // step_size`` regardless of which
    micros a rank owns; this is what the trainer's pre-flight check uses
    to detect a too-small dataset.
    """

    def test_balanced_one_micro_per_rank(self):
        """PP=4, M=4, mbs=1: each rank yields exactly one index per step chunk."""
        base = list(range(80))  # 20 steps' worth at step_chunk=4.
        for rank in range(4):
            owned = mpipe_owned_micros(4, 4, rank)
            sampler = PPRankOwnedSampler(
                base_sampler=base, owned_micros=owned,
                micro_batch_size=1, pp_micro_batch_num=4,
            )
            yielded = list(iter(sampler))
            # Each step chunk is [4k, 4k+1, 4k+2, 4k+3]; rank `r` owns micro
            # `r`, so it yields the (4k+r)-th index of every chunk.
            expected = [4 * k + rank for k in range(20)]
            self.assertEqual(yielded, expected, (
                f"rank {rank}: yielded {yielded[:8]}... expected {expected[:8]}..."
            ))
            self.assertEqual(len(sampler), len(yielded), (
                f"__len__={len(sampler)} != actual yields={len(yielded)} (rank {rank})"
            ))

    def test_micro_batch_size_greater_than_one(self):
        """mbs=2: each owned micro pulls 2 contiguous indices from the chunk."""
        base = list(range(64))  # step_chunk = 4*2 = 8 → 8 step chunks.
        sampler = PPRankOwnedSampler(
            base_sampler=base, owned_micros=frozenset({1}),
            micro_batch_size=2, pp_micro_batch_num=4,
        )
        yielded = list(iter(sampler))
        # Micro 1 occupies chunk slots [2, 3] of every 8-index chunk.
        expected = [8 * k + 2 for k in range(8)] + [8 * k + 3 for k in range(8)]
        expected = sorted(expected)
        self.assertEqual(sorted(yielded), expected)
        self.assertEqual(len(yielded), 16)
        self.assertEqual(len(sampler), 16)

    def test_round_robin_overflow_when_m_gt_pp(self):
        """PP=4, M=8: round-robin owns {r, r+NT}; each rank yields 2 per chunk."""
        base = list(range(80))  # 10 step chunks of 8.
        for rank in range(4):
            owned = mpipe_owned_micros(4, 8, rank)
            self.assertEqual(owned, frozenset({rank, rank + 4}))
            sampler = PPRankOwnedSampler(
                base_sampler=base, owned_micros=owned,
                micro_batch_size=1, pp_micro_batch_num=8,
            )
            yielded = list(iter(sampler))
            expected = sorted(
                [8 * k + m for k in range(10) for m in (rank, rank + 4)]
            )
            self.assertEqual(sorted(yielded), expected, f"rank {rank} round-robin yield")
            self.assertEqual(len(sampler), 2 * 10)

    def test_coverage_invariant_across_ranks(self):
        """Union of all ranks' yields == base sampler, with no duplicates.

        This is the disk-shard correctness contract: every sample is read
        exactly once across the PP group (the whole point of the feature).
        """
        for pp, mbn in [(4, 4), (4, 8), (2, 4), (1, 4)]:
            base = list(range(mbn * 5 * 1))  # 5 step chunks.
            seen = []
            for r in range(pp):
                owned = mpipe_owned_micros(pp, mbn, r)
                if not owned:
                    continue
                s = PPRankOwnedSampler(
                    base_sampler=list(base), owned_micros=owned,
                    micro_batch_size=1, pp_micro_batch_num=mbn,
                )
                seen.extend(iter(s))
            self.assertEqual(
                sorted(seen), list(base),
                f"PP={pp} M={mbn}: union of per-rank yields != base sampler",
            )

    def test_drops_partial_trailing_chunk(self):
        """Indices past the last complete chunk are dropped (matches ``drop_last=True``)."""
        # 10 indices, step_chunk = 4 → 2 full chunks (indices 0..7); 8, 9 dropped.
        base = list(range(10))
        sampler = PPRankOwnedSampler(
            base_sampler=base, owned_micros=frozenset({0}),
            micro_batch_size=1, pp_micro_batch_num=4,
        )
        yielded = list(iter(sampler))
        self.assertEqual(yielded, [0, 4])
        self.assertEqual(len(sampler), 2)

    def test_len_matches_iter_for_imbalanced_lengths(self):
        """``__len__`` must equal the actual yield count — the trainer's
        pre-flight ``available_steps = base_len // step_chunk`` check relies
        on this; a mismatch would re-introduce "dataloader exhausted".
        """
        for base_size in (4, 7, 8, 15, 16, 40, 41, 79, 80):
            for owned in (frozenset({0}), frozenset({2, 3}), frozenset({0, 1, 2, 3})):
                sampler = PPRankOwnedSampler(
                    base_sampler=list(range(base_size)),
                    owned_micros=owned,
                    micro_batch_size=1,
                    pp_micro_batch_num=4,
                )
                self.assertEqual(
                    len(list(iter(sampler))), len(sampler),
                    f"base={base_size} owned={sorted(owned)}: __len__ != actual",
                )

    def test_steps_supported_matches_dataset_sizing(self):
        """Regression test for the user-reported "dataloader exhausted" bug.

        With ``data.load: single``, each training step globally consumes
        ``pp_micro_batch_num * micro_batch_size * dp_size`` samples; the
        dummy dataset is sized ``max_steps * global_batch_size``. When
        ``global_batch_size == pp_micro_batch_num * micro_batch_size * dp_size``
        the sampler supplies exactly ``max_steps`` per rank and training
        completes. When ``global_batch_size`` is too small the sampler
        under-supplies and the trainer's pre-flight check must fire.
        """
        max_steps, pp, mbn, mbs, dp = 20, 4, 4, 1, 1

        # Balanced: global_batch_size == M*mbs*dp == 4; total = 80.
        gbs = mbn * mbs * dp
        base_len = (max_steps * gbs) // dp
        step_chunk = mbn * mbs
        available_steps = base_len // step_chunk
        self.assertEqual(available_steps, max_steps, (
            f"balanced: gbs={gbs} should feed exactly max_steps={max_steps}"
        ))

        # Imbalanced: lower gbs to 2 → only 10 steps supported, must be < max_steps.
        gbs_small = 2
        base_len_small = (max_steps * gbs_small) // dp
        available_small = base_len_small // step_chunk
        self.assertLess(available_small, max_steps, (
            f"unbalanced: gbs={gbs_small} should under-feed max_steps={max_steps}"
        ))
        for r in range(pp):
            owned = mpipe_owned_micros(pp, mbn, r)
            sampler = PPRankOwnedSampler(
                base_sampler=list(range(base_len_small)),
                owned_micros=owned,
                micro_batch_size=mbs,
                pp_micro_batch_num=mbn,
            )
            # Each rank's __len__ is full_chunks * len(owned) * mbs ==
            # available_small * len(owned) * mbs, so steps == available_small.
            steps_for_rank = len(sampler) // (len(owned) * mbs)
            self.assertEqual(steps_for_rank, available_small, (
                f"rank {r}: supports {steps_for_rank} steps, expected {available_small}"
            ))


class TestPPRankOwnedSamplerStateForwarding(unittest.TestCase):
    """``set_epoch`` / ``state_dict`` / ``load_state_dict`` forward to the base."""

    class _StatefulBase:
        """Records ``set_epoch`` calls; exposes ``state_dict``/``load_state_dict``."""
        def __init__(self) -> None:
            """Start with no recorded epochs and no restored state."""
            self.epoch_calls = []
            self.saved = None

        def __iter__(self) -> iter:
            """Yield nothing; only the state hooks are under test."""
            return iter([])

        def __len__(self) -> int:
            """Report an empty stream."""
            return 0

        def set_epoch(self, epoch: int) -> None:
            """Record the forwarded epoch."""
            self.epoch_calls.append(epoch)

        def state_dict(self) -> dict:
            """Return the recorded epochs."""
            return {"epoch": list(self.epoch_calls)}

        def load_state_dict(self, state: dict) -> None:
            """Capture the state handed back by the wrapper."""
            self.saved = state

    def test_set_epoch_forwarded(self):
        base = self._StatefulBase()
        sampler = PPRankOwnedSampler(base, {0}, 1, 4)
        sampler.set_epoch(5)
        sampler.set_epoch(7)
        self.assertEqual(base.epoch_calls, [5, 7])

    def test_state_dict_wraps_base(self):
        base = self._StatefulBase()
        base.epoch_calls = [3]
        sampler = PPRankOwnedSampler(base, {0}, 1, 4)
        self.assertEqual(sampler.state_dict(), {"base": {"epoch": [3]}})

    def test_load_state_dict_unwraps_base(self):
        base = self._StatefulBase()
        sampler = PPRankOwnedSampler(base, {0}, 1, 4)
        sampler.load_state_dict({"base": {"epoch": [42]}})
        self.assertEqual(base.saved, {"epoch": [42]})

    def test_stateless_base_no_crash(self):
        """A base sampler without ``state_dict`` must not break the wrapper."""
        sampler = PPRankOwnedSampler([1, 2, 3], {0}, 1, 4)
        self.assertEqual(sampler.state_dict(), {})
        sampler.load_state_dict({"base": {"anything": 1}})  # no-op, no crash


if __name__ == "__main__":
    unittest.main()
