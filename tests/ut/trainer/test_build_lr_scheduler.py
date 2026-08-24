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
"""Unit tests for ``BaseTrainer._build_lr_scheduler``.

LR schedule is pure CPU math but lives behind the trainer's lifecycle so
without dedicated tests it's only validated by end-to-end loss curves —
slow, expensive, and ambiguous when a regression hits. UT pins the
contracts:

1. **Warmup ramp** — linear ``lr * step/warmup`` for ``step < warmup_steps``.
2. **Ceil warmup** — ``warmup_steps = ceil(total * warmup_ratio)`` so a
   fractional ratio rounds *up* (matches the standard convention; rounding
   down would under-warm and risk early-step divergence on large LR).
3. **Cosine decay** — at the end of warmup ``lr = lr_max``; at ``current==
   total`` the schedule lands on ``lr_min`` (not zero) via the asymptote
   formula.
4. **Constant decay** — ``lr_decay_style="constant"`` keeps the post-warmup
   LR flat at ``lr_max`` for the rest of training.
5. **lr_min asymptote** — ``lr_min / lr_max`` becomes the floor of the
   cosine envelope; if ``lr_max <= 0`` the ratio is clamped to 0 (no NaN).
"""
# pylint: disable=protected-access
import math
import os
import types
import unittest
from unittest.mock import patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import torch
from torch import nn

from hyper_parallel.trainer import base as trainer_base
from hyper_parallel.trainer.base import BaseTrainer, TrainerState


def _make_args(*, max_steps=10, lr=1.0, lr_min=0.1,
               lr_warmup_ratio=0.2, lr_decay_style="cosine"):
    return types.SimpleNamespace(
        model=types.SimpleNamespace(name="toy"),
        train=types.SimpleNamespace(
            max_steps=max_steps,
            optimizer=types.SimpleNamespace(
                lr=lr,
                lr_min=lr_min,
                lr_warmup_ratio=lr_warmup_ratio,
                lr_decay_style=lr_decay_style,
                weight_decay=0.0,
                eps=1e-8,
                betas=(0.9, 0.999),
                foreach=None,
            ),
        ),
    )


def _build_trainer(args):
    with patch.object(trainer_base, "get_spec", return_value=object()):
        trainer = BaseTrainer(args)
    trainer.state = TrainerState(max_steps=args.train.max_steps)
    # Need a real optimizer for LambdaLR — single trivial parameter.
    p = nn.Parameter(torch.zeros(1))
    trainer.optimizer = torch.optim.SGD([p], lr=args.train.optimizer.lr)
    return trainer


def _collect_lrs(trainer, steps):
    """Step the optimizer ``steps`` times and return the resulting LR series."""
    lrs = [trainer.lr_scheduler.get_last_lr()[0]]
    for _ in range(steps):
        trainer.lr_scheduler.step()
        lrs.append(trainer.lr_scheduler.get_last_lr()[0])
    return lrs


class TestWarmupRamp(unittest.TestCase):
    """During warmup, LR climbs linearly from 0 to ``lr_max``."""

    def test_linear_warmup_reaches_max_at_warmup_end(self):
        """At ``step == warmup_steps``, LR equals the configured ``lr_max``."""
        args = _make_args(max_steps=10, lr=2.0, lr_warmup_ratio=0.3)
        trainer = _build_trainer(args)
        trainer._build_lr_scheduler()
        # warmup_steps = ceil(10 * 0.3) = 3
        lrs = _collect_lrs(trainer, steps=5)
        # step 0 / 3 = 0; step 1 / 3 = 0.667 of lr_max
        self.assertAlmostEqual(lrs[0], 0.0, places=6, msg=(
            f"warmup start must be 0, got {lrs[0]}"
        ))
        self.assertAlmostEqual(lrs[1], 2.0 / 3, places=6, msg=(
            f"warmup step 1 must be lr_max * 1/3, got {lrs[1]}"
        ))
        self.assertAlmostEqual(lrs[3], 2.0, places=6, msg=(
            f"warmup must reach lr_max at step==warmup_steps, got {lrs[3]}"
        ))

    def test_ceil_warmup_rounds_up_fractional_ratio(self):
        """A ratio that produces a non-integer warmup must round up (ceil)."""
        # 10 * 0.25 = 2.5 → ceil → 3
        args = _make_args(max_steps=10, lr=1.0, lr_warmup_ratio=0.25)
        trainer = _build_trainer(args)
        trainer._build_lr_scheduler()
        lrs = _collect_lrs(trainer, steps=3)
        # If warmup_steps was 2 (floor), step 2 would be at lr_max already.
        # With ceil(2.5)=3, step 2 must be < lr_max.
        self.assertLess(lrs[2], 1.0, (
            f"ceil(2.5)=3 warmup, step 2 must be < lr_max=1.0, got {lrs[2]}"
        ))
        self.assertAlmostEqual(lrs[3], 1.0, places=6, msg=(
            f"step 3 must reach lr_max=1.0 with ceil warmup, got {lrs[3]}"
        ))


class TestCosineDecay(unittest.TestCase):
    """Post-warmup cosine envelope ends at ``lr_min``, not at zero."""

    def test_cosine_ends_at_lr_min(self):
        """At ``current_step == total_steps`` LR equals ``lr_min`` (asymptote)."""
        args = _make_args(
            max_steps=10, lr=1.0, lr_min=0.1, lr_warmup_ratio=0.0,
            lr_decay_style="cosine",
        )
        trainer = _build_trainer(args)
        trainer._build_lr_scheduler()
        # 10 steps — final get_last_lr after the 10th step.
        lrs = _collect_lrs(trainer, steps=10)
        # At current_step == total_steps: progress=1, cos(pi)=-1, decay=0.
        # min_ratio = lr_min/lr_max = 0.1; lr = 0.1 + (1-0.1)*0 = 0.1.
        self.assertAlmostEqual(lrs[-1], 0.1, places=6, msg=(
            f"cosine bottom must equal lr_min=0.1, got {lrs[-1]}"
        ))

    def test_cosine_midpoint_is_half_decay(self):
        """At the midpoint of the post-warmup window, cos(pi/2)=0 → lr halfway."""
        args = _make_args(
            max_steps=20, lr=1.0, lr_min=0.0, lr_warmup_ratio=0.0,
            lr_decay_style="cosine",
        )
        trainer = _build_trainer(args)
        trainer._build_lr_scheduler()
        lrs = _collect_lrs(trainer, steps=20)
        # progress = 10/20 = 0.5, cos(pi*0.5)=0, decay = 0.5*(1+0)=0.5
        self.assertAlmostEqual(lrs[10], 0.5, places=6, msg=(
            f"cosine at midpoint must yield 0.5*lr_max, got {lrs[10]}"
        ))


class TestConstantDecay(unittest.TestCase):
    """``decay_style='constant'`` keeps post-warmup LR flat."""

    def test_constant_holds_lr_max_post_warmup(self):
        """After warmup, the LR must stay at ``lr_max`` for every remaining step."""
        args = _make_args(
            max_steps=8, lr=3.0, lr_min=0.5, lr_warmup_ratio=0.25,
            lr_decay_style="constant",
        )
        trainer = _build_trainer(args)
        trainer._build_lr_scheduler()
        lrs = _collect_lrs(trainer, steps=8)
        # warmup_steps = ceil(8*0.25) = 2. Steps 2..8 must equal lr_max=3.0.
        for step, lr in enumerate(lrs):
            if step >= 2:
                self.assertAlmostEqual(lr, 3.0, places=6, msg=(
                    f"constant decay: step {step} must be lr_max=3.0, got {lr}"
                ))


class TestLrMinAsymptote(unittest.TestCase):
    """``lr_min`` defines the cosine envelope floor; degenerate ``lr_max<=0`` is safe."""

    def test_negative_lr_max_clamps_min_ratio_to_zero(self):
        """If ``lr_max <= 0``, the ratio collapses to 0 to avoid NaN.

        Defensive guard in ``base.py:630`` ``min_ratio = lr_min/lr_max if lr_max > 0 else 0.0``.
        Without it a configuration mistake (negative LR) would produce NaN
        scaling and propagate silently into the optimizer step.
        """
        args = _make_args(max_steps=5, lr=0.0, lr_min=0.1, lr_warmup_ratio=0.0)
        trainer = _build_trainer(args)
        trainer._build_lr_scheduler()
        lrs = _collect_lrs(trainer, steps=5)
        for step, lr in enumerate(lrs):
            self.assertFalse(math.isnan(lr), (
                f"step {step}: LR must not be NaN even with lr_max=0, got {lr}"
            ))


if __name__ == "__main__":
    unittest.main()
