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
"""Unit tests for ``hyper_parallel.compile.parallel_config.ParallelConfig``.

Covers the rules the dataclass asserts and that have regressed before:

1. Defaults match the documented intent (FSDP on, overlap on, degrees
   unresolved until runtime).
2. ``__post_init__`` rejects negative ``tp_size`` and non-positive explicit
   ``fsdp_degree``.
3. ``validate()`` re-runs the same checks after a caller mutates a field
   (the trainer back-fills ``fsdp_degree`` from the mesh in
   ``GraphTrainer._init_device_mesh``).
4. The dataclass is torch-free at import time (no top-level
   ``torch.distributed`` import) so it can be imported anywhere.
"""

import unittest

from hyper_parallel.compile.parallel_config import ParallelConfig


class TestParallelConfigDefaults(unittest.TestCase):
    """Defaults reflect graph-mode's FSDP-focused intent."""

    def test_defaults(self):
        """Test default config: FSDP on, overlap on, degrees unresolved."""
        cfg = ParallelConfig()
        self.assertTrue(
            cfg.enable_overlap,
            (f"enable_overlap default should be True, got {cfg.enable_overlap}"),
        )
        self.assertTrue(
            cfg.fsdp_enabled,
            (f"fsdp_enabled default should be True, got {cfg.fsdp_enabled}"),
        )
        self.assertIsNone(
            cfg.fsdp_degree,
            (
                f"fsdp_degree default should be None (resolve at runtime), "
                f"got {cfg.fsdp_degree}"
            ),
        )
        self.assertEqual(
            cfg.tp_size, 1, (f"tp_size default should be 1, got {cfg.tp_size}")
        )

    def test_explicit_construction(self):
        """Test explicit construction forwards every kwarg."""
        cfg = ParallelConfig(
            enable_overlap=False,
            fsdp_enabled=False,
            fsdp_degree=8,
            tp_size=2,
        )
        self.assertFalse(cfg.enable_overlap)
        self.assertFalse(cfg.fsdp_enabled)
        self.assertEqual(cfg.fsdp_degree, 8)
        self.assertEqual(cfg.tp_size, 2)


class TestParallelConfigValidation(unittest.TestCase):
    """``__post_init__`` catches misconfigurations early."""

    def test_rejects_non_positive_tp_size(self):
        """Test ``tp_size < 1`` raises ValueError at construction."""
        for bad in (0, -1, -4):
            with self.assertRaises(ValueError) as ctx:
                ParallelConfig(tp_size=bad)
            self.assertIn(
                "tp_size",
                str(ctx.exception),
                (
                    f"error for tp_size={bad} should mention tp_size, "
                    f"got: {ctx.exception}"
                ),
            )

    def test_rejects_non_positive_fsdp_degree(self):
        """Test explicit ``fsdp_degree < 1`` raises ValueError."""
        for bad in (0, -1, -8):
            with self.assertRaises(ValueError) as ctx:
                ParallelConfig(fsdp_degree=bad)
            self.assertIn(
                "fsdp_degree",
                str(ctx.exception),
                (
                    f"error for fsdp_degree={bad} should mention fsdp_degree, "
                    f"got: {ctx.exception}"
                ),
            )

    def test_fsdp_degree_none_allowed(self):
        """Test ``fsdp_degree=None`` is the documented auto-resolve sentinel."""
        cfg = ParallelConfig(fsdp_degree=None)
        self.assertIsNone(cfg.fsdp_degree)

    def test_validate_after_mutation(self):
        """Test ``validate()`` re-runs checks after a caller mutates a field.

        ``GraphTrainer._init_device_mesh`` writes ``fsdp_degree`` from the
        mesh sub-size after construction; ``validate()`` is the public hook
        for re-checking invariants then.
        """
        cfg = ParallelConfig()
        cfg.fsdp_degree = 4
        cfg.validate()  # should not raise

        cfg.fsdp_degree = -2
        with self.assertRaises(ValueError):
            cfg.validate()


class TestParallelConfigTorchFree(unittest.TestCase):
    """The dataclass must not import torch at module top.

    The previous implementation did ``import torch.distributed as dist`` at
    module scope and probed ``dist.is_initialized()`` inside a
    ``fsdp_enabled`` property — that turned FSDP on whenever distributed
    was initialized, with no opt-out, and made the dataclass unimportable
    without torch installed. The refactor moves the dist probe into
    ``FSDPPass.run`` (where it belongs) and turns ``fsdp_enabled`` into a
    plain config field.
    """

    def test_module_does_not_import_torch_at_top(self):
        """Test no top-level torch import in ``parallel_config`` module."""
        import hyper_parallel.compile.parallel_config as mod
        import sys

        # torch / torch.distributed must not be a side-effect of importing
        # the config module. (It may be imported by *something else* in the
        # process, so we check the module's own globals, not sys.modules.)
        self.assertNotIn(
            "torch",
            mod.__dict__,
            (
                "parallel_config module should not bind 'torch' in its globals; "
                "the dist probe belongs in FSDPPass.run, not the config dataclass"
            ),
        )
        self.assertNotIn(
            "dist",
            mod.__dict__,
            (
                "parallel_config module should not bind 'dist' (= torch.distributed) "
                "in its globals"
            ),
        )


if __name__ == "__main__":
    unittest.main()
