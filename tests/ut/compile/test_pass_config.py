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
"""Unit tests for ``hyper_parallel.compile.pass_config.PassConfig``.

Covers the rules the dataclass asserts and that have regressed before:

1. Defaults match the documented intent (FSDP on, overlap on, degrees
   unresolved until runtime).
2. ``__post_init__`` rejects non-positive explicit ``fsdp_degree``.
3. ``validate()`` re-runs the same checks after a caller mutates a field
   (the compiler back-fills ``fsdp_degree`` from the mesh in
   ``GraphCompiler._init_device_mesh``).
4. The dataclass is torch-free at import time (no top-level
   ``torch.distributed`` import) so it can be imported anywhere.
"""

import unittest

from hyper_parallel.compile.pass_config import PassConfig


class TestPassConfigDefaults(unittest.TestCase):
    """Defaults reflect graph-mode's FSDP-focused intent."""

    def test_defaults(self):
        """Test default config: FSDP on, overlap on, degrees unresolved."""
        cfg = PassConfig()
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
            cfg.dp_mode,
            "fsdp",
            f"dp_mode default should be 'fsdp' (fully_shard), got {cfg.dp_mode!r}",
        )
        self.assertIsNone(
            cfg.dp_replicate_degree,
            (
                f"dp_replicate_degree default should be None, "
                f"got {cfg.dp_replicate_degree}"
            ),
        )
        # PP defaults: off by default (opt-in), degrees unresolved, safe
        # microbatch of 1.
        self.assertFalse(
            cfg.pp_enabled,
            (f"pp_enabled default should be False (opt-in), got {cfg.pp_enabled}"),
        )
        self.assertIsNone(
            cfg.pp_degree,
            (
                f"pp_degree default should be None (resolve at runtime), got {cfg.pp_degree}"
            ),
        )
        self.assertEqual(
            cfg.pp_microbatch_size,
            1,
            (f"pp_microbatch_size default should be 1, got {cfg.pp_microbatch_size}"),
        )
        self.assertEqual(
            cfg.pp_schedule,
            "gpipe",
            (f"pp_schedule default should be 'gpipe', got {cfg.pp_schedule!r}"),
        )
        self.assertEqual(
            cfg.pp_chunks,
            1,
            (f"pp_chunks default should be 1, got {cfg.pp_chunks}"),
        )

    def test_explicit_construction(self):
        """Test explicit construction forwards every kwarg."""
        cfg = PassConfig(
            enable_overlap=False,
            fsdp_enabled=False,
            fsdp_degree=8,
            dp_mode="hsdp",
            dp_replicate_degree=4,
            pp_enabled=True,
            pp_degree=4,
            pp_microbatch_size=2,
            pp_schedule="1f1b",
        )
        self.assertFalse(cfg.enable_overlap)
        self.assertFalse(cfg.fsdp_enabled)
        self.assertEqual(cfg.fsdp_degree, 8)
        self.assertEqual(cfg.dp_mode, "hsdp")
        self.assertEqual(cfg.dp_replicate_degree, 4)
        self.assertTrue(cfg.pp_enabled)
        self.assertEqual(cfg.pp_degree, 4)
        self.assertEqual(cfg.pp_microbatch_size, 2)
        self.assertEqual(cfg.pp_schedule, "1f1b")


class TestPassConfigDpMode(unittest.TestCase):
    """``dp_mode`` canonicalization + validation (simplefsdp parity)."""

    def test_accepts_canonical_modes(self):
        """Test the three canonical modes pass through unchanged."""
        for mode in ("fsdp", "ddp", "hsdp"):
            self.assertEqual(PassConfig(dp_mode=mode).dp_mode, mode)

    def test_normalizes_simplefsdp_aliases(self):
        """Test simplefsdp spellings normalize to the canonical modes."""
        self.assertEqual(PassConfig(dp_mode="fully_shard").dp_mode, "fsdp")
        self.assertEqual(PassConfig(dp_mode="replicate").dp_mode, "ddp")
        self.assertEqual(PassConfig(dp_mode="hybrid_shard").dp_mode, "hsdp")
        self.assertEqual(PassConfig(dp_mode="HSDP").dp_mode, "hsdp")

    def test_rejects_unknown_mode(self):
        """Test an unknown ``dp_mode`` raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            PassConfig(dp_mode="tensor_parallel")
        self.assertIn("dp_mode", str(ctx.exception))

    def test_rejects_non_positive_replicate_degree(self):
        """Test explicit ``dp_replicate_degree < 1`` raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            PassConfig(dp_mode="ddp", dp_replicate_degree=0)
        self.assertIn("dp_replicate_degree", str(ctx.exception))


class TestPassConfigValidation(unittest.TestCase):
    """``__post_init__`` catches misconfigurations early."""

    def test_rejects_non_positive_fsdp_degree(self):
        """Test explicit ``fsdp_degree < 1`` raises ValueError."""
        for bad in (0, -1, -8):
            with self.assertRaises(ValueError) as ctx:
                PassConfig(fsdp_degree=bad)
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
        cfg = PassConfig(fsdp_degree=None)
        self.assertIsNone(cfg.fsdp_degree)

    def test_rejects_unknown_pp_schedule(self):
        """Test an unknown ``pp_schedule`` name raises ValueError."""
        for bad in ("zbv", "1F-1B", ""):
            with self.assertRaises(ValueError) as ctx:
                PassConfig(pp_schedule=bad)
            self.assertIn(
                "pp_schedule",
                str(ctx.exception),
                f"error for pp_schedule={bad!r} should mention pp_schedule",
            )

    def test_pp_chunks_validation(self):
        """Test ``pp_chunks`` bounds and its interleaved-schedule requirement."""
        for bad in (0, -1):
            with self.assertRaises(ValueError) as ctx:
                PassConfig(pp_chunks=bad)
            self.assertIn("pp_chunks", str(ctx.exception))
        for bad_schedule in ("gpipe", "1f1b"):
            with self.assertRaises(ValueError) as ctx:
                PassConfig(pp_chunks=2, pp_schedule=bad_schedule)
            self.assertIn("pp_chunks", str(ctx.exception))
        for good in ("vpp", "interleaved_1f1b"):
            cfg = PassConfig(pp_chunks=2, pp_schedule=good)
            self.assertEqual(cfg.pp_chunks, 2)
            self.assertEqual(cfg.pp_schedule, good)

    def test_validate_after_mutation(self):
        """Test ``validate()`` re-runs checks after a caller mutates a field.

        ``GraphCompiler._init_device_mesh`` writes ``fsdp_degree`` from the
        mesh sub-size after construction; ``validate()`` is the public hook
        for re-checking invariants then.
        """
        cfg = PassConfig()
        cfg.fsdp_degree = 4
        cfg.validate()  # should not raise

        cfg.fsdp_degree = -2
        with self.assertRaises(ValueError):
            cfg.validate()


class TestPassConfigInductorBackend(unittest.TestCase):
    """``inductor_backend`` validation and its Inductor-mode pairing."""

    def test_defaults_to_default_backend(self):
        """Test the default backend leaves terminal compilation opt-in."""
        cfg = PassConfig()
        self.assertEqual(cfg.inductor_backend, "default")
        self.assertEqual(cfg.inductor_mode, "off")

    def test_accepts_and_canonicalizes_dvm(self):
        """Test the DVM backend is accepted alongside a compiling mode."""
        for mode in ("regional", "full"):
            cfg = PassConfig(inductor_mode=mode, inductor_backend="DVM")
            self.assertEqual(cfg.inductor_backend, "dvm")
            self.assertEqual(cfg.inductor_mode, mode)

    def test_rejects_unknown_backend(self):
        """Test an unknown backend spelling raises ValueError."""
        with self.assertRaisesRegex(ValueError, "inductor_backend"):
            PassConfig(inductor_mode="regional", inductor_backend="akg")

    def test_dvm_requires_compiling_inductor_mode(self):
        """Test the DVM backend cannot be paired with ``inductor_mode='off'``."""
        with self.assertRaisesRegex(ValueError, "inductor_mode"):
            PassConfig(inductor_backend="dvm")


class TestPassConfigTorchFree(unittest.TestCase):
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
        """Test no top-level torch import in ``pass_config`` module."""
        import hyper_parallel.compile.pass_config as mod  # pylint: disable=C0415

        # torch / torch.distributed must not be a side-effect of importing
        # the config module. (It may be imported by *something else* in the
        # process, so we check the module's own globals, not sys.modules.)
        self.assertNotIn(
            "torch",
            mod.__dict__,
            (
                "pass_config module should not bind 'torch' in its globals; "
                "the dist probe belongs in FSDPPass.run, not the config dataclass"
            ),
        )
        self.assertNotIn(
            "dist",
            mod.__dict__,
            (
                "pass_config module should not bind 'dist' (= torch.distributed) "
                "in its globals"
            ),
        )


if __name__ == "__main__":
    unittest.main()
