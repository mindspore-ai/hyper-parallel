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
"""Unit tests for ``hyper_parallel.trainer.parallel_dims``.

``ParallelDims`` is the fail-fast validator that catches misconfigurations
**before** any model is built. Coverage here pins down:

1. Construction-time inference (``dp_shard=-1`` auto-fill) and product check.
2. Per-degree positivity and HSDP / EP / Ulysses rules.
3. ``from_config`` legacy ``dp`` → ``dp_shard`` migration.
4. ``validate_against_model`` cross-checks (heads/tp, kv_heads/tp,
   experts/ep, seq_len/(cp*tp)).
5. Convenience properties + ``summary`` string shape.
6. ``build_mesh`` invokes ``init_device_mesh`` with the canonical dim order
   and wires the ``"fsdp" / "dp" / "loss"`` flatten aliases for each
   parallel composition (1D-FSDP, HSDP, HSDP+CP).

``build_mesh`` is exercised against a fake ``init_device_mesh`` so the test
runs without distributed init.
"""
# pylint: disable=protected-access
import os
import types
import unittest
from unittest.mock import patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.trainer.parallel_dims import ParallelDims


_FLATTEN_ALIASES = ("fsdp", "dp", "loss")


class _FakeMesh:
    """Minimal stand-in for ``DeviceMesh`` exercising the flatten-alias path.

    ``_register_flatten_aliases`` reads ``_get_root_mesh().get_flatten_mapping()``
    and calls ``mesh[dims].flatten(alias)`` — this fake records each
    ``(dims, alias)`` pair so tests can assert wiring without a real PG.
    """

    def __init__(self, mesh_dim_names):
        self.mesh_dim_names = tuple(mesh_dim_names)
        self.flatten_calls = []
        self._flatten_map = {}

    def __getitem__(self, key):
        return _FakeChild(parent=self, key=key)

    def _get_root_mesh(self):
        return self

    def get_flatten_mapping(self):
        return dict(self._flatten_map)


class _FakeChild:
    def __init__(self, parent, key):
        self.parent = parent
        self.key = key

    def flatten(self, alias):
        self.parent.flatten_calls.append((self.key, alias))
        self.parent._flatten_map[alias] = self


class TestConstructionAndValidation(unittest.TestCase):
    """Construction-time inference and product-check rules."""

    def test_default_single_card_world_one_is_valid(self):
        """All-1 dims with world_size=1 must construct without raising."""
        pd = ParallelDims(world_size=1)
        self.assertEqual(pd.dp_size, 1)
        self.assertEqual(pd.non_dp_size, 1)

    def test_negative_degree_raises_value_error(self):
        """Any non-Ulysses degree < 1 is invalid."""
        with self.assertRaises(ValueError):
            ParallelDims(dp_replicate=0, world_size=1)
        with self.assertRaises(ValueError):
            ParallelDims(tp=-2, world_size=1)

    def test_dp_shard_zero_is_invalid(self):
        """``dp_shard`` may be -1 (auto) or >=1, never 0."""
        with self.assertRaises(ValueError):
            ParallelDims(dp_shard=0, world_size=1)

    def test_product_mismatch_raises_with_helpful_message(self):
        """Product != world_size must surface a clear error mentioning ``dp_shard=-1``."""
        with self.assertRaises(ValueError) as ctx:
            ParallelDims(dp_replicate=2, dp_shard=3, world_size=8)
        msg = str(ctx.exception)
        self.assertIn("world_size(8)", msg, f"Expected world size in error, got {msg!r}")
        self.assertIn("dp_shard=-1", msg, (
            f"Error must hint at auto-infer (-1), got {msg!r}"
        ))

    def test_dp_shard_auto_infer_fills_remaining(self):
        """``dp_shard=-1`` is filled from world_size / (dp_rep * cp * tp * pp * ep)."""
        pd = ParallelDims(dp_shard=-1, tp=2, cp=2, world_size=16)
        # 16 / (1 * 2 * 2 * 1 * 1) = 4
        self.assertEqual(pd.dp_shard, 4, f"Expected dp_shard=4, got {pd.dp_shard}")

    def test_dp_shard_auto_infer_non_divisible_raises(self):
        """Non-divisible auto-infer must raise — not silently round."""
        with self.assertRaises(ValueError):
            ParallelDims(dp_shard=-1, tp=3, world_size=8)

    def test_pp_not_implemented(self):
        """``pp > 1`` must fail fast with ``NotImplementedError`` (no model wired yet)."""
        with self.assertRaises(NotImplementedError):
            ParallelDims(pp=2, world_size=2)

    def test_ep_etp_rule(self):
        """When ``ep > 1``, ``etp`` must equal ``tp`` or 1.

        ``etp`` is not part of the product check (it's a sub-dim of ``ep``),
        so use world_size = dp_replicate*dp_shard*cp*tp*pp*ep = 1*1*1*2*1*2 = 4
        to isolate the etp validation from the product check.
        """
        with self.assertRaises(ValueError) as ctx:
            ParallelDims(tp=2, ep=2, etp=4, world_size=4)
        self.assertIn("etp=4", str(ctx.exception))
        # etp == tp is fine.
        ParallelDims(tp=2, ep=2, etp=2, world_size=4)
        # etp == 1 is fine.
        ParallelDims(tp=2, ep=2, etp=1, world_size=4)

    def test_ulysses_must_divide_cp(self):
        """``ulysses_degree`` must be <= cp and divide cp."""
        with self.assertRaises(ValueError):
            ParallelDims(cp=4, ulysses_degree=8, world_size=4)
        with self.assertRaises(ValueError):
            ParallelDims(cp=4, ulysses_degree=3, world_size=4)
        # Divides cleanly.
        ParallelDims(cp=4, ulysses_degree=2, world_size=4)


class TestFromConfig(unittest.TestCase):
    """``from_config`` maps the legacy ``dp`` field to ``dp_shard``."""

    def test_legacy_dp_field_maps_to_dp_shard(self):
        """Legacy single ``dp=4`` becomes ``dp_shard=4`` (FSDP behavior)."""
        cfg = types.SimpleNamespace(dp=4)
        pd = ParallelDims.from_config(cfg, world_size=4)
        self.assertEqual(pd.dp_shard, 4, f"Legacy dp→dp_shard mapping failed: {pd.dp_shard}")
        self.assertEqual(pd.dp_replicate, 1)

    def test_explicit_dp_shard_overrides_legacy(self):
        """If both ``dp_shard`` and legacy ``dp`` are set, ``dp_shard`` wins."""
        cfg = types.SimpleNamespace(dp=4, dp_shard=2, dp_replicate=2)
        pd = ParallelDims.from_config(cfg, world_size=4)
        self.assertEqual(pd.dp_shard, 2, f"Explicit dp_shard must win, got {pd.dp_shard}")
        self.assertEqual(pd.dp_replicate, 2)

    def test_etp_defaults_to_tp_when_missing(self):
        """``etp`` defaults to ``tp`` when not supplied."""
        cfg = types.SimpleNamespace(tp=2, ep=2, dp=1)
        pd = ParallelDims.from_config(cfg, world_size=4)
        self.assertEqual(pd.etp, 2, f"etp must default to tp=2, got {pd.etp}")


class TestValidateAgainstModel(unittest.TestCase):
    """Cross-check parallel dims against model hyperparameters."""

    def _model(self, **cfg_kwargs):
        return types.SimpleNamespace(config=types.SimpleNamespace(**cfg_kwargs))

    def test_no_config_returns_silently(self):
        """A model without ``.config`` is allowed (no checks possible)."""
        pd = ParallelDims(tp=2, world_size=2)
        pd.validate_against_model(types.SimpleNamespace(), seq_len=128)  # no raise

    def test_heads_not_divisible_by_tp_raises(self):
        """``num_attention_heads % tp != 0`` must fail with a fix-it hint."""
        pd = ParallelDims(tp=8, world_size=8)
        with self.assertRaises(ValueError) as ctx:
            pd.validate_against_model(self._model(num_attention_heads=12), seq_len=None)
        self.assertIn("num_attention_heads=12", str(ctx.exception))

    def test_kv_heads_not_divisible_by_tp_raises(self):
        """GQA constraint: ``num_key_value_heads % tp != 0`` must fail."""
        pd = ParallelDims(tp=4, world_size=4)
        with self.assertRaises(ValueError):
            pd.validate_against_model(
                self._model(num_attention_heads=16, num_key_value_heads=6),
                seq_len=None,
            )

    def test_num_experts_not_divisible_by_ep_raises(self):
        """``num_experts % ep != 0`` must fail."""
        pd = ParallelDims(ep=2, tp=1, etp=1, world_size=2)
        with self.assertRaises(ValueError):
            pd.validate_against_model(self._model(num_experts=7), seq_len=None)

    def test_seq_len_must_be_divisible_by_cp_times_tp(self):
        """``seq_len % (cp * tp) != 0`` must fail with a hint about the divisor."""
        pd = ParallelDims(tp=2, cp=2, world_size=4)
        with self.assertRaises(ValueError) as ctx:
            pd.validate_against_model(
                self._model(num_attention_heads=4, num_key_value_heads=4),
                seq_len=130,
            )
        self.assertIn("cp*tp=4", str(ctx.exception))

    def test_valid_config_passes(self):
        """Sanity: a divisible config validates without raising."""
        pd = ParallelDims(tp=2, cp=2, world_size=4)
        pd.validate_against_model(
            self._model(
                num_attention_heads=16, num_key_value_heads=4, num_experts=4,
            ),
            seq_len=2048,
        )


class TestConvenienceAndSummary(unittest.TestCase):
    """Boolean properties + summary line shape."""

    def test_properties(self):
        pd = ParallelDims(dp_replicate=2, dp_shard=2, tp=2, world_size=8)
        self.assertEqual(pd.dp_size, 4)
        self.assertEqual(pd.non_dp_size, 2)
        self.assertTrue(pd.fsdp_enabled)
        self.assertTrue(pd.tp_enabled)
        self.assertFalse(pd.cp_enabled)
        self.assertFalse(pd.ep_enabled)
        self.assertFalse(pd.pp_enabled)

    def test_summary_contains_canonical_fields(self):
        """``summary`` must surface the parallel-dim factors and world size."""
        pd = ParallelDims(dp_shard=2, tp=2, world_size=4)
        summary = pd.summary()
        for token in ("dp_replicate=1", "dp_shard=2", "tp=2", "world=4"):
            self.assertIn(token, summary, (
                f"summary must contain '{token}', got {summary!r}"
            ))


class TestBuildMesh(unittest.TestCase):
    """``build_mesh`` calls ``init_device_mesh`` with canonical dim order and aliases."""

    @staticmethod
    def _make_fake_init():
        """Return a stub ``init_device_mesh`` that records its inputs and returns a fake mesh."""
        calls = {}

        def _fake_init(device_type, mesh_shape, mesh_dim_names):
            calls["device_type"] = device_type
            calls["mesh_shape"] = mesh_shape
            calls["mesh_dim_names"] = mesh_dim_names
            return _FakeMesh(mesh_dim_names)

        return _fake_init, calls

    def test_single_card_creates_degenerate_dp_shard_mesh(self):
        """All-1 dims → 1D ``(world_size,)`` ``dp_shard`` mesh so FSDP code still runs."""
        fake_init, calls = self._make_fake_init()
        pd = ParallelDims(world_size=1)
        with patch("hyper_parallel.trainer.parallel_dims.init_device_mesh", fake_init):
            mesh = pd.build_mesh("npu")
        self.assertEqual(calls["mesh_shape"], (1,))
        self.assertEqual(calls["mesh_dim_names"], ("dp_shard",))
        self.assertIsInstance(mesh, _FakeMesh)

    def test_canonical_dim_order(self):
        """Dim order must be ``dp_replicate → dp_shard → ep → cp → tp → pp``."""
        fake_init, calls = self._make_fake_init()
        pd = ParallelDims(
            dp_replicate=2, dp_shard=2, ep=2, cp=2, tp=2, world_size=32,
        )
        with patch("hyper_parallel.trainer.parallel_dims.init_device_mesh", fake_init):
            pd.build_mesh("npu")
        self.assertEqual(
            calls["mesh_dim_names"],
            ("dp_replicate", "dp_shard", "ep", "cp", "tp"),
            f"canonical order broken, got {calls['mesh_dim_names']!r}",
        )

    def test_pure_fsdp_registers_fsdp_dp_loss_aliases(self):
        """Pure 1D FSDP (only ``dp_shard``) registers ``fsdp``/``dp``/``loss`` all pointing at dp_shard."""
        fake_init, _ = self._make_fake_init()
        pd = ParallelDims(dp_shard=4, world_size=4)
        with patch("hyper_parallel.trainer.parallel_dims.init_device_mesh", fake_init):
            mesh = pd.build_mesh("npu")
        flat_aliases = {alias for _, alias in mesh.flatten_calls}
        for required in _FLATTEN_ALIASES:
            self.assertIn(required, flat_aliases, (
                f"Pure FSDP missing alias '{required}', got {flat_aliases}"
            ))

    def test_hsdp_dp_alias_combines_replicate_and_shard(self):
        """HSDP: ``dp`` alias must combine ``(dp_replicate, dp_shard)``."""
        fake_init, _ = self._make_fake_init()
        pd = ParallelDims(dp_replicate=2, dp_shard=2, world_size=4)
        with patch("hyper_parallel.trainer.parallel_dims.init_device_mesh", fake_init):
            mesh = pd.build_mesh("npu")
        dp_calls = [src for src, alias in mesh.flatten_calls if alias == "dp"]
        self.assertEqual(len(dp_calls), 1, (
            f"Expected one 'dp' flatten call, got {mesh.flatten_calls!r}"
        ))
        self.assertEqual(dp_calls[0], ("dp_replicate", "dp_shard"), (
            f"'dp' alias must combine replicate+shard, got {dp_calls[0]!r}"
        ))

    def test_cp_active_loss_alias_includes_cp(self):
        """With ``cp > 1`` the ``loss`` alias must fold ``cp`` into the data-parallel mesh."""
        fake_init, _ = self._make_fake_init()
        pd = ParallelDims(dp_shard=2, cp=2, world_size=4)
        with patch("hyper_parallel.trainer.parallel_dims.init_device_mesh", fake_init):
            mesh = pd.build_mesh("npu")
        loss_calls = [src for src, alias in mesh.flatten_calls if alias == "loss"]
        self.assertEqual(len(loss_calls), 1, (
            f"Expected one 'loss' flatten call, got {mesh.flatten_calls!r}"
        ))
        self.assertEqual(loss_calls[0], ("dp_shard", "cp"), (
            f"'loss' alias with cp must combine dp_shard+cp, got {loss_calls[0]!r}"
        ))

    def test_hsdp_plus_cp_loss_alias_folds_three_axes(self):
        """HSDP + CP: ``loss`` must combine ``(dp_replicate, dp_shard, cp)`` — three-axis flatten.

        Real production combo (HSDP outer + FSDP inner + CP for long-context)
        that the trainer relies on for global token reduction. A regression
        that drops the cp axis from the loss group would under-count tokens
        on CP-sharded ranks and inflate per-token loss numerically.
        """
        fake_init, _ = self._make_fake_init()
        pd = ParallelDims(dp_replicate=2, dp_shard=2, cp=2, world_size=8)
        with patch("hyper_parallel.trainer.parallel_dims.init_device_mesh", fake_init):
            mesh = pd.build_mesh("npu")
        loss_calls = [src for src, alias in mesh.flatten_calls if alias == "loss"]
        self.assertEqual(len(loss_calls), 1, (
            f"Expected one 'loss' flatten call, got {mesh.flatten_calls!r}"
        ))
        self.assertEqual(loss_calls[0], ("dp_replicate", "dp_shard", "cp"), (
            f"HSDP+CP 'loss' alias must fold all three axes, got {loss_calls[0]!r}"
        ))

    def test_pure_tp_skips_fsdp_alias(self):
        """Pure TP (no dp_shard, no dp_replicate) → ``fsdp`` / ``dp`` aliases NOT registered.

        With ``tp=2, world=2``, ``dp_shard`` filters out at base-name selection
        (size 1) so the trainer's later ``mesh["fsdp"]`` call must explicitly
        handle the absence. This test pins that absence so a future "always
        register fsdp" regression would surface here, not deep in trainer
        plumbing.
        """
        fake_init, _ = self._make_fake_init()
        pd = ParallelDims(tp=2, world_size=2)
        with patch("hyper_parallel.trainer.parallel_dims.init_device_mesh", fake_init):
            mesh = pd.build_mesh("npu")
        aliases = {alias for _, alias in mesh.flatten_calls}
        self.assertNotIn("fsdp", aliases, (
            f"Pure TP must NOT register 'fsdp' alias (no dp_shard axis), "
            f"got aliases={aliases}"
        ))
        self.assertNotIn("dp", aliases, (
            f"Pure TP must NOT register 'dp' alias either, got aliases={aliases}"
        ))

    def test_dp_shard_minus_one_with_pp_raises_not_implemented(self):
        """``dp_shard=-1`` first auto-infers, then ``pp>1`` check still fires.

        Pins the validation ordering: auto-infer succeeds (world=2 / (pp=2)=1)
        so ``dp_shard`` is set to 1, then the ``pp>1`` guard raises with the
        right error class. A regression that re-ordered the checks would
        silently accept the misconfig at ``__post_init__`` and crash later
        inside ``parallelize_module``.
        """
        with self.assertRaises(NotImplementedError):
            ParallelDims(dp_shard=-1, pp=2, world_size=2)

    def test_etp_rule_silent_when_ep_disabled(self):
        """``ep == 1`` — the ``etp == tp or 1`` rule must NOT fire.

        Documented as "only meaningful when ep > 1"; pins the silent-pass
        behaviour so a regression that drops the ``if self.ep > 1`` guard
        would surface as an unexpected ValueError here.
        """
        # etp=5 is normally invalid (not tp and not 1), but with ep=1 the
        # whole rule must be skipped. Use world_size = tp = 2 to satisfy
        # the product constraint.
        ParallelDims(tp=2, ep=1, etp=5, world_size=2)  # must NOT raise

    def test_dp_shard_minus_one_no_slack_clamps_to_one(self):
        """``world_size`` equal to the non-DP product → auto-infer yields 1.

        Boundary: ``world_size // non_dp == 1`` after the ``max(..., 1)`` clamp;
        the product check then passes with the inferred ``dp_shard=1``.
        Regression guard for ``parallel_dims.py:155``.
        """
        pd = ParallelDims(dp_shard=-1, tp=2, world_size=2)
        self.assertEqual(pd.dp_shard, 1, (
            f"dp_shard=-1 with no slack must clamp to 1, got {pd.dp_shard}"
        ))


if __name__ == "__main__":
    unittest.main()
