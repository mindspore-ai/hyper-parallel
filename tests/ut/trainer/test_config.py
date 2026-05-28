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
"""Unit tests for ``hyper_parallel.trainer.config``.

The config layer is the trainer's only user-facing surface, so a regression
here breaks every entry script. Coverage focuses on the rules that the
schema docstring asserts and that have failed before:

1. ``_string_to_bool`` accepts the documented alias set (``yes/no/y/n/on/off/...``).
2. ``_deep_update`` merges nested dicts without clobbering siblings.
3. ``_resolve_field_type`` walks nested dataclasses and unwraps ``Optional[X]``.
4. ``_coerce_cli_value`` routes bool fields through the alias parser while
   leaving numeric/string fields on the int → float → str heuristic.
5. ``_validate_top_level`` rejects every legacy flat key with a migration
   hint so users get a real error instead of a silent default.
6. ``_instantiate_recursive`` builds nested dataclasses from dicts, warns on
   typos, and converts string booleans inline.
7. ``parse_args`` end-to-end happy path: YAML + CLI dot-path overrides.
8. ``HyperTrainerConfig`` post-init computes ``train_steps`` from ``max_steps``.
"""
# pylint: disable=protected-access
import io
import logging
import os
import sys
import tempfile
import textwrap
import unittest
from dataclasses import dataclass, field
from typing import Optional

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.trainer.config import (
    AcceleratorConfig,
    DataConfig,
    HyperTrainerConfig,
    ModelConfig,
    TrainConfig,
    _coerce_cli_value,
    _deep_update,
    _instantiate_recursive,
    _resolve_field_type,
    _string_to_bool,
    _validate_top_level,
    parse_args,
)


@dataclass
class _Inner:
    value: int = 1
    flag: bool = False
    nullable: Optional[int] = None


@dataclass
class _Outer:
    inner: _Inner = field(default_factory=_Inner)
    name: str = "default"


class TestStringToBool(unittest.TestCase):
    """``_string_to_bool`` accepts the documented alias set."""

    def test_true_aliases(self):
        for alias in ("true", "yes", "y", "on", "1", "t", "TRUE", "Yes"):
            self.assertTrue(
                _string_to_bool(alias),
                f"Expected '{alias}' to map to True",
            )

    def test_false_aliases(self):
        for alias in ("false", "no", "n", "off", "0", "f", "FALSE", "No"):
            self.assertFalse(
                _string_to_bool(alias),
                f"Expected '{alias}' to map to False",
            )

    def test_bool_passthrough(self):
        self.assertTrue(_string_to_bool(True))
        self.assertFalse(_string_to_bool(False))

    def test_invalid_string_raises(self):
        with self.assertRaises(ValueError):
            _string_to_bool("maybe")

    def test_non_string_non_bool_raises(self):
        with self.assertRaises(ValueError):
            _string_to_bool(42)


class TestDeepUpdate(unittest.TestCase):
    """``_deep_update`` merges nested dicts in place."""

    def test_merges_nested_without_clobbering_siblings(self):
        src = {"a": {"b": 1, "c": 2}, "d": 9}
        overrides = {"a": {"c": 3}}
        result = _deep_update(src, overrides)
        self.assertEqual(result["a"], {"b": 1, "c": 3}, (
            f"_deep_update must not clobber 'b', got {result!r}"
        ))
        self.assertEqual(result["d"], 9, (
            f"_deep_update must not touch sibling 'd', got {result!r}"
        ))

    def test_override_replaces_non_dict_values(self):
        src = {"a": 1}
        overrides = {"a": [1, 2]}
        result = _deep_update(src, overrides)
        self.assertEqual(result["a"], [1, 2])


class TestResolveFieldType(unittest.TestCase):
    """``_resolve_field_type`` walks dot-path through dataclasses and unwraps Optional."""

    def test_walks_nested_dataclass(self):
        self.assertIs(_resolve_field_type(_Outer, "inner.value"), int)
        self.assertIs(_resolve_field_type(_Outer, "inner.flag"), bool)

    def test_unwraps_optional(self):
        self.assertIs(_resolve_field_type(_Outer, "inner.nullable"), int)

    def test_unknown_path_returns_none(self):
        self.assertIsNone(_resolve_field_type(_Outer, "inner.does_not_exist"))


class TestCoerceCliValue(unittest.TestCase):
    """``_coerce_cli_value`` routes bool fields through the alias parser."""

    def test_bool_field_uses_alias_parser(self):
        """A bool field accepts ``yes`` / ``no`` directly."""
        self.assertTrue(_coerce_cli_value("yes", "inner.flag", _Outer))
        self.assertFalse(_coerce_cli_value("off", "inner.flag", _Outer))

    def test_int_field_parses_int(self):
        self.assertEqual(_coerce_cli_value("42", "inner.value", _Outer), 42)

    def test_string_field_keeps_string(self):
        self.assertEqual(_coerce_cli_value("foo", "name", _Outer), "foo")

    def test_falls_back_to_float_for_decimal_numbers(self):
        """Decimal CLI values for non-bool fields must coerce to float."""
        # name is str so float parsing won't apply; use a path with an int field
        # to confirm int wins, then a free-form CLI key (unresolved type) for float.
        self.assertEqual(_coerce_cli_value("3.5", "unknown.path", _Outer), 3.5)


class TestValidateTopLevel(unittest.TestCase):
    """Top-level YAML keys are restricted to ``model``/``data``/``train``."""

    def test_accepts_canonical_keys(self):
        _validate_top_level({"model": {}, "data": {}, "train": {}})

    def test_rejects_legacy_flat_keys_with_migration_hint(self):
        with self.assertRaises(ValueError) as ctx:
            _validate_top_level({"parallel": {}, "optim": {}})
        msg = str(ctx.exception)
        self.assertIn("'parallel:'", msg, f"Migration hint missing for 'parallel', got {msg!r}")
        self.assertIn("train.accelerator", msg)
        self.assertIn("'optim:'", msg)
        self.assertIn("train.optimizer", msg)

    def test_rejects_unknown_key_without_hint(self):
        with self.assertRaises(ValueError) as ctx:
            _validate_top_level({"garbage_key": {}})
        self.assertIn("garbage_key", str(ctx.exception))


class TestInstantiateRecursive(unittest.TestCase):
    """``_instantiate_recursive`` materialises nested dataclasses from dicts."""

    def test_builds_nested_dataclass(self):
        instance = _instantiate_recursive(
            _Outer, {"inner": {"value": 7, "flag": True}, "name": "x"}
        )
        self.assertEqual(instance.inner.value, 7)
        self.assertTrue(instance.inner.flag)
        self.assertEqual(instance.name, "x")

    def test_string_bool_is_normalised(self):
        instance = _instantiate_recursive(_Outer, {"inner": {"flag": "yes"}})
        self.assertTrue(instance.inner.flag, (
            f"String 'yes' must coerce to bool True, got {instance.inner.flag!r}"
        ))

    def test_unknown_key_emits_warning_with_suggestion(self):
        """Unknown dict keys must surface a ``difflib`` closest-match warning, not a silent drop."""
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.WARNING)
        cfg_logger = logging.getLogger("hyper_parallel.trainer.config")
        cfg_logger.addHandler(handler)
        cfg_logger.setLevel(logging.WARNING)
        try:
            _instantiate_recursive(_Outer, {"naem": "typo"})  # "naem" → "name"
        finally:
            cfg_logger.removeHandler(handler)
        output = stream.getvalue()
        self.assertIn("naem", output, f"Warning must reference the bad key, got {output!r}")
        self.assertIn("name", output, (
            f"Warning must include closest-match suggestion 'name', got {output!r}"
        ))


class TestHyperTrainerConfigPostInit(unittest.TestCase):
    """Post-init must mirror ``train.max_steps`` to ``train_steps``."""

    def test_train_steps_mirrors_max_steps(self):
        cfg = HyperTrainerConfig(train=TrainConfig(max_steps=321))
        self.assertEqual(cfg.train_steps, 321, (
            f"train_steps must mirror train.max_steps, got {cfg.train_steps}"
        ))

    def test_defaults_are_strict_three_tier(self):
        """Default construction holds the three top-level slots: model/data/train."""
        cfg = HyperTrainerConfig()
        self.assertIsInstance(cfg.model, ModelConfig)
        self.assertIsInstance(cfg.data, DataConfig)
        self.assertIsInstance(cfg.train, TrainConfig)
        self.assertIsInstance(cfg.train.accelerator, AcceleratorConfig)


class TestParseArgs(unittest.TestCase):
    """End-to-end ``parse_args`` happy path: YAML + CLI overrides + bool aliases."""

    def setUp(self):
        self._saved_argv = sys.argv
        self._saved_env = os.environ.get("LOCAL_RANK")
        os.environ.pop("LOCAL_RANK", None)

    def tearDown(self):
        sys.argv = self._saved_argv
        if self._saved_env is None:
            os.environ.pop("LOCAL_RANK", None)
        else:
            os.environ["LOCAL_RANK"] = self._saved_env

    def _write_yaml(self, body: str, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(body))

    def test_yaml_plus_cli_override_with_dot_paths(self):
        """CLI dot-path overrides YAML, bool aliases coerce, local_rank lifted from env."""
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = os.path.join(tmp, "cfg.yaml")
            self._write_yaml(
                """
                model:
                  name: qwen3_5
                data:
                  type: dummy
                  max_seq_len: 512
                train:
                  max_steps: 10
                  accelerator:
                    tp: 1
                    reshard_after_forward: true
                  mixed_precision:
                    enabled: false
                """,
                yaml_path,
            )
            os.environ["LOCAL_RANK"] = "3"
            sys.argv = [
                "prog",
                yaml_path,
                "--train.accelerator.tp=2",
                "--train.mixed_precision.enabled=yes",
                "--train.max_steps=99",
            ]
            cfg = parse_args(HyperTrainerConfig)

        self.assertEqual(cfg.model.name, "qwen3_5")
        self.assertEqual(cfg.data.max_seq_len, 512)
        self.assertEqual(cfg.train.max_steps, 99, (
            f"CLI override must replace YAML max_steps, got {cfg.train.max_steps}"
        ))
        self.assertEqual(cfg.train_steps, 99, (
            f"train_steps must mirror overridden max_steps, got {cfg.train_steps}"
        ))
        self.assertEqual(cfg.train.accelerator.tp, 2)
        self.assertTrue(cfg.train.mixed_precision.enabled, (
            f"'yes' must coerce to True for bool field, got {cfg.train.mixed_precision.enabled!r}"
        ))
        self.assertEqual(cfg.train.local_rank, 3, (
            f"local_rank must be lifted from LOCAL_RANK env, got {cfg.train.local_rank}"
        ))

    def test_missing_yaml_file_falls_back_to_defaults(self):
        """A missing YAML path warns but does not raise."""
        sys.argv = ["prog", "/no/such/file.yaml"]
        cfg = parse_args(HyperTrainerConfig)
        self.assertEqual(cfg.model.name, "qwen3_5")  # default

    def test_invalid_bool_string_in_cli_raises(self):
        """An unrecognised bool alias on a bool field must surface a ValueError.

        Regression guard: ``_coerce_cli_value`` falls back to the int→float→str
        heuristic when the bool parser rejects the value, so the failure has
        to happen later inside ``_instantiate_recursive``. A regression that
        silently swallows the error would default the bool to False without
        warning.
        """
        sys.argv = ["prog", "--train.mixed_precision.enabled=maybe"]
        with self.assertRaises(ValueError):
            parse_args(HyperTrainerConfig)

    def test_unknown_cli_field_emits_warning_with_suggestion(self):
        """CLI dot-path with a typo'd leaf must warn (closest match) and not raise.

        Production scenario: a user types ``--train.accelerator.dp_shrad=8``;
        without a warning the value is silently dropped and the job runs with
        the default. The ``_instantiate_recursive`` warning is the only signal.
        """
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.WARNING)
        cfg_logger = logging.getLogger("hyper_parallel.trainer.config")
        cfg_logger.addHandler(handler)
        cfg_logger.setLevel(logging.WARNING)
        try:
            # ``dp_shrad`` is a typo for ``dp_shard``; difflib should match it.
            sys.argv = ["prog", "--train.accelerator.dp_shrad=8"]
            parse_args(HyperTrainerConfig)
        finally:
            cfg_logger.removeHandler(handler)
        output = stream.getvalue()
        self.assertIn("dp_shrad", output, (
            f"Warning must reference the bad CLI key, got {output!r}"
        ))
        self.assertIn("dp_shard", output, (
            f"Warning must include closest-match suggestion 'dp_shard', got {output!r}"
        ))


class TestUnknownCliFieldsAreDropped(unittest.TestCase):
    """A CLI key with a typo at a level-2 field is dropped (with warning)."""

    def setUp(self):
        self._saved_argv = sys.argv
        self._saved_env = os.environ.get("LOCAL_RANK")
        os.environ.pop("LOCAL_RANK", None)

    def tearDown(self):
        sys.argv = self._saved_argv
        if self._saved_env is None:
            os.environ.pop("LOCAL_RANK", None)
        else:
            os.environ["LOCAL_RANK"] = self._saved_env

    def test_typoed_field_does_not_set_correct_field(self):
        """``--train.accelerator.dp_shrad=8`` must NOT silently set ``dp_shard``."""
        sys.argv = ["prog", "--train.accelerator.dp_shrad=8"]
        cfg = parse_args(HyperTrainerConfig)
        # dp_shard is Optional[int] defaulting to None — verify it's still None.
        self.assertIsNone(cfg.train.accelerator.dp_shard, (
            f"Typoed CLI key must NOT bind to 'dp_shard', "
            f"got {cfg.train.accelerator.dp_shard!r}"
        ))


if __name__ == "__main__":
    unittest.main()
