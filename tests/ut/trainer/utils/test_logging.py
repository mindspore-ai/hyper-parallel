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
"""Unit tests for ``hyper_parallel.trainer.utils.logging``.

Covers the four invariants that make this module load-bearing for the
trainer:

1. Rank lookup falls back through ``platform`` → ``RANK`` env → ``LOCAL_RANK``
   env → ``0`` (never raises).
2. ``init_logger`` is idempotent and replaces handlers on re-init.
3. ``info_rank0`` / ``warning_rank0`` fire **only** on rank 0; ``logger.info``
   /``logger.warning`` fire on every rank (the bug the previous global
   filter caused).
4. ``info_once`` / ``warning_once`` dedupe via ``lru_cache`` so the same
   message is logged at most once per ``(logger, msg)`` pair.
"""
# pylint: disable=protected-access
import io
import logging
import os
import unittest
from unittest.mock import patch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

from hyper_parallel.trainer.utils import logging as hp_logging
from hyper_parallel.trainer.utils.logging import (
    _get_rank,
    _install_logger_methods,
    get_logger,
    info_once,
    info_rank0,
    init_logger,
    warning_once,
    warning_rank0,
)


def _reset_once_caches() -> None:
    """Clear LRU caches so ``*_once`` tests are independent."""
    hp_logging._info_once_cached.cache_clear()
    hp_logging._warning_once_cached.cache_clear()


class TestRankLookup(unittest.TestCase):
    """``_get_rank`` must never raise and must follow the documented order."""

    def setUp(self):
        # Snapshot the env keys we mutate so other tests stay isolated.
        self._saved_env = {k: os.environ.get(k) for k in ("RANK", "LOCAL_RANK")}
        for k in ("RANK", "LOCAL_RANK"):
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_rank_from_platform_when_initialised(self):
        """Platform path takes priority when ``get_rank()`` succeeds."""
        with patch("hyper_parallel.get_platform") as mock_get_platform:
            mock_get_platform.return_value.get_rank.return_value = 3
            self.assertEqual(_get_rank(), 3)

    def test_rank_falls_back_to_rank_env_when_platform_fails(self):
        """Platform raising ``RuntimeError`` (uninit'd PG) falls through to env."""
        os.environ["RANK"] = "5"
        with patch("hyper_parallel.get_platform") as mock_get_platform:
            mock_get_platform.return_value.get_rank.side_effect = RuntimeError("no pg")
            self.assertEqual(_get_rank(), 5)

    def test_rank_falls_back_to_local_rank_then_zero(self):
        """``LOCAL_RANK`` is the last env fallback; missing → 0."""
        with patch("hyper_parallel.get_platform") as mock_get_platform:
            mock_get_platform.return_value.get_rank.side_effect = RuntimeError("no pg")
            os.environ["LOCAL_RANK"] = "7"
            self.assertEqual(_get_rank(), 7)
            os.environ.pop("LOCAL_RANK")
            self.assertEqual(_get_rank(), 0)


class TestInitLogger(unittest.TestCase):
    """``init_logger`` must reset handlers and inject the rank attribute."""

    def setUp(self):
        self._root_handlers = list(logging.getLogger().handlers)

    def tearDown(self):
        root = logging.getLogger()
        for h in list(root.handlers):
            root.removeHandler(h)
        for h in self._root_handlers:
            root.addHandler(h)

    def test_init_logger_is_idempotent(self):
        """Calling ``init_logger`` twice must not stack handlers."""
        init_logger(stream=io.StringIO())
        first_count = len(logging.getLogger().handlers)
        init_logger(stream=io.StringIO())
        second_count = len(logging.getLogger().handlers)
        self.assertEqual(first_count, second_count, (
            f"init_logger should replace handlers, got "
            f"first={first_count}, second={second_count}"
        ))

    def test_rank_injector_populates_record_rank(self):
        """The rank-injecting filter must add ``record.rank`` (never drop record)."""
        stream = io.StringIO()
        init_logger(stream=stream, fmt="rank=%(rank)s %(message)s")
        with patch.object(hp_logging, "_get_rank", return_value=2):
            logging.getLogger("test_rank_injector").info("hello")
        output = stream.getvalue()
        self.assertIn("rank=2", output, f"Expected rank injected, got output={output!r}")
        self.assertIn("hello", output, f"Expected message preserved, got output={output!r}")


class TestRankZeroHelpers(unittest.TestCase):
    """``info_rank0`` / ``warning_rank0`` must fire only on rank 0."""

    def setUp(self):
        _install_logger_methods()
        self.stream = io.StringIO()
        init_logger(stream=self.stream, fmt="%(levelname)s %(message)s")
        self.logger = get_logger("test_rank_zero")
        self.logger.setLevel(logging.DEBUG)

    def tearDown(self):
        root = logging.getLogger()
        for h in list(root.handlers):
            root.removeHandler(h)

    def test_info_rank0_only_fires_on_rank_zero(self):
        """``info_rank0`` must produce output on rank 0 and be silent elsewhere."""
        with patch.object(hp_logging, "_get_rank", return_value=0):
            self.logger.info_rank0("rank0_msg")
        with patch.object(hp_logging, "_get_rank", return_value=1):
            self.logger.info_rank0("rank1_msg")
        output = self.stream.getvalue()
        self.assertIn("rank0_msg", output, f"rank-0 message missing, got={output!r}")
        self.assertNotIn("rank1_msg", output, (
            f"rank-1 message must not be emitted by info_rank0, got={output!r}"
        ))

    def test_warning_rank0_only_fires_on_rank_zero(self):
        """``warning_rank0`` must produce output on rank 0 and be silent elsewhere."""
        with patch.object(hp_logging, "_get_rank", return_value=0):
            self.logger.warning_rank0("warn_rank0")
        with patch.object(hp_logging, "_get_rank", return_value=4):
            self.logger.warning_rank0("warn_rank4")
        output = self.stream.getvalue()
        self.assertIn("warn_rank0", output, f"rank-0 warning missing, got={output!r}")
        self.assertNotIn("warn_rank4", output, (
            f"non-rank-0 warning must be filtered, got={output!r}"
        ))

    def test_plain_info_fires_on_every_rank(self):
        """Regression: a non-rank-0 ``logger.info`` must NOT be filtered."""
        with patch.object(hp_logging, "_get_rank", return_value=3):
            self.logger.info("rank3_error")
        output = self.stream.getvalue()
        self.assertIn("rank3_error", output, (
            f"plain logger.info must fire on every rank, got={output!r}"
        ))


class TestOnceHelpers(unittest.TestCase):
    """``info_once`` / ``warning_once`` dedupe by (logger, message)."""

    def setUp(self):
        _install_logger_methods()
        _reset_once_caches()
        self.stream = io.StringIO()
        init_logger(stream=self.stream, fmt="%(levelname)s %(message)s")
        self.logger = get_logger("test_once")
        self.logger.setLevel(logging.DEBUG)

    def tearDown(self):
        _reset_once_caches()
        root = logging.getLogger()
        for h in list(root.handlers):
            root.removeHandler(h)

    def test_info_once_dedupes_same_message(self):
        """Identical messages on the same logger are emitted once."""
        with patch.object(hp_logging, "_get_rank", return_value=0):
            self.logger.info_once("only_once")
            self.logger.info_once("only_once")
            self.logger.info_once("only_once")
        emitted = self.stream.getvalue().count("only_once")
        self.assertEqual(emitted, 1, f"Expected 1 emission, got {emitted}")

    def test_info_once_args_formatting(self):
        """``info_once`` interpolates ``args`` and dedupes the formatted string."""
        with patch.object(hp_logging, "_get_rank", return_value=0):
            self.logger.info_once("step=%d lr=%s", 1, "1e-4")
            self.logger.info_once("step=%d lr=%s", 1, "1e-4")
            self.logger.info_once("step=%d lr=%s", 2, "1e-4")
        output = self.stream.getvalue()
        self.assertEqual(output.count("step=1 lr=1e-4"), 1, (
            f"Expected step=1 emitted once, output={output!r}"
        ))
        self.assertEqual(output.count("step=2 lr=1e-4"), 1, (
            f"Expected step=2 emitted once, output={output!r}"
        ))

    def test_warning_once_dedupes_independently_of_info_once(self):
        """``info_once`` and ``warning_once`` have separate caches."""
        with patch.object(hp_logging, "_get_rank", return_value=0):
            self.logger.info_once("shared_msg")
            self.logger.warning_once("shared_msg")
            self.logger.warning_once("shared_msg")
        output = self.stream.getvalue()
        # info + warning each emitted once; warning second call deduped.
        self.assertEqual(output.count("INFO shared_msg"), 1, (
            f"info_once should fire once, output={output!r}"
        ))
        self.assertEqual(output.count("WARNING shared_msg"), 1, (
            f"warning_once should fire once, output={output!r}"
        ))

    def test_once_suppressed_on_non_rank_zero(self):
        """``*_once`` only fire on rank 0 but still consume the dedupe slot."""
        with patch.object(hp_logging, "_get_rank", return_value=1):
            self.logger.info_once("hidden")
        output = self.stream.getvalue()
        self.assertNotIn("hidden", output, (
            f"info_once must not emit on non-rank-0, output={output!r}"
        ))


class TestLoggerMethodInstallation(unittest.TestCase):
    """The helpers must be installed onto ``logging.Logger`` exactly once."""

    def test_install_is_idempotent_and_attaches_methods(self):
        """Repeated ``_install_logger_methods`` calls keep a single set of bindings."""
        _install_logger_methods()
        first_info = logging.Logger.info_rank0
        _install_logger_methods()
        second_info = logging.Logger.info_rank0
        self.assertIs(first_info, second_info, (
            f"info_rank0 binding must be stable across installs, "
            f"first={first_info!r}, second={second_info!r}"
        ))
        for name in ("info_rank0", "warning_rank0", "info_once", "warning_once"):
            self.assertTrue(hasattr(logging.Logger, name), (
                f"logging.Logger must expose {name} after _install_logger_methods()"
            ))

    def test_module_level_helpers_callable_as_methods(self):
        """The free functions must be the same objects bound on Logger."""
        _install_logger_methods()
        self.assertIs(logging.Logger.info_rank0, info_rank0)
        self.assertIs(logging.Logger.warning_rank0, warning_rank0)
        self.assertIs(logging.Logger.info_once, info_once)
        self.assertIs(logging.Logger.warning_once, warning_once)


if __name__ == "__main__":
    unittest.main()
