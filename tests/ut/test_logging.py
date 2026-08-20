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
"""Unit tests for the centralized hyper_parallel.tools.logging system."""
import io
import logging
import os
import unittest

import hyper_parallel.tools.logging as hp_logging
from hyper_parallel.tools.logging import (
    HP_LOG_CONFIG_ENV,
    configure,
    get_logger,
    logger as manager,
    logging_enabled,
    set_format,
    set_level,
)


class TestHyperParallelLogging(unittest.TestCase):
    """Component-aware, format-decoupled, env-configurable logging."""

    def setUp(self):
        """Snapshot module globals mutated by tests so they can be restored."""
        self._saved_registry = dict(hp_logging._registry)
        self._saved_format = hp_logging._LOG_FORMAT
        self._saved_datefmt = hp_logging._DATE_FORMAT
        os.environ.pop(HP_LOG_CONFIG_ENV, None)

    def tearDown(self):
        """Restore registry and format, drop any loggers a test registered."""
        for name in list(hp_logging._registry):
            if name not in self._saved_registry:
                logging.getLogger(f"{hp_logging._NAMESPACE}.{name}").handlers = []
        hp_logging._registry.clear()
        hp_logging._registry.update(self._saved_registry)
        hp_logging._warned_unknown.clear()
        hp_logging._LOG_FORMAT = self._saved_format
        hp_logging._DATE_FORMAT = self._saved_datefmt
        # Re-apply the restored format only to handlers owned by this module.
        # pytest 9.1 attaches capture handlers to non-propagating loggers;
        # changing their formatter would make ordinary records require the
        # module-specific ``hp_component`` field.
        formatter = hp_logging._build_formatter()
        for component_logger in hp_logging._registry.values():
            for handler in component_logger.handlers:
                if any(
                    isinstance(item, hp_logging._ContextFilter)
                    for item in handler.filters
                ):
                    handler.setFormatter(formatter)
        os.environ.pop(HP_LOG_CONFIG_ENV, None)

    def test_get_logger_is_stable_and_namespaced(self):
        """Repeated lookups return the same namespaced logger."""
        first = get_logger("UnitA")
        second = get_logger("UnitA")
        self.assertIs(first, second)
        self.assertEqual(first.name, "hyper_parallel.UnitA")
        self.assertFalse(first.propagate)

    def test_off_by_default(self):
        """A freshly registered component is silent until enabled."""
        log = get_logger("UnitQuiet")
        self.assertEqual(log.level, logging.WARNING)
        self.assertFalse(logging_enabled("UnitQuiet", logging.INFO))

    def test_component_label_is_explicit_and_path_independent(self):
        """The component is the label passed in, not derived from any module path."""
        # Same label from anywhere -> one shared logger.
        self.assertIs(get_logger("FSDP"), get_logger("FSDP"))
        self.assertEqual(get_logger("FSDP").name, f"{hp_logging._NAMESPACE}.FSDP")
        # Distinct labels -> distinct loggers.
        self.assertIsNot(get_logger("FSDP"), get_logger("DTensor"))
        # No-arg falls back to the generic component (never derived from a path).
        self.assertEqual(
            get_logger().name, f"{hp_logging._NAMESPACE}.{hp_logging._DEFAULT_COMPONENT}"
        )

    def test_env_config_sets_per_component_levels(self):
        """HP_LOG_CONFIG enables and levels each component independently."""
        os.environ[HP_LOG_CONFIG_ENV] = "UnitB:INFO,UnitC:DEBUG"
        self.assertEqual(get_logger("UnitB").level, logging.INFO)
        self.assertEqual(get_logger("UnitC").level, logging.DEBUG)

    def test_env_config_bare_name_enables_at_info(self):
        """A component named without a level is enabled at INFO."""
        os.environ[HP_LOG_CONFIG_ENV] = "UnitD"
        self.assertEqual(get_logger("UnitD").level, logging.INFO)

    def test_configure_and_set_level_programmatically(self):
        """configure() and set_level() drive levels from code."""
        configure("UnitE:WARNING")
        self.assertEqual(get_logger("UnitE").level, logging.WARNING)
        set_level("UnitE", "DEBUG")
        self.assertEqual(get_logger("UnitE").level, logging.DEBUG)

    def test_manager_facade_mirrors_module_functions(self):
        """The imported `logger` manager exposes the same operations."""
        manager.set_level("UnitF", "INFO")
        self.assertEqual(manager.get_logger("UnitF").level, logging.INFO)
        self.assertTrue(manager.enabled("UnitF", logging.INFO))

    def test_default_format_has_level_component_and_callsite(self):
        """Records render '[LEVEL] [HP-component]: date [file: line] msg' on stdout."""
        log = get_logger("UnitG")
        stream = io.StringIO()
        log.handlers[0].stream = stream
        set_level("UnitG", "INFO")

        log.info("payload=%s", 7)

        out = stream.getvalue()
        self.assertIn("[INFO]", out)
        self.assertIn("[HP-UnitG]:", out)
        self.assertIn("test_logging.py:", out)  # call-site file
        self.assertIn("payload=7", out)
        self.assertNotIn("RANK", out)

    def test_set_format_is_decoupled_from_components(self):
        """Changing the global format updates already-registered handlers."""
        log = get_logger("UnitH")
        stream = io.StringIO()
        log.handlers[0].stream = stream
        set_level("UnitH", "INFO")

        # pytest 9.1 attaches its capture handler to existing non-propagating
        # loggers. Keep those external handlers out of the registry while
        # exercising set_format(), so the test cannot mutate pytest state.
        detached = {}
        for component_logger in hp_logging._registry.values():
            owned = [
                handler
                for handler in component_logger.handlers
                if any(
                    isinstance(item, hp_logging._ContextFilter)
                    for item in handler.filters
                )
            ]
            detached[component_logger] = [
                handler for handler in component_logger.handlers if handler not in owned
            ]
            component_logger.handlers = owned
        try:
            set_format(fmt="HP|%(hp_component)s|%(message)s")
        finally:
            for component_logger, handlers in detached.items():
                component_logger.handlers.extend(handlers)

        log.info("hi")

        self.assertIn("HP|UnitH|hi", stream.getvalue())

    def test_invalid_level_rejected(self):
        """An unknown level name fails loudly."""
        with self.assertRaisesRegex(ValueError, "Invalid logging level"):
            set_level("UnitI", "verbose")

    def test_known_component_matching_is_case_insensitive(self):
        """fsdp / Fsdp / FSDP resolve to one logger with the canonical FSDP label."""
        # Identity checks don't change the logger's level, so the shared FSDP
        # logger other tests rely on is left untouched.
        canonical = get_logger("FSDP")
        self.assertIs(get_logger("fsdp"), canonical)
        self.assertIs(get_logger("Fsdp"), canonical)
        self.assertEqual(canonical.name, "hyper_parallel.FSDP")
        # Env config is canonicalized too: a lower-case spec maps to the canonical label.
        self.assertEqual(hp_logging._parse_config("fsdp:DEBUG"), {"FSDP": logging.DEBUG})

    def test_unknown_component_still_works_but_warns_once(self):
        """A typo'd component is usable (never blocked) and warns exactly once."""
        hp_logging._warned_unknown.discard("FDSP")
        with self.assertLogs("hyper_parallel.tools.logging", level="WARNING") as cm:
            log1 = get_logger("FDSP")
            log2 = get_logger("FDSP")  # second call must not warn again
        self.assertEqual(log1.name, "hyper_parallel.FDSP")  # still works
        self.assertIs(log1, log2)
        self.assertEqual(len(cm.output), 1)  # warns exactly once
        self.assertIn("unknown component", cm.output[0])
        self.assertIn("FDSP", cm.output[0])


if __name__ == "__main__":
    unittest.main()
