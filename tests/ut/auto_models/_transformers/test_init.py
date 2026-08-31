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
"""Tests for lazy exports from the Transformers integration package."""
import os
import subprocess
import sys
import textwrap
import unittest


_EXPORT_MODULES = (
    "hyper_parallel.auto_models._transformers.auto_model",
    "hyper_parallel.auto_models._transformers.checkpoint_loader",
)


def _run_isolated(script: str) -> subprocess.CompletedProcess:
    """Run ``script`` in a fresh interpreter with the local checkout."""
    env = os.environ.copy()
    env["HYPER_PARALLEL_PLATFORM"] = "torch"
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    )
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


class TestLazyTransformersExports(unittest.TestCase):
    """Public Transformers integration symbols load only when accessed."""

    def test_importing_registry_does_not_load_export_modules(self):
        """
        Feature: lightweight registry import
        Description: import get_hf_config through its direct module path
        Expectation: AutoModel and checkpoint implementation modules remain unloaded
        """
        script = textwrap.dedent(
            f"""
            import sys
            from hyper_parallel.auto_models._transformers.registry import get_hf_config
            import hyper_parallel.auto_models._transformers as integration

            loaded = [name for name in {_EXPORT_MODULES!r} if name in sys.modules]
            assert not loaded, loaded
            assert get_hf_config is not None
            assert set(integration.__all__).issubset(dir(integration))
            """
        )

        result = _run_isolated(script)

        self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)

    def test_lazy_export_is_resolved_and_cached(self):
        """
        Feature: lazy export resolution
        Description: resolve CheckpointManager using a mocked module import
        Expectation: the owner module is imported once and the symbol is cached
        """
        script = textwrap.dedent(
            """
            from types import SimpleNamespace
            from unittest.mock import patch
            import hyper_parallel.auto_models._transformers as integration

            sentinel = object()
            owner_module = SimpleNamespace(CheckpointManager=sentinel)
            with patch.object(integration, "_import_module", return_value=owner_module) as importer:
                assert integration.CheckpointManager is sentinel
                assert integration.CheckpointManager is sentinel
                importer.assert_called_once_with(".checkpoint_loader", integration.__name__)
            """
        )

        result = _run_isolated(script)

        self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)

    def test_unknown_attribute_raises_attribute_error(self):
        """
        Feature: unknown package attribute
        Description: access a name outside the public lazy export map
        Expectation: AttributeError is raised without importing implementation modules
        """
        script = textwrap.dedent(
            """
            import hyper_parallel.auto_models._transformers as integration

            try:
                integration.NotATransformersIntegrationSymbol
            except AttributeError:
                pass
            else:
                raise AssertionError("Unknown attribute did not raise AttributeError")
            """
        )

        result = _run_isolated(script)

        self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
