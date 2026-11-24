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
"""Hardware-independent CLI contract tests for the official FM-Agent adapter."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


def _find_adapter() -> Path | None:
    """Locate source tooling when CI has not copied the test tree away from it."""
    for ancestor in Path(__file__).resolve().parents:
        candidate = ancestor / ".agent/skills/formal-verification/scripts/verify.py"
        if candidate.is_file():
            return candidate
    return None


SCRIPT = _find_adapter()
if SCRIPT is None:
    raise unittest.SkipTest("Formal-verification source tooling is absent from this standalone test tree")
REPO_ROOT = SCRIPT.parents[4]
MODULE_SPEC = importlib.util.spec_from_file_location("formal_verify_adapter", SCRIPT)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Cannot load adapter: {SCRIPT}")
verify_module = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(verify_module)

ENGINE_FIXTURE = '''import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("repo")
for flag in ("--isolate", "--all-bugs", "--only-spec", "--estimate", "--resume"):
    parser.add_argument(flag, action="store_true")
parser.add_argument("--incremental")
parser.add_argument("--submodule", nargs="+")
parser.add_argument("--domain-knowledge", nargs="+")
parser.add_argument("--bug-validator")
args = parser.parse_args()
behavior = os.environ.get("HP_FM_TEST_BEHAVIOR")
if behavior == "failure":
    raise SystemExit(7)
if behavior == "no-artifact":
    raise SystemExit(0)
workspace = Path(args.repo) / ("fm_agent_estimate" if args.estimate else "fm_agent")
workspace.mkdir(exist_ok=True)
record = vars(args) | {
    "cwd": str(Path.cwd()), "ambient_resume": os.environ.get("FM_AGENT_RESUME"),
    "python_utf8": os.environ.get("PYTHONUTF8"), "io_encoding": os.environ.get("PYTHONIOENCODING"),
}
(workspace / "invocation.json").write_text(json.dumps(record), encoding="utf-8")
if args.estimate:
    (workspace / "estimate.json").write_text("{}", encoding="utf-8")
else:
    with (workspace / "version.log").open("a", encoding="utf-8") as stream:
        stream.write("completed-test-run\\n")
'''


class TestVerifyAdapter(unittest.TestCase):
    """Exercise argument parsing, portability, result preservation and process failures."""

    def setUp(self) -> None:
        """Create an isolated Git framework, configuration and executable CLI fixture."""
        self.resources = contextlib.ExitStack()
        self.addCleanup(self.resources.close)
        directory = self.resources.enter_context(tempfile.TemporaryDirectory(prefix="hp formal verification "))
        self.base = Path(directory).resolve()
        self.repo = self.base / "training repo"
        self.repo.mkdir()
        (self.repo / "src/distributed").mkdir(parents=True)
        (self.repo / "src/distributed/geometry.py").write_text("def shard(): return 1\n", encoding="utf-8")
        for arguments in (
            ["init", "--quiet"], ["add", "."],
            ["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "--quiet", "-m", "fixture"],
        ):
            subprocess.run(["git", "-C", str(self.repo), *arguments], check=True, capture_output=True)
        self.engine = self.base / "FM Agent"
        self.engine.mkdir()
        (self.engine / "main.py").write_text(ENGINE_FIXTURE, encoding="utf-8")
        self.config_dir = self.base / "profile config"
        self.config_dir.mkdir()
        for name in ("knowledge.md", "validator.md", "extra.md"):
            (self.config_dir / name).write_text("# Training contracts\n", encoding="utf-8")
        self.config = self.config_dir / "profiles.json"
        self.config_value = {
            "framework": "AnotherTrainingFramework", "domain_knowledge": ["knowledge.md"],
            "bug_validator": "validator.md",
            "profiles": {"distributed": {"submodules": ["src/distributed"], "focus": "Shard geometry"}},
        }
        self.write_config()

    def write_config(self) -> None:
        """Persist the test framework configuration."""
        self.config.write_text(json.dumps(self.config_value), encoding="utf-8")

    def arguments(self, action: str, *extra: str) -> list[str]:
        """Return independent-framework CLI arguments using paths with spaces."""
        return [
            action, "--repo", str(self.repo), "--config", str(self.config),
            "--profile", "distributed", "--engine-root", str(self.engine), *extra,
        ]

    def plan(self, *extra: str) -> dict:
        """Build a launch plan from parsed options."""
        return verify_module.build_plan(verify_module.create_parser().parse_args(self.arguments("plan", *extra)))

    def run_adapter(self, *extra: str) -> int:
        """Execute the adapter against a strict local engine fixture."""
        return verify_module.main(self.arguments("run", *extra))

    def test_custom_framework_and_config_relative_knowledge(self) -> None:
        plan = self.plan()
        self.assertEqual(plan["framework"], "AnotherTrainingFramework")
        self.assertEqual(plan["domain_knowledge"], [str(self.config_dir / "knowledge.md")])
        self.assertEqual(plan["submodules"], ["src/distributed"])
        self.assertEqual(plan["analysis_status"], "not_run")

    def test_default_hyperparallel_profiles_resolve(self) -> None:
        config_path = SCRIPT.parents[1] / "profiles.json"
        config = verify_module.load_config(config_path)
        for name in config["profiles"]:
            with self.subTest(profile=name):
                args = verify_module.create_parser().parse_args(["plan", "--profile", name])
                plan = verify_module.build_plan(args)
                self.assertEqual(plan["framework"], "HyperParallel")
                self.assertTrue(plan["submodules"])

    def test_clean_and_dirty_revision(self) -> None:
        self.assertFalse(self.plan()["has_local_changes"])
        (self.repo / "src/distributed/geometry.py").write_text("changed\n", encoding="utf-8")
        self.assertTrue(self.plan()["has_local_changes"])
        self.assertEqual(len(self.plan()["head"]), 40)

    def test_plan_does_not_start_engine_or_create_outputs(self) -> None:
        with mock.patch.object(verify_module, "check_engine", side_effect=RuntimeError("engine called")):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(verify_module.main(self.arguments("plan")), 0)
            self.assertEqual(json.loads(output.getvalue())["analysis_status"], "not_run")
        self.assertFalse((self.repo / "fm_agent").exists())

    def test_plan_without_engine_is_available(self) -> None:
        args = verify_module.create_parser().parse_args(self.arguments("plan"))
        args.engine_root = None
        plan = verify_module.build_plan(args)
        self.assertIsNone(plan["engine_root"])
        self.assertEqual(plan["command"][1], "<FM-Agent>/main.py")
        with self.assertRaisesRegex(ValueError, "FM_AGENT_HOME"):
            verify_module.check_engine(plan)

    def test_scope_collapses_duplicates_and_descendants(self) -> None:
        scopes = verify_module.resolve_scope(self.repo, ["src/distributed", "src", "src"])
        self.assertEqual(scopes, ["src"])

    def test_invalid_scope_is_rejected(self) -> None:
        (self.repo / ".agent").mkdir()
        (self.repo / "fm_agent").mkdir()
        for path in ("", ".", "missing", "../FM Agent", ".agent", "fm_agent", "src/distributed/geometry.py"):
            with self.subTest(scope=path), self.assertRaises(ValueError):
                verify_module.resolve_scope(self.repo, [path])

    def test_submodule_override(self) -> None:
        self.assertEqual(self.plan("--submodule", "src")["submodules"], ["src"])

    def test_unknown_profile_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown profile"):
            self.plan("--profile", "unknown")

    def test_repository_subdirectory_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "root of a Git"):
            self.plan("--repo", str(self.repo / "src"))

    def test_invalid_configuration_shapes_are_rejected(self) -> None:
        for value in ([], {}, {"framework": "x", "profiles": []}, {**self.config_value, "domain_knowledge": []}):
            with self.subTest(config=value):
                self.config.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(ValueError):
                    verify_module.load_config(self.config)

    def test_missing_or_non_markdown_knowledge_is_rejected(self) -> None:
        for path in ("missing.md", "profiles.json", ""):
            with self.subTest(path=path):
                self.config_value["domain_knowledge"] = [path]
                self.write_config()
                with self.assertRaises(ValueError):
                    verify_module.load_config(self.config)

    def test_extra_knowledge_is_passed(self) -> None:
        plan = self.plan("--knowledge", str(self.config_dir / "extra.md"))
        self.assertIn(str(self.config_dir / "extra.md"), plan["domain_knowledge"])
        self.assertIn(str(self.config_dir / "extra.md"), plan["command"])

    def test_estimate_omits_incompatible_upstream_flags(self) -> None:
        command = self.plan("--mode", "estimate")["command"]
        self.assertIn("--estimate", command)
        for flag in ("--isolate", "--all-bugs", "--only-spec", "--resume"):
            self.assertNotIn(flag, command)

    def test_verify_is_isolated_and_validates_all_candidates(self) -> None:
        command = self.plan()["command"]
        self.assertIn("--isolate", command)
        self.assertIn("--all-bugs", command)

    def test_spec_does_not_claim_bug_verification(self) -> None:
        command = self.plan("--mode", "spec")["command"]
        self.assertIn("--isolate", command)
        self.assertIn("--only-spec", command)
        self.assertNotIn("--all-bugs", command)

    def test_resume_and_incremental_are_verify_only(self) -> None:
        for mode in ("estimate", "spec"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "require --mode verify"):
                self.plan("--mode", mode, "--resume")

    def test_incremental_requires_existing_intent(self) -> None:
        with self.assertRaisesRegex(ValueError, "intent file"):
            self.plan("--incremental", str(self.base / "missing.md"))

    def test_doctor_calls_real_help_without_analysis(self) -> None:
        self.assertEqual(verify_module.main(self.arguments("doctor")), 0)
        self.assertFalse((self.repo / "fm_agent").exists())

    def test_doctor_reports_missing_dependencies(self) -> None:
        (self.engine / "main.py").write_text("raise SystemExit('missing dependency')\n", encoding="utf-8")
        with self.assertLogs(verify_module.LOGGER, level="ERROR") as messages:
            self.assertEqual(verify_module.main(self.arguments("doctor")), 2)
        self.assertIn("missing dependency", "\n".join(messages.output))

    def test_doctor_reports_incompatible_cli(self) -> None:
        (self.engine / "main.py").write_text("print('--submodule')\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing options"):
            verify_module.check_engine(self.plan())

    def test_dry_run_never_checks_or_launches_engine(self) -> None:
        with mock.patch.object(verify_module, "check_engine", side_effect=RuntimeError("engine called")):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(self.run_adapter("--dry-run"), 0)
        self.assertFalse((self.repo / "fm_agent").exists())

    def test_real_subprocess_preserves_paths_and_resets_ambient_resume(self) -> None:
        with mock.patch.dict(os.environ, {"FM_AGENT_RESUME": "1"}):
            self.assertEqual(self.run_adapter(), 0)
        invocation = json.loads((self.repo / "fm_agent/invocation.json").read_text(encoding="utf-8"))
        self.assertEqual(invocation["repo"], str(self.repo))
        self.assertEqual(invocation["cwd"], str(self.engine))
        self.assertIsNone(invocation["ambient_resume"])
        self.assertEqual(invocation["python_utf8"], "1")
        self.assertEqual(invocation["io_encoding"], "utf-8")
        self.assertEqual(invocation["submodule"], ["src/distributed"])

    def test_existing_results_are_not_overwritten(self) -> None:
        self.assertEqual(self.run_adapter(), 0)
        before = (self.repo / "fm_agent/version.log").read_bytes()
        with self.assertLogs(verify_module.LOGGER, level="ERROR"):
            self.assertEqual(self.run_adapter(), 2)
        self.assertEqual((self.repo / "fm_agent/version.log").read_bytes(), before)

    def test_resume_reuses_existing_results(self) -> None:
        self.assertEqual(self.run_adapter(), 0)
        self.assertEqual(self.run_adapter("--resume"), 0)
        invocation = json.loads((self.repo / "fm_agent/invocation.json").read_text(encoding="utf-8"))
        self.assertTrue(invocation["resume"])
        self.assertTrue(invocation["all_bugs"])

    def test_missing_resume_baseline_is_rejected(self) -> None:
        with self.assertLogs(verify_module.LOGGER, level="ERROR"):
            self.assertEqual(self.run_adapter("--resume"), 2)

    def test_incremental_requires_completed_baseline(self) -> None:
        intent = self.base / "intent.md"
        intent.write_text("Correct gradient scaling\n", encoding="utf-8")
        (self.repo / "fm_agent").mkdir()
        with self.assertLogs(verify_module.LOGGER, level="ERROR"):
            self.assertEqual(self.run_adapter("--incremental", str(intent)), 2)
        (self.repo / "fm_agent/version.log").write_text("baseline\n", encoding="utf-8")
        manifest = {name: self.plan()[name] for name in ("framework", "mode", "submodules")}
        (self.repo / "fm_agent/training_adapter.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.assertEqual(self.run_adapter("--incremental", str(intent)), 0)

    def test_resume_rejects_changed_scope(self) -> None:
        self.assertEqual(self.run_adapter(), 0)
        with self.assertLogs(verify_module.LOGGER, level="ERROR"):
            self.assertEqual(self.run_adapter("--resume", "--submodule", "src"), 2)

    def test_resume_rejects_spec_only_workspace(self) -> None:
        self.assertEqual(self.run_adapter("--mode", "spec"), 0)
        with self.assertLogs(verify_module.LOGGER, level="ERROR"):
            self.assertEqual(self.run_adapter("--resume"), 2)

    def test_engine_exit_code_is_preserved(self) -> None:
        with mock.patch.dict(os.environ, {"HP_FM_TEST_BEHAVIOR": "failure"}):
            self.assertEqual(self.run_adapter(), 7)

    def test_zero_exit_without_completion_is_failure(self) -> None:
        with mock.patch.dict(os.environ, {"HP_FM_TEST_BEHAVIOR": "no-artifact"}):
            with self.assertLogs(verify_module.LOGGER, level="ERROR"):
                self.assertEqual(self.run_adapter(), 1)

    def test_stale_resume_artifact_does_not_mask_setup_failure(self) -> None:
        self.assertEqual(self.run_adapter(), 0)
        with mock.patch.dict(os.environ, {"HP_FM_TEST_BEHAVIOR": "no-artifact"}):
            with self.assertLogs(verify_module.LOGGER, level="ERROR"):
                self.assertEqual(self.run_adapter("--resume"), 1)

    def test_estimate_artifact_is_required(self) -> None:
        self.assertEqual(self.run_adapter("--mode", "estimate"), 0)
        self.assertTrue((self.repo / "fm_agent_estimate/estimate.json").is_file())
        with mock.patch.dict(os.environ, {"HP_FM_TEST_BEHAVIOR": "no-artifact"}):
            with self.assertLogs(verify_module.LOGGER, level="ERROR"):
                self.assertEqual(self.run_adapter("--mode", "estimate"), 1)

    def test_cli_is_independent_of_training_imports(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), *self.arguments("plan")],
            check=True, capture_output=True, text=True, encoding="utf-8",
        )
        self.assertEqual(json.loads(result.stdout)["framework"], "AnotherTrainingFramework")

    def test_import_does_not_load_torch_or_framework(self) -> None:
        """Keep planning usable without importing training runtime dependencies."""
        code = (
            "import json, runpy, sys; runpy.run_path(sys.argv[1]); "
            "print(json.dumps([name for name in sys.modules "
            "if name.split('.')[0] in ('torch', 'hyper_parallel')]))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code, str(SCRIPT)],
            check=True, capture_output=True, text=True, encoding="utf-8",
        )
        self.assertEqual(json.loads(result.stdout), [])

    def test_copied_test_tree_collects_without_agent_sources(self) -> None:
        """Skip source-only tests cleanly when CI copies the test directory alone."""
        directory = self.base / "standalone tests/ut/agent/formal_verification"
        directory.mkdir(parents=True)
        copied_test = directory / "test_verify.py"
        shutil.copyfile(__file__, copied_test)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--noconftest", str(copied_test), "-q"],
            check=False, capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        self.assertEqual(result.returncode, 5, msg=f"Expected empty collection exit=5, got={result.returncode}")
        self.assertIn("1 skipped", result.stdout)


if __name__ == "__main__":
    unittest.main()
