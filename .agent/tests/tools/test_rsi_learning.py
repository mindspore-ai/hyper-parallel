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
"""Behavioral tests for portable, evidence-backed RSI consolidation."""
from contextlib import ExitStack
import importlib.util
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tests.common.mark_utils import arg_mark


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("rsi_learning_test", ROOT / ".agent/scripts/rsi_learning.py")
ENGINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENGINE)


@arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"], level_mark="level0",
          card_mark="dryrun", essential_mark="essential")
class RsiLearningTest(unittest.TestCase):
    """Exercise real files, subprocess validation, rollback, and report coverage."""

    def setUp(self) -> None:
        """Create a minimal repository and one immutable experimental result."""
        self.resources = ExitStack()
        self.addCleanup(self.resources.close)
        self.root = Path(self.resources.enter_context(tempfile.TemporaryDirectory()))
        self.repo = self.root / "repo"
        self.run = self.root / "run"
        self.owner = ".agent/skills/perf-playbook/SKILL.md"
        for name in (".agent/skills/perf-playbook", ".agent/scripts", "hyper_parallel", ".agent/tests/tools"):
            (self.repo / name).mkdir(parents=True)
        (self.repo / self.owner).write_text("# Performance tuning\n", encoding="utf-8")
        self.run.mkdir()
        ENGINE.save(self.run / "evidence.json", {"passed": True, "scope": "logical initial state"})
        ENGINE.save(self.run / "rsi/experiments.json", {"experiments": [
            {"id": "trial-1", "path": "evidence.json", "sha256": ENGINE.file_hash(self.run / "evidence.json"),
             "decision": "keep"}]})

    def lesson(self, **changes: object) -> dict:
        """Build a lesson with an actionable rule and bounded evidence scope."""
        result = {
            "id": "logical-state-identity", "title": "Logical state identity across layouts", "decision": "new",
            "owner": "perf-playbook", "form": "procedure", "qualification": "validated",
            "keywords": ["logical", "initial", "identity"],
            "trigger": "The numerical comparison changes the shard layout.",
            "mechanism": "Rank-local random initialization can change the logical parameter values.",
            "claim": "Match logical initial weights before comparing numerical trajectories across layouts.",
            "applicability": "Distributed comparisons of two shard layouts with equivalent model semantics.",
            "limits": "This check does not establish convergence or elementwise gradient equivalence.",
            "reason": "The experiment exposed an initialization confounder.",
            "evidence": [{"path": "evidence.json", "sha256": ENGINE.file_hash(self.run / "evidence.json")}],
        }
        result.update(changes)
        return result

    def review(self, lessons: list[dict]) -> dict:
        """Review the current corpus and all experiment outcomes."""
        current = ENGINE.inventory(self.repo)
        for lesson in lessons:
            lesson.setdefault("related_review", [dict(match, relationship="different", reason="Distinct scope")
                                                 for match in ENGINE.related(self.repo, current, lesson)])
        value = {"schema": "hp-rsi-learning-review/v1", "inventory_digest": current["digest"], "lessons": lessons,
                 "coverage": [{"event_id": row["id"], "sha256": row["sha256"],
                               "lesson_ids": [lesson["id"] for lesson in lessons],
                               "reason": "Reviewed the complete trial and its failed hypotheses"}
                              for row in ENGINE.events(self.run)]}
        ENGINE.save(self.run / "rsi/review.json", value)
        return value

    def sync(self, phase: str = "acceptance", mode: str = "acceptance") -> dict:
        """Run the same public boundary used by both Harnesses."""
        return ENGINE.synchronize(self.repo, self.run, phase, mode)

    def test_default_defers_until_acceptance_and_is_idempotent(self) -> None:
        """A default run queues changes, applies at acceptance, and never duplicates them."""
        self.review([self.lesson()])
        with patch.object(ENGINE, "inventory", side_effect=AssertionError("Unexpected corpus scan")):
            queued = self.sync("iteration")
        self.assertEqual(queued["state"], "QUEUED")
        self.assertEqual(queued["queued_experiments"], 1)
        self.assertFalse((self.repo / ENGINE.REGISTRY).exists())
        accepted = self.sync()
        self.assertTrue(accepted["complete"])
        self.assertEqual(accepted["lessons"][0]["decision"], "new")
        checksum = ENGINE.file_hash(self.repo / ENGINE.REGISTRY)
        self.assertTrue(self.sync()["complete"])
        self.assertEqual(checksum, ENGINE.file_hash(self.repo / ENGINE.REGISTRY))
        links = (self.repo / self.owner).read_text().count("references/experience/logical-state-identity.md")
        self.assertEqual(links, 1)
        self.assertTrue((self.run / "RSI_REPORT.html").exists())
        self.assertTrue((self.run / "rsi/report.json").exists())

    def test_continuous_applies_between_trials(self) -> None:
        """The optional mode applies evidence without waiting for final acceptance."""
        self.review([self.lesson()])
        result = self.sync("iteration", "continuous")
        self.assertTrue(result["complete"])
        self.assertTrue((self.repo / ENGINE.REGISTRY).exists())
        with self.assertRaisesRegex(ValueError, "frozen"):
            ENGINE.initialize(self.run, "acceptance")

    def test_missing_review_blocks_final_delivery(self) -> None:
        """A report cannot silently claim learning completion before Agent review."""
        result = self.sync()
        self.assertFalse(result["complete"])
        self.assertEqual(result["state"], "REVIEW_REQUIRED")
        self.assertTrue((self.run / "rsi/review-task.json").exists())
        self.assertEqual(ENGINE.main(["sync", "--repo", str(self.repo), "--run-dir", str(self.run),
                                      "--phase", "acceptance"]), 2)

    def test_review_can_find_no_new_experience(self) -> None:
        """A complete review with no useful novelty produces a valid empty report."""
        self.review([])
        result = self.sync()
        self.assertTrue(result["complete"])
        self.assertEqual(result["changed_files"], {})
        self.assertEqual(result["reviewed_experiments"], 1)
        self.assertFalse((self.repo / ENGINE.REGISTRY).exists())

    def test_failures_must_be_accounted_for(self) -> None:
        """Adding a failed trial invalidates a previously complete retrospective."""
        self.review([])
        index = ENGINE.read(self.run / "rsi/experiments.json")
        index["experiments"].append({**index["experiments"][0], "id": "crashed-trial", "decision": "crash"})
        ENGINE.save(self.run / "rsi/experiments.json", index)
        result = self.sync()
        self.assertFalse(result["complete"])
        self.assertIn("every experiment", result["reason"])

    def test_legacy_baseline_is_reviewed_with_candidate_receipts(self) -> None:
        """A later candidate must not hide a baseline recorded only in the TSV ledger."""
        (self.run / "results.tsv").write_text(
            "timestamp\tcommit\tmetric\tstatus\tdescription\n"
            "t0\tabc\t10\tkeep\tbaseline\n"
            "t1\txxxxxxx\t\tcrash\tfailed candidate\n", encoding="utf-8")
        (self.run / "candidates.jsonl").write_text('{"status":"crash","commit":"def"}\n', encoding="utf-8")
        observed = ENGINE.events(self.run)
        self.assertEqual([row["decision"] for row in observed], ["keep", "crash"])
        self.review([])
        self.assertEqual(self.sync()["reviewed_experiments"], 2)

    def test_exact_duplicate_is_reused_across_sessions(self) -> None:
        """Equivalent registered claims are reused even under a different proposal ID."""
        self.review([self.lesson()])
        self.assertTrue(self.sync()["complete"])
        checksum = ENGINE.file_hash(self.repo / ENGINE.REGISTRY)
        (self.run / "rsi/applied.json").unlink()
        self.review([self.lesson(id="same-rule-different-title")])
        repeated = self.sync()
        self.assertTrue(repeated["complete"])
        self.assertEqual(repeated["lessons"][0]["decision"], "reuse")
        self.assertEqual(checksum, ENGINE.file_hash(self.repo / ENGINE.REGISTRY))

    def test_legacy_experience_is_reused_without_a_registry_copy(self) -> None:
        """An existing Skill rule need not be registered again to count as reused."""
        path = self.repo / self.owner
        path.write_text("# Logical initial identity\nExisting identity procedure.\n", encoding="utf-8")
        lesson = self.lesson(decision="reuse", related_review=[{
            "path": self.owner, "sha256": ENGINE.file_hash(path), "relationship": "same", "reason": "Already covered"}])
        self.review([lesson])
        result = self.sync()
        self.assertTrue(result["complete"])
        self.assertEqual(result["lessons"][0]["files"], [self.owner])
        self.assertEqual(result["changed_files"], {})

    def test_semantic_match_must_not_be_claimed_as_new(self) -> None:
        """A reviewed same-scope match requires reuse or a concrete improvement."""
        path = self.repo / self.owner
        path.write_text("# Logical initial identity\n", encoding="utf-8")
        lesson = self.lesson(related_review=[{
            "path": self.owner, "sha256": ENGINE.file_hash(path), "relationship": "same", "reason": "Same rule"}])
        self.review([lesson])
        result = self.sync()
        self.assertFalse(result["complete"])
        self.assertIn("reuse or update", result["reason"])

    def test_improve_existing_rule_in_place(self) -> None:
        """Updating legacy content changes its location rather than adding another note."""
        path = self.repo / self.owner
        path.write_text("# Logical initial identity\nCompare common replicated tensors.\n", encoding="utf-8")
        lesson = self.lesson(decision="update", related_review=[{
            "path": self.owner, "sha256": ENGINE.file_hash(path), "relationship": "extends",
            "reason": "Add placement identity for rank-local external tables"}], changes=[{
                "path": self.owner, "before_sha256": ENGINE.file_hash(path),
                "content": path.read_text() + "Compare external tables using their declared placement identity.\n"}])
        self.review([lesson])
        result = self.sync()
        self.assertTrue(result["complete"])
        self.assertEqual(result["lessons"][0]["decision"], "update")
        self.assertFalse((path.parent / "references/experience").exists())

    def test_deployment_details_are_rejected_but_raw_evidence_stays_local(self) -> None:
        """Cluster names, addresses, credentials and user directories never enter learned rules."""
        for text in ("Run on pod11", "Use 10.2.3.4", "Read /home/operator/checkpoint", "Use D:/private/run",
                     "password=examplepassword", "Connect to worker.internal"):
            with self.subTest(text=text):
                self.review([self.lesson(trigger=text)])
                result = self.sync()
                self.assertFalse(result["complete"])
                self.assertFalse((self.repo / ENGINE.REGISTRY).exists())

    def test_unqualified_and_local_lessons_can_be_deferred_or_rejected(self) -> None:
        """Unmeasured suggestions and site-specific facts remain in the local report."""
        self.review([self.lesson(decision="defer", qualification="untried"),
                     self.lesson(id="site-network", decision="reject", reason="Only applies to pod11")])
        result = self.sync()
        self.assertTrue(result["complete"])
        self.assertEqual(result["changed_files"], {})

    def test_unqualified_result_cannot_be_promoted(self) -> None:
        """Screening without qualification cannot create a reusable recommendation."""
        self.review([self.lesson(qualification="screened")])
        self.assertIn("must be deferred", self.sync()["reason"])

    def test_stale_repository_or_evidence_requires_a_new_review(self) -> None:
        """Changes after review invalidate its architecture or experiment evidence."""
        self.review([self.lesson()])
        (self.repo / self.owner).write_text("# Changed by another contributor\n", encoding="utf-8")
        self.assertIn("Repository changed", self.sync()["reason"])
        self.review([self.lesson()])
        (self.run / "evidence.json").write_text('{}', encoding="utf-8")
        self.assertIn("evidence changed", self.sync()["reason"])

    def test_unknown_owner_and_traversal_cannot_write(self) -> None:
        """Routes must resolve into the current architecture, without escaping the repository."""
        self.review([self.lesson(owner="removed-skill")])
        self.assertIn("Owner Skill", self.sync()["reason"])
        self.review([self.lesson(changes=[{"path": "../escape.py", "before_sha256": None, "content": "x = 1\n"}])])
        self.assertFalse(self.sync()["complete"])
        self.assertFalse((self.root / "escape.py").exists())

    def test_consolidation_cannot_edit_training_runtime(self) -> None:
        """Learning can cite the framework but keeps automated patches in the agent directory."""
        name = "hyper_parallel/trainer.py"
        self.review([self.lesson(changes=[{"path": name, "before_sha256": None, "content": "x = 1\n"}])])
        self.assertIn("discovered .agent code area", self.sync()["reason"])
        self.assertFalse((self.repo / name).exists())

    def code_lesson(self, fail: bool = False) -> dict:
        """Prepare an executable guard with a genuine positive and negative regression."""
        source = "def validate(value):\n    if value < 0:\n        raise ValueError('negative')\n    return value\n"
        test = ("import runpy\nimport unittest\n"
                "guard = runpy.run_path('.agent/scripts/guard.py')['validate']\n"
                "class GuardTest(unittest.TestCase):\n"
                "    def test_positive(self):\n        self.assertEqual(guard(2), 2)\n"
                "    def test_negative(self):\n        self.assertRaises(ValueError, guard, -1)\n"
                "if __name__ == '__main__':\n    unittest.main()\n")
        if fail:
            source = source.replace("value < 0", "value > 0")
        return self.lesson(form="harness", changes=[
            {"path": ".agent/scripts/guard.py", "before_sha256": None, "content": source},
            {"path": ".agent/tests/tools/test_guard.py", "before_sha256": None, "content": test}],
            regression_tests=[".agent/tests/tools/test_guard.py"],
            validation=[{"argv": ["python", ".agent/tests/tools/test_guard.py"], "timeout_seconds": 30}])

    def test_executable_experience_is_validated_and_reported(self) -> None:
        """The tool executes a new guard's real regression before retaining it."""
        lesson = self.code_lesson()
        license_link = "# http://www.apache.org/licenses/LICENSE-2.0\n"
        lesson["changes"][0]["content"] = license_link + lesson["changes"][0]["content"]
        self.review([lesson])
        result = self.sync()
        self.assertTrue(result["complete"])
        self.assertEqual(result["validation"][0]["exit_code"], 0)
        self.assertIn(".agent/scripts/guard.py", result["changed_files"])

    def test_failed_regression_rolls_back_all_repository_changes(self) -> None:
        """A failed guard cannot leave a lesson, index, or partially changed Skill behind."""
        before = (self.repo / self.owner).read_bytes()
        self.review([self.code_lesson(fail=True)])
        result = self.sync()
        self.assertFalse(result["complete"])
        self.assertEqual((self.repo / self.owner).read_bytes(), before)
        self.assertFalse((self.repo / ENGINE.REGISTRY).exists())
        self.assertFalse((self.repo / ".agent/scripts/guard.py").exists())
        self.assertTrue((self.run / "rsi/validation-logical-state-identity-0.log").exists())

    def test_executable_changes_without_regression_are_rejected(self) -> None:
        """A procedure label cannot bypass validation of an executable artifact."""
        lesson = self.code_lesson()
        lesson.pop("regression_tests")
        self.review([lesson])
        self.assertIn("require regression", self.sync()["reason"])

    def test_pending_training_prevents_consolidation(self) -> None:
        """Repository changes wait for remote work and cleanup to reach a boundary."""
        ENGINE.save(self.run / "controller.json", {"pending": "active-trial", "attempts": []})
        result = self.sync()
        self.assertFalse(result["complete"])
        self.assertIn("pending", result["reason"])

    def test_measured_source_cannot_change_during_consolidation(self) -> None:
        """A passing learning regression cannot invalidate the accepted source identity."""
        ENGINE.save(self.run / "run.json", {"target_files": [".agent/scripts/guard.py"]})
        self.review([self.code_lesson()])
        result = self.sync()
        self.assertFalse(result["complete"])
        self.assertIn("measured source", result["reason"])

    def test_autoresearch_cli_consolidates_without_staging_unrelated_work(self) -> None:
        """Continuous learning integrates with the actual CLI and commits only its own files."""
        self.run = self.repo / "experiment"
        cli = ROOT / ".agent/skills/autoresearch/scripts/autoresearch.py"
        for args in (("init", "-q", "-b", "main"), ("config", "user.email", "test@example.com"),
                     ("config", "user.name", "test")):
            subprocess.run(["git", *args], cwd=self.repo, check=True, capture_output=True)
        (self.repo / "target.py").write_text("VALUE = 1\n", encoding="utf-8")
        (self.repo / "bench.py").write_text("print('total_ms=100')\n", encoding="utf-8")

        def invoke(*args: str) -> subprocess.CompletedProcess:
            return subprocess.run([sys.executable, str(cli), *args, "--run-dir", str(self.run)],
                                  cwd=self.repo, capture_output=True, text=True, check=False)

        initialized = invoke("init", "--target", "target.py", "--benchmark-cmd",
                             shlex.join([sys.executable, "bench.py"]),
                             "--rsi-mode", "continuous")
        self.assertEqual(initialized.returncode, 0, f"exit={initialized.returncode}, output={initialized.stderr}")
        subprocess.run(["git", "add", "."], cwd=self.repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-qm", "fixture"], cwd=self.repo, check=True, capture_output=True)
        baseline = invoke("baseline")
        self.assertEqual(baseline.returncode, 0,
                         f"exit={baseline.returncode}, output={baseline.stdout}, error={baseline.stderr}")
        self.assertEqual(invoke("finalize").returncode, 2)
        ENGINE.save(self.run / "evidence.json", {"passed": True})
        self.review([self.lesson()])
        unrelated = self.repo / "unrelated.txt"
        unrelated.write_text("Other contributor's work\n", encoding="utf-8")
        subprocess.run(["git", "add", "unrelated.txt"], cwd=self.repo, check=True, capture_output=True)
        learned = invoke("learn")
        self.assertEqual(learned.returncode, 0, f"exit={learned.returncode}, output={learned.stderr}")
        staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=self.repo,
                                check=True, capture_output=True, text=True).stdout.strip()
        self.assertEqual(staged, "unrelated.txt")
        self.assertEqual(invoke("finalize").returncode, 0)
        existing = self.repo / self.owner
        edited = existing.read_text() + "Another contributor's pending clarification.\n"
        existing.write_text(edited, encoding="utf-8")
        self.review([self.lesson(id="memory-capacity-check", keywords=["capacity", "memory"],
                                 claim="Check live allocation capacity before retaining additional tensors.")])
        self.assertEqual(invoke("learn").returncode, 0)
        report = ENGINE.read(self.run / "rsi/report.json")
        self.assertFalse(report["complete"])
        self.assertIn("pre-existing edits", report["reason"])
        self.assertEqual(existing.read_text(), edited)


if __name__ == "__main__":
    unittest.main()
