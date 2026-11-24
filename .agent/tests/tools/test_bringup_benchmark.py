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
"""Unit tests of the bring-up asset benchmark library.

Covers the catalog contract, the deterministic suite's verdict mapping
(including "expected fail" cells and blind-spot accounting), trial scoring
from the machine-readable pair, the calibration arithmetic that sets the
minimum detectable difference, the three-valued A/B rule, the A/A and
injection self-checks, and snapshot materialization on a throwaway git
repository.
"""

import importlib.util
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from tests.common.mark_utils import arg_mark

_SCRIPTS = (Path(__file__).resolve().parents[3]
            / ".agent" / "skills" / "bringup-benchmark" / "scripts")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bench_lib = _load("bench_lib")
bench_cli = _load("bench")

_CATALOG = {
    "schema": "bringup-bench/catalog/v1",
    "models": [
        {"id": "tied-model", "family": "fam-a", "category": "tied-embeddings",
         "checks": {"released_layout": "pass", "tied_save": "fail"}},
        {"id": "dense-model", "family": "fam-b", "category": "standard-dense",
         "checks": {"released_layout": "pass", "tied_save": "pass"}},
    ],
}


def _write_catalog(root: Path, payload=None) -> Path:
    path = root / "catalog.yaml"
    path.write_text(yaml.safe_dump(payload or _CATALOG, sort_keys=False),
                    encoding="utf-8")
    return path


def _write_paths(root: Path, ids) -> Path:
    path = root / "paths.yaml"
    path.write_text(yaml.safe_dump({i: str(root / i) for i in ids}),
                    encoding="utf-8")
    return path


def _fake_assets_root(root: Path, scripts=("check_released_layout.py",
                                           "check_tied_save.py")) -> Path:
    scripts_dir = (root / "assets" / ".agent" / "skills"
                   / "hf-model-integration" / "scripts")
    scripts_dir.mkdir(parents=True)
    for name in scripts:
        (scripts_dir / name).write_text("# placeholder\n", encoding="utf-8")
    return root / "assets"


def _trial_dir(root: Path, gaps_lines, meta_overrides=None) -> Path:
    trial = root / "trial"
    trial.mkdir(exist_ok=True)
    meta = {"schema": "bringup-bench/trial-meta/v1", "model_id": "tied-model",
            "assets_ref": "a" * 40, "agent_model": "test-model",
            "stages": {"S0": "done"}, "inventions": 1,
            "gates": [{"name": "layout", "verdict": "pass", "evidence": "x"},
                      {"name": "smoke", "verdict": "env_stop", "evidence": "y"},
                      {"name": "matrix", "verdict": "blocked", "evidence": "z"}]}
    meta.update(meta_overrides or {})
    (trial / "trial_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (trial / "gaps.jsonl").write_text("\n".join(gaps_lines) + "\n",
                                      encoding="utf-8")
    return trial


def _gap(gap_id="G1", category="missing") -> str:
    return json.dumps({"id": gap_id, "category": category, "stage": "S1",
                       "indicted_file": "a.md", "summary": "s", "evidence": "e"})


def _score(total, ref="a" * 40, model="tied-model"):
    return bench_lib.TrialScore(
        trial_dir="t", model_id=model, assets_ref=ref, gaps_total=total,
        gaps_by_category={"wrong": total, "missing": 0, "contradictory": 0},
        gates_passed=1, gates_failed=0, env_stops=0, inventions=0)


class TestCatalog(unittest.TestCase):
    """Catalog loading, validation, and site-path resolution."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_load_and_resolve(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            models = bench_lib.load_catalog(_write_catalog(root))
            self.assertEqual([m.model_id for m in models],
                             ["tied-model", "dense-model"])
            paths = bench_lib.resolve_model_dirs(
                models, _write_paths(root, ["tied-model", "dense-model"]))
            self.assertEqual(paths["tied-model"], root / "tied-model")

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_rejects_malformed(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            bad_schema = dict(_CATALOG, schema="other/v9")
            with self.assertRaisesRegex(bench_lib.BenchError, "schema"):
                bench_lib.load_catalog(_write_catalog(root, bad_schema))
            bad_check = {"schema": _CATALOG["schema"], "models": [
                {"id": "m", "family": "f", "category": "c",
                 "checks": {"unknown_check": "pass"}}]}
            with self.assertRaisesRegex(bench_lib.BenchError, "unknown check"):
                bench_lib.load_catalog(_write_catalog(root, bad_check))
            bad_expect = {"schema": _CATALOG["schema"], "models": [
                {"id": "m", "family": "f", "category": "c",
                 "checks": {"tied_save": "maybe"}}]}
            with self.assertRaisesRegex(bench_lib.BenchError, "pass\\|fail"):
                bench_lib.load_catalog(_write_catalog(root, bad_expect))
            with self.assertRaisesRegex(bench_lib.BenchError, "no models"):
                bench_lib.load_catalog(_write_catalog(
                    root, {"schema": _CATALOG["schema"], "models": []}))

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_paths_have_no_default(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            models = bench_lib.load_catalog(_write_catalog(root))
            old = os.environ.pop(bench_lib.PATHS_ENV, None)
            try:
                with self.assertRaisesRegex(bench_lib.BenchError, "paths"):
                    bench_lib.resolve_model_dirs(models, None)
                incomplete = _write_paths(root, ["tied-model"])
                with self.assertRaisesRegex(bench_lib.BenchError, "dense-model"):
                    bench_lib.resolve_model_dirs(models, incomplete)
            finally:
                if old is not None:
                    os.environ[bench_lib.PATHS_ENV] = old


class TestSuite(unittest.TestCase):
    """Deterministic layer: verdict mapping and blind-spot accounting."""

    def _run(self, exit_codes, scripts=None):
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            models = bench_lib.load_catalog(_write_catalog(root))
            dirs = {m.model_id: root for m in models}
            assets = (_fake_assets_root(root, scripts)
                      if scripts is not None else _fake_assets_root(root))

            def _runner(cmd, _timeout):
                script = Path(cmd[1]).name
                model_dir = cmd[3]
                return exit_codes[(Path(model_dir).name, script)], "tail"

            return bench_lib.run_suite(assets, models, dirs, runner=_runner)

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_expected_fail_counts_ok(self):
        codes = {(d, s): 0 for d in ("tied-model", "dense-model")
                 for s in ("check_released_layout.py", "check_tied_save.py")}
        # The tied model's save detector fires: exit 2 is the CORRECT verdict.
        codes[("tied-model", "check_tied_save.py")] = 2
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            models = bench_lib.load_catalog(_write_catalog(root))
            dirs = {m.model_id: root / m.model_id for m in models}
            for path in dirs.values():
                path.mkdir()
            assets = _fake_assets_root(root)

            def _runner(cmd, _timeout):
                return codes[(Path(cmd[3]).name, Path(cmd[1]).name)], "tail"

            card = bench_lib.run_suite(assets, models, dirs, runner=_runner)
        self.assertEqual(card["ok_cells"], 4)
        self.assertEqual(card["total_cells"], 4)
        self.assertEqual(card["covered_categories"],
                         ["standard-dense", "tied-embeddings"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_skip_and_unavailable_never_ok(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            models = bench_lib.load_catalog(_write_catalog(root))
            dirs = {m.model_id: root for m in models}
            assets = _fake_assets_root(root, scripts=("check_tied_save.py",))

            def _runner(_cmd, _timeout):
                return 1, "missing inputs"

            card = bench_lib.run_suite(assets, models, dirs, runner=_runner)
        by_check = {}
        for cell in card["cells"]:
            by_check.setdefault(cell["check"], []).append(cell)
        self.assertTrue(all(c["verdict"] == "unavailable" and not c["ok"]
                            for c in by_check["released_layout"]))
        self.assertTrue(all(c["verdict"] == "skipped" and not c["ok"]
                            for c in by_check["tied_save"]))


class TestTrialScore(unittest.TestCase):
    """Agent layer: scoring the machine-readable trial pair."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_scores_counts_and_gates(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            trial = _trial_dir(Path(tmp), [
                _gap("G1", "missing"), _gap("G2", "wrong"), _gap("G3", "wrong")])
            score = bench_lib.score_trial(trial)
        self.assertEqual(score.gaps_total, 3)
        self.assertEqual(score.gaps_by_category,
                         {"wrong": 2, "missing": 1, "contradictory": 0})
        self.assertEqual((score.gates_passed, score.gates_failed,
                          score.env_stops, score.inventions), (1, 0, 1, 1))

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_rejects_malformed_trials(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            trial = _trial_dir(root, ['{"id": "G1"}'])
            with self.assertRaisesRegex(bench_lib.BenchError, "gap missing"):
                bench_lib.score_trial(trial)
            trial = _trial_dir(root, [_gap(category="typo")])
            with self.assertRaisesRegex(bench_lib.BenchError, "category"):
                bench_lib.score_trial(trial)
            trial = _trial_dir(root, [_gap()], {"schema": "other/v0"})
            with self.assertRaisesRegex(bench_lib.BenchError, "schema"):
                bench_lib.score_trial(trial)
            trial = _trial_dir(root, [_gap()], {"gates": [
                {"name": "g", "verdict": "sorta", "evidence": "x"}]})
            with self.assertRaisesRegex(bench_lib.BenchError, "verdict"):
                bench_lib.score_trial(trial)

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_mapping_form_gates_accepted(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            trial = _trial_dir(Path(tmp), [_gap()], {"gates": {
                "layout": "pass", "smoke": "env_stop", "matrix": "fail"}})
            score = bench_lib.score_trial(trial)
        self.assertEqual((score.gates_passed, score.gates_failed,
                          score.env_stops), (1, 1, 1))

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_non_object_gate_rows_refused(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            trial = _trial_dir(Path(tmp), [_gap()], {"gates": ["pass"]})
            with self.assertRaisesRegex(bench_lib.BenchError, "each gate"):
                bench_lib.score_trial(trial)
            trial = _trial_dir(Path(tmp), [_gap()], {"gates": "pass"})
            with self.assertRaisesRegex(bench_lib.BenchError, "list or a name"):
                bench_lib.score_trial(trial)

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_score_cli_roundtrip(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            trial = _trial_dir(root, [_gap()])
            out = root / "score.json"
            self.assertEqual(bench_cli.main(
                ["score", "--trial-dir", str(trial), "--out", str(out)]), 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(payload["gaps_total"], 1)
        self.assertEqual(payload["schema"], "bringup-bench/trial-score/v1")


class TestCalibrationAndCompare(unittest.TestCase):
    """Resolution arithmetic and the three-valued A/B rule."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_calibrate_golden(self):
        resolution = bench_lib.calibrate([_score(t) for t in (10, 12, 11, 10, 12)])
        self.assertEqual(resolution["n"], 5)
        self.assertAlmostEqual(resolution["gaps"]["mean"], 11.0)
        self.assertAlmostEqual(resolution["gaps"]["sd"], 1.0)
        self.assertEqual(resolution["gaps"]["spread"], 2)
        self.assertEqual(resolution["min_detectable_gap_difference"], 2)

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_calibrate_rejects_mixed_or_tiny(self):
        with self.assertRaisesRegex(bench_lib.BenchError, "at least two"):
            bench_lib.calibrate([_score(10)])
        with self.assertRaisesRegex(bench_lib.BenchError, "share assets_ref"):
            bench_lib.calibrate([_score(10), _score(11, ref="b" * 40)])

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_three_valued_verdicts(self):
        resolution = {"min_detectable_gap_difference": 2}
        arm_a = [_score(10), _score(10)]
        self.assertEqual(bench_lib.compare_trials(
            arm_a, [_score(13), _score(13)], resolution)["verdict"], "worse")
        self.assertEqual(bench_lib.compare_trials(
            arm_a, [_score(7), _score(7)], resolution)["verdict"], "better")
        self.assertEqual(bench_lib.compare_trials(
            arm_a, [_score(11), _score(12)], resolution)["verdict"],
            "indistinguishable")
        with self.assertRaisesRegex(bench_lib.BenchError, "both arms"):
            bench_lib.compare_trials(arm_a, [], resolution)


class TestSelfChecks(unittest.TestCase):
    """A/A identity and injection detection on suite scorecards."""

    @staticmethod
    def _card(verdicts):
        cells = [{"model": m, "check": c, "verdict": v,
                  "expect": "pass", "ok": v == "pass"}
                 for (m, c), v in verdicts.items()]
        return {"cells": cells}

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_aa_and_injection(self):
        base = {("m1", "released_layout"): "pass", ("m1", "tied_save"): "pass"}
        same = bench_lib.selfcheck_aa(self._card(base), self._card(base))
        self.assertTrue(same["identical"])
        damaged = {**base, ("m1", "released_layout"): "fail"}
        aa = bench_lib.selfcheck_aa(self._card(base), self._card(damaged))
        self.assertFalse(aa["identical"])
        injection = bench_lib.selfcheck_injection(
            self._card(base), self._card(damaged), "drop-something")
        self.assertTrue(injection["detected"])
        self.assertTrue(injection["flips"][0]["regression"])
        blind = bench_lib.selfcheck_injection(
            self._card(base), self._card(base), "unseen-category")
        self.assertFalse(blind["detected"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_mismatched_cells_refused(self):
        card_a = self._card({("m1", "tied_save"): "pass"})
        card_b = self._card({("m2", "tied_save"): "pass"})
        with self.assertRaisesRegex(bench_lib.BenchError, "different cells"):
            bench_lib.diff_suites(card_a, card_b)


class TestMask(unittest.TestCase):
    """Family masking: quarantine moves, disclosure, and guard rails."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_mask_moves_and_discloses(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            snap = root / "snap"
            (snap / "hyper_parallel" / "models" / "foofam").mkdir(parents=True)
            (snap / "hyper_parallel" / "models" / "foofam" / "adapter.py").write_text("x = 1", encoding="utf-8")
            (snap / "hyper_parallel" / "models" / "otherfam").mkdir()
            (snap / "tests" / "ut" / "auto_models" / "models").mkdir(parents=True)
            (snap / "tests" / "ut" / "auto_models" / "models"
             / "test_foofam.py").write_text("y = 2", encoding="utf-8")
            (snap / "docs").mkdir()
            (snap / "docs" / "guide.md").write_text("mentions FooFam in prose", encoding="utf-8")
            quarantine = root / "quarantine"
            manifest = bench_lib.mask_model_family(snap, "FooFam", quarantine)
        self.assertEqual(sorted(manifest["moved"]),
                         ["hyper_parallel/models/foofam",
                          "tests/ut/auto_models/models/test_foofam.py"])
        self.assertEqual(manifest["residual_mentions"], ["docs/guide.md"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_mask_leaves_tree_usable_and_guards_inputs(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            snap = root / "snap"
            (snap / "hyper_parallel" / "models" / "foofam").mkdir(parents=True)
            (snap / "hyper_parallel" / "models" / "otherfam").mkdir()
            quarantine = root / "q"
            manifest = bench_lib.mask_model_family(snap, "foofam", quarantine)
            self.assertTrue((snap / "hyper_parallel" / "models" / "otherfam").is_dir())
            self.assertTrue((quarantine / "hyper_parallel" / "models" / "foofam").is_dir())
            again = bench_lib.mask_model_family(snap, "foofam", quarantine)
            self.assertEqual(again["moved"], [])
            with self.assertRaisesRegex(bench_lib.BenchError, "at least 3"):
                bench_lib.mask_model_family(snap, "ab", quarantine)
            with self.assertRaisesRegex(bench_lib.BenchError, "outside the snapshot"):
                bench_lib.mask_model_family(snap, "foofam", snap / "inner")
        self.assertEqual(manifest["family"], "foofam")


class TestSnapshot(unittest.TestCase):
    """Snapshot materialization and injection application on a tiny repo."""

    @staticmethod
    def _git(repo, *args):
        subprocess.run(["git", "-C", str(repo), *args], check=True,
                       capture_output=True, text=True)

    @arg_mark(plat_marks=["cpu_linux", "cpu_windows", "cpu_macos"],
              level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_snapshot_and_inject(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            self._git(repo, "init", "-q")
            self._git(repo, "config", "user.email", "t@test")
            self._git(repo, "config", "user.name", "t")
            (repo / "asset.txt").write_text("original\n", encoding="utf-8")
            self._git(repo, "add", "asset.txt")
            self._git(repo, "commit", "-q", "-m", "seed")
            patch = root / "damage.patch"
            patch.write_text(
                "--- a/asset.txt\n+++ b/asset.txt\n@@ -1 +1 @@\n"
                "-original\n+damaged\n", encoding="utf-8")

            clean = root / "snap-clean"
            bench_lib.snapshot_assets(repo, "HEAD", clean)
            self.assertEqual((clean / "asset.txt").read_text(encoding="utf-8"),
                             "original\n")
            injected = root / "snap-injected"
            bench_lib.snapshot_assets(repo, "HEAD", injected, inject=patch)
            self.assertEqual((injected / "asset.txt").read_text(encoding="utf-8"),
                             "damaged\n")
            with self.assertRaisesRegex(bench_lib.BenchError, "not empty"):
                bench_lib.snapshot_assets(repo, "HEAD", clean)
            bad_patch = root / "bad.patch"
            bad_patch.write_text(
                "--- a/asset.txt\n+++ b/asset.txt\n@@ -1 +1 @@\n"
                "-no-such-line\n+x\n", encoding="utf-8")
            broken = root / "snap-broken"
            with self.assertRaisesRegex(bench_lib.BenchError, "did not apply"):
                bench_lib.snapshot_assets(repo, "HEAD", broken, inject=bad_patch)
            # Release the worktrees so Windows can delete the tempdir.
            for snap in (clean, injected, broken):
                subprocess.run(["git", "-C", str(repo), "worktree", "remove",
                                "--force", str(snap)], check=False,
                               capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
