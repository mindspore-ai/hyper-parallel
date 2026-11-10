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
"""CPU behavioral tests: real subprocess stages, invalid evidence, recovery and budgets."""

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory, mkdtemp
import unittest
from unittest.mock import patch

from tests.common.mark_utils import arg_mark


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / ".agent/skills/mfu-riser/scripts"
if sys.platform == "win32":
    raise unittest.SkipTest("MFU Riser uses POSIX process groups and file locks")


def load_modules() -> dict:
    """Load standalone scripts without polluting the test runner's module namespace."""
    loaded = {}
    with patch.object(sys, "path", [str(SCRIPTS)] + sys.path), patch.dict(sys.modules):
        for name in ("collect_state", "riser_contract", "mfu_riser", "audit_rank_steps"):
            spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
    return loaded


MODULES = load_modules()
ENGINE = MODULES["mfu_riser"]
CONTRACT = MODULES["riser_contract"]
COLLECT = MODULES["collect_state"]
AUDIT = MODULES["audit_rank_steps"]

# This backend deliberately generates synthetic evidence to exercise orchestration,
# not to claim model correctness or accelerator performance.
BACKEND = r'''
import hashlib
import json
import os
from pathlib import Path
import sys

root = Path(os.environ["RISER_TRIAL_DIR"])
context = json.loads(Path(os.environ["RISER_CONTEXT"]).read_text())
settings = json.loads((Path.cwd() / "candidate.json").read_text())
phase = sys.argv[1]
with (root / "stage-order.txt").open("a") as log:
    log.write(phase + "\n")
if settings.get("fail_phase") == phase:
    raise SystemExit(7)
if phase == "profile":
    state = {"schema": "mfu-riser-state/v1", "valid": True,
             "measurement_binding": {"trial_token": context["trial_token"],
                                     "source_digest": context["source_digest"]},
             "profiling": [{"trace": {"steps": 2, "context": {"Stage": 100}},
                             "checks": [{"passed": True}],
                             "provenance": {"schedule": {"warmup": 1}}}],
             "operators": [{"status": "PASS", "hotspots": [{"operator": "test-only"}]}]}
    if settings.get("stale_profile"):
        state["measurement_binding"]["trial_token"] = "old-profile"
    (root / "state.json").write_text(json.dumps(state))
if phase == "validate":
    workload = context["config"]["workload"]
    (root / "audit.json").write_text(json.dumps({"synthetic_test_evidence": True}))
    check = {"passed": True, "evidence": "audit.json",
             "sha256": hashlib.sha256((root / "audit.json").read_bytes()).hexdigest()}
    checks = {key: dict(check) for key in ["real_weights", "all_ranks_complete", "finite_loss_grad",
              "input_equivalence", "numerical_equivalence", "independent_flops", "timing_boundary",
              "fusion_active", "fallback_correct"]}
    first, last = workload["window"]
    rate = settings.get("mfu", 10)
    scale = workload["global_batch_size"] / 2
    samples = [{"step": step, "time_s": 10 * scale / rate, "model_tokens": 1000 * scale,
                "model_flops": 20e12 * scale, "mfu_percent": rate} for step in range(first, last + 1)]
    if settings.get("varied_times"):
        for row in samples[1:]:
            row["time_s"] *= 2
            row["mfu_percent"] /= 2
    if settings.get("bad_flops"):
        samples[0]["model_flops"] *= 2
    if settings.get("inflate_work"):
        for row in samples:
            row["model_flops"] *= 2
            row["mfu_percent"] *= 2
    if settings.get("nan"):
        samples[0]["time_s"] = float("nan")
    measurement = {"schema": "mfu-riser-measurement/v1", "trial_token": context["trial_token"],
                   "workload": workload, "profiler_during_benchmark": False,
                   "runtime": [{"rank": rank, "source_digest": context["source_digest"],
                                "steps_completed": workload["train_steps"]} for rank in range(2)],
                   "samples": samples, "checks": checks, "profile_state": "state.json"}
    policy = context["config"].get("performance_policy")
    if policy is not None:
        level = policy["validation_level"]
        measurement["validation_level"] = level
        checks.update(model_semantics=dict(check), candidate_active=dict(check))
        if level == "screening":
            checks.pop("numerical_equivalence")
            checks["implementation_correctness"] = dict(check)
        reference = 20e12 * scale * settings.get("architecture_flops_factor", 1)
        audited = {"trial_token": context["trial_token"], "source_digest": context["source_digest"],
                   "model_config_sha256": workload["model_config_sha256"],
                   "convention": workload["flops_convention"],
                   "steps": [{"step": step, "model_tokens": 1000 * scale,
                              "components": {"attention": reference / 4, "experts": reference * 3 / 4},
                              "approximate_model_flops": 20e12 * scale} for step in range(first, last + 1)]}
        if settings.get("stale_flops_audit"):
            audited["trial_token"] = "stale"
        audit_path = root / "flops-audit.json"
        audit_path.write_text(json.dumps(audited))
        checks["independent_flops"] = {"passed": True, "evidence": audit_path.name,
                                     "sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest()}
    if settings.get("executed_work"):
        executed = [{"step": first, "executed_flops": 30e12 * scale}]
        if settings.get("duplicate_executed_step"):
            executed += executed
        measurement["executed_work"] = {**check, "method": "estimate", "coverage": "fixture arithmetic",
                                        "samples": executed}
        measurement["allocation"] = {**check, "device_seconds": 12}
    if settings.get("failed_check"):
        checks[settings["failed_check"]]["passed"] = False
    if settings.get("missing_check"):
        checks.pop(settings["missing_check"])
    if settings.get("stale"):
        measurement["trial_token"] = "old-run"
    if settings.get("changed_workload"):
        measurement["workload"]["routing"] = "fixed"
    if settings.get("missing_rank"):
        measurement["runtime"].pop()
    (root / "measurement.json").write_text(json.dumps(measurement))
'''


class RiserLoopTest(unittest.TestCase):
    """Run isolated controller sessions against a deterministic CPU backend."""

    def setUp(self) -> None:
        """Allocate an isolated directory and register cleanup for each controller session."""
        self.root = Path(mkdtemp(prefix="mfu-riser-test-"))
        self.addCleanup(shutil.rmtree, self.root)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.backend = self.root / "backend.py"
        self.backend.write_text(BACKEND)
        self.settings()
        self.run = self.root / "run"
        self.evidence = self.root / "prior.json"
        self.evidence.write_text('{"synthetic": true}')
        self.config = {
            "schema": "mfu-riser/v1", "workload": {
                "model": "SYNTHETIC-CPU-TEST", "weights_id": "fixture", "dataset_id": "fixture",
                "optimizer": "fixture", "precision": "bf16", "routing": "native",
                "global_batch_size": 2, "world_size": 2, "nodes": ["fixture"],
                "peak_tflops_per_device": 100, "flops_convention": "fixture-model-only",
                "timing_scope": "fixture-complete-step", "train_steps": 30, "window": [11, 30]},
            "target_mfu_percent": 14, "max_trials": 8, "max_seconds": 60,
            "command_timeout_seconds": 10, "noise_fraction": 0.02,
            "validator_files": [str(self.backend)],
            "commands": {phase: [sys.executable, str(self.backend), phase]
                         for phase in ("precheck", "benchmark", "profile", "validate", "cleanup")}}
        ENGINE.initialize(self.run, self.config)

    def settings(self, **values: object) -> None:
        """Write candidate configuration, which is included in the source snapshot."""
        (self.workspace / "candidate.json").write_text(json.dumps(values))

    def plan(self, kind: str = "candidate") -> dict:
        """Build a test proposal linked to the current best."""
        state = CONTRACT.read_json(self.run / "controller.json")
        return {"kind": kind, "parent": state["best"], "workspace": str(self.workspace),
                "source_paths": ["candidate.json"], "operator_scope": "compute_fusion",
                "actions": [] if kind == "baseline" else ["test-fusion"],
                "observation": "test observation", "mechanism": "test mechanism",
                "expected_effect": "test prediction", "falsifier": "test falsifier",
                "compatibility": "test compatibility", "evidence": [str(self.evidence)],
                "knowledge": [str(self.evidence)]}

    def baseline(self) -> dict:
        """Record the initial synthetic baseline."""
        return ENGINE.run_trial(self.run, self.plan("baseline"))

    def enable_performance_policy(self, level: str = "screening", method: str = "architecture_aware") -> None:
        """Start an opted-in series with a known architecture and a short screening window."""
        self.run = self.root / "performance"
        self.config["workload"].update(model_config_sha256="a" * 64, sequence_length=1024)
        if level == "screening":
            self.config["workload"].update(train_steps=8, window=[4, 8])
        self.config["performance_policy"] = {
            "validation_level": level, "low_mfu_threshold_percent": 10,
            "tunable_workload_fields": ["sequence_length", "global_batch_size"],
            "flops_accounting": {"method": method, "max_relative_error": 0.02},
        }
        ENGINE.initialize(self.run, self.config)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_screening_uses_critical_path_and_cannot_confirm_target(self):
        """Feature: Performance screening.
        Description: Screen a communication candidate above a low-MFU baseline.
        Expectation: Require semantics, implementation, and engagement evidence before selecting the candidate.
        """
        self.enable_performance_policy()
        self.settings(mfu=8)
        baseline = self.baseline()
        self.assertEqual(baseline["decision"], "baseline")
        status = ENGINE.status(self.run)
        self.assertEqual(status["next_action"], "plan_from_critical_path")
        self.assertAlmostEqual(status["required_throughput_multiplier"], 14 / 8)
        plan = {**self.plan(), "operator_scope": "communication"}
        for check in ("model_semantics", "implementation_correctness", "candidate_active"):
            with self.subTest(check=check):
                self.settings(mfu=15, failed_check=check)
                invalid = ENGINE.run_trial(self.run, plan)
                self.assertEqual(invalid["decision"], "crash")
                self.assertIn(check, invalid["reason"])
        self.settings(mfu=15)
        candidate = ENGINE.run_trial(self.run, plan)
        self.assertEqual(candidate["decision"], "keep")
        confirm = {**self.plan("confirm"), "operator_scope": "communication"}
        repeated = ENGINE.run_trial(self.run, confirm)
        self.assertFalse(repeated["target_met"])
        self.assertEqual(ENGINE.status(self.run)["next_action"], "start_acceptance_series")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_capacity_series_requires_declared_fields_and_its_own_baseline(self):
        """Feature: Workload capacity series.
        Description: Link a global-batch change and try changes to frozen fields.
        Expectation: Record an independent baseline and reject changes to model, routing, or device count.
        """
        self.enable_performance_policy()
        parent_run = self.run
        parent = self.baseline()
        child = copy.deepcopy(self.config)
        child["workload"]["global_batch_size"] = 4
        for key, value in (("routing", "fixed"), ("model_config_sha256", "b" * 64), ("world_size", 4)):
            with self.subTest(key=key):
                invalid = copy.deepcopy(child)
                invalid["workload"][key] = value
                with self.assertRaisesRegex(ValueError, "outside declared workload tuning"):
                    ENGINE.initialize(self.root / "invalid-child", invalid, parent_run)
        self.run = self.root / "capacity-child"
        state = ENGINE.initialize(self.run, child, parent_run)
        self.assertIsNone(state["best"])
        self.assertEqual(state["parent_series"]["trial"], parent["id"])
        self.assertEqual(state["parent_series"]["changed_workload_fields"], ["global_batch_size"])
        capacity = self.baseline()
        self.assertEqual(capacity["decision"], "baseline")
        self.assertEqual(capacity["metrics"]["samples"][0]["model_tokens"], 2000)
        self.assertEqual(ENGINE.status(parent_run)["best"], parent["id"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_acceptance_replays_screened_source_and_confirms_independently(self):
        """Feature: Performance acceptance.
        Description: Promote an optimizer candidate with its screened source.
        Expectation: Reject a changed source and require a second accepted measurement for the target.
        """
        self.enable_performance_policy()
        self.baseline()
        self.settings(mfu=15)
        screened = ENGINE.run_trial(self.run, {**self.plan(), "operator_scope": "optimizer"})
        self.assertEqual(screened["decision"], "keep")
        parent_run = self.run
        child = copy.deepcopy(self.config)
        child["performance_policy"]["validation_level"] = "acceptance"
        child["workload"].update(train_steps=30, window=[11, 30])
        self.run = self.root / "acceptance-child"
        ENGINE.initialize(self.run, child, parent_run)
        self.settings(mfu=20)
        with self.assertRaisesRegex(ValueError, "screened source"):
            self.baseline()
        self.settings(mfu=15)
        accepted = self.baseline()
        self.assertEqual(accepted["decision"], "baseline")
        self.assertEqual(accepted["plan"]["operator_scope"], "optimizer")
        self.assertEqual(accepted["plan"]["actions"], screened["plan"]["actions"])
        repeat = ENGINE.run_trial(self.run, {**self.plan("confirm"), "operator_scope": "optimizer"})
        self.assertEqual(repeat["decision"], "confirmed")
        self.assertTrue(repeat["target_met"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_mfu_approximation_requires_a_fresh_architecture_audit(self):
        """Feature: FLOPs audit.
        Description: Compare an approximate model-work calculation with a component breakdown.
        Expectation: Reject excessive error and stale audit evidence.
        """
        self.enable_performance_policy(method="approximation")
        self.settings(architecture_flops_factor=1.01)
        baseline = self.baseline()
        self.assertEqual(baseline["decision"], "baseline")
        for settings, reason in (
            ({"architecture_flops_factor": 1.1}, "exceeds declared error"),
            ({"stale_flops_audit": True}, "different trial"),
        ):
            with self.subTest(settings=settings):
                self.settings(mfu=20, **settings)
                rejected = ENGINE.run_trial(self.run, self.plan())
                self.assertEqual(rejected["decision"], "crash")
                self.assertIn(reason, rejected["reason"])
                self.assertEqual(ENGINE.status(self.run)["best"], baseline["id"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_hfu_uses_the_unprofiled_time_of_the_sampled_steps(self):
        """Feature: Executed work reporting.
        Description: Combine one executed-work sample with a longer benchmark window.
        Expectation: Use the matching step time, report coverage, and reject duplicate steps.
        """
        self.enable_performance_policy("acceptance")
        self.settings(executed_work=True, varied_times=True)
        result = self.baseline()
        self.assertEqual(result["decision"], "baseline")
        metrics = result["metrics"]
        self.assertAlmostEqual(metrics["hfu_percent"], 15)
        self.assertAlmostEqual(metrics["tokens_per_second_per_device"], 10000 / 39)
        self.assertEqual(metrics["hfu_method"], "estimate")
        self.assertEqual(metrics["hfu_steps"], [11])
        self.assertEqual(metrics["allocated_device_seconds"], 12)
        self.assertGreater(result["stage_seconds"]["benchmark"], 0)
        self.assertIn("estimate", ENGINE.report(self.run).read_text())
        self.settings(mfu=15, executed_work=True, duplicate_executed_step=True)
        rejected = ENGINE.run_trial(self.run, self.plan())
        self.assertEqual(rejected["decision"], "crash")
        self.assertIn("duplicate executed work step", rejected["reason"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_full_loop_requires_confirmation_then_stops(self):
        """Feature: Trial lifecycle.
        Description: Run a baseline, an improvement, and an independent confirmation.
        Expectation: Accept the target only after confirmation and stop further trials.
        """
        initial = self.baseline()
        self.assertEqual(initial["decision"], "baseline")
        self.settings(mfu=15)
        improved = ENGINE.run_trial(self.run, self.plan())
        self.assertEqual(improved["decision"], "keep")
        self.assertAlmostEqual(improved["metrics"]["mfu_percent"], 15)
        self.assertEqual(ENGINE.status(self.run)["next_action"], "confirm_best")
        confirmed = ENGINE.run_trial(self.run, self.plan("confirm"))
        self.assertEqual(confirmed["decision"], "confirmed")
        self.assertEqual(ENGINE.status(self.run)["stop_reason"], "target_confirmed")
        with self.assertRaisesRegex(ValueError, "stopped"):
            ENGINE.run_trial(self.run, self.plan())
        stages = (self.run / "trials" / improved["id"] / "stage-order.txt").read_text().splitlines()
        self.assertEqual(stages, ["precheck", "benchmark", "profile", "validate", "cleanup"])
        self.assertIn("confirmed", ENGINE.report(self.run).read_text())

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_noise_and_failures_do_not_replace_best(self):
        """Feature: Candidate selection.
        Description: Try an improvement within noise and several invalid measurements.
        Expectation: Retain the validated best result.
        """
        initial = self.baseline()
        self.settings(mfu=10.1)
        result = ENGINE.run_trial(self.run, self.plan())
        self.assertEqual(result["decision"], "discard")
        for setting in ({"failed_check": "numerical_equivalence"}, {"missing_check": "fusion_active"},
                        {"stale": True}, {"bad_flops": True}, {"nan": True}, {"inflate_work": True}):
            with self.subTest(setting=setting):
                self.settings(mfu=25, **setting)
                result = ENGINE.run_trial(self.run, self.plan())
                self.assertEqual(result["decision"], "crash")
                self.assertIsNone(result["metrics"])
                self.assertEqual(ENGINE.status(self.run)["best"], initial["id"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_workload_and_rank_coverage_are_required(self):
        """Feature: Measurement binding.
        Description: Change the recorded workload, rank coverage, or profiling token.
        Expectation: Reject invalid measurements.
        """
        self.baseline()
        for setting in ({"changed_workload": True}, {"missing_rank": True}, {"stale_profile": True}):
            self.settings(mfu=25, **setting)
            result = ENGINE.run_trial(self.run, self.plan())
            self.assertEqual(result["decision"], "crash")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_failed_command_stops_later_phases_and_cleans_up(self):
        """Feature: Failure cleanup.
        Description: Inject a benchmark command failure.
        Expectation: Skip later measurement stages and run cleanup.
        """
        self.settings(fail_phase="benchmark")
        result = self.baseline()
        self.assertEqual(result["decision"], "crash")
        stages = (self.run / "trials" / result["id"] / "stage-order.txt").read_text().splitlines()
        self.assertEqual(stages, ["precheck", "benchmark", "cleanup"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_cleanup_failure_requires_recovery(self):
        """Feature: Interrupted trial recovery.
        Description: Fail trial cleanup and invoke job-scoped recovery.
        Expectation: Block further trials while the failed cleanup is pending.
        """
        self.settings(fail_phase="cleanup")
        failed = self.baseline()
        self.assertEqual(failed["decision"], "crash")
        self.assertEqual(ENGINE.status(self.run)["stop_reason"], "pending_trial_requires_recovery")
        self.settings()
        recovered = ENGINE.recover(self.run)
        self.assertEqual(recovered["decision"], "crash")
        self.assertIsNone(ENGINE.status(self.run)["best"])
        self.assertEqual(ENGINE.status(self.run)["next_action"], "baseline")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_stale_parent_and_modified_confirmation_are_rejected(self):
        """Feature: Source and parent binding.
        Description: Try a stale parent and change the confirmation source.
        Expectation: Reject both proposals before launching them.
        """
        self.baseline()
        stale = self.plan()
        self.settings(mfu=15)
        ENGINE.run_trial(self.run, self.plan())
        with self.assertRaisesRegex(ValueError, "stale best"):
            ENGINE.run_trial(self.run, stale)
        self.settings(mfu=16)
        with self.assertRaisesRegex(ValueError, "Confirmation source"):
            ENGINE.run_trial(self.run, self.plan("confirm"))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_config_and_acceptance_scripts_cannot_drift(self):
        """Feature: Frozen run contract.
        Description: Edit the validator or configuration after initialization.
        Expectation: Reject trials with modified acceptance inputs.
        """
        self.backend.write_text(BACKEND + "\n# changed validator\n")
        with self.assertRaisesRegex(ValueError, "Acceptance scripts changed"):
            self.baseline()
        self.backend.write_text(BACKEND)
        changed = copy.deepcopy(self.config)
        changed["workload"]["peak_tflops_per_device"] = 80
        ENGINE.save(self.run / "config.json", changed)
        with self.assertRaisesRegex(ValueError, "config changed"):
            self.baseline()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_trial_and_time_budgets_are_enforced(self):
        """Feature: Bounded search.
        Description: Exhaust the attempt limit and then the time limit.
        Expectation: Return the applicable stop reason.
        """
        config = copy.deepcopy(self.config)
        config["max_trials"] = 1
        other = self.root / "limited"
        ENGINE.initialize(other, config)
        self.run = other
        self.baseline()
        self.settings(mfu=9)
        ENGINE.run_trial(self.run, self.plan())
        self.assertEqual(ENGINE.status(self.run)["stop_reason"], "trial_budget_exhausted")
        state = CONTRACT.read_json(self.run / "controller.json")
        state["deadline"] = 0
        ENGINE.save(self.run / "controller.json", state)
        self.assertEqual(ENGINE.status(self.run)["stop_reason"], "time_budget_exhausted")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_no_empty_or_short_window(self):
        """Feature: Acceptance measurement window.
        Description: Submit invalid or too short acceptance windows.
        Expectation: Reject windows without sufficient post-warmup steps.
        """
        for window in ([1, 30], [11, 12], [20, 19]):
            config = copy.deepcopy(self.config)
            config["workload"]["window"] = window
            with self.assertRaisesRegex(ValueError, "Window"):
                CONTRACT.validate_config(config)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_command_timeout_records_crash_and_runs_cleanup(self):
        """Feature: Command timeout.
        Description: Run a precheck that exceeds the configured timeout.
        Expectation: Record the crash and complete cleanup.
        """
        config = copy.deepcopy(self.config)
        config["command_timeout_seconds"] = 1
        config["commands"]["precheck"] = [sys.executable, "-c", "import time; time.sleep(30)"]
        other = self.root / "timeout-run"
        ENGINE.initialize(other, config)
        self.run = other
        result = self.baseline()
        self.assertEqual(result["decision"], "crash")
        self.assertIn("TimeoutExpired", result["reason"])
        self.assertEqual(result["cleanup"], "PASS")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_unstable_confirmation_does_not_claim_success(self):
        """Feature: Independent confirmation.
        Description: Compare a repetition outside the noise band.
        Expectation: Classify the repetition as unstable.
        """
        decision = CONTRACT.decide({"mfu_percent": 18}, {"metrics": {"mfu_percent": 15}},
                                   self.config, "confirm")
        self.assertEqual(decision, "unstable")


class RankAuditTest(unittest.TestCase):
    """Reused JSONL gates distinguish scalar parity from full tensor equivalence."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_raw_records_verify_work_and_leave_unmeasured_checks_unknown(self):
        """Feature: Rank-step auditing.
        Description: Compare measured scalar records with an independent work calculation.
        Expectation: Detect inconsistent MFU and identify the checks that still lack evidence.
        """
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics = {"performance/step_time": 1, "performance/model_tflops_per_device": 20,
                       "performance/mfu_percent": 20}
            meter = {"step": 1, "rank": 0, "metrics": metrics, "batches": [{"model_tokens": 100}]}
            row = {"step": 1, "rank": 0, "loss_local": 1, "grad_norm_pre_clip": 2, "input_hashes": ["actual"]}
            (root / "steps_rank0.jsonl").write_text(json.dumps(row) + "\n")
            (root / "meter_steps_rank0.jsonl").write_text(json.dumps(meter) + "\n")
            expected = {"derivation": "fixture", "source": "fixture",
                        "samples": [{"step": 1, "model_tokens": 100, "model_flops": 20e12}]}
            work = root / "work.json"
            work.write_text(json.dumps(expected))
            args = argparse.Namespace(run_dir=root, reference_dir=None, expected_work=work,
                                      world_size=1, steps=1, peak=100, window=[1, 1], rtol=0, atol=0)
            result = AUDIT.audit(args)
            self.assertIsNone(result["scalar_parity"])
            self.assertIn("module/tensor gradient parity", result["unverified"])
            self.assertIn("microbatch input coverage", result["unverified"])
            args.reference_dir = root
            self.assertTrue(AUDIT.audit(args)["scalar_parity"])
            meter["metrics"]["performance/mfu_percent"] = 40
            (root / "meter_steps_rank0.jsonl").write_text(json.dumps(meter) + "\n")
            with self.assertRaisesRegex(ValueError, "Logged MFU"):
                AUDIT.audit(args)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_accumulation_checks_coverage_even_when_reference_matches(self):
        """Feature: Accumulated input auditing.
        Description: Audit two microbatches, then truncate matched inputs or meter records.
        Expectation: Reject missing coverage, preserve ordering, and reject invalid counts.
        """
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference"
            reference.mkdir()
            metrics = {"performance/step_time": 1, "performance/model_tflops_per_device": 20,
                       "performance/mfu_percent": 20}
            meter = {"step": 1, "rank": 0, "metrics": metrics,
                     "batches": [{"model_tokens": 50}, {"model_tokens": 50}]}
            row = {"step": 1, "rank": 0, "loss_local": 1, "grad_norm_pre_clip": 2,
                   "input_hashes": ["microbatch-a", "microbatch-b"]}
            meter_path = root / "meter_steps_rank0.jsonl"
            candidate_path = root / "steps_rank0.jsonl"
            reference_path = reference / "steps_rank0.jsonl"
            meter_path.write_text(json.dumps(meter) + "\n")
            candidate_path.write_text(json.dumps(row) + "\n")
            reference_path.write_text(json.dumps(row) + "\n")
            work = root / "work.json"
            work.write_text(json.dumps({"derivation": "fixture", "source": "fixture",
                                        "samples": [{"step": 1, "model_tokens": 100, "model_flops": 20e12}]}))
            args = argparse.Namespace(run_dir=root, reference_dir=reference, expected_work=work,
                                      world_size=1, steps=1, peak=100, window=[1, 1], rtol=0, atol=0,
                                      microbatches_per_update=2)
            result = AUDIT.audit(args)
            self.assertTrue(result["scalar_parity"])
            self.assertEqual(result["all_rank_microbatches"], 2)
            self.assertNotIn("microbatch input coverage", result["unverified"])
            reversed_row = dict(row, input_hashes=list(reversed(row["input_hashes"])))
            reference_path.write_text(json.dumps(reversed_row) + "\n")
            self.assertFalse(AUDIT.audit(args)["scalar_parity"])
            missing = dict(row, input_hashes=row["input_hashes"][:1])
            for label, path in (("candidate", candidate_path), ("reference", reference_path)):
                candidate_path.write_text(json.dumps(row) + "\n")
                reference_path.write_text(json.dumps(row) + "\n")
                path.write_text(json.dumps(missing) + "\n")
                with self.subTest(missing=label), self.assertRaisesRegex(ValueError, "microbatch input coverage"):
                    AUDIT.audit(args)
            candidate_path.write_text(json.dumps(missing) + "\n")
            with self.assertRaisesRegex(ValueError, "microbatch input coverage"):
                AUDIT.audit(args)
            candidate_path.write_text(json.dumps(row) + "\n")
            reference_path.write_text(json.dumps(row) + "\n")
            meter["batches"] = [{"model_tokens": 100}]
            meter_path.write_text(json.dumps(meter) + "\n")
            with self.assertRaisesRegex(ValueError, "microbatch meter coverage"):
                AUDIT.audit(args)
            for count in (0, -1, 1.5, True):
                args.microbatches_per_update = count
                with self.subTest(count=count), self.assertRaisesRegex(ValueError, "positive integer"):
                    AUDIT.audit(args)


class OperatorEvidenceTest(unittest.TestCase):
    """Operator grouping must retain shapes and exclude communication launch kernels."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_actual_shapes_and_durations_without_invented_shares(self):
        """Feature: Operator evidence.
        Description: Group measured operators by shape and include a communication launch.
        Expectation: Preserve shape groups and exclude communication from compute hotspots.
        """
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "kernel_details.csv"
            path.write_text("Name,Type,Accelerator Core,Duration(us),Input Shapes\n"
                            "x,MatMul,AI_CORE,10,[2x3]\n"
                            "y,MatMul,AI_CORE,20,[2x3]\n"
                            "z,MatMul,AI_CORE,40,[4x3]\n"
                            "a,allreduceAicpuKernel,AI_CPU,999,[2x3]\n")
            result = COLLECT.operator_shapes(path)
            self.assertEqual(result["group_count"], 2)
            self.assertEqual(result["hotspots"][0]["summed_device_us"], 40)
            self.assertEqual(result["hotspots"][1]["calls"], 2)
            self.assertIsNone(result["hotspots"][0]["input_dtypes"])
            self.assertNotIn("wall_share", result["hotspots"][0])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="dryrun", essential_mark="essential")
    def test_missing_checks_or_warmup_cannot_pass(self):
        """Feature: Profile validity.
        Description: Pass empty or incomplete reports to the profile checker.
        Expectation: Require warmup and populated invariant checks.
        """
        self.assertTrue(COLLECT.check_reports([]))
        self.assertTrue(COLLECT.check_reports([{"trace": {"steps": 2, "context": {"Stage": 10}}}]))
