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
"""CPU regression tests for agent evidence and strategy execution contracts."""

import copy
import importlib
import importlib.util
import io
import json
import sqlite3
from contextlib import closing
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

ROOT = Path(__file__).resolve().parents[3]
SKILLS = ROOT / ".agent/skills"


def load_script(skill: str, name: str):
    """Load an entrypoint with its sibling imports scoped to this operation."""
    directory = SKILLS / skill / "scripts"
    spec = importlib.util.spec_from_file_location(name, directory / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    # Standalone CLIs use bare sibling imports; restore both path and modules after loading.
    with patch.object(sys, "path", [str(directory)] + sys.path), patch.dict(sys.modules):
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


EVIDENCE = load_script("routing-validator", "_routing_evidence")
TOKEN = load_script("routing-validator", "_token_integrity")
ROUTING = load_script("routing-validator", "validate_routing")
COMM = load_script("moe-profiler", "_moe_comm_analysis")
LOAD = load_script("moe-profiler", "_moe_load_analysis")
EXTRACT = load_script("moe-profiler", "extract_moe_facts")
OPT = load_script("strategy-optimizer", "run_optimization")
SCHED = load_script("strategy-optimizer", "distributed_scheduler")
LINKS = load_script("strategy-optimizer", "check_skill_architecture")
SAFETY = load_script("strategy-optimizer", "safety_checks")


def capture() -> dict:
    """Return independent pre/post-dispatch and gradient observations."""
    return {
        "routing": [{"input_expert_indices": [1, 0, 1, 0], "token_permutation": [1, 3, 0, 2],
                     "tokens_per_expert": [2, 2], "expert_assignments": [0, 0, 1, 1]}],
        "state_dict": {"layers.3.experts.0.weight": [[1.0, 2.0]],
                       "layers.3.experts.1.weight": [[3.0, 4.0]]},
        "gradients": {"layers.3.experts.0.weight": [[0.1, 0.2]],
                      "layers.3.experts.1.weight": [[0.3, 0.4]]},
    }


class TemporaryTest(unittest.TestCase):
    """Keep generated evidence isolated from the checkout."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def write(self, name: str, content: str) -> Path:
        """Write a fixture and return its path."""
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path


class RoutingEvidenceTest(TemporaryTest):
    """Missing or inconsistent evidence cannot produce a correctness PASS."""

    def test_weights_and_empty_records_do_not_pass(self) -> None:
        """Parameter values alone establish neither routing nor gradients."""
        self.assertEqual(EVIDENCE.check_routing({})["status"], "UNVERIFIED")
        data = capture()
        data.pop("gradients")
        self.assertEqual(EVIDENCE.gradient_evidence(data)["status"], "UNVERIFIED")

    def test_permutation_must_link_to_original_assignments(self) -> None:
        """A bijection alone is insufficient when it routes slots to wrong experts."""
        data = capture()
        self.assertEqual(EVIDENCE.check_routing(data)["status"], "PASS")
        for permutation in ([0, 1, 2, 3], [1, 1, 0, 2], [1.5, 3, 0, 2]):
            with self.subTest(permutation=permutation):
                data["routing"][0]["token_permutation"] = permutation
                self.assertEqual(EVIDENCE.check_routing(data)["status"], "FAIL")

    def test_gradient_full_keys_shapes_and_finiteness(self) -> None:
        """Partial, mis-shaped, zero and nonfinite gradients fail."""
        data = capture()
        self.assertEqual(EVIDENCE.gradient_evidence(data)["status"], "PASS")
        for value in (None, [0.1, 0.2], [[0.0, 0.0]], [[float("nan"), 0.2]]):
            with self.subTest(value=value):
                data["gradients"]["layers.3.experts.1.weight"] = value
                self.assertEqual(EVIDENCE.gradient_evidence(data)["status"], "FAIL")

    def test_tokens_need_independent_input_and_topk(self) -> None:
        """Each record needs explicit independent counts, without stale metadata."""
        good = "input_tokens=2 top_k=2 tokens_per_expert=[2,2]"
        checker = TOKEN.TokenIntegrityChecker()
        log = self.write("rank0.log", good + "\n")
        self.assertEqual(checker.check([str(log)])["status"], "PASS")
        log.write_text(good + "\ntokens_per_expert=[2,2]\n")
        self.assertEqual(checker.check([str(log)])["status"], "UNVERIFIED")
        log.write_text("input_tokens=3 top_k=2 tokens_per_expert=[2,2]\n")
        self.assertEqual(checker.check([str(log)])["status"], "FAIL")

    def test_missing_required_checks_never_pass(self) -> None:
        """Optional passing checks cannot compensate for absent mandatory checks."""
        for strict, expected in ((False, "UNVERIFIED"), (True, "FAIL")):
            validator = ROUTING.RoutingValidator(None, None, self.directory, strict)
            with redirect_stdout(io.StringIO()):
                result = validator.generate_report({"expert_coverage": {"status": "PASS"}}, {})
            self.assertEqual(result["overall_status"], expected)

    def test_every_checkpoint_shard_is_checked(self) -> None:
        """A valid first shard cannot hide invalid routing in another shard."""
        good = self.write("rank0.json", json.dumps(capture()))
        data = capture()
        data["routing"][0]["token_permutation"] = [0, 1, 2, 3]
        bad = self.write("rank1.json", json.dumps(data))
        validator = ROUTING.RoutingValidator(None, None, self.directory)
        with redirect_stdout(io.StringIO()):
            result = validator.check_routing_correctness({"checkpoint_files": [str(good), str(bad)]})
        self.assertEqual(result["status"], "FAIL")

    def test_invalid_top_level_capture_is_rejected(self) -> None:
        """Bad JSON structure fails at the input boundary."""
        path = self.write("invalid.json", "[]")
        with self.assertRaisesRegex(ValueError, "mapping"):
            EVIDENCE.load_capture(str(path))


class ProfilerEvidenceTest(TemporaryTest):
    """Extract observed dimensions and units and keep the artifact index complete."""

    def test_sparse_layers_and_rank_step_filter(self) -> None:
        """Filtering does not create zero-based layers or unobserved steps."""
        log = self.write("rank2.log", "step=5 Layer 7 tokens_per_expert=[1,3]\n"
                         "step=9 Layer 11 tokens_per_expert=[9,1]\n")
        result = LOAD.LoadAnalyzer(steps=1, ranks=[2]).analyze([str(log)])
        self.assertEqual((result["num_steps"], result["num_layers"]), (1, 1))
        self.assertEqual(result["summary"][0][0], 7)
        self.assertIsNone(result["timeline"][0][-1])
        self.assertEqual(LOAD.LoadAnalyzer(ranks=[1]).analyze([str(log)])["num_steps"], 0)

    def test_communication_does_not_invent_compute_or_world_size(self) -> None:
        """Absent metadata and step timing remain null."""
        log = self.write("rank0.log", "step=1\n[MoE] Layer 7 dispatch: 2.0ms, 4.0MB\n")
        result = COMM.CommunicationAnalyzer().analyze([str(log)], [])
        self.assertEqual(result["timing"][0][5], 16.0)
        self.assertIsNone(result["world_size"])
        self.assertIsNone(result["volume"][0][3])
        self.assertIsNone(result["breakdown"][0][1])
        self.assertIsNone(result["breakdown"][0][4])

    def test_json_profile_units_and_observed_rank_count(self) -> None:
        """Profile-only evidence retains rank and converts explicit units."""
        profile = self.write("trace.json", json.dumps({"ops": [
            {"name": "AllToAll", "duration_us": 2000, "bytes": 4000000, "rank": 3, "step": 5}]}))
        result = COMM.CommunicationAnalyzer().analyze([], [str(profile)])
        self.assertEqual(result["timing"][0][3:6], [2.0, 4.0, 16.0])
        self.assertEqual(result["num_ranks"], 1)

    def test_sqlite_explicit_units(self) -> None:
        """SQLite operations are parsed with their recorded duration scale."""
        path = self.directory / "profile.db"
        with closing(sqlite3.connect(str(path))) as connection, connection:
            connection.execute("CREATE TABLE ops(name TEXT, duration_ns REAL, bytes INTEGER, "
                               "rank INTEGER, step INTEGER)")
            connection.execute("INSERT INTO ops VALUES ('AllToAll', 2000000, 4000000, 0, 1)")
        result = COMM.CommunicationAnalyzer().analyze([], [str(path)])
        self.assertEqual(result["timing"][0][3:6], [2.0, 4.0, 16.0])

    def test_malformed_profile_is_warning_without_fake_events(self) -> None:
        """Unsupported and invalid profiler input cannot fabricate measurements."""
        for ops in ({"bad": 1}, [1], [{"name": "AllToAll", "duration_ms": -1}]):
            profile = self.write("trace.json", json.dumps({"ops": ops}))
            result = COMM.CommunicationAnalyzer().analyze([], [str(profile)])
            self.assertEqual(result["timing"], [])
            self.assertTrue(result["warnings"])

    def test_partial_extraction_indexes_all_fifteen_artifacts(self) -> None:
        """Repeated partial extraction preserves unique, existing artifact paths."""
        log = self.write("logs/rank0.log", "training started\n")
        extractor = EXTRACT.MoEFactsExtractor(log.parent, None, None, self.directory / "output")
        sources = {"log_files": [str(log)], "profile_files": [], "checkpoint_files": []}
        with redirect_stdout(io.StringIO()):
            facts = extractor.extract_facts(sources)
            for _ in range(2):
                extractor.generate_artifacts(facts, sources)
                extractor.validate_output()
        index = json.loads((extractor.output_dir / "moe_facts.json").read_text())
        self.assertEqual(len(set(index["artifacts"])), 15)
        self.assertEqual(index["evidence_status"], "PARTIAL")
        self.assertEqual(index["summary"]["gradient_flow"], "UNVERIFIED")
        self.assertIsNone(index["summary"]["max_straggler_slowdown"])
        self.assertTrue(all((extractor.output_dir / name).is_file() for name in index["artifacts"]))


class OptimizerTest(TemporaryTest):
    """Mock device execution while retaining candidate/config/result logic."""

    def test_ep_overlaps_world_and_invalid_degrees_rejected(self) -> None:
        """EP does not multiply world size; invalid divisors are rejected."""
        result = OPT._fit_to_devices({"tp": 2, "ep": 2}, 8)
        self.assertEqual((result["dp"], result["edp_shard"]), (4, 4))
        for proposal in ({"ep": 0}, {"tp": 1.5}, {"dp_shard": 0}, {"ep": 3}):
            self.assertIsNone(OPT._fit_to_devices(proposal, 8))

    def test_incomplete_and_over_memory_results_cannot_win(self) -> None:
        """Selection requires complete measurements within the memory cap."""
        rows = [
            {"candidate_id": "incomplete", "success": True, "metrics": {"throughput_tokens_per_sec": 999}},
            {"candidate_id": "over", "success": True, "metrics": {
                "throughput_tokens_per_sec": 300, "step_time_ms": 10, "memory_peak_gb": 70}},
            {"candidate_id": "valid", "success": True, "metrics": {
                "throughput_tokens_per_sec": 130, "step_time_ms": 20, "memory_peak_gb": 50}},
        ]
        _, best, met = OPT._evaluate_results(rows, 100, 1.2, 60)
        self.assertEqual(best["candidate_id"], "valid")
        self.assertTrue(met)

    def test_template_fields_and_production_length_are_preserved(self) -> None:
        """Benchmark overrides stay out of the exported production length."""
        template = self.write("template.yaml", yaml.safe_dump({
            "model": {"_target_": "example.Model"}, "optimizer": {"_target_": "example.Optimizer"},
            "dataset": {"path": "preserved"}, "training": {"train_iters": 1000, "train_samples": 4000},
            "activation_checkpoint": {"mode": "selective", "layers": [1, 2]}}))
        strategy = OPT._fit_to_devices({"tp": 2, "ep": 2, "activation_checkpoint": "full"}, 8)
        candidate = {"id": "candidate", "strategy": strategy}
        output = OPT.write_candidate_config(self.directory / "configs", candidate, "model", 5, "npu", template)
        data = yaml.safe_load(output.read_text())
        self.assertEqual(data["training"], {"train_iters": 5, "train_samples": None})
        self.assertEqual(data["activation_checkpoint"], {"mode": "full"})
        self.assertEqual(data["dataset"], {"path": "preserved"})
        self.assertNotIn("parallelism", data)
        output = OPT.write_candidate_config(self.directory / "prod", candidate, "model", None, "npu", template)
        self.assertEqual(yaml.safe_load(output.read_text())["training"]["train_iters"], 1000)

    def test_plan_has_no_measured_winner(self) -> None:
        """Planning cannot produce a measured winner or production config."""
        out = self.directory / "plan"
        args = ["optimizer", "--model", "fixture", "--devices", "8", "--baseline-throughput", "100",
                "--baseline-config", "TP=2,EP=2,DP=4", "--plan-only", "--output-dir", str(out)]
        with patch.object(sys, "argv", args), redirect_stdout(io.StringIO()):
            self.assertEqual(OPT.main(), 0)
        result = json.loads((out / "optimization_result.json").read_text())
        self.assertEqual(result["final_result"]["status"], "PLANNED")
        self.assertIsNone(result["final_result"]["optimal_candidate"])
        self.assertFalse((out / "production_config.yaml").exists())

    def test_loop_refines_oom_and_exports_measured_winner(self) -> None:
        """A failed round updates constraints before a successful second round."""
        template = self.write("train.yaml", yaml.safe_dump({
            "model": {"_target_": "example.Model"}, "optimizer": {}, "training": {"train_iters": 1000}}))
        out = self.directory / "run"
        calls = []

        def candidates(model, hardware, baseline, psa_config=None, constraints=None):
            calls.append(copy.deepcopy(constraints))
            return OPT._heuristic_candidates(baseline, hardware["device_count"], constraints)

        def benchmark(items, **kwargs):
            if len(calls) == 1:
                return [{"candidate_id": item["candidate_id"], "success": False, "error": "OOM"} for item in items]
            return [{"candidate_id": item["candidate_id"], "success": True, "metrics": {
                "throughput_tokens_per_sec": 130, "step_time_ms": 10, "memory_peak_gb": 40}} for item in items]

        args = ["optimizer", "--model", "fixture", "--devices", "8", "--baseline-throughput", "100",
                "--baseline-config", "TP=2,EP=2,DP=4", "--psa-config", "unused-search.yaml",
                "--config-template", str(template), "--output-dir", str(out)]
        with patch.object(sys, "argv", args), patch.object(OPT, "run_safety_checks", return_value=(True, "test")):
            with patch.object(OPT, "generate_psa_candidates", side_effect=candidates):
                with patch.object(OPT, "launch_batch_validation", side_effect=benchmark):
                    with redirect_stdout(io.StringIO()):
                        self.assertEqual(OPT.main(), 0)
        result = json.loads((out / "optimization_result.json").read_text())
        self.assertEqual(len(result["iterations"]), 2)
        self.assertEqual(calls[1]["activation_checkpoint"], "full")
        self.assertTrue(calls[1]["full_shard"])
        self.assertEqual(result["final_result"]["status"], "SUCCESS")
        self.assertEqual(yaml.safe_load((out / "production_config.yaml").read_text())["training"]["train_iters"], 1000)

    def test_psa_adapter_contract_and_dp_validation(self) -> None:
        """PSA search receives constraints and cannot silently change its chosen DP."""
        config = types.SimpleNamespace(cluster_spec={"num_nodes": 1, "cards_per_node": 8},
                                       constraint={}, search_space={}, estimator={})
        adapter = types.ModuleType("hyper_parallel.auto_parallel.config_adapter")
        adapter.read_search_config = Mock(return_value=config)
        adapter.validate_strict = Mock()
        adapter.search_strategies = Mock(return_value={"tp": 2, "cp": 1, "pp": 1, "ep": 2, "dp": 4})
        with patch.dict(sys.modules, {adapter.__name__: adapter}):
            result = OPT.generate_psa_candidates({}, {"device_count": 8}, {}, Path("search.yaml"), {"max_tp": 2})
            self.assertEqual(result[0]["strategy"]["dp"], 4)
            self.assertTrue(all(value <= 2 for value in config.search_space["tensor_parallel_degree"]))
            adapter.search_strategies.return_value["dp"] = 2
            with self.assertRaisesRegex(ValueError, "incompatible"):
                OPT.generate_psa_candidates({}, {"device_count": 8}, {}, Path("search.yaml"))


class HardwareContractTest(unittest.TestCase):
    """Check A3 topology and logical device counting without loading Torch."""

    def test_a3_node_and_bandwidth_contract(self) -> None:
        """Load the actual device-free model while bypassing package Torch initialization."""
        packages = ("hyper_parallel", "hyper_parallel.auto_parallel",
                    "hyper_parallel.auto_parallel.sapp_nd", "hyper_parallel.auto_parallel.sapp_nd.nd",
                    "hyper_parallel.auto_parallel.sapp_nd.nd.common")
        with patch.dict(sys.modules):
            for name in packages:
                package = types.ModuleType(name)
                package.__path__ = [str(ROOT.joinpath(*name.split(".")))]
                sys.modules[name] = package
            hardware = importlib.import_module("hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware")
            self.assertEqual(hardware.Device_A3.intra_node_num(), 8)
            self.assertEqual(hardware.Device_A3.levels_used(8), 0)
            self.assertEqual(hardware.Device_A3.levels_used(16), 1)
            self.assertEqual(hardware.get_cp_bandwidth("intra-node", "A3"), 196)
            self.assertEqual(hardware.get_cp_bandwidth("cross-node", "A3"), 196)

    def test_runtime_counts_visible_logical_devices(self) -> None:
        """Count logical dies instead of physical board rows or table headers."""
        result = types.SimpleNamespace(returncode=0, stdout="runtime notice\n8\n", stderr="")
        with patch.object(SAFETY.subprocess, "run", return_value=result) as query:
            self.assertTrue(SAFETY.check_device_availability("npu", 8)[0])
            self.assertIn("torch.npu.device_count()", query.call_args.args[0][-1])
            self.assertFalse(SAFETY.check_device_availability("npu", 9)[0])


class SchedulerTest(TemporaryTest):
    """Check actual Trainer log syntax and subprocess lifecycle."""

    def test_timestamp_and_duplicate_records_do_not_count_as_steps(self) -> None:
        """Timestamp minutes and duplicate rank lines cannot satisfy step counts."""
        rows = [f"2026-10-30 12:34:56 INFO 50%|###| {step}/2 [tokens/s=100 step_time=0.2s "
                "memory/device_max_allocated_gb=10]" for step in (1, 1, 2)]
        path = self.write("train.log", "\n".join(rows))
        result = SCHED.extract_metrics_from_log(path, 2)
        self.assertEqual(result["total_steps"], 2)
        self.assertEqual(result["step_time_ms"], 200)
        self.assertIn("error", SCHED.extract_metrics_from_log(path, 3))

    def test_zero_exit_without_metrics_fails_and_command_matches_trainer(self) -> None:
        """The positional config and overrides reach Trainer; empty logs fail."""
        process = Mock(returncode=0)
        with patch.object(SCHED.subprocess, "Popen", return_value=process) as launch, redirect_stdout(io.StringIO()):
            result = SCHED.launch_validation_benchmark(
                str(self.directory / "train.yaml"), 8, str(self.directory / "out"),
                master_port=12345, validation_steps=2)
        self.assertFalse(result["success"])
        args = launch.call_args.args[0]
        self.assertIn("--training.train_iters=2", args)
        self.assertIn("--training.train_samples=null", args)
        self.assertEqual(args[args.index("--module") + 2], str(self.directory / "train.yaml"))
        self.assertEqual(launch.call_args.kwargs["cwd"], ROOT)

    def test_timeout_kills_process_group_and_reaps_launcher(self) -> None:
        """Timeout cleans the process group rather than leaving distributed workers."""
        process = Mock(pid=12345)
        process.wait.side_effect = [subprocess.TimeoutExpired("torchrun", 1), None]
        sigkill = 9
        with patch.object(SCHED.subprocess, "Popen", return_value=process), \
             patch.object(SCHED.os, "killpg", create=True) as kill, \
             patch.object(SCHED.signal, "SIGKILL", sigkill, create=True), redirect_stdout(io.StringIO()):
            result = SCHED.launch_validation_benchmark("train.yaml", 8, str(self.directory),
                                                       master_port=12345, timeout=1)
        self.assertFalse(result["success"])
        kill.assert_called_once_with(12345, sigkill)
        self.assertEqual(process.wait.call_count, 2)


class LinkCheckerTest(TemporaryTest):
    """Supporting docs resolve their own relative destinations."""

    def test_nested_missing_reference_fails(self) -> None:
        """A valid SKILL.md cannot hide a broken nested reference."""
        self.write("SKILL.md", "# Skill\n")
        source = self.write("references/nested/doc.md", "[bad][target]\n\n[target]: missing.md#section\n")
        self.assertEqual(LINKS.broken_links(self.directory), [(source, "missing.md#section")])

    def test_encoded_external_and_fenced_links(self) -> None:
        """URL decoding works and external/example destinations are excluded."""
        self.write("references/space name.md", "# Reference\n")
        self.write("SKILL.md", "[ok](references/space%20name.md#anchor)\n[web](https://example.com)\n"
                   "[mail](mailto:example@example.com)\n[anchor](#local)\n"
                   "```markdown\n[example](missing.md)\n```\n")
        self.assertEqual(LINKS.broken_links(self.directory), [])


if __name__ == "__main__":
    unittest.main()
