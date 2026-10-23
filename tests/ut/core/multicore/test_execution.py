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
"""Host-only execution inspection, sampling lifecycle and complete-rank coverage."""

import json
from contextlib import redirect_stdout
import gc
import io
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import torch

from hyper_parallel.core.multicore.diagnostics import (
    execution_monitor, inspect_execution, schedule, summarize_execution_reports,
)
from hyper_parallel.core.multicore.diagnostics.common.call import LaunchContext
from hyper_parallel.core.multicore.diagnostics.__main__ import main as summarize_main
from hyper_parallel.core.multicore.diagnostics.common.session import (
    _activate_session, _deactivate_session, _get_active_session, _get_execution_observer,
)
from hyper_parallel.core.multicore.modules.mega_moe import module as moe_module
from hyper_parallel.core.multicore.modules.mega_moe.module import MegaMoeExperts
from hyper_parallel.core.multicore.modules.mega_moe import route as moe_route
from hyper_parallel.core.multicore.modules.mega_moe.route import _dynamic_summary
from hyper_parallel.core.multicore.modules.mega_moe.spec import MegaMoeSpec
from tests.common.mark_utils import arg_mark


class TestExecutionReports(unittest.TestCase):
    """Exercise public sessions and the actual Host producer paths without a device."""

    # pylint: disable=protected-access

    def setUp(self) -> None:
        """Create independent models and context-managed output storage."""
        self.output = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.output)
        self.model = torch.nn.ModuleList([self._experts(), self._experts()])
        self.addCleanup(lambda: [module.close() for module in self.model])

    @staticmethod
    def _experts() -> MegaMoeExperts:
        return MegaMoeExperts(hidden_size=16, intermediate_size=8, num_experts=4, top_k=2)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_inspection_is_lazy_and_sharing_is_visible_from_nonfirst_member(self) -> None:
        """Feature: Host execution diagnostics.

        Description: Inspection exposes unbound groups and does not bind or inspect tensor contents.
        Expectation: Unbound resources stay unbound; both members report the same shared group.
        """
        with patch.object(moe_module, "build_mega_moe_plan", side_effect=AssertionError("unexpected plan")):
            before = inspect_execution(self.model)
            MegaMoeExperts.share_execution_resources(self.model)
            after = inspect_execution(self.model)
        self.assertEqual(before["resource_group_count"], 2)
        self.assertEqual(len(before["findings"]), 1)
        self.assertEqual(after["resource_group_count"], 1)
        self.assertFalse(after["findings"])
        second = after["execution_modules"]["1"]
        self.assertTrue(second["resource_shared"])
        self.assertEqual(second["resource_members"], 2)
        self.assertFalse(second["resource_bound"])
        self.assertIsNone(second["token_capacity"])
        json.dumps(after)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_meta_and_optimizer_options_are_reported_without_materialization(self) -> None:
        """Feature: Host execution diagnostics.

        Description: Report optimizer options independently of the MoE backend and preserve meta state.
        Expectation: Meta weights remain meta and resolved optimizer flags are retained.
        """
        model = torch.nn.Linear(4, 3, device="meta")
        optimizer = SimpleNamespace(param_groups=[{"fused": True, "foreach": False, "lr": torch.tensor(0.1)}])
        result = inspect_execution(model, optimizer=optimizer)
        self.assertEqual(result["parameters"]["meta"], ["weight", "bias"])
        self.assertEqual(result["parameters"]["layouts"][0]["shape"], [3, 4])
        self.assertTrue(result["optimizer"]["param_groups"][0]["fused"])
        self.assertEqual(result["optimizer"]["param_groups"][0]["lr"], {"type": "torch.Tensor"})
        self.assertTrue(model.weight.is_meta)
        self.assertEqual(result["resource_group_count"], 0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_existing_schedule_selects_complete_steps_and_bounds_records(self) -> None:
        """Feature: Host execution diagnostics.

        Description: Only explicit completed active steps are written; excess events are visible.
        Expectation: Only the selected completed step is saved and event truncation is explicit.
        """
        monitor = execution_monitor(self.model, output_dir=self.output, max_events_per_step=1,
                                    schedule=schedule(wait=1, warmup=1, active=1, repeat=1))
        self.assertFalse(list(self.output.iterdir()))
        with monitor:
            for step in range(4):
                observer = _get_execution_observer()
                if observer is not None:
                    observer.record_event("test", {"value": step})
                    observer.record_event("test", {"value": 99})
                monitor.step()
        self.assertIsNone(_get_active_session())
        files = list(monitor.directory.glob("step_*.json"))
        self.assertEqual([path.name for path in files], ["step_000002.json"])
        row = json.loads(files[0].read_text())
        self.assertEqual(row["events"], [{"kind": "test", "value": 2}])
        self.assertEqual(row["dropped_events"], 1)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_exception_does_not_publish_pending_step_or_leak_session(self) -> None:
        """Feature: Host execution diagnostics.

        Description: An interrupted training body cannot look like a complete sample.
        Expectation: Pending data is not accepted and a subsequent session can enter.
        """
        monitor = execution_monitor(self.model, output_dir=self.output)
        with self.assertRaisesRegex(RuntimeError, "training failed"):
            with monitor:
                monitor.record_event("pending", {})
                raise RuntimeError("training failed")
        self.assertIsNone(_get_active_session())
        self.assertFalse(list(monitor.directory.glob("step_*.json")))
        self.assertEqual(json.loads((monitor.directory / "session.json").read_text())["status"], "failed")
        with execution_monitor(self.model, output_dir=self.output) as retry:
            retry.step()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_entry_failure_and_nested_tools_preserve_session_ownership(self) -> None:
        """Feature: Host execution diagnostics.

        Description: Invalid schedules and nested diagnostics release only their own ownership.
        Expectation: Failed entry releases itself and nested entry preserves the outer owner.
        """
        with self.assertRaisesRegex(TypeError, "DiagnosticAction"):
            with execution_monitor(self.model, output_dir=self.output, schedule=lambda _: None):
                self.fail("invalid session entered")
        self.assertIsNone(_get_active_session())
        owner = SimpleNamespace()
        _activate_session("profiler", owner)
        try:
            with self.assertRaisesRegex(RuntimeError, "nested"):
                with execution_monitor(self.model, output_dir=self.output):
                    self.fail("nested session entered")
            self.assertIs(_get_active_session(), owner)
        finally:
            _deactivate_session(owner)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_monitor_leaves_kernel_arguments_unchanged(self) -> None:
        """Feature: Host execution diagnostics.

        Description: Host observations never request instrumented graphs or device buffers.
        Expectation: The normal runtime and fallback arguments are reused by identity.
        """
        normal = object()
        context = LaunchContext(SimpleNamespace(normal_tensor=normal), (), "forward", None)
        with execution_monitor(self.model, output_dir=self.output) as monitor:
            call = monitor.prepare_call(context)
            self.assertIs(call.runtime_config, normal)
            self.assertIs(call.diagnostic_buffer, normal)
            self.assertIsNone(call.event_counters)
            monitor.step()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_closed_monitor_does_not_keep_a_replaced_model_alive(self) -> None:
        """Feature: Execution observation lifecycle.

        Description: Retain a finished monitor after dropping its original model.
        Expectation: Diagnostic metadata does not prevent model collection.
        """
        model = torch.nn.Linear(4, 3)
        reference = weakref.ref(model)
        monitor = execution_monitor(model, output_dir=self.output)
        with monitor:
            monitor.step()
        del model
        gc.collect()
        self.assertIsNone(reference())

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_plan_and_route_producers_report_hits_and_owner_skew(self) -> None:
        """Feature: Host execution diagnostics.

        Description: Real Host summary and cache paths reuse existing metadata, including empty owners.
        Expectation: One build then one hit and the exact two-source owner counts are observed.
        """
        resources = moe_module._MegaMoeExecutionResources.__new__(moe_module._MegaMoeExecutionResources)
        resources.spec = MegaMoeSpec(256, 16, 8, 4, 2, None, 512, 2, None, 0, 24,
                                    max_local_num_tokens=256)
        resources._token_capacity = 256
        resources._plans = {}
        payload = torch.tensor([[0, 2, 31, 2, 2, 0, 0], [0, 1, 31, 1, 1, 0, 0]])
        with execution_monitor(self.model, output_dir=self.output) as monitor:
            with patch.object(moe_module, "build_mega_moe_plan",
                              side_effect=lambda spec, device: SimpleNamespace(spec=spec)) as build:
                first = resources.plan_for_tokens(128, "cpu")
                self.assertIs(first, resources.plan_for_tokens(128, "cpu"))
                self.assertEqual(build.call_count, 1)
            self.assertEqual(_dynamic_summary(payload, resources.spec), (6, 128, 6))
            self.assertEqual(_dynamic_summary(torch.zeros_like(payload), resources.spec), (1, 128, 0))
            monitor.step()
        row = json.loads((monitor.directory / "step_000000.json").read_text())
        self.assertEqual([event["cache_hit"] for event in row["events"] if event["kind"] == "plan"], [False, True])
        route = next(event for event in row["events"] if event["kind"] == "route")
        self.assertEqual(route["source_tokens"], [2, 1])
        self.assertEqual(route["received_rows"], [6, 0])
        self.assertEqual(route["maximum_over_mean"], 2.0)
        empty = [event for event in row["events"] if event["kind"] == "route"][-1]
        self.assertEqual(empty["source_tokens"], [0, 0])
        self.assertEqual(empty["maximum_over_mean"], 0.0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_small_and_large_route_tables_emit_the_same_observation(self) -> None:
        """Feature: Host execution diagnostics.

        Description: Exercise both metadata policies through their common route readback boundary.
        Expectation: The same existing metadata yields one identical route event for each policy.
        """
        spec = MegaMoeSpec(256, 16, 8, 4, 2, None, 512, 2, None, 0, 24, auto_tokens=True)
        payload = torch.tensor([[0, 2, 31, 2, 2, 0, 0], [0, 1, 31, 1, 1, 0, 0]])
        with execution_monitor(self.model, output_dir=self.output) as monitor:
            for threshold in (1024, 0):
                with patch.object(moe_route, "_HOST_ROUTE_MAX_COUNTS", threshold):
                    summary, _ = moe_route._read_dynamic_route(payload, spec)
                    self.assertEqual(summary, (6, 128, 6))
            monitor.step()
        row = json.loads((monitor.directory / "step_000000.json").read_text())
        events = [event for event in row["events"] if event["kind"] == "route"]
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0], events[1])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_inactive_plan_cache_does_not_start_a_diagnostic_timer(self) -> None:
        """Feature: Host execution diagnostics.

        Description: Default execution does not add clocks, copies or observers.
        Expectation: Ordinary cache access does not invoke the observation timer.
        """
        resources = moe_module._MegaMoeExecutionResources.__new__(moe_module._MegaMoeExecutionResources)
        resources.spec = SimpleNamespace(plan_tokens=128)
        resources._token_capacity = 128
        resources._plans = {128: SimpleNamespace(spec=resources.spec)}
        with patch.object(moe_module.time, "perf_counter", side_effect=AssertionError("unexpected timer")):
            resources.plan_for_tokens(128, "cpu")
        self.assertIsNone(_get_execution_observer())

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_from_module_rejects_meta_before_creating_an_executor(self) -> None:
        """Feature: Host execution diagnostics.

        Description: Parameter conversion requires actual initialized values, not just shapes.
        Expectation: A clear materialization error is raised before copying meta weights.
        """
        source = torch.nn.Module()
        source.w1 = torch.nn.Parameter(torch.empty(4, 8, 16, device="meta"))
        source.w3 = torch.nn.Parameter(torch.empty(4, 8, 16, device="meta"))
        source.w2 = torch.nn.Parameter(torch.empty(4, 16, 8, device="meta"))
        with self.assertRaisesRegex(ValueError, "materialize and initialize"):
            MegaMoeExperts.from_module(source, top_k=2)


class TestReportCoverage(unittest.TestCase):
    """Ensure missing future steps and repeated processes cannot be accepted."""

    def setUp(self) -> None:
        """Isolate the rank-report files for each coverage scenario."""
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)

    def _write_rank(self, rank: int, steps: list[int], suffix: str = "", dropped: int = 0) -> None:
        directory = self.root / f"rank_{rank:04d}{suffix}"
        directory.mkdir()
        (directory / "identity.json").write_text(json.dumps({"schema_version": 1, "rank": rank, "world_size": 2}))
        for step in steps:
            (directory / f"step_{step:06d}.json").write_text(json.dumps({
                "schema_version": 1, "step": step, "events": [], "dropped_events": dropped,
            }))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_every_expected_rank_and_step_is_required(self) -> None:
        """Feature: Host execution diagnostics.

        Description: Coverage of present steps alone does not prove the requested capture completed.
        Expectation: Missing future steps and missing ranks are reported as incomplete.
        """
        self._write_rank(0, [3, 14])
        self._write_rank(1, [3])
        summary = summarize_execution_reports(self.root, world_size=2, expected_steps=[3, 14, 45])
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["steps"]["45"], [])
        self.assertEqual(summary["steps"]["14"], [0])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_complete_and_duplicate_runs_are_distinguished(self) -> None:
        """Feature: Host execution diagnostics.

        Description: A second process for an already collected rank invalidates the run.
        Expectation: Complete coverage passes and repeated rank sessions invalidate it.
        """
        self._write_rank(0, [3])
        self._write_rank(1, [3])
        self.assertTrue(summarize_execution_reports(self.root, world_size=2, expected_steps=[3])["complete"])
        self._write_rank(1, [3], suffix="_retry")
        summary = summarize_execution_reports(self.root, world_size=2, expected_steps=[3])
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["duplicate_ranks"], [1])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_offline_command_writes_summary_and_signals_missing_steps(self) -> None:
        """Feature: Offline execution reports.

        Description: Invoke the CLI on the same collected files with complete and missing steps.
        Expectation: A summary is written and incomplete coverage returns a nonzero exit status.
        """
        self._write_rank(0, [3])
        self._write_rank(1, [3])
        with redirect_stdout(io.StringIO()):
            arguments = ["summarize", str(self.root), "--world-size", "2", "--steps", "3"]
            self.assertEqual(summarize_main(arguments), 0)
            arguments[-1] = "3,45"
            self.assertEqual(summarize_main(arguments), 2)
        self.assertFalse(json.loads((self.root / "summary.json").read_text())["complete"])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_dropped_records_are_not_accepted_as_complete_diagnostics(self) -> None:
        """Feature: Host execution diagnostics.

        Description: A bounded session reports that its event sample was truncated.
        Expectation: Truncated events keep the coverage report incomplete.
        """
        self._write_rank(0, [3], dropped=1)
        self._write_rank(1, [3])
        summary = summarize_execution_reports(self.root, world_size=2, expected_steps=[3])
        self.assertFalse(summary["complete"])
        self.assertEqual(len(summary["truncated_reports"]), 1)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_unknown_schema_is_rejected_instead_of_becoming_zero_event_counts(self) -> None:
        """Feature: Execution report format validation.

        Description: Mix an unrelated old diagnostic file with the current report format.
        Expectation: The foreign schema is reported as invalid rather than accepted with zero counters.
        """
        self._write_rank(0, [3])
        self._write_rank(1, [3])
        foreign = self.root / "rank_0001/step_000003.json"
        foreign.write_text(json.dumps({"step": 3, "records": []}))
        summary = summarize_execution_reports(self.root, world_size=2, expected_steps=[3])
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["invalid_reports"], [str(foreign)])
