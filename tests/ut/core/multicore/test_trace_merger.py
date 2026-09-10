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
"""Unit tests for offline framework and MegaKernel Chrome Trace merging."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from hyper_parallel.core.multicore import profiler as multicore_profiler
from hyper_parallel.core.multicore.trace_merger import merge_chrome_traces


def _outer_event(
        name,
        timestamp,
        duration,
        *,
        pid=7,
        tid=3,
        device=True,
):
    return {
        "name": name,
        "cat": "kernel" if device else "cpu_op",
        "ph": "X",
        "pid": pid,
        "tid": tid,
        "ts": timestamp,
        "dur": duration,
        "args": {},
    }


def _internal_event(
        name,
        timestamp,
        duration,
        *,
        invocation_id=None,
        core_type="AIC",
        block_id=0,
):
    args = {
        "core_type": core_type,
        "block_id": block_id,
        "task_id": 1,
        "desc_id": 0x10002,
    }
    if invocation_id is not None:
        args["invocation_id"] = invocation_id
    return {
        "name": name,
        "cat": "MegaKernelInternal",
        "ph": "X",
        "pid": 0,
        "tid": 1000 + block_id,
        "ts": timestamp,
        "dur": duration,
        "args": args,
    }


def _window_trace():
    return {
        "traceEvents": [
            _internal_event("GMM1", 10.0, 5.0, invocation_id=0),
            _internal_event(
                "SwiGLU",
                20.0,
                10.0,
                invocation_id=0,
                core_type="AIV",
                block_id=1,
            ),
            _internal_event("GMM2Grad", 100.0, 4.0, invocation_id=1),
        ],
        "megaKernelCycleTrace": {
            "schemaVersion": 1,
            "windowIndex": 0,
            "invocations": [
                {
                    "invocationId": 0,
                    "direction": "forward",
                    "kernelName": "HyperMegaMoe",
                    "rank": 0,
                    "deviceId": 6,
                },
                {
                    "invocationId": 1,
                    "direction": "backward",
                    "kernelName": "HyperMegaMoeGrad",
                    "rank": 0,
                    "deviceId": 6,
                },
            ],
        },
    }


class TestMergeChromeTraces(unittest.TestCase):
    """Validate public offline trace merging without framework imports."""

    def test_public_api_merges_multi_invocation_window_and_writes_output(self):
        """Map each invocation to the strongest matching outer Device event."""
        framework_trace = {
            "displayTimeUnit": "ns",
            "traceEvents": [
                {
                    "name": "thread_sort_index",
                    "ph": "M",
                    "pid": 7,
                    "tid": 3,
                    "args": {"sort_index": 10},
                },
                _outer_event("HyperMegaMoe", 900.0, 200.0, device=False),
                _outer_event("HyperMegaMoe_abc123", 1000.0, 50.0),
                _outer_event("HyperMegaMoeGrad_def456", 2000.0, 20.0),
            ],
        }
        standalone_trace = _window_trace()
        original_framework_trace = copy.deepcopy(framework_trace)
        original_standalone_trace = copy.deepcopy(standalone_trace)

        with tempfile.TemporaryDirectory() as output_dir:
            output_path = Path(output_dir) / "nested" / "merged.json"
            merged = multicore_profiler.merge_chrome_traces(
                framework_trace,
                standalone_trace,
                output_path,
            )
            persisted = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertIs(multicore_profiler.merge_chrome_traces, merge_chrome_traces)
        self.assertEqual(merged, persisted)
        self.assertEqual(framework_trace, original_framework_trace)
        self.assertEqual(standalone_trace, original_standalone_trace)
        self.assertEqual(merged["displayTimeUnit"], "ns")
        report = merged["megaKernelTraceMerge"]
        self.assertEqual(report["mergedInvocationCount"], 2)
        self.assertEqual(report["mergedEventCount"], 3)
        self.assertFalse(report["usedFallbackKernelSelection"])
        self.assertEqual(
            [alignment["outerKernelName"] for alignment in report["alignments"]],
            ["HyperMegaMoe_abc123", "HyperMegaMoeGrad_def456"],
        )

        internal_events = [
            event
            for event in merged["traceEvents"]
            if event.get("cat") == "MegaKernelInternal"
        ]
        self.assertEqual([event["ts"] for event in internal_events], [1000.0, 1010.0, 2000.0])
        self.assertEqual(
            [event["args"]["parent_kernel"] for event in internal_events],
            ["HyperMegaMoe_abc123", "HyperMegaMoe_abc123", "HyperMegaMoeGrad_def456"],
        )
        self.assertEqual({event["pid"] for event in internal_events}, {7})
        self.assertEqual(internal_events[0]["args"]["cycle_trace_device_id"], 6)

    def test_event_list_and_single_invocation_trace_are_supported(self):
        """Normalize event-list framework exports and legacy single captures."""
        framework_events = [
            _outer_event(
                "aclnnMegaMoe_MegaMoe_MegaMoe",
                "1234.5",
                "40.0",
                pid=99,
            )
        ]
        standalone_trace = {
            "traceEvents": [_internal_event("GMM1", 3.0, 8.0)],
            "megaKernelCycleTrace": {
                "schemaVersion": 1,
                "rank": 1,
                "deviceId": 7,
                "kernelName": "MegaMoe",
            },
        }

        with tempfile.TemporaryDirectory() as output_dir:
            merged = merge_chrome_traces(
                framework_events,
                standalone_trace,
                Path(output_dir) / "merged.json",
            )

        event = next(
            item for item in merged["traceEvents"]
            if item.get("cat") == "MegaKernelInternal"
        )
        self.assertEqual(event["ts"], 1234.5)
        self.assertEqual(event["pid"], 99)
        self.assertEqual(event["args"]["cycle_trace_rank"], 1)
        self.assertEqual(merged["megaKernelTraceMerge"]["mergedInvocationCount"], 1)

    def test_cann_mix_aic_task_is_recognized_as_a_device_kernel(self):
        """Recognize the category-free Task Type emitted by CANN 9.1 msprof."""
        framework_trace = [{
            "name": "aclnnHyperMegaMoe_HyperMegaMoe_HyperMegaMoe",
            "cat": None,
            "ph": "X",
            "pid": 7,
            "tid": 43,
            "ts": "1788773332531494.453",
            "dur": 9529.571,
            "args": {
                "Task Type": "MIX_AIC",
                "Physic Stream Id": 43,
                "Task Id": 56,
            },
        }]
        standalone_trace = {
            "traceEvents": [_internal_event("Compute", 0.0, 20.0)],
            "megaKernelCycleTrace": {
                "schemaVersion": 1,
                "kernelName": "MegaMoe",
            },
        }

        with tempfile.TemporaryDirectory() as output_dir:
            merged = merge_chrome_traces(
                framework_trace,
                standalone_trace,
                Path(output_dir) / "merged.json",
            )

        alignment = merged["megaKernelTraceMerge"]["alignments"][0]
        self.assertTrue(alignment["outerKernelIsDevice"])
        self.assertEqual(alignment["outerKernelTid"], 43)

    def test_inferred_name_fallback_records_approximate_alignment(self):
        """Select a similar Device kernel and surface an explicit warning."""
        framework_trace = {
            "traceEvents": [
                _outer_event("unrelated_first", 100.0, 25.0),
                _outer_event("mangled_device_symbol", 200.0, 25.0),
            ]
        }
        standalone_trace = {
            "traceEvents": [_internal_event("Compute", 10.0, 20.0)],
            "megaKernelCycleTrace": {
                "schemaVersion": 1,
                "kernelName": "NameMissingFromFramework",
            },
        }

        with tempfile.TemporaryDirectory() as output_dir:
            merged = merge_chrome_traces(
                framework_trace,
                standalone_trace,
                Path(output_dir) / "merged.json",
                kernel_index=1,
            )

        report = merged["megaKernelTraceMerge"]
        self.assertTrue(report["usedFallbackKernelSelection"])
        self.assertEqual(report["alignments"][0]["outerKernelName"], "mangled_device_symbol")
        self.assertTrue(report["warnings"])

    def test_host_wrapper_alignment_requires_explicit_opt_in(self):
        """Reject misleading Host alignment unless diagnostics opt into it."""
        framework_trace = {
            "traceEvents": [
                _outer_event("HyperMegaMoe", 100.0, 50.0, device=False),
            ]
        }
        standalone_trace = {
            "traceEvents": [_internal_event("Compute", 10.0, 20.0)],
            "megaKernelCycleTrace": {
                "schemaVersion": 1,
                "kernelName": "HyperMegaMoe",
            },
        }

        with tempfile.TemporaryDirectory() as output_dir:
            output_path = Path(output_dir) / "merged.json"
            with self.assertRaisesRegex(ValueError, "host-only"):
                merge_chrome_traces(framework_trace, standalone_trace, output_path)
            merged = merge_chrome_traces(
                framework_trace,
                standalone_trace,
                output_path,
                allow_host_wrapper=True,
            )

        report = merged["megaKernelTraceMerge"]
        self.assertFalse(report["alignments"][0]["outerKernelIsDevice"])
        self.assertTrue(report["warnings"])

    def test_invalid_schema_regex_index_and_count_fail_explicitly(self):
        """Reject malformed schema and ambiguous outer-event selections."""
        framework_trace = {
            "traceEvents": [_outer_event("HyperMegaMoe", 100.0, 50.0)]
        }
        standalone_trace = {
            "traceEvents": [_internal_event("Compute", 10.0, 20.0)],
            "megaKernelCycleTrace": {
                "schemaVersion": 1,
                "kernelName": "HyperMegaMoe",
            },
        }

        with tempfile.TemporaryDirectory() as output_dir:
            output_path = Path(output_dir) / "merged.json"
            invalid_schema = copy.deepcopy(standalone_trace)
            invalid_schema["megaKernelCycleTrace"]["schemaVersion"] = 2
            with self.assertRaisesRegex(ValueError, "schemaVersion"):
                merge_chrome_traces(framework_trace, invalid_schema, output_path)
            with self.assertRaisesRegex(ValueError, "Invalid kernel regex"):
                merge_chrome_traces(
                    framework_trace,
                    standalone_trace,
                    output_path,
                    kernel_pattern="[",
                )
            with self.assertRaisesRegex(ValueError, "must not be empty"):
                merge_chrome_traces(
                    framework_trace,
                    standalone_trace,
                    output_path,
                    kernel_pattern="",
                )
            with self.assertRaisesRegex(ValueError, "kernel_index"):
                merge_chrome_traces(
                    framework_trace,
                    standalone_trace,
                    output_path,
                    kernel_index=-1,
                )
            invalid_number = copy.deepcopy(standalone_trace)
            invalid_number["traceEvents"][0]["ts"] = float("nan")
            with self.assertRaisesRegex(ValueError, "must be finite"):
                merge_chrome_traces(framework_trace, invalid_number, output_path)
            with self.assertRaisesRegex(ValueError, "Not enough outer"):
                merge_chrome_traces(framework_trace, _window_trace(), output_path)


if __name__ == "__main__":
    unittest.main()
