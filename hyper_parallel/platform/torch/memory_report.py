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
"""CSV serialization for FakeTensor memory reports."""

import csv
import os
import tempfile
from pathlib import Path
from typing import Any, Dict


_FAKE_MEMORY_POOL_TYPE = "FakeTensorLogicalMemoryPool"
_MEMORY_VISUALIZER_POOL_TYPE = "DefaultEnhancedAscendMemoryPool"
_CUDA_MEMORY_POOL_TYPE = "CUDALogicalMemoryPool"
_CSV_FIELDS = (
    "start_time_stamp",
    "end_time_stamp",
    "start_plot_time_stamp",
    "end_plot_time_stamp",
    "device_addr",
    "stream_id",
    "pool_type",
    "size",
    "actual_used_memory",
    "actual_peak_memory",
    "file_name",
    "line_num",
    "type",
    "producer_task",
    "task_name",
    "node_name",
    "graph_name",
    "user_tasks",
    "last_user_task",
    "python_stack",
    "is_persistent",
    "is_small",
)


def build_memory_csv_rows(report: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Build visualization-compatible logical memory-block CSV rows.

    Args:
        report: Successful Dry-run memory report.

    Returns:
        One row per completed allocation lifetime.

    Raises:
        ValueError: If the report does not describe a successful run.
    """
    if report.get("status") != "ok":
        raise ValueError("CSV output requires a successful memory report")
    rows = []
    for block in report.get("memory_blocks", []):
        user_tasks = list(block.get("user_tasks", []))
        row = {field: block.get(field, "") for field in _CSV_FIELDS}
        if row["pool_type"] == _FAKE_MEMORY_POOL_TYPE:
            target_device = report.get("metadata", {}).get("target_device", "npu")
            row["pool_type"] = (
                _MEMORY_VISUALIZER_POOL_TYPE
                if target_device == "npu"
                else _CUDA_MEMORY_POOL_TYPE
            )
            user_tasks = []
        row["user_tasks"] = "{" + "-".join(str(task) for task in user_tasks) + "}"
        row["last_user_task"] = user_tasks[-1] if user_tasks else ""
        rows.append(row)
    return rows


def write_memory_csv(report: Dict[str, Any], output_path: str) -> str:
    """Atomically write one memory report as CSV.

    Args:
        report: Successful Dry-run memory report.
        output_path: Destination CSV path.

    Returns:
        Absolute destination path.
    """
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = build_memory_csv_rows(report)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as report_file:
            writer = csv.DictWriter(report_file, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, destination)
    except Exception:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise
    return str(destination)


__all__ = ["build_memory_csv_rows", "write_memory_csv"]
