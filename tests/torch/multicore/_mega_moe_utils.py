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
"""Device evidence shared by the MegaMoe acceptance workers."""

import gc
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu

from hyper_parallel.core.multicore._loader import get_multicore_paths
from hyper_parallel.core.multicore.shmem._bindings import _require_library
from hyper_parallel.core.multicore.shmem import lifecycle
from tests.common.port_utils import allocate_port


def start_shmem_lifetime() -> None:
    """Isolate bootstrap state between independent acceptance scenarios."""
    # Inspect the internal owner registry to catch leaks from a previous scenario.
    assert not lifecycle._PROCESS_STATE.initialized, (  # pylint: disable=protected-access
        "Previous scenario retained an active SHMEM owner."
    )
    dist.barrier()
    endpoint = [f"tcp://127.0.0.1:{allocate_port()}" if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(endpoint, src=0)
    os.environ["SHMEM_IP_PORT"] = endpoint[0]


def memory_sample() -> dict[str, int]:
    """Measure allocator usage separately from the external symmetric heap."""
    torch.npu.synchronize()
    free_bytes, total_bytes = torch.npu.mem_get_info()
    return {
        "allocated_bytes": torch.npu.memory_allocated(),
        "reserved_bytes": torch.npu.memory_reserved(),
        "device_used_bytes": total_bytes - free_bytes,
        # The external SHMEM heap is absent from Torch allocator statistics.
        "shmem_heap_bytes": lifecycle._PROCESS_STATE.heap_size or 0,  # pylint: disable=protected-access
    }


def collect_memory() -> dict[str, int]:
    """Collect released graphs before sampling stable allocation usage."""
    gc.collect()
    return memory_sample()


def assert_stable_memory(samples: list[dict[str, int]]) -> None:
    """Reject retained allocation growth after fixed-shape warmup."""
    allocated = [sample["allocated_bytes"] for sample in samples]
    allowance = 1024 * 1024
    assert max(allocated) - min(allocated) <= allowance, (
        f"rank={dist.get_rank()}: live memory failed to stabilize: "
        f"samples={allocated}, allowed_variation={allowance}."
    )


def environment_identity() -> dict:
    """Record the exact software, chip and loaded vendor used by a device run."""
    root = Path(os.getenv("HP_VALIDATION_REPO", str(Path(__file__).resolve().parents[3])))
    vendor, adapter = get_multicore_paths()
    cann_root = Path(os.environ["ASCEND_HOME_PATH"])
    version_files = [cann_root / "version.cfg", cann_root / "version.info", cann_root / "opp/version.info"]
    versions = {
        str(path): path.read_text(encoding="utf-8")
        for path in version_files if path.is_file()
    }
    native_files = [adapter, *vendor.rglob("*.so"), *vendor.rglob("*.o")]
    shmem_root = _require_library().parents[2]
    native_files.extend(shmem_root.rglob("*.so"))
    return {
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "worker_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(__file__).parent.glob("*mega_moe*.py")
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "chip": torch.npu.get_device_name(),
        "device_limits": torch.npu.get_device_limit(torch.npu.current_device()),
        "cann_root": str(cann_root),
        "cann_version_files": versions,
        "native_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in native_files
        },
    }


def write_evidence(result: dict) -> None:
    """Persist every rank's acceptance data and the software identity."""
    rank_results = [None] * dist.get_world_size()
    dist.all_gather_object(rank_results, result)
    if dist.get_rank() == 0:
        path = Path(os.environ["HP_MEGA_MOE_LEVEL1_RESULT"])
        path.parent.mkdir(parents=True, exist_ok=True)
        evidence = {"environment": environment_identity(), "ranks": rank_results}
        path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        print(f"MegaMoe Level1 evidence: {path}", flush=True)
