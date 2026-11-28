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
"""CPU allocation boundaries, container quotas and pre-Torch launcher behavior."""

from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from rl.utils.cpu_threads import detect_cpu_resources, resolve_cpu_threads


class TestCPUResources(unittest.TestCase):
    """Use synthetic proc/sys trees so the CI host's quotas do not affect assertions."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        self.proc = self.root / "proc"
        self.cpus = self.root / "cpus"
        self.mount = self.root / "cgroup mounts"
        for cpu in range(8):
            self.write(self.cpus / f"cpu{cpu}/topology/thread_siblings_list", str(cpu))
        self.affinity = patch("rl.utils.cpu_threads.os.sched_getaffinity", return_value=set(range(8)))
        self.affinity.start()
        self.addCleanup(self.affinity.stop)

    @staticmethod
    def write(path: Path, value: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value + "\n", encoding="utf-8")

    def mount_controller(self, *, unified: bool, member: str, root: str = "/") -> None:
        controller = "" if unified else "cpu,cpuacct"
        self.write(self.proc / "self/cgroup", f"0:{controller}:{member}")
        filesystem = "cgroup2 cgroup rw" if unified else "cgroup cgroup rw,cpu,cpuacct"
        escaped = str(self.mount).replace(" ", "\\040")
        self.write(self.proc / "self/mountinfo", f"1 0 0:1 {root} {escaped} rw - {filesystem}")

    def detect(self) -> dict:
        return detect_cpu_resources(proc_root=self.proc, cpu_root=self.cpus)

    def test_affinity_and_smt_are_counted_before_dividing(self) -> None:
        for cpu in range(8):
            self.write(self.cpus / f"cpu{cpu}/topology/thread_siblings_list", f"{cpu % 4},{cpu % 4 + 4}")
        with patch("rl.utils.cpu_threads.os.cpu_count", return_value=192), \
                patch("rl.utils.cpu_threads.os.sched_getaffinity", return_value={0, 1, 4, 5}):
            resources = self.detect()
        self.assertEqual(resources["affinity_cpus"], 4)
        self.assertEqual(resources["physical_cores"], 2)
        self.assertEqual(resources["available_cores"], 2)

    def test_v2_parent_quota_constrains_unlimited_child(self) -> None:
        self.mount_controller(unified=True, member="/job/worker")
        self.write(self.mount / "job/worker/cpu.max", "max 100000")
        self.write(self.mount / "job/cpu.max", "350000 100000")
        self.write(self.mount / "cpu.max", "600000 100000")
        resources = self.detect()
        self.assertEqual(resources["quota_cpus"], 3.5)
        self.assertEqual(resources["available_cores"], 3)

    def test_v1_parent_quota_constrains_unlimited_child(self) -> None:
        self.mount_controller(unified=False, member="/job/worker")
        for relative, quota in (("job/worker", "-1"), ("job", "250000"), ("", "400000")):
            self.write(self.mount / relative / "cpu.cfs_quota_us", quota)
            self.write(self.mount / relative / "cpu.cfs_period_us", "100000")
        resources = self.detect()
        self.assertEqual(resources["quota_cpus"], 2.5)
        self.assertEqual(resources["available_cores"], 2)

class TestCPUThreadBudget(unittest.TestCase):
    """Manual overrides and role isolation remain explicit and independently testable."""

    def test_shared_launcher_budget_is_divided_by_local_ranks(self) -> None:
        """Divide the shared CPU quota by local ranks and apply the thread cap."""
        for cores, ranks, threads in ((192, 8, 8), (64, 8, 4), (31, 4, 3), (8, 1, 4)):
            with self.subTest(cores=cores, ranks=ranks):
                plan = resolve_cpu_threads({"available_cores": cores}, ranks,
                                           cpu_offload=True, environment={})
                self.assertEqual(plan["reserved_cores"], (cores + 1) // 2)
                self.assertEqual(plan["threads_per_rank"], threads)
                self.assertEqual(plan["training_threads"], threads * ranks)
        single = resolve_cpu_threads({"available_cores": 8}, 1, cpu_offload=True,
                                     environment={"HYPER_RL_CPU_RESERVE": "0"})
        self.assertEqual(single["threads_per_rank"], 4)

    def test_device_training_and_manual_limits(self) -> None:
        """Respect manual thread limits and choose role-appropriate defaults."""
        cases = [({}, False, 1, "auto"), ({"OMP_NUM_THREADS": "3"}, False, 3, "OMP_NUM_THREADS"),
                 ({"OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "2"}, True, 2, "MKL_NUM_THREADS"),
                 ({"OMP_NUM_THREADS": "16"}, True, 16, "OMP_NUM_THREADS"),
                 ({"OMP_NUM_THREADS": "auto", "MKL_NUM_THREADS": "auto"}, True, 4, "auto"),
                 ({"HYPER_RL_CPU_RESERVE": "48"}, True, 2, "auto")]
        for environment, offload, threads, source in cases:
            with self.subTest(environment=environment):
                plan = resolve_cpu_threads({"available_cores": 64}, 8, cpu_offload=offload,
                                           environment=environment)
                self.assertEqual(plan["threads_per_rank"], threads)
                self.assertEqual(plan["source"], source)

    def test_tiny_budget_keeps_one_thread_per_process(self) -> None:
        with self.assertLogs("rl.utils.cpu_threads", level="WARNING"):
            plan = resolve_cpu_threads({"available_cores": 1}, 8, cpu_offload=True, environment={})
        self.assertEqual(plan["threads_per_rank"], 1)
