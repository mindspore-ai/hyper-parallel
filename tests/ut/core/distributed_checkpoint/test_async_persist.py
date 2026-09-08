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
"""UT for :mod:`hyper_parallel.core.distributed_checkpoint.async_persist`."""
# pylint: disable=wrong-import-position
import importlib
import os
import queue
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

import torch

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
import hyper_parallel.platform.platform as _platform_mod

_platform_mod.platform = None

import hyper_parallel.core.distributed_checkpoint.async_persist as staging_mod

importlib.reload(staging_mod)

# Only ``build_staged_state_dict`` is imported by name: setUp reloads the module under test,
# which rebinds its enums, so everything compared against them is reached through the module.
from hyper_parallel.core.distributed_checkpoint.async_persist import build_staged_state_dict
from hyper_parallel.core.distributed_checkpoint.metadata import Metadata
from hyper_parallel.core.distributed_checkpoint.planner import SavePlan
from hyper_parallel.core.dtensor.device_mesh import _DEVICE_MESH_MAP
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.layout import Layout
from hyper_parallel.core.dtensor.placement_types import RaggedShard
from hyper_parallel.platform.platform import EXISTING_COMM_GROUPS


class TestAsyncStaging(unittest.TestCase):
    """Tests for async checkpoint staging helpers."""

    def setUp(self) -> None:
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        _platform_mod.platform = None
        importlib.reload(staging_mod)

    def test_build_staged_state_dict_copies_tensors_to_cpu(self):
        """
        Feature: build_staged_state_dict host staging.
        Description: Stage a nested state dict with a GPU-resident tensor (if available).
        Expectation: Staged copy is on CPU, equal in value, and independent object.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        weight = torch.nn.Parameter(torch.ones(2, 3, device=device) * 3.0)
        original = {"model": {"weight": weight}}
        staged = build_staged_state_dict(original)
        staged_weight = staged["model"]["weight"]
        self.assertIsNot(staged_weight, weight)
        self.assertFalse(staged_weight.is_cuda)
        torch.testing.assert_close(staged_weight, weight.cpu())

    def test_copy_tensor_to_cpu_uses_platform_detach(self):
        """
        Feature: Cross-backend tensor staging.
        Description: Stage a tensor whose detached value only exposes the shared ``to`` API.
        Expectation: The platform detach hook is used and the CPU transfer requests an independent copy.
        """
        source = object()
        staged = object()
        to_calls = []

        class _DetachedTensor:
            def to(self, *args, **kwargs):
                """Record the backend-neutral CPU copy request."""
                to_calls.append((args, kwargs))
                return staged

        detached = _DetachedTensor()
        with patch.object(staging_mod.platform, "detach", return_value=detached) as detach:
            result = staging_mod._copy_tensor_to_cpu(source)

        detach.assert_called_once_with(source)
        self.assertEqual(to_calls, [(('cpu',), {'copy': True})])
        self.assertIs(result, staged)

    def test_build_staged_state_dict_deep_copies_bytes(self):
        """
        Feature: build_staged_state_dict bytes isolation.
        Description: Stage state dict containing a bytearray leaf.
        Expectation: Staged bytes are equal but not the same mutable buffer object.
        """
        buf = bytearray(b"checkpoint-meta")
        staged = build_staged_state_dict({"meta": buf})
        self.assertEqual(staged["meta"], bytes(buf))
        self.assertIsInstance(staged["meta"], bytes)
        self.assertIsNot(staged["meta"], buf)

    def test_build_staged_state_dict_preserves_nested_structure(self):
        """
        Feature: build_staged_state_dict structural round-trip.
        Description: Nested dict/list optimizer-style state with tensor leaves.
        Expectation: Staged dict mirrors keys and nesting of the input.
        """
        state = {
            "model": {"w": torch.zeros(2)},
            "optim": [{"step": 1, "exp_avg": torch.zeros(2)}],
        }
        staged = build_staged_state_dict(state)
        self.assertIn("w", staged["model"])
        self.assertEqual(staged["optim"][0]["step"], 1)
        self.assertEqual(tuple(staged["optim"][0]["exp_avg"].shape), (2,))

    def test_build_staged_state_dict_preserves_ragged_global_shape(self):
        """Ragged DTensor staging keeps the explicit logical global shape."""
        _DEVICE_MESH_MAP.clear()
        EXISTING_COMM_GROUPS.clear()
        with patch(
                "hyper_parallel.core.dtensor.device_mesh.platform.get_rank",
                return_value=0,
        ):
            mesh = Layout((2,), ("ragged",), init_backend=False).mesh
            tensor = DTensor.from_local(
                torch.arange(48),
                mesh,
                (RaggedShard(dims=(0, 1), local_units=(1, 3)),),
                shape=(6, 4, 8),
            )
            staged = build_staged_state_dict({"weight": tensor})["weight"]

        self.assertEqual(tuple(staged.shape), (6, 4, 8))
        torch.testing.assert_close(staged.to_local(), tensor.to_local())


class _FinishedProc:
    """Stand-in for the persist child process, already exited when it is joined."""

    def __init__(self, exitcode: int = 0) -> None:
        """Record the exit code the join thread reports when no result arrives."""
        self.exitcode = exitcode
        self.joined = False

    def is_alive(self) -> bool:
        """The child has already exited by the time the join thread looks."""
        return False

    def join(self) -> None:
        """Record that the join thread waited for the child."""
        self.joined = True


class TestFileExchange(unittest.TestCase):
    """Tests for the files ranks exchange when async save communicates through storage."""

    def setUp(self) -> None:
        """Rebuild the module under test so its enums match the ones these tests pass in."""
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        _platform_mod.platform = None
        importlib.reload(staging_mod)

    @staticmethod
    def _plan(fqn: str) -> SavePlan:
        """Build a payload that survives a pickle round trip and compares by value."""
        return SavePlan(items=[], planner_data=fqn)

    def test_construct_file_batches_covers_every_file_once(self):
        """
        Feature: construct_file_batches splitting.
        Description: Ten files across at most four reader threads.
        Expectation: Four batches, together holding every file id exactly once.
        """
        batches, max_workers = staging_mod.construct_file_batches(tuple(range(10)), 4)
        self.assertEqual(max_workers, 4)
        self.assertEqual(len(batches), 4)
        self.assertEqual(sorted(f for batch in batches for f in batch), list(range(10)))

    def test_construct_file_batches_clamps_workers_to_file_count(self):
        """
        Feature: construct_file_batches worker clamping.
        Description: Two files, but room for thirty-two threads.
        Expectation: Two single-file batches and a clamped worker count.
        """
        batches, max_workers = staging_mod.construct_file_batches((0, 1), 32)
        self.assertEqual(max_workers, 2)
        self.assertEqual(batches, [(0,), (1,)])

    def test_construct_file_batches_without_files_raises(self):
        """
        Feature: construct_file_batches input validation.
        Description: An empty file list would clamp the worker count to zero.
        Expectation: ValueError instead of a division by zero later on.
        """
        with self.assertRaises(ValueError):
            staging_mod.construct_file_batches((), 8)

    def test_write_then_load_file_round_trip(self):
        """
        Feature: write_file / load_file round trip.
        Description: Write one rank's local plan, then read it back.
        Expectation: The loaded payload equals the written one.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            staging_mod.write_file(tmpdir, staging_mod.FileType.LOCAL_PLAN, 3, self._plan("rank3"))
            self.assertEqual(staging_mod.load_file(tmpdir, staging_mod.FileType.LOCAL_PLAN, 3), self._plan("rank3"))

    def test_load_file_skips_a_file_without_the_complete_flag(self):
        """
        Feature: load_file completion flag.
        Description: Truncate the trailing flag, as a crash mid-write would leave it - readers
            poll for files other ranks are still writing, so a partial file must not be read.
        Expectation: load_file reports the file as not ready instead of unpickling it.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            staging_mod.write_file(tmpdir, staging_mod.FileType.LOCAL_PLAN, 0, self._plan("rank0"))
            path = Path(tmpdir) / f"{staging_mod.FileType.LOCAL_PLAN.name}_0.pkl"
            path.write_bytes(path.read_bytes()[: -len(staging_mod._COMPLETE_FLAG)])
            self.assertIsNone(staging_mod.load_file(tmpdir, staging_mod.FileType.LOCAL_PLAN, 0))

    def test_load_file_ignores_a_file_shorter_than_the_flag(self):
        """
        Feature: load_file completion flag on a barely started file.
        Description: A file whose length is below the flag length.
        Expectation: load_file reports it as not ready rather than seeking past the start.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / f"{staging_mod.FileType.LOCAL_PLAN.name}_1.pkl").write_bytes(b"ab")
            self.assertIsNone(staging_mod.load_file(tmpdir, staging_mod.FileType.LOCAL_PLAN, 1))

    def test_load_file_consumes_storage_data_but_keeps_local_plans(self):
        """
        Feature: load_file cleanup.
        Description: Read back one file of each type.
        Expectation: The storage data file is deleted once read; the local plan file stays,
            because every rank reads it while only the coordinator reads storage data.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            staging_mod.write_file(tmpdir, staging_mod.FileType.STORAGE_DATA, 0, self._plan("storage"))
            staging_mod.write_file(tmpdir, staging_mod.FileType.LOCAL_PLAN, 0, self._plan("plan"))
            staging_mod.load_file(tmpdir, staging_mod.FileType.STORAGE_DATA, 0)
            staging_mod.load_file(tmpdir, staging_mod.FileType.LOCAL_PLAN, 0)
            self.assertFalse((Path(tmpdir) / f"{staging_mod.FileType.STORAGE_DATA.name}_0.pkl").exists())
            self.assertTrue((Path(tmpdir) / f"{staging_mod.FileType.LOCAL_PLAN.name}_0.pkl").exists())

    def test_load_file_returns_none_for_a_file_that_does_not_exist(self):
        """
        Feature: load_file on a rank that has not written yet.
        Description: Read a file id nobody wrote.
        Expectation: None, so the reader keeps polling instead of raising.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertIsNone(staging_mod.load_file(tmpdir, staging_mod.FileType.LOCAL_PLAN, 7))


class TestResolveAsyncPersistResult(unittest.TestCase):
    """Tests for the join thread that turns the child's queue payload into the future."""

    def setUp(self) -> None:
        """Rebuild the module under test and start from an empty plan cache."""
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        _platform_mod.platform = None
        importlib.reload(staging_mod)
        staging_mod.StandardSavePlanner.cached_save_result.clear()

    def tearDown(self) -> None:
        """Leave no cached plans behind for the other tests."""
        staging_mod.StandardSavePlanner.cached_save_result.clear()

    @staticmethod
    def _metadata() -> Metadata:
        """Build the metadata the child process would hand back."""
        return Metadata(state_dict_metadata={})

    def test_success_completes_the_future_and_runs_the_callback(self):
        """
        Feature: resolve_async_persist_result success path.
        Description: The child reports SUCCESS with metadata and its plan cache.
        Expectation: The future holds the metadata, the callback ran, the plan cache is merged
            into the parent process, and the child was joined.
        """
        metadata = self._metadata()
        result_queue = queue.Queue()
        result_queue.put((staging_mod.AsyncPersistStatus.SUCCESS, (metadata, {"ns": "cached_plan"})))
        future: Future = Future()
        proc = _FinishedProc()
        calls = []

        staging_mod.resolve_async_persist_result(proc, result_queue, future, lambda: calls.append(1))

        self.assertIs(future.result(timeout=0), metadata)
        self.assertEqual(calls, [1])
        self.assertEqual(staging_mod.StandardSavePlanner.cached_save_result["ns"], "cached_plan")
        self.assertTrue(proc.joined)

    def test_failure_surfaces_the_child_traceback(self):
        """
        Feature: resolve_async_persist_result failure path.
        Description: The child reports FAILURE with its formatted traceback.
        Expectation: Waiting on the future raises, carrying the child's traceback text.
        """
        result_queue = queue.Queue()
        result_queue.put((staging_mod.AsyncPersistStatus.FAILURE, "OSError: No space left on device"))
        future: Future = Future()

        staging_mod.resolve_async_persist_result(_FinishedProc(1), result_queue, future, lambda: None)

        with self.assertRaises(RuntimeError) as ctx:
            future.result(timeout=0)
        self.assertIn("No space left on device", str(ctx.exception))

    def test_a_child_that_dies_without_a_result_fails_the_future(self):
        """
        Feature: resolve_async_persist_result on a crashed child.
        Description: The child exits without putting anything on the queue.
        Expectation: The future raises and names the exit code, instead of hanging forever.
        """
        future: Future = Future()

        staging_mod.resolve_async_persist_result(_FinishedProc(-9), queue.Queue(), future, lambda: None)

        with self.assertRaises(RuntimeError) as ctx:
            future.result(timeout=0)
        self.assertIn("-9", str(ctx.exception))

    def test_a_failing_callback_fails_the_future(self):
        """
        Feature: resolve_async_persist_result callback errors.
        Description: Persistence succeeded but the user callback raises.
        Expectation: The future raises rather than reporting a clean save.
        """
        result_queue = queue.Queue()
        result_queue.put((staging_mod.AsyncPersistStatus.SUCCESS, (self._metadata(), {})))
        future: Future = Future()

        def boom() -> None:
            """Stand in for a user callback that fails after a successful persist."""
            raise ValueError("callback exploded")

        staging_mod.resolve_async_persist_result(_FinishedProc(), result_queue, future, boom)

        with self.assertRaises(RuntimeError) as ctx:
            future.result(timeout=0)
        self.assertIn("callback exploded", str(ctx.exception))

    def test_an_unexpected_status_fails_the_future(self):
        """
        Feature: resolve_async_persist_result unknown payload.
        Description: The queue carries a status the join thread does not know.
        Expectation: The future raises and names the unexpected status.
        """
        result_queue = queue.Queue()
        result_queue.put(("not_a_status", None))
        future: Future = Future()

        staging_mod.resolve_async_persist_result(_FinishedProc(), result_queue, future, lambda: None)

        with self.assertRaises(RuntimeError) as ctx:
            future.result(timeout=0)
        self.assertIn("unexpected status", str(ctx.exception))

    def test_async_save_response_get_result_returns_the_metadata(self):
        """
        Feature: AsyncSaveResponse.get_result.
        Description: Wait on a response whose persist future already completed.
        Expectation: The metadata the child produced is returned.
        """
        metadata = self._metadata()
        future: Future = Future()
        future.set_result(metadata)
        self.assertIs(staging_mod.AsyncSaveResponse(persist_completion=future).get_result(timeout=0), metadata)


class TestCopyDispatch(unittest.TestCase):
    """Tests for which copy handler DataCopier picks for a given object."""

    def setUp(self) -> None:
        """Rebuild the module under test and snapshot its handler registry."""
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        _platform_mod.platform = None
        importlib.reload(staging_mod)
        self._registry_backup = dict(staging_mod.DataCopier._registry)

    def tearDown(self) -> None:
        """Restore the registry the test registered its own types into."""
        staging_mod.DataCopier._registry.clear()
        staging_mod.DataCopier._registry.update(self._registry_backup)

    def test_exact_type_wins(self):
        """
        Feature: DataCopier.dispatch exact type lookup.
        Description: Dispatch a plain torch tensor and a DTensor.
        Expectation: Each one resolves to the handler registered for its own type.
        """
        dispatch = staging_mod.DataCopier.dispatch
        self.assertIs(dispatch(torch.zeros(2)), staging_mod.DataCopier._registry[staging_mod.platform.Tensor])
        self.assertIs(dispatch({}), staging_mod.DataCopier._registry[dict])

    def test_subclass_resolves_to_the_most_derived_registered_type(self):
        """
        Feature: DataCopier.dispatch subclass lookup.
        Description: Register a base class and then a derived one, and dispatch an instance of
            a further subclass - the shape DTensor has, since it is registered after
            platform.Tensor and derives from it.
        Expectation: The derived handler wins. Resolving in registration order instead would
            stage a DTensor subclass as a plain tensor, dropping its mesh and placements.
        """
        class _Base:
            pass

        class _Derived(_Base):
            pass

        class _Leaf(_Derived):
            pass

        staging_mod.DataCopier.register(_Base)(lambda obj: "base")
        staging_mod.DataCopier.register(_Derived)(lambda obj: "derived")

        self.assertEqual(staging_mod.DataCopier.dispatch(_Leaf())("x"), "derived")

    def test_unregistered_type_has_no_handler(self):
        """
        Feature: DataCopier.dispatch fallback.
        Description: Dispatch an object of a type nothing was registered for.
        Expectation: None, so copy() falls back to its generic deep copy.
        """
        class _Unregistered:
            pass

        self.assertIsNone(staging_mod.DataCopier.dispatch(_Unregistered()))


class TestWaitsForTheChildProcess(unittest.TestCase):
    """Tests how the save path waits on work another process is still doing."""

    def setUp(self) -> None:
        """Rebuild the module under test."""
        os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"
        _platform_mod.platform = None
        importlib.reload(staging_mod)

    def test_the_join_wait_blocks_on_the_queue_with_a_timeout(self):
        """
        Feature: resolve_async_persist_result waiting for the child.
        Description: A queue that is empty once and then holds the result. This wait runs in
            the training process for as long as the child persists.
        Expectation: It waits with a blocking get plus timeout, never a bare get_nowait poll.
        """
        metadata = Metadata(state_dict_metadata={})
        payload = (staging_mod.AsyncPersistStatus.SUCCESS, (metadata, {}))
        timeouts = []

        class _SlowQueue:
            """Queue that reports empty once before handing over the payload."""

            def __init__(self) -> None:
                """Start with no wait recorded yet."""
                self.calls = 0

            def get(self, timeout: Optional[float] = None) -> Any:
                """Record the timeout and hand over the payload on the second wait."""
                timeouts.append(timeout)
                self.calls += 1
                if self.calls == 1:
                    raise queue.Empty
                return payload

            def get_nowait(self) -> Any:
                """A spinning implementation would come through here."""
                raise AssertionError("resolve_async_persist_result must not spin on get_nowait")

        class _LiveThenDoneProc:
            """Child process that is still alive during the first empty round."""

            exitcode = 0

            def is_alive(self) -> bool:
                """Stay alive so the wait loops instead of giving up."""
                return True

            def join(self) -> None:
                """Joining a finished child is a no-op here."""

        future: Future = Future()
        staging_mod.resolve_async_persist_result(_LiveThenDoneProc(), _SlowQueue(), future, lambda: None)

        self.assertIs(future.result(timeout=0), metadata)
        self.assertTrue(all(t is not None and t > 0 for t in timeouts), f"expected timeouts, got {timeouts}")

    def test_the_file_wait_retries_the_next_round_without_sleeping(self):
        """
        Feature: batch_read_worker waiting for the ranks still writing.
        Description: A file that is not readable on the first round and readable on the second.
        Expectation: The worker picks it up on the next round and never sleeps in between -
            the round-robin exists to read every file the moment it lands.
        """
        reads = []
        sleeps = []

        # pylint: disable=W0613
        def fake_load_file(checkpoint_id: str, file_type: Any, file_id: int) -> Optional[str]:
            """Report the file as not ready the first time it is asked for."""
            reads.append(file_id)
            return None if len(reads) == 1 else "plan"

        with patch.object(staging_mod, "load_file", fake_load_file), \
                patch.object(staging_mod.time, "sleep", sleeps.append):
            results = staging_mod.batch_read_worker((0,), "ckpt", staging_mod.FileType.LOCAL_PLAN)

        self.assertEqual(results, ["plan"])
        self.assertEqual(reads, [0, 0])
        self.assertEqual(sleeps, [])


if __name__ == "__main__":
    unittest.main()
