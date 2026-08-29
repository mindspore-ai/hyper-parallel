# Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tensor staging and persist for asynchronous distributed checkpoint save.

Staging materializes a host-side snapshot of ``state_dict`` so ``async_save`` can
run persistence (plan + I/O) on a copy while training continues on the original tensors.
"""

import os
import copy
import pickle
import dataclasses
import queue
import time
import traceback
import multiprocessing as mp
from pathlib import Path
from enum import Enum, auto
from argparse import Namespace
from collections.abc import Callable
from typing import Any, Optional, Union
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.distributed_checkpoint.metadata import (
    Metadata,
    CHUNK_INFO,
)
from hyper_parallel.core.distributed_checkpoint.planner import (
    SavePlan,
    SavePlanner,
)
from hyper_parallel.core.distributed_checkpoint.storage import (
    StorageWriter,
    WriteResult,
)
from hyper_parallel.core.distributed_checkpoint.standard_planner import (
    StandardSavePlanner,
)
from hyper_parallel.core.distributed_checkpoint.util import (
    dcp_timer_decorator,
    logger,
    platform,
)

_COMPLETE_FLAG = b"\x00__PICKLE_WRITE_COMPLETE__\x00"

# The join wait sits in the *training* process for as long as the child persists,
# so it blocks on the queue with a timeout rather than spinning.
_PERSIST_RESULT_POLL_SECONDS = 0.1


class DataCopier:
    """Type dispatched copy of ``state_dict`` values into host memory.

    Copy handlers are registered per type through :meth:`register`; :meth:`copy` resolves the
    handler of an object by exact type first and by ``isinstance`` afterwards, falling back to
    a generic recursive deep copy for objects no handler was registered for.
    """

    _registry = {}

    @classmethod
    def register(cls, *types):
        def decorator(func):
            for t in types:
                cls._registry[t] = func
            return func

        return decorator

    @classmethod
    def dispatch(cls, obj: Any) -> Optional[Callable]:
        """
        Resolve the copy handler of ``obj``: its exact type first, its most derived
        registered base class afterwards.

        Registration order must not decide the handler. ``DTensor`` is registered after
        ``platform.Tensor`` and is a subclass of it, so scanning in registration order would
        hand a ``DTensor`` subclass to the plain tensor handler, staging it as a bare local
        tensor with its mesh and placements dropped.

        Args:
            obj (Any): Object to find a copy handler for.

        Returns:
            Optional[Callable]: The handler, or None when no registered type matches.
        """
        if type(obj) in cls._registry:
            return cls._registry[type(obj)]

        best_type = None
        for r_type in cls._registry:
            if not isinstance(obj, r_type):
                continue
            # Unrelated matches (neither one a subclass of the other) keep the first found.
            if best_type is None or issubclass(r_type, best_type):
                best_type = r_type
        return cls._registry[best_type] if best_type is not None else None

    @classmethod
    def copy(cls, obj):
        """
        Copy ``obj`` with its registered handler, or with a generic deep copy as fallback.

        Args:
            obj (Any): Object to copy, of any type held by a state dict.

        Returns:
            Any: Copy of ``obj``, with tensor leaves residing in host memory.
        """
        copy_func = cls.dispatch(obj)
        if copy_func:
            return copy_func(obj)
        logger.warning(
            "The user define object %s not implement copy method and register "
            "to DataCopier. So, the general copy method will be used, which may cause "
            "unexpected copy behavior and incorrect data storage. Please verify this on your own.",
            type(obj))
        return cls._copy_others(obj)

    @staticmethod
    def _copy_others(obj):
        obj_c = copy.deepcopy(obj)
        for name, value in obj_c.__dict__.items():
            if callable(value):
                continue
            setattr(obj_c, name, DataCopier.copy(value))
        return obj_c


def _copy_tensor_to_cpu(tensor: platform.Tensor) -> platform.Tensor:
    """Return a host-memory copy of a framework tensor, detached from autograd where applicable."""
    # ``to("cpu")`` is supported on both Torch and MindSpore tensor APIs used by HyperParallel.
    t = tensor.detach().clone() if tensor.is_cpu else tensor.detach().cpu()
    if hasattr(tensor, CHUNK_INFO):
        setattr(t, CHUNK_INFO, getattr(tensor, CHUNK_INFO))
    return t


@DataCopier.register(int, float, complex, bool, str, type(None), Enum, platform.dtype)
def _copy_const(obj):
    return obj


@DataCopier.register(bytes, bytearray)
def _copy_bytes(obj):
    return bytes(obj)


@DataCopier.register(np.ndarray)
def _copy_ndarray(obj):
    return obj.copy()


@DataCopier.register(platform.Tensor)
def _copy_tensor(obj):
    result = _copy_tensor_to_cpu(obj)
    return result


@DataCopier.register(DTensor)
def _copy_dtensor(obj):
    """Stage a DTensor by copying its local shard to host memory, keeping mesh and placements."""
    staged_local = _copy_tensor_to_cpu(obj.to_local())
    # ``shape`` is required for RaggedShard: an unevenly-sharded DTensor cannot
    # recover its logical global shape from the local shard alone.
    result = DTensor.from_local(
        staged_local,
        obj.device_mesh,
        obj.placements,
        shape=tuple(obj.shape),
    )
    return result


@DataCopier.register(Namespace)
def _copy_namespace(obj):
    c = DataCopier.copy(vars(obj))
    return Namespace(**c)


@DataCopier.register(list)
def _copy_list(obj):
    result = [DataCopier.copy(item) for item in obj]
    return result


@DataCopier.register(tuple)
def _copy_tuple(obj):
    result = tuple(DataCopier.copy(item) for item in obj)
    return result


@DataCopier.register(dict)
def _copy_dict(obj):
    result = {}
    for k, v in obj.items():
        result[DataCopier.copy(k)] = DataCopier.copy(v)
    return result


class AsyncPersistStatus(Enum):
    """Queue payload status from :func:`_async_persist_worker` to the parent join thread."""

    SUCCESS = auto()
    FAILURE = auto()


class FileType(Enum):

    LOCAL_PLAN = auto()
    STORAGE_DATA = auto()


@dataclasses.dataclass
class AsyncSaveResponse:
    """Result of :func:`async_save`.

    Host staging runs synchronously before :func:`async_save` returns; only checkpoint
    **persistence** is asynchronous. ``persist_completion`` completes when the child
    process finishes :func:`_save_impl` (plan, collectives, disk I/O) and supplies
    :class:`Metadata`.
    """

    persist_completion: Future[Metadata]

    def get_result(self, timeout: int = None) -> Optional[Metadata]:
        return self.persist_completion.result(timeout=timeout)


@dcp_timer_decorator
def build_staged_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Build a deep structural copy of ``state_dict`` with tensor / DTensor data
    copied to host memory so the original can be mutated while checkpoint I/O runs.

    Uses the same flattening rules as :class:`StandardSavePlanner`.

    Args:
        state_dict (dict[str, Any]): Nested or flat training state dict.

    Returns:
        dict[str, Any]: New dict with identical nesting and keys; tensor leaves are staging copies.
    """
    return DataCopier.copy(state_dict)


def cleanup_and_reinit_process_group(
    master_addr: str,
    master_port: int,
    rank: int,
    world_size: int
):
    """
    Cleanup hccl process group and reinitialize gloo process group.
    """
    # torch is imported lazily so this module stays importable on a MindSpore-only install.
    import torch.distributed as dist  # pylint: disable=import-outside-toplevel
    import torch.distributed.distributed_c10d as c10d  # pylint: disable=import-outside-toplevel

    # Resetting the private c10d state is the only way to drop the process group inherited
    # from the parent process, so the accesses below are deliberate.
    # pylint: disable=protected-access

    # Step 1: clear C++ ProcessGroupRegistry (avoids "already registered" error).
    if hasattr(c10d, "_unregister_all_process_groups"):
        try:
            c10d._unregister_all_process_groups()
        except Exception:  # pylint: disable=broad-except
            pass

    # Step 2: reset Python-layer _world to an uninitialised state.
    if hasattr(c10d, "_World"):
        try:
            c10d._world = c10d._World()
        except Exception:  # pylint: disable=broad-except
            pass
    # Also reset the module-level init-method string (present in all versions).
    if hasattr(c10d, "_default_pg_init_method"):
        c10d._default_pg_init_method = None
    for attr in ("_default_pg", "_pg_map", "_pg_names", "_pg_group_ranks",
                 "_pg_backend_config", "_group_count", "_tags_to_pg",
                 "_pg_to_tag"):
        if not hasattr(c10d, attr):
            continue
        old_val = getattr(c10d, attr)
        if isinstance(old_val, dict):
            setattr(c10d, attr, {})
        elif isinstance(old_val, int):
            setattr(c10d, attr, 0)
        else:
            setattr(c10d, attr, None)

    # Step 3: build a fresh TCPStore on the dedicated gloo port and init.
    # rank-0 child is the store master; all others connect as clients.
    store = dist.PrefixStore(
        prefix=mp.current_process().name,
        store=dist.TCPStore(
            host_name=master_addr,
            port=master_port
        ),
    )

    # Step 4: init new process group with backend "gloo".
    dist.init_process_group(
        backend="gloo",
        store=store,
        rank=rank,
        world_size=world_size,
    )


@dcp_timer_decorator
def execute_async_persist_with_gloo(
    result_queue: mp.Queue,
    staged: dict[str, Any],
    checkpoint_id: Optional[Union[Path, str]],
    storage_writer: Optional[StorageWriter],
    planner: Optional[SavePlanner],
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
) -> None:
    """Child-process entry for gloo-based async save.
    Reinitialises a **CPU-only gloo process group** on a dedicated port so that
    all collective communication (all_gather_object for plan exchange,
    all_gather_object for write_results) can run inside the child process,
    exactly mirroring what :func:`_save_impl` does synchronously.
    The parent process only stages tensors to CPU; all planning, collectives,
    disk I/O and metadata finalisation happen here.
    Design rationale
    ----------------
    * The training group uses NCCL (or hccl on Ascend) and lives in C++ global
      state that is invalidated across ``fork``.  We never touch it.
    * A fresh ``gloo`` group is initialised via a ``TCPStore`` client on the training
      rendezvous itself (``master_addr:master_port``), namespaced by a ``PrefixStore``
      keyed on this process name. The name carries a per-``async_save`` counter, so each
      save gets its own key space on the same store; every rank must therefore reach the
      same number of gloo saves, or the prefixes diverge and the rendezvous hangs.
    * After :func:`_save_impl` completes the gloo group is destroyed so resources
      are released cleanly.

    Args:
        result_queue: IPC queue for returning status / metadata to parent join thread.
        staged: CPU-resident staged state dict (output of build_staged_state_dict).
        checkpoint_id: Checkpoint directory path.
        storage_writer: Pre-constructed StorageWriter (or None → FileSystemWriter).
        planner: Pre-constructed SavePlanner (or None → StandardSavePlanner).
        rank: Global rank of this process (inherited from parent).
        world_size: Total world size (inherited from parent).
        master_addr: TCP rendezvous address for the gloo store.
        master_port: Training master port; the child connects to that store as a client.
    """
    # Lazy imports: torch keeps this module framework-agnostic, and ``api`` imports this
    # module at load time, so importing it here is what breaks the import cycle.
    import torch.distributed as dist  # pylint: disable=import-outside-toplevel
    from hyper_parallel.core.distributed_checkpoint.api import _save_impl  # pylint: disable=import-outside-toplevel
    try:
        # Clear the original communication group information and establish the gloo communication.
        cleanup_and_reinit_process_group(
            master_addr,
            master_port,
            rank,
            world_size
        )
        meta = _save_impl(
            staged,
            checkpoint_id=checkpoint_id,
            storage_writer=storage_writer,
            planner=planner,
        )
        # The plan cache is also returned and can be reused for the next dcp async save.
        result_queue.put((AsyncPersistStatus.SUCCESS, (meta, StandardSavePlanner.cached_save_result)))
    except Exception:  # pylint: disable=broad-except
        result_queue.put((AsyncPersistStatus.FAILURE, traceback.format_exc()))
    finally:
        # Clean up the gloo group we created; ignore errors (process is exiting).
        if dist.is_initialized():
            dist.destroy_process_group()


@dcp_timer_decorator
def execute_async_persist(
    result_queue: mp.Queue,
    staged: dict[str, Any],
    checkpoint_id: Optional[Union[Path, str]],
    storage_writer: Optional[StorageWriter],
    planner: Optional[SavePlanner],
    no_dist: bool,
    use_collectives: bool,
    use_storage_comm: bool,
) -> None:
    """
    Perform asynchronous flushing to disk,
    which is not suitable for collective communication,
    or use the file system for collective communication.

    Args:
        result_queue: IPC queue for returning status / metadata to parent join thread.
        staged: CPU-resident staged state dict (output of build_staged_state_dict).
        checkpoint_id: Checkpoint directory path.
        storage_writer: Pre-constructed StorageWriter (or None → FileSystemWriter).
        planner: Pre-constructed SavePlanner (or None → StandardSavePlanner).
        no_dist: Whether it is a distributed scenario or a standalone scenario.
        use_collectives: Total world size (inherited from parent).
        use_storage_comm: TCP rendezvous address for the gloo store.
    """
    # ``api`` imports this module at load time, so this import has to stay function local.
    from hyper_parallel.core.distributed_checkpoint.api import _save_impl  # pylint: disable=import-outside-toplevel
    try:
        meta = _save_impl(
            staged,
            checkpoint_id=checkpoint_id,
            storage_writer=storage_writer,
            planner=planner,
            no_dist=no_dist,
            use_collectives=use_collectives,
            use_storage_comm=use_storage_comm
        )
        # The plan cache is also returned and can be reused for the next dcp async save.
        result_queue.put((AsyncPersistStatus.SUCCESS, (meta, StandardSavePlanner.cached_save_result)))
    except Exception:  # pylint: disable=broad-except
        result_queue.put((AsyncPersistStatus.FAILURE, traceback.format_exc()))


def resolve_async_persist_result(
        proc: mp.Process,
        result_queue: mp.Queue,
        persist_future: Future[None],
        async_callback: Callable[[], None],
) -> None:
    """Join persist ``proc`` and complete ``persist_future`` (runs on a background thread)."""
    result = None
    while result is None:
        try:
            result = result_queue.get(timeout=_PERSIST_RESULT_POLL_SECONDS)
        except queue.Empty:
            if proc.is_alive():
                continue
            # The child may have put its result and exited while the wait above was
            # timing out, so look once more before giving up on it.
            try:
                result = result_queue.get_nowait()
            except queue.Empty:
                pass
            break

    proc.join()
    if persist_future.done():
        return

    if result is None:
        persist_future.set_exception(
            RuntimeError(
                f"async_persist process exited with code {proc.exitcode} and no result on queue"
            )
        )
        return

    status, payload = result
    if status == AsyncPersistStatus.SUCCESS:
        try:
            async_callback()
            StandardSavePlanner.cached_save_result.update(payload[1])
        except Exception:  # pylint: disable=broad-except
            persist_future.set_exception(RuntimeError(traceback.format_exc()))
        else:
            persist_future.set_result(payload[0])
    elif status == AsyncPersistStatus.FAILURE:
        persist_future.set_exception(RuntimeError(payload))
    else:
        persist_future.set_exception(
            RuntimeError(f"async_persist queue returned unexpected status: {status!r}")
        )


def batch_read_worker(file_batch: tuple[int], checkpoint_id, file_type) -> list:
    """
    Read one batch of files, waiting for the files that the other ranks have not written yet.

    Args:
        file_batch (tuple[int]): Contiguous file ids this worker is responsible for.
        checkpoint_id (str): Checkpoint directory holding the files.
        file_type (FileType): Which kind of file to read, local plan or storage data.

    Returns:
        list: Content of every file of ``file_batch``, in the order of the batch.

    Raises:
        TimeoutError: If some files are still unreadable after 1800 seconds.
    """
    results = [None] * len(file_batch)
    file_to_read = file_batch
    first_file_id = file_batch[0]

    # Read files in a round-robin manner,
    # If a file has not been generated yet, skip it and proceed to the next file.
    # Until all files have been read.
    timeout = 1800
    deadline = time.monotonic() + timeout
    while True:
        if not file_to_read:
            break
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"After waiting for 1800 seconds, "
                f"the {file_type.name} files: {file_to_read} still have not been read completely."
            )
        file_not_read = []
        for file_id in file_to_read:
            ret = load_file(checkpoint_id, file_type, file_id)
            if ret is None:
                file_not_read.append(file_id)
            else:
                results[file_id - first_file_id] = ret
        file_to_read = tuple(file_not_read)

    return results


@dcp_timer_decorator
def parallel_read_files(
    file_list: tuple[int],
    checkpoint_id: str,
    file_type: FileType,
    max_workers: int = 32
) -> list:
    """
    Split the total file list into multiple batches
    based on the number of parallel threads,
    then perform multi-threaded parallel reading to improve read efficiency.
    """
    if not file_list:
        return []

    batches, max_workers = construct_file_batches(file_list, max_workers)
    logger.info(
        "parallel read files, file type: %s, file count: %d, max read workers: %d",
        file_type.name, len(file_list), max_workers
    )

    # Use the thread pool to read data concurrently, improving the read efficiency.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        all_results = tuple(
            executor.map(lambda b: batch_read_worker(b, checkpoint_id, file_type), batches)
        )
    return [result for batch_results in all_results for result in batch_results]


def construct_file_batches(file_list: tuple[int], max_workers: int) -> (list[tuple[int]], int):
    """
    Split the total file list into multiple batches
    based on the number of parallel threads
    """
    n = len(file_list)
    max_workers = min(max_workers, n)
    if max_workers == 0:
        raise ValueError("Max workers number for parallel read batch file is 0!")
    batch_size = (n + max_workers - 1) // max_workers
    # Split the file list into max_workers sublists.
    batches = [
        file_list[i * batch_size: (i + 1) * batch_size]
        for i in range(max_workers)
    ]
    batches = [b for b in batches if b]
    return batches, max_workers


@dcp_timer_decorator
def gather_all_results_from_storage(
    checkpoint_id: str,
    is_coordinator: bool,
    world_size: int,
    results: Union[SavePlan, WriteResult],
    file_type: FileType,
    file_id: int
) -> list:
    """
    All node data is collected from the storage,
    which is classified into local plan and storage data.
    In the first step, each rank writes its own data.
    In the second step, for local plan, each rank reads the file written by all ranks.
    For storage data, only rank 0 reads the file written by all ranks.
    """
    logger.info("gather all results from storage, file type: %s", file_type.name)
    write_file(checkpoint_id, file_type, file_id, results)
    if not is_coordinator and file_type != FileType.LOCAL_PLAN:
        return []

    # For LOCAL_PLAN, each node has 16 rank processes reading simultaneously.
    if file_type == FileType.LOCAL_PLAN:
        max_workers = min(16, max(os.cpu_count() // 32, 1))
    else:
        max_workers = min(64, max(os.cpu_count() // 2, 1))
    file_list = tuple(range(0, world_size))
    gathered = parallel_read_files(file_list,
                                   checkpoint_id,
                                   file_type,
                                   max_workers=max_workers)
    return gathered


def assemble_file_path(checkpoint_id: str, file_type: FileType, file_id: int) -> str:
    """
    Assemble file path to exchanges its results through.

    Args:
        checkpoint_id (str): Checkpoint directory holding the exchange files.
        file_type (FileType): Which kind of file, local plan or storage data.
        file_id (int): Rank that owns the file.

    Returns:
        str: Path of that rank's file.
    """
    return f"{checkpoint_id}/{file_type.name}_{file_id}.pkl"


def load_file(checkpoint_id: str, file_type: FileType, file_id: int):
    """
    Reads files from the storage.
    Currently, this function is mainly used to
    asynchronously save local plan and storage data files.
    """
    file_path = assemble_file_path(checkpoint_id, file_type, file_id)

    try:
        with open(file_path, "rb") as f:
            flag_len = len(_COMPLETE_FLAG)
            f.seek(0, os.SEEK_END)
            size = f.tell()
            # determining whether the file length is greater than the flag length
            if size < flag_len:
                return None
            f.seek(size - flag_len)
            # determining whether the file has a write complete flag
            if f.read(flag_len) != _COMPLETE_FLAG:
                return None
            f.seek(0)
            result = pickle.load(f)
        if file_type == FileType.STORAGE_DATA:
            os.remove(file_path)
        return result
    except FileNotFoundError:
        return None



@dcp_timer_decorator
def write_file(
    checkpoint_id: str,
    file_type: FileType,
    file_id: int,
    content: Union[SavePlan, WriteResult]
):
    """
    Write files to the storage.
    Currently, this function is mainly used to
    asynchronously save local plan and storage data files.
    """
    final_path = assemble_file_path(checkpoint_id, file_type, file_id)

    # To avoid read-write conflicts, the completion flag is appended last: a reader that
    # opens the file mid-write finds no flag at the end and retries on its next round.
    with open(final_path, "wb") as f:
        pickle.dump(content, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.write(_COMPLETE_FLAG)

    logger.info("write file type: %s, file path: %s", file_type.name, final_path)
