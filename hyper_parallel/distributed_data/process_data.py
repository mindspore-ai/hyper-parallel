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
"""Independent data collectives with native shared CPU tensor delivery."""

from __future__ import annotations

import logging
import os
import pickle
import random
import re
import signal
import time
import traceback
import uuid
from concurrent.futures import Future
from contextlib import ExitStack
from datetime import timedelta
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from multiprocessing.util import Finalize
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils._import_utils import import_dill

from hyper_parallel.distributed_data.balance_logging import logger as balance_logger
from hyper_parallel.distributed_data.transport import DataGroups, DataPlaneTransport

_SOURCE_PICKLE = import_dill() or pickle
_REPLY_TIMEOUT = 300


def _serialize_source(value: tuple) -> bytes:
    """Capture callback globals for the fresh interpreter when dill is available."""
    if _SOURCE_PICKLE is not pickle:
        return _SOURCE_PICKLE.dumps(value, recurse=True)
    try:
        return pickle.dumps(value)
    except (AttributeError, pickle.PicklingError) as exc:
        raise TypeError("Use callbacks from an importable module when dill is unavailable. "
                        f"Could not serialize the data source: {exc}") from exc


def _store_descriptor() -> dict[str, Any]:
    """Reuse the launcher's rendezvous without reading launcher-specific variables."""
    # There is no public accessor for an initialized group's rendezvous store.
    store = dist.distributed_c10d._get_default_store()  # pylint: disable=protected-access
    while isinstance(store, dist.PrefixStore):
        store = store.underlying_store
    if isinstance(store, dist.TCPStore):
        return {"kind": "tcp", "address": store.host, "port": store.port}
    if isinstance(store, dist.FileStore):
        return {"kind": "file", "path": store.path}
    raise ValueError("Independent data processes require a TCPStore or FileStore rendezvous.")


def _child_environment(backends: tuple[str | None, ...]) -> dict[str, str]:
    """Build data-process environment overrides and validate its HCCL listener range."""
    env = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    if "hccl" not in backends:
        return env
    selected = os.environ.get("HP_DATA_HCCL_SOCKET_PORT_RANGE", "50000-50031")
    match = re.fullmatch(r"(\d+)-(\d+)", selected)
    if match is None or not 1024 <= int(match[1]) <= int(match[2]) <= 65535:
        raise ValueError("HP_DATA_HCCL_SOCKET_PORT_RANGE must be an inclusive range within 1024-65535.")
    for start, end in re.findall(r"(\d+)-(\d+)", os.environ.get("HCCL_NPU_SOCKET_PORT_RANGE", "16666-16666")):
        if max(int(start), int(match[1])) <= min(int(end), int(match[2])):
            raise ValueError("HP_DATA_HCCL_SOCKET_PORT_RANGE overlaps the training HCCL listener range.")
    env["HCCL_NPU_SOCKET_PORT_RANGE"] = selected
    return env


def _child_transport(bootstrap: dict[str, Any]) -> DataPlaneTransport:
    """Rebuild collectives while retaining logical sample-owner rank identities."""
    device = bootstrap["device"]
    if device is not None and device.type != "cpu":
        getattr(torch, device.type).set_device(device)
    ranks = bootstrap["ranks"]
    control = payload = None
    if len(ranks) > 1:
        if bootstrap["kind"] == "tcp":
            store = dist.TCPStore(bootstrap["address"], bootstrap["port"], is_master=False,
                                  timeout=timedelta(seconds=_REPLY_TIMEOUT))
        else:
            store = dist.FileStore(bootstrap["path"])
        store = dist.PrefixStore(bootstrap["prefix"], store)
        dist.init_process_group(bootstrap["backend"], store=store, rank=bootstrap["rank"],
                                world_size=len(ranks), timeout=timedelta(seconds=_REPLY_TIMEOUT))
        control = dist.group.WORLD
        if bootstrap["payload_backend"] is not None:
            payload = control if bootstrap["payload_backend"] == bootstrap["backend"] else dist.new_group(
                backend=bootstrap["payload_backend"], timeout=timedelta(seconds=_REPLY_TIMEOUT),
            )
    groups = DataGroups(ranks, control, payload, None, bootstrap["planner_rank"], len(ranks) > 1)
    return DataPlaneTransport(groups, ranks[bootstrap["rank"]], device,
                              planner_process_rank=ranks.index(bootstrap["planner_rank"]))


def _worker_main(source_path: Path, bootstrap: dict[str, Any], connection: Connection) -> None:
    """Initialize the spawned data rank and serve requests on its parent connection."""
    source = None
    try:
        os.setsid()
        os.environ.update(bootstrap["environment"])
        torch.set_num_threads(1)
        with source_path.open("rb") as handle:
            source, rng_state = _SOURCE_PICKLE.load(handle)
        balance_logger.setLevel(logging.INFO if bootstrap["balance_logging"] else logging.WARNING)
        random.setstate(rng_state[0])
        np.random.set_state(rng_state[1])
        torch.random.set_rng_state(rng_state[2])
        source.bind_transport(_child_transport(bootstrap))
        connection.send(("READY", os.getpid()))
        while True:
            command, value = connection.recv()
            if command == "CLOSE":
                connection.send(("CLOSED", None))
                break
            if command == "NEXT":
                try:
                    batch = next(source)
                except StopIteration:
                    connection.send(("EOF", None))
                else:
                    connection.send(("BATCH", batch))
            elif command == "EPOCH":
                source.set_epoch(value)
                connection.send(("EPOCH", None))
            elif command == "RESET":
                source.reset_iteration()
                connection.send(("RESET", None))
            else:
                raise ValueError(f"Unknown data process command: {command}.")
    except BaseException:
        try:
            connection.send(("ERROR", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        try:
            if source is not None:
                source.close()
        finally:
            connection.close()
            if dist.is_initialized():
                dist.destroy_process_group()


def _reap_process(process: BaseProcess) -> None:
    """Stop the data process and any DataLoader workers still in its session."""
    if process.pid is None:
        return
    for sig, stop, timeout in ((signal.SIGTERM, process.terminate, 10), (signal.SIGKILL, process.kill, None)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            # Spawn may fail or time out before the target creates its session.
            if not process.is_alive():
                return
            stop()
        process.join(timeout)


class ProcessDataWorker:
    """Keep one CPU result in flight on a private distributed producer."""

    def __init__(self, source: Any, transport: DataPlaneTransport) -> None:
        """Serialize an unstarted source and capture its data-group rendezvous."""
        self._resources = ExitStack()
        # At interpreter shutdown multiprocessing joins non-daemonic children after running finalizers.
        self._finalize = Finalize(self, self._resources.close, exitpriority=10)
        directory = self._resources.enter_context(TemporaryDirectory(prefix="hp-data-process-"))
        self._process = None
        self._connection = None
        self._pending = False
        self._deadline = None
        self._closed = False
        self._failed = False
        self._source_path = Path(directory) / "source.pkl"
        try:
            bootstrap = {"ranks": transport.ranks, "planner_rank": transport.planner_rank,
                         "rank": transport.ranks.index(source.global_rank),
                         "backend": transport.communication_backend, "payload_backend": transport.payload_backend,
                         "device": transport.communication_device,
                         "balance_logging": balance_logger.isEnabledFor(logging.INFO)}
            rng = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
            error = None
            try:
                bootstrap["environment"] = _child_environment((bootstrap["backend"], bootstrap["payload_backend"]))
                # Keep large datasets out of spawn's synchronous startup pipe.
                self._source_path.write_bytes(_serialize_source((source, rng)))
            except Exception as exc:  # pylint: disable=broad-exception-caught
                # User-defined pickle hooks may raise any exception. All ranks must still enter the gather.
                error = f"{type(exc).__name__}: {exc}"
            statuses = transport.all_gather_object((uuid.uuid4().hex, error))
            errors = [message for _, message in statuses if message is not None]
            if errors:
                raise ValueError("Data processes require a valid environment and serializable, unstarted CPU sources "
                                 "and callbacks: "
                                 + "; ".join(errors))
            if len(transport.ranks) > 1:
                bootstrap.update(_store_descriptor(), prefix="hp-data-process-" + statuses[0][0])
            self._bootstrap = bootstrap
        except BaseException:
            self.close()
            raise

    def _start(self) -> None:
        """Spawn the producer once and wait until it has bound its data transport."""
        if self._closed or self._failed:
            raise RuntimeError("Data process is closed or failed.")
        if self._process is not None:
            return
        context = torch.multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        self._connection = self._resources.enter_context(parent)
        try:
            with child:
                self._process = context.Process(
                    target=_worker_main, args=(self._source_path, self._bootstrap, child), daemon=False,
                )
                self._resources.callback(self._process.close)
                self._resources.callback(_reap_process, self._process)
                self._process.start()
            status, _ = self._receive()
            if status != "READY":
                raise RuntimeError(f"Expected a ready data process, received {status}.")
        except BaseException:
            self._failed = True
            self.close()
            raise

    def _receive(self) -> tuple[str, Any]:
        """Wait for one reply and propagate a timeout, child exit or remote traceback."""
        deadline = time.monotonic() + _REPLY_TIMEOUT
        try:
            while not self._connection.poll(0.1):
                if self._process.exitcode is not None:
                    raise RuntimeError(f"Data process exited with status {self._process.exitcode}.")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Data process did not reply within the data timeout.")
            status, value = self._connection.recv()
            if status == "ERROR":
                raise RuntimeError(f"Data process failed:\n{value}")
            return status, value
        except EOFError as exc:
            self._failed = True
            raise RuntimeError("Data process closed its connection before replying.") from exc
        except BaseException:
            self._failed = True
            raise

    def prefetch(self) -> None:
        """Start one CPU result without waiting for preparation or communication."""
        self._start()
        if not self._pending:
            self._send("NEXT")
            self._pending = True
            self._deadline = time.monotonic() + _REPLY_TIMEOUT

    def is_ready(self, timeout: float = 0.0) -> bool:
        """Poll delivery so staging can continue while another CPU window is prepared.

        Args:
            timeout: Maximum seconds to wait for a ready result.
        """
        if self._connection.poll(timeout):
            return True
        if self._process.exitcode is not None:
            self._failed = True
            raise RuntimeError(f"Data process exited with status {self._process.exitcode}.")
        if time.monotonic() >= self._deadline:
            self._failed = True
            raise TimeoutError("Data process did not reply within the data timeout.")
        return False

    def __next__(self) -> Any:
        """Receive one result through PyTorch's CPU shared-storage reducers."""
        self.prefetch()
        status, result = self._receive()
        self._pending = False
        if status == "EOF":
            raise StopIteration
        if status != "BATCH":
            raise RuntimeError(f"Expected a CPU batch, received {status}.")
        return result

    def _send(self, command: str, value: Any = None) -> None:
        try:
            self._connection.send((command, value))
        except BaseException:
            self._failed = True
            raise

    def _command(self, command: str, value: Any = None) -> None:
        self._start()
        if self._pending:
            self._receive()
            self._pending = False
        self._send(command, value)
        status, _ = self._receive()
        if status != command:
            raise RuntimeError(f"Expected {command}, received {status}.")

    def set_epoch(self, epoch: int) -> None:
        """Reset the source after all prefetched results have been drained.

        Args:
            epoch: Non-negative epoch to select before the next source read.
        """
        self._command("EPOCH", epoch)

    def reset_iteration(self) -> None:
        """Re-enter the external source without replacing its process-owned state."""
        self._command("RESET")

    def close(self) -> None:
        """Drain a published CPU result and terminate only the owned process session."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._process is not None and not self._failed and self._process.is_alive():
                if self._pending:
                    self._receive()
                self._connection.send(("CLOSE", None))
                self._receive()
                self._process.join(timeout=30)
                if self._process.is_alive():
                    raise TimeoutError("Data process did not exit after acknowledging CLOSE.")
        finally:
            self._finalize()
            self._process = None


class DataRequest:
    """One child reply, received on demand and retained across explicit drains."""

    def __init__(self, worker: Any) -> None:
        """Start one request on the persistent data worker."""
        self._worker = worker
        self._reply = Future()
        try:
            worker.prefetch()
        except BaseException as error:  # pylint: disable=broad-exception-caught
            # A successor's startup failure belongs to that request, after any batch already prepared.
            self._reply.set_exception(error)

    def is_ready(self) -> bool:
        """Poll without consuming the reply or waiting for the child."""
        return self._reply.done() or self._worker.is_ready()

    def result(self) -> Any:
        """Receive once, including EOF or a producer error."""
        if not self._reply.done():
            try:
                self._reply.set_result(next(self._worker))
            except BaseException as error:  # pylint: disable=broad-exception-caught
                # Cache the original failure so drains and delivery observe the same outcome.
                self._reply.set_exception(error)
        return self._reply.result()


def create_data_worker(source: Any, transport: DataPlaneTransport) -> Any:
    """Spawn data-plane members; model-only peers retain their empty local cursor.

    Args:
        source: Serializable, unstarted CPU source with native or external preparation.
        transport: Data-plane collectives owned by this source process.
    """
    if transport.is_member:
        return ProcessDataWorker(source, transport)
    source.bind_transport(transport)
    return source
