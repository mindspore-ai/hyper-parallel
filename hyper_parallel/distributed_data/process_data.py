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
import multiprocessing
import os
import pickle
import random
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid
from contextlib import ExitStack
from datetime import timedelta
from multiprocessing.connection import Connection
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import cloudpickle
import numpy as np
import torch
import torch.distributed as dist

from hyper_parallel.distributed_data.balance_logging import logger as balance_logger
from hyper_parallel.distributed_data.transport import DataGroups, DataPlaneTransport

_REPLY_TIMEOUT = 300


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
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    if "hccl" not in backends:
        return env
    selected = env.get("HP_DATA_HCCL_SOCKET_PORT_RANGE", "50000-50031")
    match = re.fullmatch(r"(\d+)-(\d+)", selected)
    if match is None or not 1024 <= int(match[1]) <= int(match[2]) <= 65535:
        raise ValueError("HP_DATA_HCCL_SOCKET_PORT_RANGE must be an inclusive range within 1024-65535.")
    for start, end in re.findall(r"(\d+)-(\d+)", env.get("HCCL_NPU_SOCKET_PORT_RANGE", "16666-16666")):
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


def _worker_main(capsule_path: str, descriptor: int) -> None:
    """Own CPU preparation and data collectives without importing the trainer's main module."""
    connection = Connection(descriptor)
    source = None
    try:
        torch.set_num_threads(1)
        with open(capsule_path, "rb") as handle:
            module_paths, capsule = pickle.load(handle)
        sys.path[:] = module_paths
        source_bytes, bootstrap = capsule
        # Popen does not inherit multiprocessing's authentication key; CPU FD
        # reducers need the same key in the sender and receiver.
        multiprocessing.current_process().authkey = bootstrap["authkey"]
        source, rng_state = cloudpickle.loads(source_bytes)
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


class ProcessDataWorker:
    """Keep one CPU result in flight on a private distributed producer."""

    def __init__(self, source: Any, transport: DataPlaneTransport) -> None:
        """Serialize an unstarted source and capture its data-group rendezvous."""
        self._resources = ExitStack()
        self._directory = self._resources.enter_context(TemporaryDirectory(prefix="hp-data-process-"))
        self._process = None
        self._connection = None
        self._pending = False
        self._deadline = None
        self._closed = False
        self._failed = False
        self._capsule = Path(self._directory) / "source.pkl"
        try:
            bootstrap = {"ranks": transport.ranks, "planner_rank": transport.planner_rank,
                         "rank": transport.ranks.index(source.global_rank),
                         "backend": transport.communication_backend, "payload_backend": transport.payload_backend,
                         "device": transport.communication_device,
                         "balance_logging": balance_logger.isEnabledFor(logging.INFO),
                         "authkey": bytes(multiprocessing.current_process().authkey)}
            self._env = _child_environment((bootstrap["backend"], bootstrap["payload_backend"]))
            rng = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
            error = None
            source_bytes = None
            try:
                source_bytes = cloudpickle.dumps((source, rng))
            except Exception as exc:  # Propagate source validation on every data rank.
                error = f"{type(exc).__name__}: {exc}"
            statuses = transport.all_gather_object((uuid.uuid4().hex, error))
            errors = [message for _, message in statuses if message is not None]
            if errors:
                raise ValueError("Data processes require serializable, unstarted CPU sources and callbacks: "
                                 + "; ".join(errors))
            if len(transport.ranks) > 1:
                bootstrap.update(_store_descriptor(), prefix="hp-data-process-" + statuses[0][0])
            capsule = (source_bytes, bootstrap)
            with self._capsule.open("wb") as handle:
                pickle.dump((sys.path, capsule), handle)
        except BaseException:
            self.close()
            raise

    def _start(self) -> None:
        if self._closed or self._failed:
            raise RuntimeError("Data process is closed or failed.")
        if self._process is not None:
            return
        parent, child = socket.socketpair()
        try:
            self._process = self._resources.enter_context(subprocess.Popen(
                [sys.executable, "-c",
                 ("import sys\nfrom hyper_parallel.distributed_data.process_data import _worker_main\n"
                  "_worker_main(sys.argv[1], int(sys.argv[2]))\n"),
                 str(self._capsule), str(child.fileno())],
                pass_fds=(child.fileno(),), start_new_session=True, env=self._env,
            ))
            self._connection = Connection(parent.detach())
            status, _ = self._receive()
            if status != "READY":
                raise RuntimeError(f"Expected a ready data process, received {status}.")
        except BaseException:
            self._failed = True
            raise
        finally:
            parent.close()
            child.close()

    def _receive(self) -> tuple[str, Any]:
        deadline = time.monotonic() + _REPLY_TIMEOUT
        try:
            while not self._connection.poll(0.1):
                if self._process.poll() is not None:
                    raise RuntimeError(f"Data process exited with status {self._process.returncode}.")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Data process did not reply within the data timeout.")
            status, value = self._connection.recv()
            if status == "ERROR":
                raise RuntimeError(f"Data process failed:\n{value}")
            return status, value
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
        if self._process.poll() is not None:
            self._failed = True
            raise RuntimeError(f"Data process exited with status {self._process.returncode}.")
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
            if self._process is not None and self._process.poll() is None and not self._failed:
                if self._pending:
                    self._receive()
                self._connection.send(("CLOSE", None))
                self._receive()
                self._process.wait(timeout=30)
        finally:
            if self._process is not None:
                try:
                    os.killpg(self._process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(self._process.pid, signal.SIGKILL)
                    self._process.wait()
            if self._connection is not None:
                self._connection.close()
            self._resources.close()


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
