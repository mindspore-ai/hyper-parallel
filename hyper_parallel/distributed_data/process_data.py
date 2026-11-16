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
"""One CPU producer process with ordered parent-owned data communication."""

from __future__ import annotations

import os
import pickle
import random
import signal
import socket
import subprocess
import sys
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack, nullcontext
from multiprocessing.connection import Connection
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

import cloudpickle
import numpy as np
import torch  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed_data.shared_batch import SharedBatchReader, SharedBatchWriter
from hyper_parallel.distributed_data.transport import _decode_received_payloads, _encode_payload_segment


_REPLY_TIMEOUT = 300


class ProducerTransport:
    """CPU codec whose collectives are executed by the training process."""

    def __init__(self, connection: Connection, ranks: tuple[int, ...], planner_rank: int,
                 shared_directory: str) -> None:
        """Bind an ordered channel without creating a device or process group."""
        self.ranks = ranks
        self.planner_rank = planner_rank
        self._connection = connection
        self._writer = SharedBatchWriter(shared_directory)
        self._reader = SharedBatchReader()

    def _call(self, operation: str, value: Any) -> Any:
        self._connection.send(("CALL", (operation, self._writer.write(value))))
        if not self._connection.poll(_REPLY_TIMEOUT):
            raise TimeoutError("Training process did not finish the producer's data collective.")
        status, message = self._connection.recv()
        if status != "RESULT":
            raise RuntimeError(f"Data collective failed: {message}")
        return self._reader.read(message)

    def all_gather_object(self, value: Any) -> tuple[Any, ...]:
        """Encode and decode metadata exclusively in the CPU producer."""
        if len(self.ranks) == 1:
            return (value,)
        encoded = torch.frombuffer(bytearray(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)), dtype=torch.uint8)
        payload, sizes = self._call("gather", encoded)
        rows = payload.view(len(sizes), -1).numpy()
        return tuple(pickle.loads(row[:size].tobytes()) for row, size in zip(rows, sizes))

    def broadcast_from_planner(self, value: Any) -> Any:
        """Broadcast planner output through the parent without parent-side decoding."""
        if len(self.ranks) == 1:
            return value
        encoded = torch.frombuffer(bytearray(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)), dtype=torch.uint8)
        return pickle.loads(self._call("broadcast", encoded).numpy().tobytes())

    def exchange(self, outgoing: dict[int, Any]) -> dict[Any, Any]:
        """Serialize raw samples in the producer and delegate the byte A2A."""
        unexpected = outgoing.keys() - self.ranks
        if unexpected:
            raise ValueError(f"Payload routes target ranks outside the data plane: {sorted(unexpected)}.")
        segments = [_encode_payload_segment(outgoing.get(rank, ())) for rank in self.ranks]
        sizes = tuple(map(len, segments))
        storage = bytearray(b"".join(segments))
        encoded = torch.frombuffer(storage, dtype=torch.uint8) if storage else torch.empty(0, dtype=torch.uint8)
        if len(self.ranks) == 1:
            return _decode_received_payloads(bytes(storage), sizes)
        received, splits = self._call("exchange", (encoded, sizes))
        return _decode_received_payloads(received.numpy().tobytes(), splits)

    def close(self) -> None:
        """Release the producer's own shared storage."""
        self._reader.close()
        self._writer.close()


def _cpu_rng_state() -> tuple[Any, Any, torch.Tensor]:
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state()


def _restore_cpu_rng(state: tuple[Any, Any, torch.Tensor]) -> None:
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.random.set_rng_state(state[2])


def _producer_main(capsule_path: str, descriptor: int, shared_directory: str) -> None:
    """Run a source factory in a fresh interpreter, never inheriting device state."""
    connection = Connection(descriptor)
    cleanup = ExitStack()
    cleanup.callback(connection.close)
    writer = SharedBatchWriter(shared_directory)
    cleanup.callback(writer.close)
    try:
        torch.set_num_threads(1)
        with open(capsule_path, "rb") as handle:
            factory, ranks, planner_rank, rng_state = cloudpickle.load(handle)
        _restore_cpu_rng(rng_state)
        transport = ProducerTransport(connection, ranks, planner_rank, shared_directory)
        cleanup.callback(transport.close)
        producer = factory(transport)
        cleanup.callback(producer.close)
        connection.send(("READY", os.getpid()))
        while True:
            command, value = connection.recv()
            if command == "CLOSE":
                connection.send(("CLOSED", None))
                return
            if command == "NEXT":
                try:
                    result = next(producer)
                except StopIteration:
                    connection.send(("EOF", None))
                else:
                    if any(getattr(torch, name).is_initialized() for name in ("cuda", "npu")
                           if hasattr(torch, name)):
                        raise RuntimeError("CPU data producers must not initialize an accelerator runtime.")
                    connection.send(("BATCH", writer.write(result)))
            elif command == "SAVE":
                consumed, path = value
                torch.save(producer.state_dict(consumed), path)
                connection.send(("SAVED", None))
            elif command == "LOAD":
                producer.load_state_dict(torch.load(value, map_location="cpu", weights_only=False))
                connection.send(("LOADED", None))
            elif command == "EPOCH":
                producer.set_epoch(value)
                connection.send(("EPOCH", None))
            else:
                raise ValueError(f"Unknown producer command: {command}.")
    except BaseException:  # Send the full child traceback before propagating failure to its exit status.
        try:
            connection.send(("ERROR", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        cleanup.close()


class ProcessDataProducer:
    """Own one CPU process and one lightweight communication/staging thread.

    A single pending request bounds speculative source progress. Tensor bytes
    cross the process boundary through shared storage; only descriptors and
    small control values use the pipe. Full source state moves only on save/load.
    """

    def __init__(self, factory: Callable[..., Any], transport: Any, stage: Callable[..., Any] | None = None) -> None:
        """Serialize an unstarted CPU source before any background work begins."""
        self._resources = ExitStack()
        self._directory = self._resources.enter_context(TemporaryDirectory(prefix="hp-producer-"))
        # Parent ownership also reclaims child mappings after an uncatchable exit.
        self._shared_directory = self._resources.enter_context(
            TemporaryDirectory(prefix="hp-producer-", dir="/dev/shm"))
        self._transport = transport
        self._stage = stage
        self._device = transport.communication_device
        self._stream = None
        self._process = None
        self._connection = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hp-data-communication")
        self._future: Future | None = None
        self._writer = SharedBatchWriter(self._shared_directory)
        self._reader = SharedBatchReader()
        self._closed = False
        self._failed = False
        self._capsule = Path(self._directory) / "source.pkl"
        try:
            with self._capsule.open("wb") as handle:
                cloudpickle.dump((factory, transport.ranks, transport.planner_rank, _cpu_rng_state()), handle)
        except BaseException:
            self.close()
            raise

    def _start(self) -> None:
        if self._closed:
            raise RuntimeError("CPU producer is closed.")
        if self._process is not None:
            return
        parent, child = socket.socketpair()
        try:
            self._process = self._resources.enter_context(subprocess.Popen(
                [sys.executable, "-c",
                 "import sys\nfrom hyper_parallel.distributed_data.process_data import _producer_main\n"
                 "_producer_main(sys.argv[1], int(sys.argv[2]), sys.argv[3])\n",
                 str(self._capsule), str(child.fileno()), self._shared_directory],
                pass_fds=(child.fileno(),), start_new_session=True,
                env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"),
            ))
            self._connection = Connection(parent.detach())
            status, _ = self._receive()
            if status != "READY":
                raise RuntimeError(f"CPU producer did not start: {status}.")
        finally:
            child.close()
            parent.close()

    def _receive(self) -> tuple[str, Any]:
        deadline = time.monotonic() + _REPLY_TIMEOUT
        while not self._connection.poll(0.1):
            if self._process.poll() is not None:
                raise RuntimeError(f"CPU producer exited with status {self._process.returncode}.")
            if time.monotonic() >= deadline:
                raise TimeoutError("CPU producer did not reply within the data timeout.")
        try:
            status, value = self._connection.recv()
        except (EOFError, OSError) as error:
            raise RuntimeError("CPU producer disconnected before returning its result.") from error
        if status == "ERROR":
            self._failed = True
            raise RuntimeError(f"CPU producer failed:\n{value}")
        return status, value

    def _stream_context(self) -> Any:
        if self._device is None or self._device.type == "cpu":
            return nullcontext()
        accelerator = getattr(torch, self._device.type)
        accelerator.set_device(self._device)
        if self._stream is None:
            self._stream = accelerator.Stream(device=self._device)
        return accelerator.stream(self._stream)

    def _request(self, command: str, value: Any = None) -> tuple[str, Any]:
        self._connection.send((command, value))
        while True:
            status, message = self._receive()
            if status != "CALL":
                break
            operation, descriptor = message
            try:
                request = self._reader.read(descriptor)
                with self._stream_context():
                    response = self._transport.exchange_cpu_request(operation, request)
                self._connection.send(("RESULT", self._writer.write(response)))
            except BaseException:
                self._failed = True
                self._connection.send(("ERROR", traceback.format_exc()))
                raise
        if status == "BATCH":
            if self._stage is not None:
                getattr(torch, self._stage.device.type).set_device(self._stage.device)
            batch, stats = self._reader.read(message, pin_memory=self._stage is not None)
            message = (self._stage(batch) if self._stage is not None else batch, stats)
        return status, message

    def prefetch(self) -> None:
        """Start at most one speculative step, including device communication and H2D."""
        if self._future is None:
            self._start()
            self._future = self._executor.submit(self._request, "NEXT")

    def wait(self) -> tuple[str, Any] | None:
        """Finish the pending step without consuming or discarding its result."""
        try:
            return self._future.result(timeout=_REPLY_TIMEOUT) if self._future is not None else None
        except BaseException:
            self._failed = True
            raise

    def take(self) -> Any:
        """Consume the pending result after all communication has completed."""
        self.prefetch()
        status, result = self.wait()
        self._future = None
        if status == "EOF":
            raise StopIteration
        if status != "BATCH":
            raise RuntimeError(f"Expected a producer batch, received {status}.")
        return result

    def state_dict(self, consumed: int) -> dict[str, Any]:
        """Capture progress at the delivered cursor, including one speculative batch."""
        self._start()
        self.wait()
        path = str(Path(self._directory) / "checkpoint.pt")
        self._executor.submit(self._request, "SAVE", (consumed, path)).result(timeout=_REPLY_TIMEOUT)
        return torch.load(path, map_location="cpu", weights_only=False)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore only after draining the previous pending request."""
        self._start()
        self.wait()
        self._future = None
        path = str(Path(self._directory) / "restore.pt")
        torch.save(state, path)
        self._executor.submit(self._request, "LOAD", path).result(timeout=_REPLY_TIMEOUT)

    def set_epoch(self, epoch: int) -> None:
        """Discard speculative work before resetting the producer's source epoch."""
        self._start()
        self.wait()
        self._future = None
        self._executor.submit(self._request, "EPOCH", epoch).result(timeout=_REPLY_TIMEOUT)

    def close(self) -> None:
        """Drain outstanding work and stop only this producer and its own workers."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._process is not None and self._process.poll() is None and not self._failed:
                self.wait()
                self._executor.submit(self._request, "CLOSE").result(timeout=_REPLY_TIMEOUT)
                if self._process.wait(timeout=30):
                    raise RuntimeError(f"CPU producer cleanup failed with status {self._process.returncode}.")
        finally:
            if self._process is not None:
                # Source DataLoader workers may outlive an abruptly killed producer.
                try:
                    os.killpg(self._process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(self._process.pid, signal.SIGKILL)
                    self._process.wait()
            try:
                self._executor.shutdown(wait=True)
                if self._stream is not None:
                    self._stream.synchronize()
            finally:
                if self._connection is not None:
                    self._connection.close()
                self._reader.close()
                self._writer.close()
                self._resources.close()
