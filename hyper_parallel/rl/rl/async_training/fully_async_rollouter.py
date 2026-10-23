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
"""Independent CPU producer; only complete, behavior-versioned GRPO groups cross IPC."""
import concurrent.futures
import os
import multiprocessing as mp
import queue
import time
import traceback
from dataclasses import replace
from typing import Any, Mapping

from rl.agentic.envs.environment import load_agentic_module
from rl.utils.async_audit import EventTrace
from rl.async_training.protocol import AsyncTrainingConfig
from rl.async_training.protocol import SampleGroup, SamplingTask, ExperienceBuffer
from rl.roles.model_setup import load_tokenizer
from rl.dataset.data_source import StatefulPromptSource
from rl.roles.rollout.client import AsyncGenerationClient
from rl.roles.rollout.collection import CandidateCollector


class FullyAsyncRollouter:
    """Own the prompt iterator and independent task sessions in a spawned process."""

    def __init__(self, config: Mapping[str, Any], connection: Any, results: Any, source: dict | None) -> None:
        self.config = config
        self.settings = AsyncTrainingConfig.from_mapping(config["async_training"])
        self.trace = EventTrace(self.settings.audit_dir, "rollouter")
        self.connection, self.results = connection, results
        self.candidates = config["rollout"]["num_return_sequences"]
        self.source = StatefulPromptSource(config["data"], load_tokenizer(config["model"]),
                                           config["train"]["seed"], source)
        if config["agentic"].get("runner", "internal") == "internal":
            load_agentic_module(config["agentic"]["module_path"])
        self.client = AsyncGenerationClient.from_config(config, self.settings.operation_timeout)
        self.groups: dict[int, tuple[SamplingTask, float, list]] = {}
        self.collector = CandidateCollector(config, self.client)
        self.next_probe = 0.0

    def _next_prompt(self, task: SamplingTask) -> Any:
        record = self.source.next_record()
        self.trace.emit("source_read", identity=task.identity, source_prompt_id=str(record.prompt_id),
                        epoch=self.source.epoch)
        return replace(record, prompt_id=str(task.identity),
                       metadata={**record.metadata, "source_prompt_id": record.prompt_id})

    def _generate_candidate(self, task: SamplingTask, prompt: Any, candidate: int) -> tuple:
        """Retry infrastructure failures while preserving the candidate's episode identity."""
        self.trace.emit("episode_start", identity=task.identity, candidate=candidate,
                        episode_id=f"{task.identity}:{candidate}")
        for attempt in range(self.settings.max_candidate_retries + 1):
            try:
                rows = self.collector.generate(task, prompt, candidate)
                break
            except (RuntimeError, OSError) as error:
                self.trace.emit("episode_failure", identity=task.identity, candidate=candidate,
                                attempt=attempt, error=str(error))
                if self.client.closed or attempt == self.settings.max_candidate_retries:
                    self.trace.emit("episode_end", identity=task.identity, candidate=candidate,
                                    episode_id=f"{task.identity}:{candidate}", failed=True, calls=0)
                    raise
        self.trace.emit("episode_end", identity=task.identity, candidate=candidate,
                        episode_id=f"{task.identity}:{candidate}", calls=len(rows),
                        policy_versions=[row.policy_version for row in rows])
        return rows

    def _accept_group(self, identity: int) -> None:
        """Complete all candidate cleanup before transferring or rejecting one group."""
        task, started, futures = self.groups[identity]
        if not all(future.done() for future in futures):
            return
        failures = [future.exception() for future in futures if future.exception() is not None]
        for error in failures:
            if not isinstance(error, (RuntimeError, OSError)):
                raise error
        self.trace.emit("sampling_end", identity=identity, policy_version=task.policy_version,
                        source_update=task.source_update, candidates=self.candidates, discarded=bool(failures))
        if failures:
            self.trace.emit("group_discarded", identity=identity, reason="infrastructure",
                            errors=[str(error) for error in failures])
            self.connection.send(("discarded", task))
        else:
            finished = time.monotonic()
            rows = tuple(row for future in futures for row in future.result())
            payload = SampleGroup(task, rows, finished - started, started, finished).encode(
                self.candidates, self.settings.max_group_bytes)
            self.results.put(payload, timeout=self.settings.operation_timeout)
        del self.groups[identity]

    def run(self) -> None:
        """Run task admission and completion handling, keeping control separate from data."""
        self.connection.send(("ready", None))
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.settings.max_inflight_groups * self.candidates,
        )
        try:
            while True:
                if self.connection.poll(0.02):
                    command, value = self.connection.recv()
                    if command == "cancel":
                        break
                    if command == "stop":
                        if self.groups:
                            raise RuntimeError("Cannot stop with in-flight groups")
                        break
                    if command == "task":
                        if len(self.groups) >= self.settings.max_inflight_groups:
                            raise RuntimeError("Invalid sampling admission")
                        prompt = self._next_prompt(value)
                        started = time.monotonic()
                        self.trace.emit("sampling_start", identity=value.identity, policy_version=value.policy_version,
                                        source_update=value.source_update)
                        self.groups[value.identity] = (value, started, [
                            executor.submit(self._generate_candidate, value, prompt, index)
                            for index in range(self.candidates)])
                    elif command == "snapshot":
                        if self.groups:
                            raise RuntimeError("Source snapshot requires quiescence")
                        self.connection.send(("snapshot", self.source.state_dict()))
                    elif command != "task":
                        raise ValueError(f"Unknown sampler command: {command}")
                for identity in list(self.groups):
                    self._accept_group(identity)
                if self.trace.path is not None and time.monotonic() >= self.next_probe:
                    vllm = self.config["rollout"]["vllm"]
                    self.trace.probe_generation(f"http://{vllm.get('host', '127.0.0.1')}:{vllm['port']}")
                    self.next_probe = time.monotonic() + 2.
        finally:
            self.client.close()
            executor.shutdown(wait=True, cancel_futures=True)
            self.collector.close()


def run_rollouter(config: Mapping[str, Any], connection: Any, results: Any, source: dict | None) -> None:
    """Spawn entry: clear inherited launcher identity and report errors over the control pipe."""
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        os.environ.pop(key, None)
    try:
        FullyAsyncRollouter(config, connection, results, source).run()
    except BaseException:
        connection.send(("error", traceback.format_exc()))
        raise
    finally:
        connection.close()


class AsyncRuntime:
    """Connect a single buffer owner to a CPU-only generation producer."""

    def __init__(
        self, config: Mapping[str, Any], buffer: ExperienceBuffer, source: dict | None = None,
        target: Any = run_rollouter, control_client: Any = None,
    ) -> None:
        self.settings = buffer.settings
        self.buffer = buffer
        self.trace = EventTrace(self.settings.audit_dir, "coordinator")
        self.client = control_client
        self.codex = config.get("agentic", {}).get("runner") == "codex"
        if self.client is None:
            self.client = AsyncGenerationClient.from_config(config, self.settings.operation_timeout)
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.results = context.Queue(maxsize=self.settings.max_buffer_groups)
        self.process = context.Process(target=target, args=(config, child, self.results, source),
                                       name="HyperRL-Rollouter")
        self.messages: dict[str, Any] = {}
        self.closed = False
        self.process.start()
        child.close()
        try:
            self.wait("ready")
        except BaseException:
            self.close()
            raise

    def pump(self) -> None:
        """Receive data and control without queue-size or empty polling assumptions."""
        while True:
            try:
                self.buffer.accept(self.results.get_nowait())
            except queue.Empty:
                break
        while self.connection.poll():
            try:
                message = self.connection.recv()
            except EOFError as error:
                raise RuntimeError("Sampler control channel closed unexpectedly") from error
            if message[0] == "discarded":
                self.buffer.discard(message[1])
                self.trace.emit("group_discarded", identity=message[1].identity, reason="infrastructure")
                continue
            if message[0] == "error":
                raise RuntimeError(f"Sampler failed:\n{message[1]}")
            self.messages[message[0]] = message[1]
        if not self.process.is_alive():
            raise RuntimeError(f"Sampler exited unexpectedly: {self.process.exitcode}")

    def wait(self, kind: str | None = None) -> Any:
        """Pump until an acknowledgement arrives, or all result credits return."""
        deadline = time.monotonic() + self.settings.operation_timeout
        while True:
            self.pump()
            if kind is None and not self.buffer.reserved:
                return None
            if kind in self.messages:
                return self.messages.pop(kind)
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Sampler did not acknowledge {kind}")
            time.sleep(0.01)

    def fill(self, version: int, source_update: int) -> None:
        """Grant only pre-reserved tasks; one owner allocates monotonic identities."""
        self.pump()
        while True:
            task = self.buffer.reserve(version, source_update)
            if task is None:
                break
            self.trace.emit("admit", identity=task.identity, policy_version=version, source_update=source_update)
            self.connection.send(("task", task))

    def drain(self) -> None:
        """Stop admission and collect all accepted results before returning."""
        self.wait()

    def pause(self) -> None:
        """Close server admission at model-call boundaries; tool programs keep running."""
        self.client.pause(drain=self.codex)

    def resume(self, version: int) -> None:
        """Reopen inference only after the server verifies every worker's version."""
        self.client.resume_sampling(version)

    def snapshot(self) -> dict:
        """Request a source cursor only after every reserved result was delivered."""
        self.connection.send(("snapshot", None))
        return self.wait("snapshot")

    def close(self, graceful: bool = False) -> None:
        """Stop normally after drain, or terminate a failed job with bounded joins."""
        if self.closed:
            return
        self.closed = True
        if self.process.is_alive():
            try:
                self.connection.send(("stop" if graceful else "cancel", None))
            except (BrokenPipeError, EOFError, OSError):
                # The child may exit after reporting the primary error.
                pass
            self.process.join(timeout=5 if graceful else min(30., self.settings.operation_timeout))
        for stop in (self.process.terminate, self.process.kill):
            if self.process.is_alive():
                stop()
                self.process.join(timeout=5)
        self.connection.close()
        self.results.cancel_join_thread()
        self.results.close()
        self.client.close()
