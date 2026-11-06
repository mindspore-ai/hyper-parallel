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
"""Coordinator-side complete-group sampling; reuse the existing CPU producer."""
import time
from dataclasses import asdict
from typing import Any, Mapping, Optional

from rl.async_runtime.producer import AsyncRuntime
from rl.async_runtime.protocol import ExperienceBuffer
from rl.config.schema import AsyncTrainingConfig
from rl.utils.async_audit import EventTrace


class ColocateAsyncRollouter:
    """Own sampling credits on rank zero, never models or training collectives.

    The spawned producer is the existing FullyAsyncRollouter. This object only
    coordinates its data stream and partial-generation admission.
    """

    def __init__(self, config: Mapping[str, Any], settings: AsyncTrainingConfig,
                 restored: Optional[dict] = None) -> None:
        """Restore bounded credits and start the existing CPU sampling transport."""
        self.settings = settings
        self.candidates = config["rollout"]["num_return_sequences"]
        self.buffer = ExperienceBuffer(settings, self.candidates)
        self.trace = EventTrace(settings.audit_dir, "colocate_sampler")
        if restored is not None:
            self.buffer.load_state_dict(restored["buffer"])
        producer_config = {key: value for key, value in config.items() if key != "colocate_async"}
        producer_config["async_training"] = asdict(settings)
        self.runtime = AsyncRuntime(producer_config, self.buffer, None if restored is None else restored["source"])

    def sample_batch(self, version: int, update: int, count: int) -> tuple[bytes, ...]:
        """Fill the bounded window and lease complete groups without releasing credits."""
        if version != update + 1 or count <= 0 or count > self.settings.max_buffer_groups:
            raise ValueError("Invalid colocated batch size or published version")
        deadline = time.monotonic() + self.settings.operation_timeout
        while True:
            self.runtime.pump()
            self.buffer.evict_stale(update)
            self.runtime.fill(version, update)
            # fill also pumps transport; those newly accepted rows need the same age gate.
            self.buffer.evict_stale(update)
            if len(self.buffer.ready) >= count:
                groups = self.buffer.select(count)
                self.trace.emit("batch_selected", update=update + 1,
                                identities=[group.task.identity for group in groups])
                return tuple(self.buffer.ready[key] for key in self.buffer.selected)
            if time.monotonic() >= deadline:
                raise TimeoutError("No complete colocated training batch available")
            time.sleep(0.01)

    def commit_batch(self) -> None:
        """Consume a lease only after all training ranks completed their update."""
        self.buffer.commit()
        self.trace.emit("batch_committed", consumed_groups=self.buffer.consumed_groups)

    def reject_batch_groups(self, identities: tuple[str, ...]) -> None:
        """Drop rejected complete groups and return all other leases to the ready pool."""
        rejected = {int(identity) for identity in identities}
        if not rejected or not rejected.issubset(self.buffer.selected):
            raise ValueError("Rejected groups must belong to the selected batch")
        for identity in rejected:
            del self.buffer.ready[identity]
        self.buffer.dropped_groups += len(rejected)
        self.buffer.selected = ()
        self.trace.emit("off_policy_groups_rejected", identities=sorted(rejected))

    def pause(self) -> None:
        """Close admission and settle aborted segments before the trainer sleeps vLLM."""
        self.runtime.pause()
        self.trace.emit("partial_pause_done")

    def resume(self, version: int) -> None:
        """Open admission after the server verifies the committed worker version."""
        self.runtime.resume(version)
        self.trace.emit("partial_resume_done", policy_version=version)

    def drain_and_snapshot(self) -> dict:
        """Freeze complete experience and source while generation is still available."""
        if self.buffer.selected:
            raise RuntimeError("Cannot checkpoint an uncommitted colocated update")
        self.runtime.drain()
        source = self.runtime.snapshot()
        return {"buffer": self.buffer.state_dict(), "source": source}

    def metrics(self) -> dict[str, int]:
        """Return counters without advancing transport or exposing mutable state."""
        return {"colocate/ready_groups": len(self.buffer.ready),
                "colocate/inflight_groups": len(self.buffer.reserved),
                "colocate/consumed_groups": self.buffer.consumed_groups,
                "colocate/dropped_groups": self.buffer.dropped_groups}

    def close(self, graceful: bool = False) -> None:
        """Delegate bounded producer/transport cleanup to its existing owner."""
        self.runtime.close(graceful=graceful)
