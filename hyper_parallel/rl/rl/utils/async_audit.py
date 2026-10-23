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
"""Opt-in, process-local evidence for asynchronous execution and Actor arithmetic."""
import hashlib
import json
import os
import socket
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Mapping

import torch

from hyper_parallel import SkipDTensorDispatch


def state_digest(value: Any) -> str:
    """Hash tensor values and nested runtime state without pickling device layouts."""
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        """Hash one nested value in deterministic traversal order."""
        if isinstance(item, torch.Tensor):
            tensor = getattr(item, "_local_tensor", item).detach().cpu().contiguous()
            digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, Mapping):
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr((type(item).__name__, item)).encode())

    visit(value)
    return digest.hexdigest()


class EventTrace:
    """Write complete JSON lines under a lock; processes never share a file."""

    def __init__(self, directory: str | None, role: str, rank: int = 0) -> None:
        self.directory = None if directory is None else Path(directory)
        self.role, self.rank = role, rank
        self.lock = threading.Lock()
        self.path = None
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.path = self.directory / f"{role}_rank_{rank}.jsonl"
            boot = Path("/proc/sys/kernel/random/boot_id")
            self.emit("clock", hostname=socket.gethostname(),
                      boot_id=boot.read_text(encoding="utf-8").strip() if boot.is_file() else None,
                      clock="CLOCK_MONOTONIC", schema=1)

    def emit(self, event: str, **values: Any) -> None:
        """Append one host-clock event; disabled tracing performs no IO."""
        if self.path is None:
            return
        with self.lock:
            record = {"event": event, "role": self.role, "rank": self.rank, "pid": os.getpid(),
                      "monotonic_ns": time.monotonic_ns(), "wall_time_ns": time.time_ns(), **values}
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()

    def probe_generation(self, base_url: str) -> None:
        """Record server token counters with request bounds on the same host clock."""
        if self.path is None:
            return
        started = time.monotonic_ns()
        try:
            with urllib.request.urlopen(base_url + "/metrics", timeout=2.) as response:
                payload = response.read().decode()
            finished = time.monotonic_ns()
            counters = {}
            for line in payload.splitlines():
                if not line or line.startswith("#"):
                    continue
                label, value = line.rsplit(" ", 1)
                name = label.split("{", 1)[0]
                if name in ("vllm:generation_tokens_total", "vllm:prompt_tokens_total", "vllm:num_requests_running"):
                    counters[name] = counters.get(name, 0.) + float(value)
            if counters:
                self.emit("generation_progress", request_start_ns=started, request_end_ns=finished,
                          counters=counters)
        except (OSError, ValueError) as error:
            self.emit("generation_probe_error", error=str(error))


class ActorLossAudit:
    """Observe the real Actor loss and backward hook without replacing its math."""

    def __init__(self, algorithm: Any, trace: EventTrace) -> None:
        self.algorithm, self.trace = algorithm, trace
        self.plan: list[tuple[int, int, int, int]] = []
        self.calls = 0
        self.completed = 0
        self.step = 0
        self.rollout_log_probs = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.algorithm, name)

    def begin(self, step: int, experience: Any, train: Mapping[str, Any], dp_size: int) -> None:
        """Persist immutable targets and the exact Actor micro/mini-batch plan."""
        self.step, self.calls, self.completed = step, 0, 0
        rows = experience.sequences.shape[0]
        mini, micro = train["response_mini_batch_size"], train["micro_batch_size"]
        self.plan = [(start, min(start + micro, end), mini_start, end)
                     for _ in range(train["policy_update_epochs"])
                     for mini_start in range(0, rows, mini)
                     for end in [min(mini_start + mini, rows)]
                     for start in range(mini_start, end, micro)]
        self.rollout_log_probs = experience.rollout_log_probs.detach().cpu()
        targets = {name: getattr(experience, name).detach().cpu() for name in (
            "old_log_probs", "advantages", "rewards", "loss_action_mask",
        )}
        targets.update(
            rollout_log_probs=self.rollout_log_probs,
            groups=[row.group_id for row in experience.trajectories], dp_size=dp_size, plan=self.plan,
            trajectory_ids=[row.trajectory_id for row in experience.trajectories],
            episode_ids=[getattr(row, "metadata", {}).get("episode_id", row.trajectory_id)
                         for row in experience.trajectories],
            dp_padding=[getattr(row, "metadata", {}).get("dp_padding", False) for row in experience.trajectories],
            policy_versions=[row.policy_version for row in experience.trajectories],
            behavior_segments=[getattr(row, "metadata", {}).get("behavior_segments")
                               for row in experience.trajectories],
        )
        torch.save(targets, self.trace.directory / f"batch_step_{step}_rank_{self.trace.rank}.pt")

    def compute_actor_loss(self, **kwargs: Any) -> Any:
        """Capture actual forward values and the gradient delivered by Actor.backward."""
        output = self.algorithm.compute_actor_loss(**kwargs)
        index = self.calls
        if index >= len(self.plan):
            raise RuntimeError("Actor loss calls exceed the recorded update plan")
        self.calls += 1
        start, end, mini_start, mini_end = self.plan[index]
        current = kwargs["current_log_probs"]
        record = {name: None if value is None else value.detach().cpu()
                  for name, value in kwargs.items()}
        record.update(
            start=start, end=end, mini_start=mini_start, mini_end=mini_end,
            rollout_log_probs=self.rollout_log_probs[start:end],
            ratio=(current.detach() - kwargs["old_log_probs"].detach()).exp().cpu(),
            total_loss_sum=output.total_loss_sum.detach().cpu(),
        )
        reference = kwargs["reference_log_probs"]
        if reference is not None:
            delta = reference.detach().float() - current.detach().float()
            reference_ratio = delta.exp()
            record["kl_native"] = {"reference_ratio": reference_ratio.cpu(),
                                   "raw_kl": (reference_ratio - delta - 1.0).cpu(),
                                   "source": "live forward observer"}
        destination = self.trace.directory / f"loss_step_{self.step}_rank_{self.trace.rank}_call_{index}.pt"

        def capture(gradient: torch.Tensor) -> None:
            """Save the gradient delivered by the real backward pass."""
            record["log_prob_gradient"] = gradient.detach().cpu()
            torch.save(record, destination)
            self.completed += 1

        current.register_hook(capture)
        return output

    def finish(self) -> None:
        """Reject incomplete evidence rather than reporting a partially audited update."""
        if self.calls != len(self.plan) or self.completed != self.calls:
            raise RuntimeError("Actor audit is missing forward/backward records")


def capture_runtime_state(roles: Any) -> dict[str, Any]:
    """Fingerprint the live model, optimizer, scheduler and both saved RNG states."""
    with SkipDTensorDispatch():
        result = {
            "model": state_digest(roles.model.state_dict()),
            "optimizer": state_digest(roles.optimizer.state_dict()),
            "scheduler": state_digest(None if roles.lr_scheduler is None else roles.lr_scheduler.state_dict()),
            "cpu_rng": state_digest(torch.get_rng_state()),
            "device_rng": state_digest(roles.device_handle.get_rng_state(roles.device)),
        }
    return result


def write_rollout_diagnostics(experience: Any, directory: str, step: int, dp_rank: int, response_limit: int) -> None:
    """Save consumed responses and actual group advantage signals, once per DP shard."""
    # Dataset initialization imports Actor/monitoring; defer this import until the runtime is constructed.
    from rl.dataset.episodes import generation_finish_reason  # pylint: disable=C0415

    nonzero = ((experience.advantages != 0) & experience.loss_action_mask.bool()).any(dim=-1).cpu().tolist()
    groups = {}
    for index, row in enumerate(experience.trajectories):
        if not row.metadata.get("dp_padding", False):
            groups.setdefault(row.group_id, []).append(index)
    records = []
    for identity, indices in groups.items():
        candidates = {experience.trajectories[index].metadata.get(
            "episode_id", experience.trajectories[index].trajectory_id): experience.trajectories[index].reward
                      for index in indices}
        rewards = list(candidates.values())
        constant = min(rewards) == max(rewards)
        group_signal = any(nonzero[index] for index in indices)
        for index in indices:
            row = experience.trajectories[index]
            infos = row.metadata.get("turn_infos", ())
            info = infos[-1] if infos else {}
            reason = generation_finish_reason(row)
            tokens = int(row.action_mask.sum())
            records.append({
                "step": step, "dp_rank": dp_rank, "group_id": identity, "trajectory_id": row.trajectory_id,
                "episode_id": row.metadata.get("episode_id", row.trajectory_id),
                "call_index": row.metadata.get("call_index", 0), "call_count": row.metadata.get("call_count", 1),
                "source_prompt_id": row.metadata.get("source_prompt_id", row.prompt_id),
                "prompt": "\n".join(turn.content for turn in row.turns if not turn.trainable),
                "response": "\n".join(turn.content for turn in row.turns if turn.trainable),
                "reward": row.reward, "reward_components": row.reward_components,
                "extracted_answer": info.get("extracted_answer"), "expected_answer": info.get("expected_answer"),
                "reward_scorer": info.get("reward_scorer"),
                "reward_diagnostics_available": "expected_answer" in info and info.get("reward_scorer") is not None,
                "finish_reason": reason, "response_tokens": tokens,
                "length_limit_hit": tokens >= response_limit, "generation_length_limited": reason == "length",
                "environment_truncated": row.truncated, "group_rewards": rewards,
                "group_zero_reward_variance": constant, "group_nonzero_advantage": group_signal,
                "response_nonzero_advantage": nonzero[index],
                "policy_version": row.policy_version, "behavior_segments": row.metadata.get("behavior_segments"),
            })
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"step_{step}_rank_{dp_rank}.jsonl"
    pending = path.with_suffix(".jsonl.tmp")
    pending.write_text("".join(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in records),
                       encoding="utf-8")
    pending.replace(path)


def record_async_state(trainer: Any, event: str, **metadata: Any) -> None:
    """Capture model/optimizer/RNG evidence only when the existing audit option is enabled."""
    if trainer.settings.audit_dir is not None:
        state = trainer.roles.run_synchronized("audit " + event, lambda: capture_runtime_state(trainer.roles))
        trainer.trace.emit(event, update=trainer.roles.state.global_step, digests=state, **metadata)


def record_async_rollout(trainer: Any, experience: Any, step: int) -> None:
    """Persist the same consumed calls on TP owners and synchronize any diagnostic IO failure."""
    def write_samples() -> None:
        """Keep one diagnostics file per DP owner."""
        if trainer.roles.parallel_dims.tp_rank == 0:
            root = trainer.settings.audit_dir or trainer.config["train"]["checkpoint"]["output_dir"]
            write_rollout_diagnostics(experience, str(Path(root).parent / "rollouts"), step,
                                      trainer.roles.parallel_dims.dp_rank, trainer.config["rollout"]["max_new_tokens"])

    trainer.roles.run_synchronized("async rollout diagnostics", write_samples)


def record_async_checkpoint(trace: EventTrace, directory: Path, snapshot: dict) -> None:
    """Report the durable commit; this helper does not save or mutate checkpoint state."""
    trace.emit("checkpoint_complete", update=snapshot["learner_update"], directory=str(directory), async_state={
        **{key: snapshot[key] for key in ("learner_update", "policy_version", "source_update", "optimizer_steps")},
        **{key: snapshot["buffer"][key] for key in ("next_identity", "consumed_groups")},
        "buffered_groups": len(snapshot["buffer"]["ready"]),
    })


def install_async_audit(roles: Any, trace: EventTrace, enabled: bool) -> ActorLossAudit | None:
    """Attach the existing opt-in backward observer without changing the Actor math."""
    if not enabled:
        return None
    observer = ActorLossAudit(roles.actor.algorithm, trace)
    roles.actor.algorithm = observer
    return observer


def begin_async_update(trainer: Any, step: int, experience: Any) -> None:
    """Record fixed targets and exact optimizer start after synchronized preparation."""
    if trainer.loss_audit is not None:
        trainer.roles.run_synchronized("async audit targets", lambda: trainer.loss_audit.begin(
            step, experience, trainer.config["train"], trainer.roles.parallel_dims.dp_size))
    trainer.trace.emit("update_start", update=step,
                       groups=sorted({row.group_id for row in experience.trajectories}),
                       behavior_versions=sorted({segment["policy_version"]
                           for row in experience.trajectories
                           for segment in row.metadata.get("behavior_segments", (
                               {"policy_version": row.policy_version},))}))


def finish_async_update(trainer: Any) -> None:
    """Check every observed backward before reporting a successful optimizer update."""
    if trainer.loss_audit is not None:
        trainer.roles.run_synchronized("async audit completion", trainer.loss_audit.finish)


def record_async_selection(trace: EventTrace, groups: tuple[Any, ...], update: int) -> None:
    """Expose selected identities and actual behavior ages without changing the selection."""
    trace.emit("batch_selected", update=update,
               groups=[{"identity": group.task.identity, "policy_version": group.task.policy_version,
                        "source_update": group.task.source_update,
                        "behavior_source_updates": list(group.source_updates)} for group in groups])


def record_publication_timing(trainer: Any, started: float, admitted_at: float) -> None:
    """Keep publication timing metrics separate from the commit/admission transaction."""
    trainer.roles.tracker.log({"async/publication_seconds": time.monotonic() - started,
                              "async/publication_wait_seconds": admitted_at - started,
                              "async/publication_transfer_seconds": time.monotonic() - admitted_at},
                             step=trainer.roles.state.global_step)
