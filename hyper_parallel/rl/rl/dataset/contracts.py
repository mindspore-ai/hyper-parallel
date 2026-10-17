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
"""Token-first dataset and training contracts for the Hyper-RL runtime."""
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import torch


@dataclass(frozen=True)
class RouterTrace:
    """CPU expert choices at unshifted input positions, including prompt tokens."""

    expert_ids: torch.Tensor
    valid_mask: torch.Tensor
    layer_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        """Reject malformed routes before batching or distributed model execution."""
        if not isinstance(self.expert_ids, torch.Tensor) or self.expert_ids.ndim != 3:
            raise ValueError("router expert_ids must have shape [sequence, layers, top_k]")
        if self.expert_ids.device.type != "cpu" or self.expert_ids.dtype not in (
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
        ):
            raise ValueError("router expert_ids must be a CPU integer tensor")
        if not isinstance(self.valid_mask, torch.Tensor) or self.valid_mask.device.type != "cpu":
            raise ValueError("router valid_mask must be a CPU boolean tensor")
        if self.valid_mask.dtype != torch.bool or tuple(self.valid_mask.shape) != (self.expert_ids.shape[0],):
            raise ValueError("router valid_mask must align with input positions")
        if (not isinstance(self.layer_ids, tuple) or not self.layer_ids
                or any(not isinstance(layer, int) or isinstance(layer, bool) or layer < 0 for layer in self.layer_ids)
                or tuple(sorted(set(self.layer_ids))) != self.layer_ids
                or len(self.layer_ids) != self.expert_ids.shape[1] or self.expert_ids.shape[2] == 0):
            raise ValueError("router layer_ids must be ordered unique non-negative IDs matching the layer dimension")
        self._validate_expert_ids()

    def _validate_expert_ids(self) -> None:
        """Require non-negative, distinct experts at every valid input position."""
        # Torch sort also allocates int64 indices; bound its temporary storage for long prompts.
        for start in range(0, self.expert_ids.shape[0], 4096):
            selected = self.expert_ids[start:start + 4096][self.valid_mask[start:start + 4096]]
            if bool((selected < 0).any()):
                raise ValueError("router expert IDs must be non-negative at valid positions")
            if selected.shape[-1] > 1 and bool((selected.sort(dim=-1).values.diff(dim=-1) == 0).any()):
                raise ValueError("router top-k expert IDs must be unique at valid positions")


def resize_router_trace(trace: RouterTrace, token_count: int) -> RouterTrace:
    """Crop generated tokens or append unrouted terminal observation positions."""
    if not isinstance(token_count, int) or isinstance(token_count, bool) or token_count < 0:
        raise ValueError("router trace target token count must be a non-negative integer")
    current = trace.expert_ids.shape[0]
    if token_count == current:
        return trace
    if token_count < current:
        return RouterTrace(trace.expert_ids[:token_count], trace.valid_mask[:token_count], trace.layer_ids)
    return RouterTrace(torch.cat((trace.expert_ids, trace.expert_ids.new_zeros(
        (token_count - current, *trace.expert_ids.shape[1:])))),
        torch.cat((trace.valid_mask, trace.valid_mask.new_zeros(token_count - current))), trace.layer_ids)


@dataclass(frozen=True)
class Message:
    """One conversational message before tokenization."""
    role: Literal["system", "user", "assistant", "tool", "environment"]
    content: str


@dataclass(frozen=True)
class PromptRecord:
    """Stable input record understood by rollout and environments."""
    prompt_id: str
    messages: tuple[Message, ...]
    ground_truth: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Turn:
    """One token-aligned observation or action in a trajectory."""
    role: Literal["system", "user", "assistant", "tool", "environment"]
    content: str
    token_start: int
    token_end: int
    trainable: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Trajectory:
    """Canonical output of both one-shot and multi-turn rollout."""
    trajectory_id: str
    prompt_id: str
    group_id: Optional[str]
    policy_version: int
    turns: tuple[Turn, ...]
    token_ids: Any
    attention_mask: Any
    action_mask: Any
    rollout_log_probs: Optional[Any]
    reward: float
    reward_components: dict[str, float]
    done: bool
    truncated: bool
    terminal_reason: str
    metadata: dict[str, Any] = field(default_factory=dict)
    worker_policy_version: Optional[int] = None
    router_trace: Optional[RouterTrace] = None

    def __post_init__(self) -> None:
        """Fail early when token-aligned fields drift apart."""
        if self.policy_version < 0:
            raise ValueError("trajectory policy_version must be non-negative")
        if self.worker_policy_version is not None and self.worker_policy_version != self.policy_version:
            raise ValueError(
                "trajectory worker policy version must match its requested policy version"
            )
        token_count = int(self.token_ids.numel())
        if int(self.attention_mask.numel()) != token_count:
            raise ValueError("trajectory attention_mask must align with token_ids")
        if int(self.action_mask.numel()) != token_count:
            raise ValueError("trajectory action_mask must align with token_ids")
        if self.router_trace is not None:
            if not isinstance(self.router_trace, RouterTrace) or self.router_trace.expert_ids.shape[0] != token_count:
                raise ValueError("trajectory router_trace must align with token_ids")
            if bool((self.router_trace.valid_mask & ~self.attention_mask.detach().cpu().bool()).any()):
                raise ValueError("trajectory router_trace must not select padding tokens")
        if bool((self.action_mask.bool() & ~self.attention_mask.bool()).any().item()):
            raise ValueError("trajectory action_mask must not select padding tokens")
        if self.rollout_log_probs is not None and int(self.rollout_log_probs.numel()) != token_count - 1:
            raise ValueError(
                "trajectory rollout_log_probs must align with next-token positions"
            )


@dataclass(frozen=True)
class ExperienceBatch:
    """Padded tensor batch consumed by an algorithm and the actor worker.

    ``action_mask`` spans the complete sequence. Optional algorithm-specific
    fields are populated only when their requirements request them.
    """
    trajectories: tuple[Trajectory, ...]
    sequences: Any
    attention_mask: Any
    action_mask: Any
    rewards: Any
    old_log_probs: Optional[Any]
    responses: tuple[str, ...]
    generation_seconds: float
    advantages: Optional[Any] = None
    returns: Optional[Any] = None
    values: Optional[Any] = None
    reference_log_probs: Optional[Any] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    worker_policy_version: Optional[int] = None
    bootstrap_values: Optional[Any] = None
    teacher_log_probs: Optional[Any] = None
    teacher_keys: Optional[tuple[str, ...]] = None
    rollout_log_probs: Optional[Any] = None
    router_traces: Optional[tuple[Optional[RouterTrace], ...]] = None

    def __post_init__(self) -> None:
        """Validate the shared token dimensions before algorithm code runs."""
        if self.sequences.ndim != 2:
            raise ValueError("experience sequences must be rank two")
        sequence_shape = tuple(self.sequences.shape)
        for name in ("attention_mask", "action_mask"):
            if tuple(getattr(self, name).shape) != sequence_shape:
                raise ValueError(f"experience {name} must align with sequences")
        self._validate_router_traces(sequence_shape)
        if bool((self.action_mask.bool() & ~self.attention_mask.bool()).any().item()):
            raise ValueError("experience action_mask must not select padding tokens")
        if tuple(self.rewards.shape) != (self.sequences.shape[0],):
            raise ValueError("experience rewards must contain one value per sequence")
        if self.bootstrap_values is not None and tuple(self.bootstrap_values.shape) != (self.sequences.shape[0],):
            raise ValueError("experience bootstrap_values must contain one value per sequence")
        log_prob_shape = (self.sequences.shape[0], self.sequences.shape[1] - 1)
        if self.old_log_probs is not None and tuple(self.old_log_probs.shape) != log_prob_shape:
            raise ValueError(
                "experience old_log_probs must align with next-token positions"
            )
        self._validate_rollout_scores(log_prob_shape)
        for field_name in ("advantages", "returns", "values", "reference_log_probs"):
            value = getattr(self, field_name)
            if value is not None and tuple(value.shape) != log_prob_shape:
                raise ValueError(
                    f"experience {field_name} must align with next-token positions"
                )
        self._validate_teacher_scores(log_prob_shape)
        if bool(self.action_mask[:, 0].any().item()):
            raise ValueError("the first sequence token cannot be a trainable action")
        if self.trajectories:
            trajectory_versions = {
                trajectory.worker_policy_version for trajectory in self.trajectories
            }
            if trajectory_versions != {self.worker_policy_version}:
                raise ValueError(
                    "experience worker policy version must match every trajectory"
                )

    def _validate_router_traces(self, sequence_shape: tuple[int, int]) -> None:
        """Align optional router targets with each trajectory and its unpadded inputs."""
        if self.router_traces is not None:
            if not isinstance(self.router_traces, tuple) or len(self.router_traces) != sequence_shape[0]:
                raise ValueError("experience router_traces must align with sequence rows")
            for row, trace in enumerate(self.router_traces):
                if trace is None:
                    continue
                if not isinstance(trace, RouterTrace):
                    raise ValueError("experience router_traces must contain RouterTrace rows or None")
                length = trace.expert_ids.shape[0]
                if length > sequence_shape[1]:
                    raise ValueError("experience router_trace exceeds its sequence length")
                if self.trajectories and length != self.trajectories[row].token_ids.numel():
                    raise ValueError("experience router_trace must align with its trajectory")
                if bool((trace.valid_mask & ~self.attention_mask[row, :length].detach().cpu().bool()).any()):
                    raise ValueError("experience router_trace must not select padding tokens")

    def _validate_rollout_scores(self, log_prob_shape: tuple[int, int]) -> None:
        """Validate detached sampling probabilities on supervised actions."""
        if self.rollout_log_probs is not None:
            if tuple(self.rollout_log_probs.shape) != log_prob_shape:
                raise ValueError("experience rollout_log_probs must align with next-token positions")
            if self.rollout_log_probs.requires_grad:
                raise ValueError("Rollout probabilities must be detached")
            valid = self.rollout_log_probs.masked_select(self.loss_action_mask.bool())
            if not bool(valid.isfinite().all().item()) or bool((valid > 1.0e-5).any().item()):
                raise ValueError("Rollout action log probabilities must be finite and nonpositive")

    def _validate_teacher_scores(self, log_prob_shape: tuple[int, int]) -> None:
        """Validate frozen teacher scores and row routing identities."""
        if self.teacher_log_probs is not None:
            if tuple(self.teacher_log_probs.shape) != log_prob_shape:
                raise ValueError('experience teacher_log_probs must align with next-token positions')
            if self.teacher_keys is None or len(self.teacher_keys) != log_prob_shape[0]:
                raise ValueError('experience teacher_keys must contain one identity per row')
            active_rows = self.loss_action_mask.bool().any(dim=-1).tolist()
            if any(not isinstance(key, str) or (active and not key)
                   for key, active in zip(self.teacher_keys, active_rows)):
                raise ValueError('Every supervised row must identify its teacher')
            if self.teacher_log_probs.requires_grad:
                raise ValueError('Teacher scores must be detached')
            valid = self.teacher_log_probs.masked_select(self.loss_action_mask.bool())
            if not bool(valid.isfinite().all().item()) or bool((valid > 1.0e-5).any().item()):
                raise ValueError('Teacher action log probabilities must be finite and nonpositive')
        elif self.teacher_keys is not None:
            raise ValueError('teacher_keys require teacher_log_probs')

    @property
    def loss_action_mask(self) -> Any:
        """Align full-sequence action positions with next-token log-probabilities."""
        return self.action_mask[:, 1:]
