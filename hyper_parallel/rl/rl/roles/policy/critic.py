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
"""Critic model execution and algorithm-neutral value optimization."""
from dataclasses import replace
from functools import wraps
from types import MethodType
from typing import Any, Optional

import torch
import torch.distributed as dist

from rl.algorithm.loss import RLAlgorithm
from rl.dataset.contracts import ExperienceBatch
from rl.roles.qwen3_builder import build_causal_lm
from rl.utils.monitoring.metrics import CriticUpdateMetrics

from hyper_parallel import HSDPModule, SkipDTensorDispatch, hsdp_sync_stream
from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.core.utils import clip_grad_norm_
from hyper_parallel.distributed.mesh import DistributedSetup
from hyper_parallel.distributed.recipe_spec import ModuleShardingSpec


class ValueHead(torch.nn.Module):
    """A scalar projection initialized to the same zero baseline on every rank."""

    def __init__(self, hidden_size: int, device: Any) -> None:
        """Create the trainable scalar projection before distributed wrapping."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(
            (1, hidden_size), dtype=torch.float32, device=device,
        ))

    def forward(self, hidden_states: Any) -> Any:
        """Predict one value per token without allocating vocabulary logits."""
        return hidden_states.to(self.weight.dtype).matmul(self.weight.transpose(0, 1))


def _value_forward(model: Any, input_ids: Any, attention_mask: Any = None,
                   use_cache: bool = False, **kwargs: Any) -> dict[str, Any]:
    output = model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache, **kwargs)
    return {"values": model.value_head(output.last_hidden_state).squeeze(-1).float()}


def attach_value_head(model: Any) -> Any:
    """Adapt an already-created backbone before the model builder applies parallelism."""
    model.config.tie_word_embeddings = False
    model.all_tied_weights_keys = {}
    model._tied_weights_keys = {}  # pylint: disable=protected-access
    del model.lm_head
    model.value_head = ValueHead(model.config.hidden_size, model.model.embed_tokens.weight.device)
    model._keys_to_ignore_on_load_missing = [r"^value_head\.weight$"]  # pylint: disable=protected-access
    model._keys_to_ignore_on_load_unexpected = [r"^lm_head\.weight$"]  # pylint: disable=protected-access
    model.forward = MethodType(_value_forward, model)
    original_to_empty = model.to_empty

    @wraps(original_to_empty)
    def to_empty(**kwargs: Any) -> Any:
        """Materialize all parameters and initialize the newly added head."""
        result = original_to_empty(**kwargs)
        # The head is absent in the initial causal checkpoint, including on meta builds.
        with torch.no_grad(), SkipDTensorDispatch():
            model.value_head.weight.zero_()
        return result

    model.to_empty = to_empty
    return model


def build_value_model(pretrained_model_name_or_path: str, **kwargs: Any) -> Any:
    """Request standard model loading with a pre-parallel scalar-head adaptation."""
    setup = kwargs.pop("distributed_setup", None) or DistributedSetup()
    head_spec = ModuleShardingSpec(params={"weight": {"tp": Replicate()}}, region_dispatch=False)
    setup = replace(setup, plan_overrides={**(setup.plan_overrides or {}), "value_head": head_spec})
    return build_causal_lm(
        pretrained_model_name_or_path, fused=False, distributed_setup=setup,
        model_transform=attach_value_head, **kwargs,
    )


# Role execution uses explicit compute/update APIs; forward remains the Module default.
class Critic(torch.nn.Module):  # pylint: disable=abstract-method
    """Own a value model and its optimization runtime."""

    def __init__(
        self,
        critic_model: torch.nn.Module,
        algorithm: RLAlgorithm,
        optimizer: Any,
        lr_scheduler: Optional[Any],
        device: Any,
        dp_group_info: Any,
        dp_size: int,
        micro_batch_size: int,
        response_mini_batch_size: int,
        update_epochs: int,
        max_grad_norm: float,
    ) -> None:
        """Initialize the trainable Critic and its distributed optimizer settings."""
        super().__init__()
        if not algorithm.requirements.roles.critic:
            raise ValueError(f"Algorithm '{algorithm.name}' does not require a Critic")
        if optimizer is None:
            raise ValueError("A Critic requires an optimizer")
        for name, value in (
            ("micro_batch_size", micro_batch_size),
            ("response_mini_batch_size", response_mini_batch_size),
            ("update_epochs", update_epochs),
            ("dp_size", dp_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        self.critic_model = critic_model
        self.algorithm = algorithm
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self._device = device
        self._dp_group_info = dp_group_info
        self._dp_size = dp_size
        self._micro_batch_size = micro_batch_size
        self._response_mini_batch_size = response_mini_batch_size
        self._update_epochs = update_epochs
        self._max_grad_norm = max_grad_norm

    def sequence_values(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        *, include_last: bool = False,
    ) -> torch.Tensor:
        """Return float32 values aligned with next-token positions."""
        outputs = self.critic_model(
            input_ids=sequences,
            attention_mask=attention_mask,
            use_cache=False,
        )
        values = outputs["values"] if isinstance(outputs, dict) else outputs.values
        expected = (sequences.shape[0], sequences.shape[1])
        if tuple(values.shape) != expected:
            raise ValueError(
                f"Critic values must have shape {expected}, got {tuple(values.shape)}"
            )
        return (values if include_last else values[:, :-1]).float()

    def compute_values(self, experience: ExperienceBatch, *, include_last: bool = False) -> torch.Tensor:
        """Compute detached old values in response micro-batches."""
        was_training = self.training
        self.eval()
        chunks = []
        try:
            with torch.no_grad():
                for start in range(
                    0,
                    experience.sequences.shape[0],
                    self._micro_batch_size,
                ):
                    end = min(
                        start + self._micro_batch_size,
                        experience.sequences.shape[0],
                    )
                    chunks.append(
                        self.sequence_values(
                            experience.sequences[start:end],
                            experience.attention_mask[start:end],
                            include_last=include_last,
                        )
                    )
        finally:
            self.train(was_training)
        return torch.cat(chunks, dim=0).detach()

    def forward_backward(
        self,
        experience: ExperienceBatch,
        start: int,
        end: int,
        *,
        global_tokens: int,
    ) -> Any:
        """Compute and backpropagate one Critic loss micro-batch."""
        current_values = self.sequence_values(
            experience.sequences[start:end],
            experience.attention_mask[start:end],
        )
        output = self.algorithm.compute_critic_loss(
            current_values=current_values,
            old_values=experience.values[start:end].detach(),
            returns=experience.returns[start:end].detach(),
            action_mask=experience.loss_action_mask[start:end],
        )
        scaled_loss = output.loss_sum / global_tokens * self._dp_size
        if not bool(scaled_loss.isfinite().item()):
            raise RuntimeError("Non-finite Critic loss detected")
        scaled_loss.backward()
        return output.loss_sum.detach()

    def update(self, experience: ExperienceBatch) -> CriticUpdateMetrics:
        """Run configured value epochs and optimizer steps."""
        if experience.values is None or experience.returns is None:
            raise RuntimeError(
                "Training experience must include values and returns before Critic update"
            )
        response_count = int(experience.sequences.shape[0])
        if self._response_mini_batch_size > response_count:
            raise ValueError(
                "response_mini_batch_size cannot exceed the local response count: "
                f"mini_batch={self._response_mini_batch_size}, responses={response_count}"
            )
        local_loss = experience.sequences.new_zeros(
            (), dtype=torch.float32
        )
        processed_tokens = 0
        optimizer_steps = 0
        gradient_norm_sum = 0.0
        self.train()
        for _ in range(self._update_epochs):
            for mini_start in range(0, response_count, self._response_mini_batch_size):
                mini_end = min(
                    mini_start + self._response_mini_batch_size,
                    response_count,
                )
                global_tokens = self._global_token_count(
                    experience.loss_action_mask[mini_start:mini_end]
                )
                processed_tokens += global_tokens
                self.optimizer.zero_grad(set_to_none=True)
                for start in range(mini_start, mini_end, self._micro_batch_size):
                    end = min(start + self._micro_batch_size, mini_end)
                    self._set_gradient_sync(end == mini_end)
                    local_loss = local_loss + self.forward_backward(
                        experience,
                        start,
                        end,
                        global_tokens=global_tokens,
                    )
                gradient_norm_sum += self._optimizer_step()
                optimizer_steps += 1
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        if self._dp_size > 1:
            dist.all_reduce(local_loss, group=self._dp_group_info.group)
        return CriticUpdateMetrics(
            value_loss=float(local_loss.item()) / processed_tokens,
            gradient_norm=gradient_norm_sum / optimizer_steps,
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
            valid_tokens=processed_tokens,
            optimizer_steps=optimizer_steps,
        )

    def _global_token_count(self, action_mask: Any) -> int:
        """Return the data-parallel count of valid response tokens."""
        count = torch.tensor(
            [float(action_mask.flatten().sum(dim=0).item())],
            dtype=torch.float32,
            device=self._device,
        )
        if self._dp_size > 1:
            dist.all_reduce(count, group=self._dp_group_info.group)
        result = int(count.item())
        if result <= 0:
            raise RuntimeError("Critic update has no valid action tokens")
        return result

    def _set_gradient_sync(self, is_last_micro_batch: bool) -> None:
        if isinstance(self.critic_model, HSDPModule):
            self.critic_model.set_requires_gradient_sync(is_last_micro_batch)
            self.critic_model.set_is_last_backward(is_last_micro_batch)

    def _optimizer_step(self) -> float:
        """Clip gradients and advance the Critic optimizer and scheduler."""
        hsdp_sync_stream()
        norm_limit = self._max_grad_norm if self._max_grad_norm > 0 else float("inf")
        grad_norm = clip_grad_norm_(
            self.parameters(),
            norm_limit,
            error_if_nonfinite=True,
        )
        with SkipDTensorDispatch():
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return float(grad_norm.item())
__all__ = ["Critic"]
