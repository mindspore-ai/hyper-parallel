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
"""DeepSeek V4.1 text-model recipe for logical memory simulation."""

from contextlib import AbstractContextManager, nullcontext
from types import MethodType
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
from torch.nn import functional  # pylint: disable=forbidden-backend-import
from torch.utils.checkpoint import checkpoint  # pylint: disable=forbidden-backend-import
from transformers.modeling_outputs import CausalLMOutputWithPast

from hyper_parallel.components.modules.engram import EngramModule
from hyper_parallel.core.dry_run import (
    ValueDependencyHandler,
    ValueDependencyStage,
    register_value_dependency_handler,
)
from hyper_parallel.distributed.expert_parallel.collectives import (
    _EPAllToAllPadded,
    ep_all_to_all,
)
from hyper_parallel.models import HyperAutoModelForCausalLM
from hyper_parallel.models.deepseek_v41.adapter.policies import activation_checkpointing
from hyper_parallel.models.deepseek_v41.adapter.validation.cropped_model import build_deepseek_v41_validation_config


def _balanced_sparse_lookup(
        hash_ids: torch.Tensor,
        embedding: torch.nn.Embedding,
        *,
        ep_group: Any,
        ep_rank: int,
        ep_size: int,
        padded_num_embeddings: int,
) -> torch.Tensor:
    """Replay Engram routing with balanced, value-free EP traffic."""
    num_local_rows = padded_num_embeddings // ep_size
    row_start = ep_rank * num_local_rows
    flat_ids = hash_ids.reshape(-1)
    owner = torch.div(flat_ids, num_local_rows, rounding_mode="floor")
    base, remainder = divmod(int(flat_ids.numel()), ep_size)
    counts = [base + int(index < remainder) for index in range(ep_size)]
    sort_indices = torch.argsort(owner.float(), stable=True)
    sorted_ids = flat_ids.index_select(0, sort_indices)
    received_ids = ep_all_to_all(sorted_ids, counts, counts, ep_group)
    values = embedding(received_ids - row_start)
    returned = ep_all_to_all(values, counts, counts, ep_group)
    restored = torch.zeros_like(returned)
    restored.scatter_add_(
        0,
        sort_indices.unsqueeze(1).expand(-1, returned.shape[1]),
        returned,
    )
    return restored.view(*hash_ids.shape, embedding.embedding_dim)


def _fake_padded_all_to_all(
        ctx: Any,
        x: torch.Tensor,
        send_counts: list[int],
        recv_counts: list[int],
        group: Any,
) -> torch.Tensor:
    """Replay padded all-to-all without reading a value from fake MAX reduction."""
    ctx.send_counts = send_counts
    ctx.recv_counts = recv_counts
    ctx.group = group
    ctx.pad_to = pad_to = max([*send_counts, *recv_counts, 1])
    recv = _EPAllToAllPadded._pad_and_exchange(x, send_counts, pad_to, group)
    return _EPAllToAllPadded._unpad(recv, recv_counts, pad_to)


class _DeepSeekV41FakeValues(AbstractContextManager):
    """Install DeepSeek-specific value replay only for the fake training step."""

    def __enter__(self) -> "_DeepSeekV41FakeValues":
        """Patch value-dependent Engram and padded EP boundaries."""
        self._engram_lookup = EngramModule._sparse_lookup  # pylint: disable=protected-access
        self._padded_forward = _EPAllToAllPadded.forward  # pylint: disable=protected-access
        EngramModule._sparse_lookup = staticmethod(_balanced_sparse_lookup)  # pylint: disable=protected-access
        _EPAllToAllPadded.forward = staticmethod(_fake_padded_all_to_all)  # pylint: disable=protected-access
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        """Restore production methods after the fake step."""
        del exc_type, exc_value, traceback
        EngramModule._sparse_lookup = staticmethod(self._engram_lookup)  # pylint: disable=protected-access
        _EPAllToAllPadded.forward = staticmethod(self._padded_forward)  # pylint: disable=protected-access
        return False


class _DeepSeekV41ValueDependency(ValueDependencyHandler):
    """Own DeepSeek-specific fake-value boundaries outside business modules."""

    name = "deepseek_v41"
    paths = ("balanced",)

    def discover(self, model: Any, context: Any) -> list[str]:
        """Expose one target when the parallelized model contains Engram."""
        del context
        return ["deepseek_v41"] if any(isinstance(module, EngramModule) for module in model.modules()) else []

    def install(
            self,
            stage: ValueDependencyStage,
            target: str,
            path: str,
            inputs: dict[str, Any],
            context: Any,
    ) -> AbstractContextManager:
        """Install the scoped DeepSeek fake-value adapter."""
        del target, path, inputs, context
        if stage is ValueDependencyStage.FAKE_STEP:
            return _DeepSeekV41FakeValues()
        return nullcontext()


register_value_dependency_handler(_DeepSeekV41ValueDependency.name, _DeepSeekV41ValueDependency)


def _bind_chunk_loss(model: Any, chunks: int) -> None:
    """Compute a checkpointed output-head loss without full-sequence logits."""
    if chunks < 1:
        raise ValueError("loss_chunks must be positive")

    def _loss_chunk(hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute one output-head cross-entropy chunk."""
        logits = model.lm_head(hidden).float()
        return functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
            ignore_index=-100, reduction="sum",
        )

    def _forward(current_model: Any, *, labels: torch.Tensor, **inputs: Any) -> CausalLMOutputWithPast:
        """Run the text model and checkpoint each output-head loss chunk."""
        hidden = current_model.model(**inputs).last_hidden_state
        targets = functional.pad(labels[..., 1:], (0, 1), value=-100)
        if hidden.size(1) % chunks:
            raise ValueError("CP-local sequence length must be divisible by loss_chunks")
        loss = torch.zeros((), dtype=torch.float32, device=hidden.device)
        for hidden_chunk, target_chunk in zip(hidden.chunk(chunks, dim=1), targets.chunk(chunks, dim=1)):
            loss = loss + checkpoint(_loss_chunk, hidden_chunk, target_chunk, use_reentrant=False)
        return CausalLMOutputWithPast(loss=loss / targets.ne(-100).sum().clamp_min(1))

    model.forward = MethodType(_forward, model)


def build_text_model(
        config_path: str,
        engram_assets_path: str,
        *,
        text_parameter_divisor: int = 1,
        num_routed_experts: int = 384,
        loss_chunks: int = 8,
        recompute_tier: str = "all",
        **kwargs: Any,
) -> Any:
    """Build a released-size (or explicitly cropped) text model with chunk loss.

    Args:
        config_path: Directory containing the released config.json.
        engram_assets_path: Engram validation asset generated for that config.
        text_parameter_divisor: 1 retains released text widths.
        num_routed_experts: 384 retains all released routed experts.
        loss_chunks: Number of CP-local loss chunks.
        recompute_tier: Model-owned attention recomputation policy.
        **kwargs: Normal HyperAutoModel build settings supplied by Trainer.

    Returns:
        Config-only model parallelized through the usual AutoModel path.
    """
    config = build_deepseek_v41_validation_config(
        config_path, engram_assets_path,
        text_parameter_divisor=text_parameter_divisor,
        num_routed_experts=num_routed_experts,
        exercise_post_training_indexer=False,
    )
    config.v41_model_mode = "full"
    if recompute_tier not in ("none", "reuse", "reuse_reindex", "all"):
        raise ValueError(f"Unknown recompute_tier: {recompute_tier}")
    previous_tier = activation_checkpointing.ATTENTION_RECOMPUTE_TIER
    activation_checkpointing.ATTENTION_RECOMPUTE_TIER = recompute_tier
    try:
        model = HyperAutoModelForCausalLM.from_config(config, **kwargs)
    finally:
        activation_checkpointing.ATTENTION_RECOMPUTE_TIER = previous_tier
    _bind_chunk_loss(model, loss_chunks)
    return model
