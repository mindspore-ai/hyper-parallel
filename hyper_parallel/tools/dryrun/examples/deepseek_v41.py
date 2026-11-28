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

from types import MethodType
from typing import Any, Optional

import torch
from torch.nn import functional
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import CausalLMOutputWithPast

from hyper_parallel.components.modules.engram import EngramModule
from hyper_parallel.tools.dryrun.compatibility import register_environment_patch
from hyper_parallel.tools.dryrun.runtime import install_moe_replay
from hyper_parallel.distributed.expert_parallel.collectives import (
    _EPAllToAllPadded,
    ep_all_to_all,
)
from hyper_parallel.models import HyperAutoModelForCausalLM
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
    recv = _EPAllToAllPadded._pad_and_exchange(
        x, send_counts, pad_to, group)
    return _EPAllToAllPadded._unpad(recv, recv_counts, pad_to)


def install_deepseek_v41_replay() -> None:
    """Install DeepSeek V4.1 fake-value boundaries for the rest of the process.

    Model-specific replay is requested by the model integration code below.
    Nothing is discovered from the module tree -- ``EngramModule`` also appears
    in other models whose replay needs differ -- and patches are never
    restored, because dry-run owns the process it runs in.
    """
    EngramModule._sparse_lookup = staticmethod(_balanced_sparse_lookup)
    _EPAllToAllPadded.forward = staticmethod(_fake_padded_all_to_all)


def _synthetic_document_boundaries(sequence: Any) -> tuple[int, ...]:
    """Keep document count while assigning equal spans to value-free inputs."""
    document_count = max(int(sequence.cu_seq_lens.numel()) - 1, 1)
    span, remainder = divmod(sequence.global_sequence_length, document_count)
    boundaries = [0]
    for index in range(1, document_count + 1):
        boundaries.append(boundaries[-1] + span + (index <= remainder))
    return tuple(boundaries)


def _install_csa2_patches() -> None:
    """Make CSA2 host-side validations shape-only under FakeTensor.

    Model-specific environment patch, registered from this example module so
    the generic dry-run compatibility layer never imports DeepSeek code.
    """
    from hyper_parallel.components.modules.attention.compressed_sparse_attention2 import (  # pylint: disable=C0415
        SharedCompressedPackedSequence,
    )

    sequence_type = SharedCompressedPackedSequence
    if getattr(sequence_type, "_dryrun_compatibility_installed", False):
        return

    original_document_boundaries = sequence_type.document_boundaries
    original_local_segment_starts = sequence_type.local_segment_starts

    def document_boundaries(sequence: Any) -> tuple[int, ...]:
        """Return synthetic document boundaries for value-free sequences.

        Args:
            sequence: Packed sequence whose document layout is requested.
        """
        if not hasattr(sequence.cu_seq_lens, "fake_mode"):
            return original_document_boundaries(sequence)
        if "boundaries" not in sequence._snapshot:
            sequence._snapshot["boundaries"] = _synthetic_document_boundaries(sequence)
        return sequence._snapshot["boundaries"]

    def local_segment_starts(sequence: Any, device: torch.device) -> torch.Tensor:
        """Return device-local segment starts that preserve the sequence shape.

        Args:
            sequence: Packed sequence whose local segments are requested.
            device: Device the returned starts must live on.
        """
        if not hasattr(sequence.cu_seq_lens, "fake_mode"):
            return original_local_segment_starts(sequence, device)
        cached = sequence._prepared_starts.get(torch.device(device))
        if cached is not None:
            return cached
        boundaries = torch.tensor(
            document_boundaries(sequence),
            dtype=torch.long,
            device=device,
        )
        query_positions = sequence.local_positions(device).squeeze(0)
        segment_ids = torch.bucketize(query_positions, boundaries[1:], right=True)
        result = boundaries[:-1].index_select(0, segment_ids).unsqueeze(0)
        sequence._prepared_starts[torch.device(device)] = result
        return result

    sequence_type.document_boundaries = document_boundaries
    sequence_type.local_segment_starts = local_segment_starts
    sequence_type._dryrun_compatibility_installed = True


register_environment_patch("deepseek_v41.csa2", _install_csa2_patches)


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
        num_hidden_layers: Optional[int] = None,
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
        num_hidden_layers: Optional decoder crop depth; ``None`` keeps all layers.
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
        num_hidden_layers=num_hidden_layers,
        text_parameter_divisor=text_parameter_divisor,
        num_routed_experts=num_routed_experts,
        exercise_post_training_indexer=False,
    )
    config.v41_model_mode = "full"
    if recompute_tier not in ("none", "reuse", "reuse_reindex", "all"):
        raise ValueError(f"Unknown recompute_tier: {recompute_tier}")
    model = HyperAutoModelForCausalLM.from_config(config, **kwargs)
    _bind_chunk_loss(model, loss_chunks)
    # Model-owned value replay: this recipe is the one place that knows which
    # routing profile DeepSeek V4.1 needs under FakeTensor.
    setup = kwargs.get("distributed_setup")
    if setup is not None:
        install_moe_replay(model, setup.mesh_context, "balanced")
    install_deepseek_v41_replay()
    return model
