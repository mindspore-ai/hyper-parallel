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
"""vLLM protocol around the original HP Qwen3 model and Actor replacement plan."""

from __future__ import annotations

from collections.abc import Iterable
import logging
from typing import Any

import torch
from torch import nn

from rl.unified_execution.runtimes.vllm.parallel_context import payload_from_vllm_config
from rl.unified_execution.runtimes.vllm.parallel_context import build_parallel_context
from rl.unified_execution.runtimes.hyper_parallel.builder import build_inference_model
from rl.unified_execution.models import get_model_family
from rl.unified_execution.validation import capture_training_structure
from rl.unified_execution.runtimes.vllm.paged_attention import install_paged_attention
from rl.unified_execution.runtimes.vllm.weight_loader import LogicalWeightLoader
from rl.unified_execution.batch_invariant import initialize_batch_invariant, verify_batch_invariant
from hyper_parallel.core.dtensor.dtensor import DTensor, distribute_tensor
from hyper_parallel.core.dtensor.placement_types import Replicate
from hyper_parallel.distributed._builder.tp_collective_lowering import create_tp_collective_lowerer
from hyper_parallel.distributed.recipe_spec import resolve_placements

logger = logging.getLogger(__name__)


class HyperParallelQwen3ForCausalLM(nn.Module):
    """Reuse HP embedding, decoder, RoPE, normalization, MLP and LM head.

    Only the attention interface is replaced. vLLM owns paged KV cache,
    scheduled request boundaries, sampling and the flattened token schedule.
    """

    def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
        """Build an HP model with the worker's topology and the Actor's plan."""
        super().__init__()
        payload = payload_from_vllm_config(vllm_config)
        self.unified_numeric_policy = payload.get("batch_invariant")
        if self.unified_numeric_policy is not None:
            initialize_batch_invariant(self.unified_numeric_policy)
        if vllm_config.quant_config is not None:
            raise ValueError("Unified Qwen3 does not support quantization")
        if not vllm_config.model_config.enforce_eager and payload["attention_backend"] != "flash_attention_npu_v4":
            raise ValueError("Unified Qwen3 graph execution requires FlashAttention NPU V4")
        family = get_model_family(vllm_config.model_config.hf_config.model_type)
        if family.model_type != payload.get("model_family", "qwen3"):
            raise ValueError("Worker model family differs from the unified contract")
        self.parallel_context = build_parallel_context(vllm_config, payload)
        hp_model = build_inference_model(
            vllm_config.model_config.hf_config, payload["plan_overrides"], attn_implementation="eager",
            parallel_context=self.parallel_context,
        )
        self.unified_structure_report = capture_training_structure(hp_model, payload, inference=True)
        expected = payload.get("training_structure")
        if expected is not None and self.unified_structure_report != expected:
            raise ValueError("Inference HP model differs from the Actor's realized execution structure")
        self.config = hp_model.config
        self.model = hp_model.model
        self.lm_head = hp_model.lm_head
        self.model.inference_tp_size = self.parallel_context.tp_size
        self.weight_loader = LogicalWeightLoader(self)
        self.start_layer, self.end_layer = 0, len(self.model.layers)
        self._prepare_scheduler_boundaries()
        install_paged_attention(self.model, vllm_config, prefix)
        self.verify_numerical_policy()
        self.eval()
        logger.info(
            "Unified Qwen3 uses HP model with plan=%s, structure=%s; attention core uses vLLM",
            payload["plan_fingerprint"], self.unified_structure_report["fingerprint"],
        )

    def verify_numerical_policy(self) -> dict[str, Any] | None:
        """Read actual worker state at startup, serving and publication boundaries."""
        if self.unified_numeric_policy is None:
            return None
        return {"policy": dict(self.unified_numeric_policy),
                "state": verify_batch_invariant(self.unified_numeric_policy)}

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed vLLM token IDs with the original training-side module."""
        count = input_ids.shape[0]
        input_ids = self._pad_rows(input_ids)
        tokens = self._input_rows(input_ids.unsqueeze(0), self.model.embed_tokens, argument="input")
        return self._output_rows(self.model.embed_tokens(tokens), self.model.embed_tokens).squeeze(0)[:count]

    def _pad_rows(self, tensor: torch.Tensor) -> torch.Tensor:
        """Keep SP padding valid when one compiled graph sees odd and even rows."""
        context = self.parallel_context
        multiple = context.tp_size if context.sequence_parallel else 1
        if multiple == 1:
            return tensor
        # A shape-dependent Python branch is specialized during vLLM tracing.
        # Fixed extra rows avoid zero-sized symbolic allocations at divisible sizes.
        rows = ((tensor.shape[0] + multiple - 1) // multiple) * multiple
        padded = torch.cat((tensor, tensor.new_zeros((multiple - 1, *tensor.shape[1:]))))
        return padded[:rows]

    def _input_rows(self, tensor: torch.Tensor, module: nn.Module, *,
                    argument: str = "hidden_states") -> torch.Tensor:
        """Adapt replicated scheduler rows to HP's declared production entry layout."""
        context = self.parallel_context.mesh_context
        layout = getattr(module, "_rl_input_layouts", {}).get(argument)
        if context is None or layout is None:
            return tensor
        mesh = context.device_mesh
        placements = resolve_placements(layout, mesh.mesh_dim_names)
        if all(isinstance(placement, Replicate) for placement in placements):
            return tensor
        # Each DP scheduler already owns its token rows; only local slicing is
        # needed. Broadcasting the first DP rank would replace other requests.
        return distribute_tensor(tensor, mesh, placements, src_data_rank=None).to_local()

    def _output_rows(self, tensor: torch.Tensor, module: nn.Module) -> torch.Tensor:
        """Gather the declared HP output only when returning rows to the scheduler."""
        if isinstance(tensor, DTensor):
            return tensor.reduce_partial().full_tensor()
        operation = getattr(module, "_rl_scheduler_output_op", None)
        if operation is not None:
            return operation.execute(tensor)
        context = self.parallel_context.mesh_context
        layout = getattr(module, "_rl_output_layout", None)
        if context is None or layout is None:
            return tensor
        mesh = context.device_mesh
        placements = resolve_placements(layout, mesh.mesh_dim_names)
        if all(isinstance(placement, Replicate) for placement in placements):
            return tensor
        return DTensor.from_local(tensor, mesh, placements).reduce_partial().full_tensor()

    def _prepare_scheduler_boundaries(self) -> None:
        """Reuse HP's local-tensor TP lowerer outside Dynamo tracing."""
        context = self.parallel_context.mesh_context
        if context is None:
            return
        mesh = context.device_mesh
        lowerer = create_tp_collective_lowerer(mesh, mesh.mesh_dim_names)
        if lowerer is None:
            return
        for module in (self.model.embed_tokens, self.model.norm):
            layout = getattr(module, "_rl_output_layout", None)
            if layout is not None:
                source = resolve_placements(layout, mesh.mesh_dim_names)
                target = tuple(Replicate() for _ in source)
                module._rl_scheduler_output_op = lowerer(source, target)  # pylint: disable=protected-access

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run the original decoder forward with absolute scheduled positions."""
        del kwargs
        if not torch.compiler.is_compiling():
            self.verify_numerical_policy()
        if positions.ndim != 1:
            raise ValueError("Unified Qwen3 requires one absolute position per scheduled token")
        token_count = positions.shape[0]
        for index in range(self.start_layer, self.end_layer):
            self.model.layers[index].self_attn.unified_token_count = token_count
            self.model.layers[index].self_attn.unified_sp_multiple = (
                self.parallel_context.tp_size if self.parallel_context.sequence_parallel else 1
            )
        if intermediate_tensors is not None:
            raise ValueError("Unified HP pipeline inference is not integrated")
        if inputs_embeds is not None:
            if inputs_embeds.ndim != 2 or inputs_embeds.shape[0] != positions.shape[0]:
                raise ValueError("Unified Qwen3 inputs_embeds must match the scheduled token count")
            input_ids = None
            inputs_embeds = self._input_rows(
                self._pad_rows(inputs_embeds).unsqueeze(0), self.model.layers[0].input_layernorm,
            )
        elif input_ids is None or input_ids.ndim != 1 or input_ids.shape[0] != positions.shape[0]:
            raise ValueError("Unified Qwen3 input_ids must match the scheduled token count")
        else:
            input_ids = self._input_rows(
                self._pad_rows(input_ids).unsqueeze(0), self.model.embed_tokens, argument="input",
            )
        positions = self._pad_rows(positions)
        # A prepared mapping skips HF's causal-mask construction. Each attention
        # core instead consumes vLLM metadata, including request and cache bounds.
        output = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=positions.unsqueeze(0),
            attention_mask={"full_attention": None, "sliding_attention": None},
            past_key_values=None,
            use_cache=False,
            return_dict=True,
        )
        return self._output_rows(output.last_hidden_state, self.model.norm).squeeze(0)[:token_count]

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Use HP's LM head directly; vLLM selects token rows before this call."""
        # This boundary also executes when a captured decoder bypasses Python forward.
        self.verify_numerical_policy()
        count = hidden_states.shape[0]
        hidden_states = self._input_rows(self._pad_rows(hidden_states).unsqueeze(0), self.lm_head)
        return self._output_rows(self.lm_head(hidden_states), self.lm_head).squeeze(0)[:count]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load logical HF weights through the same HP conversion descriptors."""
        return self.weight_loader.load(weights)

    def begin_weight_update(self) -> None:
        """Start a complete online publication before its first bucket arrives."""
        self.verify_numerical_policy()
        self.weight_loader.begin()

    def finish_weight_update(self) -> None:
        """Validate publication completeness before the worker resumes serving."""
        self.verify_numerical_policy()
        self.weight_loader.finish()


class HyperParallelQwen3MoeForCausalLM(HyperParallelQwen3ForCausalLM):
    """Compose the same vLLM protocol with HP's existing Qwen3-MoE model."""
