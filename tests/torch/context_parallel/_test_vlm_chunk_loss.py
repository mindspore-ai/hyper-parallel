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
"""Two-rank VLM chunk/full-logits gradient comparison for both CP algorithms."""

import copy
from types import SimpleNamespace

import torch
import torch.distributed as dist

from hyper_parallel.components.losses import ChunkedCausalLMLoss
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.distributed._builder.forward_rewriter import _ForwardRewriteRequest, _commit_forward_rewrite
from hyper_parallel.distributed.context_parallel.wrappers import sdpa_hf_cp_wrapper, sdpa_hf_ulysses_cp_wrapper
from hyper_parallel.models.qwen3_5.adapter.distributed.context_parallel import (
    qwen3_5_gdn_p2p_cp_wrapper, qwen3_5_gdn_ulysses_cp_wrapper,
)
from hyper_parallel.models.qwen3_5_moe.vlm_cp import QwenVLMParallelBatch
from hyper_parallel.trainer.runtime.loss_aggregation import count_loss_token
from tests.ut.auto_models.test_qwen_vlm_chunk_loss import make_batch, make_model


def _compare_gradients(candidate: torch.nn.Module, reference: torch.nn.Module) -> None:
    """Check every parameter after combining contributions from both CP ranks."""
    for (name, param), ref in zip(candidate.named_parameters(), reference.parameters()):
        if param.grad is None or ref.grad is None:
            raise AssertionError(f"Missing gradient: {name}")
        dist.all_reduce(param.grad)
        dist.all_reduce(ref.grad)
        torch.testing.assert_close(param.grad, ref.grad, atol=3e-6, rtol=3e-4,
                                   msg=lambda message, name=name: f"{name}: {message}")


def test_chunk_loss_cp() -> None:
    """Check every parameter gradient, image boundary and empty-supervision rank."""
    dist.init_process_group("gloo")
    try:
        torch.set_num_threads(1)
        mesh = init_device_mesh("cpu", (1, 2, 1), mesh_dim_names=("dp", "cp", "tp"))
        context = SimpleNamespace(device_mesh=mesh, cp_size=2, tp_size=1, pp_size=1, dp_size=1, dp_rank=0,
                                  cp_rank=dist.get_rank(), tp_rank=0, sequence_parallel=False, loss_parallel=False)
        modes = ((qwen3_5_gdn_ulysses_cp_wrapper, sdpa_hf_ulysses_cp_wrapper),
                 (qwen3_5_gdn_p2p_cp_wrapper, sdpa_hf_cp_wrapper))
        for gdn_wrapper, attention_wrapper in modes:
            for bind_first in (True, False):
                _check_chunk_loss_mode(mesh, context, gdn_wrapper, attention_wrapper, bind_first)
    finally:
        dist.destroy_process_group()


def _check_chunk_loss_mode(mesh, context, gdn_wrapper, attention_wrapper, bind_first):
    """Compare all image boundaries for one wrapper and binding order."""
    torch.manual_seed(42)
    reference = make_model()
    candidate = copy.deepcopy(reference)
    objective = ChunkedCausalLMLoss(chunk_size=17)
    batchers = []
    for model in (reference, candidate):
        layers = model.model.language_model.layers
        _commit_forward_rewrite(gdn_wrapper(layers[0].linear_attn, mesh, mesh["tp"], mesh["cp"], None))
        attention = layers[1].self_attn
        _commit_forward_rewrite(_ForwardRewriteRequest(
            attention, attention_wrapper(attention, mesh, mesh["tp"], mesh["cp"], None)))
        if model is candidate and bind_first:
            objective.bind_model(model)
        batcher = QwenVLMParallelBatch(mesh_context=context, device=torch.device("cpu"))
        batcher.bind_model(model)
        if model is candidate and not bind_first:
            objective.bind_model(model)
        batchers.append(batcher)
    for image_start, supervised_start in ((62, 2), (5, 65), (5, 100)):
        batch = make_batch(image_start, supervised_start)
        losses = []
        for model, batcher in zip((reference, candidate), batchers):
            model.zero_grad(set_to_none=True)
            inputs, loss_inputs = batcher(None, external_batch=batch)
            count = count_loss_token(loss_inputs)["foundation_tokens"]
            total = count.clone()
            dist.all_reduce(total)
            if model is candidate:
                torch.testing.assert_close(inputs["shift_labels"], inputs["vlm_cp_inputs"].targets)
                output = model(**objective.prepare_model_inputs(inputs, loss_inputs), use_cache=False)
                if output.logits is not None:
                    raise AssertionError("Chunk path materialized logits")
                loss = objective(model_output=output, labels=None)
            else:
                loss = model(**inputs, use_cache=False).loss
            scaled = loss * count / total.clamp_min(1)
            reduced = scaled.detach().clone()
            dist.all_reduce(reduced)
            losses.append(reduced)
            scaled.backward()
        torch.testing.assert_close(losses[0], losses[1], atol=1e-6, rtol=1e-6)
        _compare_gradients(candidate, reference)
    if dist.get_rank() == 0:
        print(f"PASS {gdn_wrapper.__name__} bind_first={bind_first}", flush=True)
