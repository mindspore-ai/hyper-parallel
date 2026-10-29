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
"""Actual HSDP acceptance for selected probabilities with uneven per-rank head chunks."""

from copy import deepcopy
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # pylint: disable=unused-import
from transformers import Qwen3Config, Qwen3ForCausalLM

from rl.roles.policy.token_log_probs_adapter import bind_token_log_probs
from rl.trainer import SyncTrainer
from rl.utils.failure_reporting import report_failure

from hyper_parallel import SkipDTensorDispatch, fully_shard, hsdp_sync_stream
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy


def local(value: torch.Tensor) -> torch.Tensor:
    """Read rank-local gradient or parameter shards without another collective."""
    return value.to_local() if isinstance(value, DTensor) else value


def run(dtype: torch.dtype) -> dict:
    """Compare identical weights, clipped token losses, gradients and updates with reshard enabled."""
    rank = dist.get_rank()
    torch.manual_seed(23)
    config = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         max_position_embeddings=256, use_cache=False, tie_word_embeddings=True)
    config._attn_implementation = "eager"  # pylint: disable=protected-access
    base = Qwen3ForCausalLM(config).to(dtype=dtype)
    mesh = init_device_mesh("npu", (dist.get_world_size(),), mesh_dim_names=("dp",))
    models = []
    for selected in (False, True):
        model = deepcopy(base).npu().train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        options = {"mesh": mesh, "mp_policy": MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32),
                   "reshard_after_forward": True, "comm_fusion": True, "comm_fusion_zero_copy": False}
        for layer in model.model.layers:
            fully_shard(layer, **options)
        fully_shard(model, **options)
        if selected:
            bind_token_log_probs(model, chunk_size=3)
        models.append(model)
    torch.manual_seed(400 + rank)
    ids = torch.randint(0, 128, (1, 37 + rank * 2), device="npu")
    mask = torch.zeros_like(ids[:, 1:], dtype=torch.bool)
    if rank != 0:
        mask[:, -(rank + 2):] = True
        mask[:, 5:8] = True
    weights = torch.randn(mask.shape, device="npu") * mask
    values = []
    gradients = []
    updates = []
    old = None
    for selected, model in zip((False, True), models):
        with SkipDTensorDispatch():
            if selected:
                probs = model(input_ids=ids, use_cache=False, return_token_log_probs=True, token_log_probs_mask=mask)
            else:
                logits = model(input_ids=ids, use_cache=False).logits[:, :-1]
                probs = logits.float().log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
            if old is None:
                old = probs.detach().clone() + 0.02
            ratio = (probs - old).exp()
            loss = -torch.minimum(ratio * weights, ratio.clamp(0.99, 1.01) * weights).sum()
        loss.backward()
        hsdp_sync_stream()
        torch.npu.synchronize()
        values.append(probs.detach().cpu() * mask.cpu())
        gradients.append({name: local(param.grad).float().cpu().clone() for name, param in model.named_parameters()})
        with SkipDTensorDispatch(), torch.no_grad():
            for param in model.parameters():
                param.add_(param.grad.to(param.dtype), alpha=-0.01)
        updates.append({name: local(param).float().cpu().clone() for name, param in model.named_parameters()})
        if not model.hsdp_scheduler.reshard_after_forward:
            raise RuntimeError("Root reshard policy changed")
        SyncTrainer._reshard_model(model)  # pylint: disable=protected-access
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(values[1], values[0], rtol=tolerance, atol=tolerance)
    gradient_max = 0.0
    for name, expected in gradients[0].items():
        actual = gradients[1][name]
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(updates[1][name], updates[0][name], rtol=tolerance, atol=tolerance)
        gradient_max = max(gradient_max, (actual - expected).abs().max().item())
    if sum(value.abs().sum().item() for value in gradients[0].values()) == 0:
        raise RuntimeError("Distributed gradient comparison was vacuous")
    return {"dtype": str(dtype), "gradient_max_abs": gradient_max, "root_reshard": True,
            "zero_action_rank": rank == 0, "sequence_length": ids.shape[1]}


def main() -> None:
    """Exercise two precisions on the allocated devices and persist every rank's result."""
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="cpu:gloo,npu:hccl")
    root = Path(os.environ["HYPER_COMPARISON_OUTPUT"])
    try:
        results = [run(dtype) for dtype in (torch.float32, torch.bfloat16)]
        (root / f"token-probs-rank-{dist.get_rank()}.json").write_text(json.dumps(results, indent=2) + "\n")
        dist.barrier()
        if dist.get_rank() == 0:
            result = {"status": "passed", "ranks": dist.get_world_size()}
            (root / "measurement.json").write_text(json.dumps(result) + "\n")
    except BaseException as error:
        report_failure(error, stage="token_log_probs_hsdp")
        raise
    finally:
        dist.destroy_process_group()


def test_token_probabilities() -> None:
    """Entry used by the framework-free distributed ST launcher."""
    main()


if __name__ == "__main__":
    main()
