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
"""Eight-card AutoModels Muon+AdamW checkpoint-resume integration test."""
# pylint: disable=E1123

import os
import shutil
from pathlib import Path
from typing import Optional, TextIO

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

from hyper_parallel import SkipDTensorDispatch
from hyper_parallel.components.checkpoint.dcp_checkpointer import (
    DistributedCheckpointer,
    initialize_optimizer_state,
)
from hyper_parallel.components.optim.mixed_precision_optimizer import (
    Float16OptimizerWithFloat16Params,
)
from hyper_parallel.core.utils import clip_grad_norm_
from hyper_parallel.data.parallel.batch_parallel import shard_batch_for_cp
from hyper_parallel.distributed.apply import apply_sharding_plan
from hyper_parallel.distributed.mesh import DistributedSetup
from hyper_parallel.models._transformers.model_builder import (
    apply_model_init_dtype,
    instantiate_infrastructure,
)
from hyper_parallel.trainer.config import (
    TrainerConfig,
    normalize_distributed_setup_overrides,
)
from hyper_parallel.trainer.config.parser import parse_training_args
from hyper_parallel.trainer.runtime.device import get_device_type
from hyper_parallel.trainer.runtime.distributed import (
    create_distributed_setup_from_config,
    destroy_process_group,
    initialize_distributed,
)
from hyper_parallel.trainer.runtime.metrics import mean_global_loss


_TEST_DIRECTORY = Path(__file__).resolve().parent
_TEST_YAML = _TEST_DIRECTORY / "test_yamls" / "qwen2_moe_muon_checkpoint_resume.yaml"
_CASE_NAME = "qwen2_moe_muon_adam_checkpoint_resume"
_OUTPUT_DIRECTORY = _TEST_DIRECTORY / f"{_CASE_NAME}_output"
_CHECKPOINT_DIRECTORY = _OUTPUT_DIRECTORY / "checkpoint_step_1"
_LOG_FILE = _OUTPUT_DIRECTORY / "resume.log"

_INIT_SEED = 31415
_DATA_SEED = 27182
_TRAINING_STEPS = 10
_GLOBAL_BATCH_SIZE = 8
_SEQUENCE_LENGTH = 16
_VOCAB_SIZE = 128
_HIDDEN_SIZE = 64
_INTERMEDIATE_SIZE = 128
_NUM_LAYERS = 4
_NUM_HEADS = 4
_NUM_EXPERTS = 4
_NUM_EXPERTS_PER_TOKEN = 2


def _parse_config() -> TrainerConfig:
    """Load the test configuration through the normal YAML parser."""
    return parse_training_args([str(_TEST_YAML)])


def _build_model_config() -> Qwen2MoeConfig:
    """Return the four-layer Qwen2-MoE configuration used by the test."""
    return Qwen2MoeConfig(
        vocab_size=_VOCAB_SIZE,
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=_INTERMEDIATE_SIZE,
        moe_intermediate_size=_INTERMEDIATE_SIZE,
        shared_expert_intermediate_size=_INTERMEDIATE_SIZE,
        num_hidden_layers=_NUM_LAYERS,
        num_attention_heads=_NUM_HEADS,
        num_key_value_heads=2,
        num_experts=_NUM_EXPERTS,
        num_experts_per_tok=_NUM_EXPERTS_PER_TOKEN,
        num_experts_shared=1,
        max_position_embeddings=_SEQUENCE_LENGTH,
        qkv_bias=True,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        use_cache=False,
    )


def _build_model(
    distributed_setup: DistributedSetup,
    device: torch.device,
    model_init_dtype: Optional[str],
) -> torch.nn.Module:
    """Build and shard one fresh Qwen2-MoE model through AutoModels."""
    torch.manual_seed(_INIT_SEED)
    model = Qwen2MoeForCausalLM(_build_model_config()).to(device=device)
    apply_model_init_dtype(model, model_init_dtype)
    model.train()

    mesh_context = distributed_setup.mesh_context
    sharding_planner, fsdp_manager = instantiate_infrastructure(
        distributed_setup=distributed_setup,
        device=device,
    )
    if fsdp_manager is None or mesh_context.device_mesh is None:
        raise RuntimeError("The resume test requires initialized FSDP and TP meshes")
    sharding_plan = sharding_planner.plan(
        model,
        mesh_context.device_mesh,
        tp_size=mesh_context.tp_size,
        cp_size=mesh_context.cp_size,
        ep_size=mesh_context.ep_size,
        sequence_parallel=mesh_context.sequence_parallel,
        loss_parallel=mesh_context.loss_parallel,
    )
    model, source_shard_info = apply_sharding_plan(
        model,
        sharding_plan,
        mesh_context,
    )
    fsdp_manager.parallelize(model, source_shard_info)
    return model


def _build_optimizer(
    config: TrainerConfig,
    model: torch.nn.Module,
) -> Float16OptimizerWithFloat16Params:
    """Build the configured Muon+AdamW optimizer with fp32 main params."""
    inner_optimizer = config.optimizer.target.build(model=model).get_optimizer()
    return Float16OptimizerWithFloat16Params(inner_optimizer, model)


def _validate_topology(distributed_setup: DistributedSetup) -> None:
    """Check the exact topology requested by the resume scenario."""
    mesh_context = distributed_setup.mesh_context
    topology = (
        mesh_context.dp_size,
        mesh_context.cp_size,
        mesh_context.tp_size,
        mesh_context.ep_size,
        mesh_context.dp_shard_size,
        mesh_context.edp_shard_size,
    )
    if topology != (4, 2, 1, 2, 2, 2):
        raise AssertionError(
            "resume test requires dp=4, cp=2, tp=1, ep=2, dp_shard=2, "
            f"edp_shard=2; got {topology}"
        )
    if mesh_context.fsdp_moe_mesh is None:
        raise RuntimeError("resume test requires an expert FSDP mesh")


def _build_batch(step_index: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic global tokens and labels for one step."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_DATA_SEED + step_index)
    tokens = torch.randint(
        _VOCAB_SIZE,
        (_GLOBAL_BATCH_SIZE, _SEQUENCE_LENGTH),
        generator=generator,
    )
    labels = torch.randint(
        _VOCAB_SIZE,
        (_GLOBAL_BATCH_SIZE, _SEQUENCE_LENGTH),
        generator=generator,
    )
    return tokens.to(device=device), labels.to(device=device)


def _local_batch(
    tokens: torch.Tensor,
    labels: torch.Tensor,
    distributed_setup: DistributedSetup,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the local DP batch and CP sequence range."""
    mesh_context = distributed_setup.mesh_context
    device_mesh = mesh_context.device_mesh
    local_batch_size = _GLOBAL_BATCH_SIZE // mesh_context.dp_size
    batch_start = device_mesh.get_local_rank("dp") * local_batch_size
    batch = shard_batch_for_cp(
        {
            "input_ids": tokens[batch_start:batch_start + local_batch_size],
            "labels": labels[batch_start:batch_start + local_batch_size],
        },
        mesh_context.cp_mesh,
    )
    local_sequence_size = _SEQUENCE_LENGTH // mesh_context.cp_size
    sequence_start = device_mesh.get_local_rank("cp") * local_sequence_size
    position_ids = torch.arange(
        sequence_start,
        sequence_start + local_sequence_size,
        device=tokens.device,
    ).unsqueeze(0).expand(local_batch_size, -1)
    return batch["input_ids"], batch["labels"], position_ids


def _step(
    model: torch.nn.Module,
    optimizer: Float16OptimizerWithFloat16Params,
    distributed_setup: DistributedSetup,
    step_index: int,
    device: torch.device,
) -> tuple[float, float]:
    """Run one deterministic forward/backward/update step."""
    tokens, labels = _build_batch(step_index, device)
    local_tokens, local_labels, position_ids = _local_batch(
        tokens,
        labels,
        distributed_setup,
    )
    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids=local_tokens, position_ids=position_ids).logits
    local_loss = F.cross_entropy(
        logits.float().reshape(-1, _VOCAB_SIZE),
        local_labels.reshape(-1),
    )
    token_count = local_labels.new_tensor(local_labels.numel())
    loss = mean_global_loss(
        local_loss,
        {"foundation_tokens": token_count},
        {"foundation_tokens": token_count},
        distributed_setup.mesh_context,
    )["foundation_loss"]
    loss.backward()
    grad_norm = clip_grad_norm_(model, max_norm=float("inf"))
    with SkipDTensorDispatch(no_skip={torch.zeros_like}):
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach().cpu()), float(grad_norm.detach().cpu())


def _write_log(
    log_file: Optional[TextIO],
    run_name: str,
    step_index: int,
    loss: float,
    grad_norm: float,
) -> None:
    """Write a rank-zero loss and gradient norm record."""
    if log_file is None:
        return
    log_file.write(
        f"run={run_name} step={step_index} loss={loss:.9e} "
        f"grad_norm={grad_norm:.9e}\n"
    )
    log_file.flush()


def _save_checkpoint(
    model: torch.nn.Module,
    optimizer: Float16OptimizerWithFloat16Params,
) -> None:
    """Persist model and optimizer state after step one."""
    DistributedCheckpointer().save(
        str(_CHECKPOINT_DIRECTORY),
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
        global_step=1,
    )


def _load_checkpoint(
    model: torch.nn.Module,
    optimizer: Float16OptimizerWithFloat16Params,
) -> None:
    """Restore model and optimizer state into a fresh runtime."""
    if not initialize_optimizer_state(optimizer):
        raise RuntimeError("Could not materialize optimizer state before resume")
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    DistributedCheckpointer().load(str(_CHECKPOINT_DIRECTORY), state)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])


def _prepare_output_directory() -> None:
    """Remove stale artifacts before the distributed test starts."""
    if dist.get_rank() == 0:
        shutil.rmtree(_OUTPUT_DIRECTORY, ignore_errors=True)
        _OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    dist.barrier()


def _cleanup_output_directory() -> None:
    """Remove checkpoint and logs after all ranks finish the comparison."""
    dist.barrier()
    if dist.get_rank() == 0:
        shutil.rmtree(_OUTPUT_DIRECTORY, ignore_errors=True)
    dist.barrier()


def test_qwen2_moe_muon_adam_checkpoint_resume() -> None:
    """Verify Muon and AdamW resume with matching loss and gradient norm."""
    config = _parse_config()
    initialize_distributed(backend=config.training.backend)
    device = torch.device(
        get_device_type(),
        int(os.environ.get("LOCAL_RANK", "0")),
    )
    distributed_setup = create_distributed_setup_from_config(config)
    normalize_distributed_setup_overrides(distributed_setup, config)
    _validate_topology(distributed_setup)
    _prepare_output_directory()
    log_file = _LOG_FILE.open("w", encoding="utf-8") if dist.get_rank() == 0 else None
    try:
        continuous_model = _build_model(
            distributed_setup,
            device,
            config.model_init_dtype,
        )
        continuous_optimizer = _build_optimizer(config, continuous_model)
        continuous_metrics = []
        for step_index in range(_TRAINING_STEPS):
            metrics = _step(
                continuous_model,
                continuous_optimizer,
                distributed_setup,
                step_index,
                device,
            )
            continuous_metrics.append(metrics)
            _write_log(log_file, "continuous", step_index + 1, *metrics)
            if step_index == 0:
                _save_checkpoint(continuous_model, continuous_optimizer)

        resumed_model = _build_model(
            distributed_setup,
            device,
            config.model_init_dtype,
        )
        resumed_optimizer = _build_optimizer(config, resumed_model)
        _load_checkpoint(resumed_model, resumed_optimizer)
        for step_index in range(1, _TRAINING_STEPS):
            resumed_metrics = _step(
                resumed_model,
                resumed_optimizer,
                distributed_setup,
                step_index,
                device,
            )
            _write_log(log_file, "resumed", step_index + 1, *resumed_metrics)
            torch.testing.assert_close(
                torch.tensor(resumed_metrics),
                torch.tensor(continuous_metrics[step_index]),
                rtol=5.0e-3,
                atol=5.0e-3,
                msg=f"step {step_index + 1} resume metrics mismatch",
            )
        if dist.get_rank() == 0:
            print(f"[{_CASE_NAME}] passed {_TRAINING_STEPS - 1} resumed steps")
    finally:
        if log_file is not None:
            log_file.close()
        _cleanup_output_directory()
        destroy_process_group()
