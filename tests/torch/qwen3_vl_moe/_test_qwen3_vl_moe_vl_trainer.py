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
"""torchrun worker tests for ``qwen3_vl_moe`` VL trainer smoke coverage."""
from __future__ import annotations

import importlib
import json
import os
import tempfile
from typing import Optional

import torch.distributed as dist

from hyper_parallel import destroy_process_group
from hyper_parallel.trainer.config import (
    AcceleratorConfig,
    CheckpointConfig,
    DataConfig,
    GradientCheckpointingConfig,
    HyperTrainerConfig,
    LoggingConfig,
    MixedPrecisionConfig,
    ModelConfig,
    OptimizerConfig,
    TrainConfig,
)
from hyper_parallel.trainer.utils.discovery import discover_model_spec
from hyper_parallel.trainer.vl_trainer import VLTrainer

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")
importlib.import_module("torch_npu")


def _output_dir(case_name: str) -> str:
    master_port = os.environ.get("MASTER_PORT", "0")
    return os.path.join(
        tempfile.gettempdir(),
        f"hp_qwen3_vl_moe_{case_name}_{master_port}",
    )


def _is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _build_args(
    case_name: str,
    world_size: int,
    vision_parallel: Optional[dict] = None,
    global_batch_size: Optional[int] = None,
) -> HyperTrainerConfig:
    """Build a minimal trainer config for the requested distributed smoke case."""
    config_overrides = {
        "vl": True,
        "text_config": {
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "mrope_section": [4, 2, 2],
            "moe_intermediate_size": 32,
            "num_experts": 8,
            "num_experts_per_tok": 2,
        },
        "vision_config": {
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "patch_size": 4,
            "temporal_patch_size": 2,
            "spatial_merge_size": 1,
            "out_hidden_size": 64,
            "num_position_embeddings": 64,
            "deepstack_visual_indexes": [],
        },
    }
    return HyperTrainerConfig(
        model=ModelConfig(
            name="qwen3_vl_moe",
            vision_parallel=vision_parallel,
            config_overrides=config_overrides,
        ),
        data=DataConfig(
            type="vl_dummy",
            max_seq_len=32,
            vl_grid_t=2,
            vl_grid_h=2,
            vl_grid_w=2,
        ),
        train=TrainConfig(
            max_steps=1,
            num_train_epochs=1,
            global_batch_size=global_batch_size or world_size,
            micro_batch_size=1,
            seed=1234,
            backend="torch",
            init_device="meta",
            local_rank=int(os.environ.get("LOCAL_RANK", "0")),
            accelerator=AcceleratorConfig(
                dp_shard=world_size,
                comm_fusion=False,
            ),
            optimizer=OptimizerConfig(
                type="adamw",
                lr=1.0e-4,
                lr_min=0.0,
                lr_decay_style="cosine",
                lr_warmup_ratio=0.0,
                max_grad_norm=1.0,
                weight_decay=0.0,
                loss_aggregation="rank_average",
            ),
            mixed_precision=MixedPrecisionConfig(enabled=False),
            gradient_checkpointing=GradientCheckpointingConfig(
                activation_checkpoint="off",
            ),
            checkpoint=CheckpointConfig(
                output_dir=_output_dir(case_name),
                save_steps=0,
                save_hf_weights=False,
            ),
            logging=LoggingConfig(
                log_steps=1,
                report_throughput=False,
            ),
        ),
    )


def _maybe_write_captured_loss(case_name: str, loss: Optional[float]) -> None:
    """Write a rank-zero loss result for launcher-side consistency checks."""
    path = os.environ.get("HP_QWEN3_VL_MOE_CAPTURE_FILE")
    if not path:
        master_port = os.environ.get("MASTER_PORT")
        if master_port:
            path = os.path.join(
                tempfile.gettempdir(),
                f"hp_qwen3_vl_moe_loss_{master_port}.json",
            )
    if not path or not _is_rank0() or loss is None:
        return
    with open(path, "w", encoding="utf-8") as file:
        json.dump({"case_name": case_name, "loss": loss}, file)


def _run_vl_case(
    case_name: str,
    world_size: int,
    vision_parallel: Optional[dict] = None,
    global_batch_size: Optional[int] = None,
) -> Optional[float]:
    """Run one distributed VL trainer step and return the captured loss when available."""
    discover_model_spec("qwen3_vl_moe")
    trainer = VLTrainer(_build_args(case_name, world_size, vision_parallel, global_batch_size))
    captured_loss: dict[str, float] = {}
    original_on_step_end = trainer.base.on_step_end

    def _capture_on_step_end(*args, **kwargs):
        loss = kwargs.get("loss")
        if loss is not None:
            captured_loss["value"] = float(loss)
        return original_on_step_end(*args, **kwargs)

    trainer.base.on_step_end = _capture_on_step_end
    trainer.train()
    if "value" not in captured_loss:
        raise AssertionError(f"{case_name} produced no captured step loss")
    loss = captured_loss["value"]
    if dist.is_initialized():
        destroy_process_group()
    return loss


def _run_vl_smoke(
    case_name: str,
    world_size: int,
    vision_parallel: Optional[dict] = None,
    global_batch_size: Optional[int] = None,
) -> None:
    _run_vl_case(case_name, world_size, vision_parallel, global_batch_size)


def test_qwen3_vl_moe_vl_dummy_smoke_1card():
    """Feature: 1-card VL trainer smoke for ``qwen3_vl_moe`` baseline."""
    _run_vl_smoke("smoke_1card", world_size=1)


def test_qwen3_vl_moe_vl_dummy_smoke_2card_dp():
    """Feature: 2-card VL trainer smoke for baseline DP/FSDP."""
    _run_vl_smoke("smoke_2card_dp", world_size=2)


def test_qwen3_vl_moe_vl_dummy_smoke_2card_vision_cp_colossal():
    """Feature: 2-card visual Encoder CP smoke in Pure Colossal mode."""
    _run_vl_smoke(
        "smoke_2card_vision_cp_colossal",
        world_size=2,
        vision_parallel={"cp": 2, "ulysses_degree": 1, "reuse_dp_shard_mesh": True},
    )


def test_qwen3_vl_moe_vl_dummy_smoke_2card_vision_dp1():
    """Feature: 2-card visual Encoder DP smoke with replicated visual params."""
    _run_vl_smoke(
        "smoke_2card_vision_dp1",
        world_size=2,
        vision_parallel={"dp_shard": 1},
    )


def test_qwen3_vl_moe_vl_dummy_smoke_2card_vision_cp_ulysses():
    """Feature: 2-card visual Encoder CP smoke in Pure Ulysses mode."""
    _run_vl_smoke(
        "smoke_2card_vision_cp_ulysses",
        world_size=2,
        vision_parallel={"cp": 2, "ulysses_degree": 2, "reuse_dp_shard_mesh": True},
    )


def test_qwen3_vl_moe_vl_dummy_smoke_2card_vision_cp_colossal_same_sample():
    """Feature: 2-card visual Encoder CP smoke with same-sample DP fanout."""
    _run_vl_smoke(
        "smoke_2card_vision_cp_colossal_same_sample",
        world_size=2,
        vision_parallel={
            "cp": 2,
            "ulysses_degree": 1,
            "reuse_dp_shard_mesh": True,
            "share_samples_across_dp": True,
        },
        global_batch_size=1,
    )


def test_qwen3_vl_moe_vl_dummy_smoke_2card_vision_async_cp_colossal():
    """Feature: 2-card visual Encoder async CP smoke in Pure Colossal mode."""
    _run_vl_smoke(
        "smoke_2card_vision_async_cp_colossal",
        world_size=2,
        vision_parallel={
            "cp": 2,
            "ulysses_degree": 1,
            "reuse_dp_shard_mesh": True,
            "async_cp": True,
        },
    )


def test_qwen3_vl_moe_vl_dummy_vision_cp_requires_reuse_opt_in_2card():
    """Feature: 2-card visual Encoder CP requires explicit dp_shard reuse opt-in."""
    try:
        _run_vl_case(
            "smoke_2card_vision_cp_requires_reuse_opt_in",
            world_size=2,
            vision_parallel={"cp": 2, "ulysses_degree": 1},
        )
    except ValueError as exc:
        if "reuse_dp_shard_mesh" not in str(exc):
            raise AssertionError(f"unexpected error message: {exc}") from exc
    else:
        raise AssertionError("vision CP without reuse_dp_shard_mesh opt-in unexpectedly succeeded")
    finally:
        if dist.is_initialized():
            destroy_process_group()


def test_qwen3_vl_moe_vl_dummy_capture_loss_1card_baseline():
    """Feature: capture 1-card baseline first-step loss for self-consistency comparison."""
    loss = _run_vl_case(
        "align_1card_baseline",
        world_size=1,
        global_batch_size=1,
    )
    _maybe_write_captured_loss("align_1card_baseline", loss)


def test_qwen3_vl_moe_vl_dummy_capture_loss_2card_dp():
    """Feature: capture 2-card baseline first-step loss for launcher-side comparison."""
    loss = _run_vl_case("align_2card_dp", world_size=2)
    _maybe_write_captured_loss("align_2card_dp", loss)


def test_qwen3_vl_moe_vl_dummy_capture_loss_2card_vision_dp1():
    """Feature: capture 2-card visual DP first-step loss."""
    loss = _run_vl_case(
        "align_2card_vision_dp1",
        world_size=2,
        vision_parallel={"dp_shard": 1},
    )
    _maybe_write_captured_loss("align_2card_vision_dp1", loss)


def test_qwen3_vl_moe_vl_dummy_capture_loss_2card_baseline_same_sample():
    """Feature: capture 2-card baseline first-step loss with same-sample DP fanout."""
    loss = _run_vl_case(
        "align_2card_baseline_same_sample",
        world_size=2,
        vision_parallel={"share_samples_across_dp": True},
        global_batch_size=1,
    )
    _maybe_write_captured_loss("align_2card_baseline_same_sample", loss)


def test_qwen3_vl_moe_vl_dummy_capture_loss_2card_vision_cp_colossal():
    """Feature: capture 2-card visual CP Pure Colossal first-step loss."""
    loss = _run_vl_case(
        "align_2card_vision_cp_colossal",
        world_size=2,
        vision_parallel={"cp": 2, "ulysses_degree": 1, "reuse_dp_shard_mesh": True},
    )
    _maybe_write_captured_loss("align_2card_vision_cp_colossal", loss)


def test_qwen3_vl_moe_vl_dummy_capture_loss_2card_vision_cp_colossal_same_sample():
    """Feature: capture same-sample visual CP Pure Colossal first-step loss."""
    loss = _run_vl_case(
        "align_2card_vision_cp_colossal_same_sample",
        world_size=2,
        vision_parallel={
            "cp": 2,
            "ulysses_degree": 1,
            "reuse_dp_shard_mesh": True,
            "share_samples_across_dp": True,
        },
        global_batch_size=1,
    )
    _maybe_write_captured_loss("align_2card_vision_cp_colossal_same_sample", loss)


def test_qwen3_vl_moe_vl_dummy_capture_loss_2card_vision_cp_ulysses():
    """Feature: capture 2-card visual CP Pure Ulysses first-step loss."""
    loss = _run_vl_case(
        "align_2card_vision_cp_ulysses",
        world_size=2,
        vision_parallel={"cp": 2, "ulysses_degree": 2, "reuse_dp_shard_mesh": True},
    )
    _maybe_write_captured_loss("align_2card_vision_cp_ulysses", loss)


def test_qwen3_vl_moe_vl_dummy_capture_loss_2card_vision_async_cp_colossal():
    """Feature: capture 2-card visual async CP Pure Colossal first-step loss."""
    loss = _run_vl_case(
        "align_2card_vision_async_cp_colossal",
        world_size=2,
        vision_parallel={
            "cp": 2,
            "ulysses_degree": 1,
            "reuse_dp_shard_mesh": True,
            "async_cp": True,
        },
    )
    _maybe_write_captured_loss("align_2card_vision_async_cp_colossal", loss)
