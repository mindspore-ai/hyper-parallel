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
"""TP2 + graph-FSDP2 prototype reusing AutoModel + TextTrainer."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset
from transformers import LlamaConfig

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from hyper_parallel.compile import (  # pylint: disable=C0413
    GraphTrainer,
    create_simple_sharding_plan,
)
from hyper_parallel.components.checkpoint.config import (  # pylint: disable=C0413
    CheckpointingConfig,
)
from hyper_parallel.components.optim.builders import AdamW  # pylint: disable=C0413
from hyper_parallel.components.optim.lr_scheduler import (  # pylint: disable=C0413
    MultiLRScheduler,
)
from hyper_parallel.models._transformers import (  # pylint: disable=C0413
    HyperAutoModelForCausalLM,
)
from hyper_parallel.models.build_options import FSDP2Config  # pylint: disable=C0413
from hyper_parallel.trainer.config import (  # pylint: disable=C0413
    AcceleratorConfig,
    DataLoaderConfig,
    DatasetConfig,
    OptimizerConfig,
    Target,
    TrainerConfig,
    TrainingConfig,
)


class TinyTextDataset(Dataset):
    """Tiny fixed-shape dataset for graph-mode smoke tests."""

    def __init__(self, *, vocab_size: int, seq_len: int, size: int) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        gen = torch.Generator().manual_seed(index)
        input_ids = torch.randint(
            low=0,
            high=self.vocab_size,
            size=(self.seq_len,),
            generator=gen,
        )
        labels = input_ids.roll(shifts=-1)
        labels[-1] = -100
        return {"input_ids": input_ids, "labels": labels}


def build_tiny_automodel(
    *,
    distributed_setup=None,
    compile_config=None,
    activation_checkpoint=None,
    swap_inputs: bool = False,
    activation_swap: str = "none",
    model_init_dtype=None,
) -> torch.nn.Module:
    """Build a tiny TP-compatible AutoModel through the shared facade."""
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    return HyperAutoModelForCausalLM.from_config(
        config,
        distributed_setup=distributed_setup,
        compile_config=compile_config,
        activation_checkpoint=activation_checkpoint,
        swap_inputs=swap_inputs,
        activation_swap=activation_swap,
        model_init_dtype=model_init_dtype,
    )


def build_dummy_dataset(
    *,
    transform=None,
    tokenizer=None,
    mesh_context=None,
    training_config=None,
) -> Dataset:
    """Build the tiny text dataset expected by the trainer."""
    del transform, tokenizer, mesh_context, training_config
    return TinyTextDataset(vocab_size=128, seq_len=16, size=8)


def build_simple_collate_fn(*, mesh_context=None):
    """Stack per-sample tensors into one local micro-batch."""
    del mesh_context

    def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            key: torch.stack([sample[key] for sample in batch], dim=0)
            for key in batch[0].keys()
        }

    return _collate


class SimpleGetBatch:
    """Minimal DataLoader-to-model batch adapter for the graph prototype."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def __call__(
        self,
        data_iterator: Any,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        batch = next(data_iterator)
        batch = {
            key: value.to(self.device, non_blocking=True)
            for key, value in batch.items()
        }
        input_ids = batch["input_ids"]
        batch["attention_mask"] = torch.ones_like(input_ids)
        batch["position_ids"] = torch.arange(
            input_ids.shape[1],
            device=input_ids.device,
        ).unsqueeze(0).expand_as(input_ids)
        loss_inputs = {"labels": batch["labels"]}
        return batch, loss_inputs


def build_simple_get_batch(
    *,
    mesh_context=None,
    device=None,
    tokenizer=None,
    data_config=None,
    pp_shared_data: bool = False,
) -> SimpleGetBatch:
    """Build the graph prototype batch adapter."""
    del mesh_context, tokenizer, data_config
    if pp_shared_data:
        raise ValueError("The tp2+fsdp2 graph prototype does not support pp_shared_data")
    return SimpleGetBatch(device=device)


def build_simple_dataloader(
    *,
    dataset: Dataset,
    collate_fn,
    batch_sampler=None,
    batch_size: int = 1,
    dp_world_size: int = 1,
    max_seq_len=None,
    seed: int = 0,
    drop_last: bool = False,
    use_background_prefetcher: bool = False,
) -> TorchDataLoader:
    """Build a plain PyTorch DataLoader for the graph prototype."""
    del dp_world_size, max_seq_len, seed, use_background_prefetcher
    if batch_sampler is not None:
        return TorchDataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
        )
    return TorchDataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=drop_last,
    )


def build_config() -> TrainerConfig:
    """Create the tp2+fsdp2 trainer config used by the graph prototype."""
    train_iters = int(os.environ.get("GRAPH_TRAIN_ITERS", "2"))
    return TrainerConfig(
        model=Target(
            _target_=build_tiny_automodel,
            target_path="__main__.build_tiny_automodel",
        ),
        optimizer=OptimizerConfig(
            target=Target(
                _target_=AdamW,
                target_path="hyper_parallel.components.optim.builders.AdamW",
                adamw_config={"lr": 1.0e-4, "adamw_weight_decay": 0.0},
            ),
        ),
        lr_scheduler=Target(
            _target_=MultiLRScheduler,
            target_path="hyper_parallel.components.optim.lr_scheduler.MultiLRScheduler",
            lr_decay_style="constant",
            lr_config={"lr": 1.0e-4, "lr_warmup_steps": 0},
        ),
        training=TrainingConfig(
            train_iters=train_iters,
            global_batch_size=4,
            micro_batch_size=1,
            logging_steps=1,
        ),
        accelerator=AcceleratorConfig(tp_size=2),
        fsdp_config=FSDP2Config(dp_shard_size=2),
        dataset=DatasetConfig(
            target=Target(
                _target_=build_dummy_dataset,
                target_path="__main__.build_dummy_dataset",
            ),
        ),
        dataloader=DataLoaderConfig(
            target=Target(
                _target_=build_simple_dataloader,
                target_path="__main__.build_simple_dataloader",
                drop_last=False,
                use_background_prefetcher=False,
            ),
            collate_fn=Target(
                _target_=build_simple_collate_fn,
                target_path="__main__.build_simple_collate_fn",
            ),
            get_batch=Target(
                _target_=build_simple_get_batch,
                target_path="__main__.build_simple_get_batch",
            ),
        ),
        checkpoint=CheckpointingConfig(save_ckpt=False),
    )


def _selected_param_shapes(model: torch.nn.Module) -> dict[str, list[int]]:
    """Collect a compact shape summary for a few representative parameters."""
    names = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "lm_head.weight",
    ]
    params = dict(model.named_parameters())
    summary = {}
    for name in names:
        if name in params:
            summary[name] = list(params[name].shape)
    return summary


def log_rank_summary(trainer: Any, stage: str) -> None:
    """Print mesh ranks and representative parameter shapes rank by rank."""
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    mesh = trainer.base.mesh
    graph_cfg = trainer.graph_executor.pass_config
    payload = {
        "rank": rank,
        "dp_rank": getattr(mesh, "dp_rank", 0),
        "tp_rank": getattr(mesh, "tp_rank", 0),
        "dp_size": getattr(mesh, "dp_size", 1),
        "tp_size": getattr(mesh, "tp_size", 1),
        "graph_fsdp_enabled": getattr(graph_cfg, "fsdp_enabled", False),
        "graph_fsdp_degree": getattr(graph_cfg, "fsdp_degree", None),
        "param_shapes": _selected_param_shapes(trainer.base.model),
    }
    for current_rank in range(world_size):
        dist.barrier()
        if rank == current_rank:
            print(f"[{stage}] {payload}", flush=True)
    dist.barrier()


def main() -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29631")

    trainer = GraphTrainer.from_text_config(
        build_config(),
        pass_plan=create_simple_sharding_plan(),
    )
    log_rank_summary(trainer, "before_train")
    trainer.train()
    if dist.is_initialized():
        log_rank_summary(trainer, "after_train")
    print(
        f"[after_train_local] rank={os.environ.get('RANK', '0')} "
        f"param_shapes={_selected_param_shapes(trainer.base.model)}",
        flush=True,
    )
    if not dist.is_initialized() or int(os.environ.get("RANK", "0")) == 0:
        print("tp2+fsdp2 graph prototype finished", flush=True)


if __name__ == "__main__":
    main()
