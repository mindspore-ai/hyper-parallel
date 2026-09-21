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

"""Real TextTrainer/EP/FSDP smoke test with a small four-layer DSV4.1 crop."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch
from typing import Any

import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, Dataset

from hyper_parallel.components.functional.sinkhorn import sinkhorn_knopps
from hyper_parallel.components.checkpoint.dcp_checkpointer import DistributedCheckpointer
from hyper_parallel.components.modules.shared_compressed_dsa_attention import _reference_sparse_attention
from hyper_parallel.models import HyperAutoModelForCausalLM
from hyper_parallel.models.deepseek_v41.adapter.megamoe_training import DeepseekV41TrainingExperts
from hyper_parallel.trainer.config.manager import parse_training_args
from examples.training_demo.train_deepseek_v41_megamoe import MegaMoeTextTrainer
from tests.ut.auto_models.models.deepseek_v41.test_deepseek_v41_crop import _tiny_config, _write_engram_assets

_TARGET = "tests.torch.multicore._test_deepseek_v41_megamoe_training."


def build_model(assets_directory: str, distributed_setup: Any = None,
                model_init_dtype: str = "float32") -> torch.nn.Module:
    """Build through the same AutoModel entrypoint used by the public recipe.

    Args:
        assets_directory: Rank-specific directory for synthetic Engram assets.
        distributed_setup: Trainer-resolved mesh and replacement configuration.
        model_init_dtype: Storage dtype before mixed-precision FSDP execution.
    """
    config = _tiny_config(_write_engram_assets(assets_directory))
    config.hidden_size = 512
    config.moe_intermediate_size = config.intermediate_size = 128
    config.swiglu_limit = 0.0
    config.v41_source_swiglu_limit = 10.0
    return HyperAutoModelForCausalLM.from_config(
        config, distributed_setup=distributed_setup, torch_dtype="bfloat16",
        attn_implementation="eager", model_init_dtype=model_init_dtype,
    )


class TokenDataset(Dataset):
    """Small deterministic, rank-sharded text samples without tokenizer assets."""

    def __len__(self) -> int:
        """Provide enough samples for three accumulated steps."""
        return 128

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one reproducible token sequence with next-token targets."""
        tokens = (torch.arange(128) + index) % 60 + 3
        return {"input_ids": tokens, "labels": tokens.roll(-1)}


def build_loader(dataset: Dataset, collate_fn: Any, batch_sampler: Any) -> DataLoader:
    """Use the Trainer's actual distributed sampler and online collator.

    Args:
        dataset: Fixed token samples.
        collate_fn: Recipe's packed text collator.
        batch_sampler: Trainer's actual DP sampler.
    """
    return DataLoader(dataset, collate_fn=collate_fn, batch_sampler=batch_sampler)


def _portable_attention(query, key_value, sparse_indices, sinks, _rope_head_dim, scale):
    """Keep real attention math while isolating MegaMoe from optional custom ops."""
    return _reference_sparse_attention(query, key_value, sparse_indices, sinks, scale)


def _recipe(work, world_size, ep_size, dispatch_mode):
    path = Path(__file__).resolve().parents[3] / "examples/training_demo/train_deepseek_v41_megamoe.yaml"
    recipe = yaml.safe_load(path.read_text(encoding="utf-8"))
    recipe["model_init_dtype"] = os.environ.get("HP_DSV41_MODEL_INIT_DTYPE", "float32")
    recipe["model"] = {"_target_": _TARGET + "build_model", "assets_directory": str(work)}
    recipe["training"].update(train_iters=3, global_batch_size=world_size * 2)
    recipe["accelerator"]["ep_size"] = ep_size
    recipe["fsdp_config"].update(dp_shard_size=world_size, edp_shard_size=world_size // ep_size)
    recipe["plan_overrides"][-1]["local_compute_fn"].update(local_num_tokens=128, dispatch_mode=dispatch_mode)
    recipe["dataset"] = {"_target_": _TARGET + "TokenDataset", "data_config": {
        "create_attention_mask_in_dataloader": True, "labels_are_shifted": True,
    }}
    # Dataset constructors have no data_config argument; keep it on a factory.
    recipe["dataset"]["_target_"] = _TARGET + "build_dataset"
    recipe["dataloader"] = {
        "_target_": _TARGET + "build_loader",
        "collate_fn": recipe["dataloader"]["collate_fn"],
        "get_batch": recipe["dataloader"]["get_batch"],
        "dataloader_type": "single",
    }
    recipe["checkpoint"]["checkpoint_dir"] = str(work.parent / "trainer-state")
    target = work / "recipe.yaml"
    target.write_text(yaml.safe_dump(recipe), encoding="utf-8")
    return parse_training_args([str(target)])


def build_dataset(data_config: dict | None = None) -> Dataset:
    """Build fixed token samples with the real batch adapter's metadata.

    Args:
        data_config: Metadata consumed separately by the batching adapter.
    """
    del data_config
    return TokenDataset()


def _local(tensor):
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def _snapshot(model):
    return {name: _local(value).detach().cpu().clone() for name, value in model.state_dict().items()}


def _check_checkpoint(base, directory):
    before = _snapshot(base.model)
    checkpointer = DistributedCheckpointer()
    payload = {"model": base.model.state_dict()}
    checkpointer.save(str(directory), payload, global_step=base.state.global_step)
    with torch.no_grad():
        for parameter in base.model.parameters():
            _local(parameter).zero_()
    restored = {"model": base.model.state_dict()}
    checkpointer.load(str(directory), restored)
    base.model.load_state_dict(restored["model"])
    for name, value in _snapshot(base.model).items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_deepseek_v41_megamoe_training() -> None:
    """Exercise real Trainer accumulation, FP32 main parameters and checkpoint."""
    root = Path(os.environ["HP_DSV41_TRAINING_OUTPUT"])
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    ep_size = int(os.environ.get("HP_DSV41_TRAINING_EP", "2"))
    mode = os.environ.get("HP_DSV41_TRAINING_MODE", "push")
    work = root / f"rank{rank}"
    work.mkdir(parents=True, exist_ok=True)
    config = _recipe(work, world, ep_size, mode)
    trainer = MegaMoeTextTrainer(config)
    base = trainer.base
    experts = [module for module in base.model.modules() if isinstance(module, DeepseekV41TrainingExperts)]
    assert len(experts) == 4, "Expected four real MegaMoe expert modules"
    assert base.num_micro_batches == 2, "Expected gradient accumulation"
    for module in experts:
        state = module.hsdp_scheduler.hsdp_state
        assert state.mesh.size() == world // ep_size, "Expert FSDP must reduce only across EDP"
        assert set(state.mesh.mesh_dim_names) <= {"edp_shard", "edp_replicate"}, "EP must not reduce expert dW"
        assert state.mp_policy.reduce_dtype == torch.float32, "Expected FP32 EDP gradient communication"
    initial = _snapshot(base.model)
    metrics = []
    target = "hyper_parallel.components.modules.shared_compressed_dsa_attention.npu_sparse_attention_with_scalar_sink"
    with patch(target, _portable_attention), \
            patch("hyper_parallel.components.functional.mhc_post.omni_training_custom_ops", None), \
            patch("hyper_parallel.components.modules.mhc.sinkhorn", sinkhorn_knopps):
        trainer.on_train_begin()
        iterator = iter(base.train_dataloader)
        for step in range(3):
            metrics.append(trainer.train_step(iterator))
            finite = all(torch.isfinite(torch.tensor(value)) for value in metrics[-1].values())
            assert finite, "Nonfinite training metric"
            if step == 1:
                _check_checkpoint(base, root / "checkpoint")
        trainer.on_train_end()
    current = _snapshot(base.model)
    updated = []
    for name, value in current.items():
        assert torch.isfinite(value).all(), f"Nonfinite parameter {name}"
        if ".mlp.experts." in name and not torch.equal(value, initial[name]):
            updated.append(name)
    assert len(updated) == 8, "Expected both expert tensors to update in all four layers"
    assert all(module._kernel is None for module in experts), "Trainer must close all native executors"
    report = {"passed": True, "world_size": world, "ep_size": ep_size, "edp_shard_size": world // ep_size,
              "dispatch_mode": mode, "model_init_dtype": config.model_init_dtype,
              "micro_batches": 2, "steps": 3, "metrics": metrics,
              "model_checkpoint_exact": True, "updated_expert_tensors": updated,
              "attention": "portable_reference_on_npu", "mhc": "pipelined_portable_math_on_npu"}
    (work / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    dist.barrier()
    dist.destroy_process_group()
