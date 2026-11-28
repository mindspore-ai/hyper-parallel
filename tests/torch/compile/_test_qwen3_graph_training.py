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
"""Train a tiny Qwen3 model through the graph-mode TextTrainer."""

from pathlib import Path

from transformers import Qwen3Config, Qwen3ForCausalLM

from hyper_parallel.trainer.config import TrainerConfig
from hyper_parallel.trainer.config.resolver import resolve_config
from hyper_parallel.trainer.text_trainer import TextTrainer


_TRAIN_STEPS = 2
_SEQUENCE_LENGTH = 32
_VOCAB_SIZE = 128
_EOD_TOKEN_ID = _VOCAB_SIZE - 1


def _build_training_config(model_dir: Path) -> TrainerConfig:
    """Resolve the smoke-test configuration independently of the source checkout."""
    learning_rate = 1.0e-4
    return resolve_config({
        "model_init_dtype": "bfloat16",
        "model": {
            "_target_": "hyper_parallel.models.HyperAutoModelForCausalLM.from_pretrained",
            "pretrained_model_name_or_path": str(model_dir),
            "load_base_model": False,
            "torch_dtype": "bfloat16",
            "attn_implementation": "sdpa",
            "force_hf": True,
        },
        "optimizer": {
            "_target_": "hyper_parallel.components.optim.builders.AdamW",
            "adamw_config": {"lr": learning_rate, "adamw_weight_decay": 0.0},
        },
        "lr_scheduler": {
            "_target_": "hyper_parallel.components.optim.lr_scheduler.MultiLRScheduler",
            "lr_decay_style": "constant",
            "lr_config": {"lr": learning_rate, "lr_warmup_steps": 0},
        },
        "training": {"train_iters": _TRAIN_STEPS, "global_batch_size": 4, "backend": "hccl", "seed": 42},
        "accelerator": {"tp_size": 2, "sequence_parallel": True},
        "fsdp_config": {"dp_shard_size": 2},
        "activation_checkpoint": {"mode": "full"},
        "compile": {"enabled": True, "use_joint_graph": True},
        "dataset": {
            "_target_": "hyper_parallel.data.text.build_dataset.build_indexed_text_dataset",
            "model_assets": {
                "tokenizer": {
                    "_target_": "hyper_parallel.data.text.build_tokenizer.AutoTokenizer.from_pretrained",
                    "tokenizer_type": "pretokenized",
                    "vocab_size": _VOCAB_SIZE,
                    "eod_token_id": _EOD_TOKEN_ID,
                },
            },
            "data_config": {
                "seq_length": _SEQUENCE_LENGTH,
                "mock_data": True,
                "simple_blend": "no",
                "labels_are_shifted": True,
                "low_level_dataset": {
                    "_target_": "hyper_parallel.data.indexed.indexed_pretrain_dataset.MockGPTDataset",
                },
            },
        },
        "dataloader": {
            "_target_": "hyper_parallel.data.batching.FixedBatchDataLoader",
            "collate_fn": {"_target_": "hyper_parallel.data.batching.build_indexed_collate_fn"},
            "get_batch": {
                "_target_": "hyper_parallel.data.batching.TextParallelBatch",
                "source_type": "indexed",
            },
        },
        "checkpoint": {"save_ckpt": False},
    })


def test_qwen3_graph_training(tmp_path: Path) -> None:
    """Check optimizer progress and the resulting FSDP and recomputation graph."""
    Qwen3Config.from_dict({
        "vocab_size": _VOCAB_SIZE,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "max_position_embeddings": _SEQUENCE_LENGTH,
        "eos_token_id": _EOD_TOKEN_ID,
        "tie_word_embeddings": False,
        "use_cache": False,
    }).save_pretrained(tmp_path)

    trainer = TextTrainer(_build_training_config(tmp_path))
    trainer.train()

    model = trainer.base.model
    assert isinstance(model, Qwen3ForCausalLM), f"Expected a Qwen3 model, got {type(model).__name__}"
    step = trainer.base.state.global_step
    assert step == _TRAIN_STEPS, f"Expected {_TRAIN_STEPS} optimizer steps, got {step}"

    compiler = trainer.base.graph_compiler
    assert compiler is not None, f"Expected a graph compiler after step {step}, got {compiler}"
    joint_graph = compiler._joint_graph  # pylint: disable=protected-access
    assert joint_graph is not None, f"Expected a captured joint graph after step {step}, got {joint_graph}"
    graph_module = joint_graph.graph_module
    assert graph_module.fsdp_sharded_param_names, f"Expected FSDP parameter shards after step {step}"
    assert any(node.meta.get("ac_recomputed") for node in graph_module.graph.nodes), (
        f"Expected recomputed nodes in the captured graph for mode={compiler.pass_config.ac_mode}"
    )
