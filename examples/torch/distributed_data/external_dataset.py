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
"""CPU example: config -> dataset contract -> device-ready dataloader.

Run with ``torchrun --master-addr=127.0.0.1 --nproc-per-node=2
examples/torch/distributed_data/external_dataset.py``.
"""

import logging
from collections.abc import Sequence
from typing import Any

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import
from torch.distributed.device_mesh import init_device_mesh  # pylint: disable=forbidden-backend-import

from hyper_parallel.distributed_data import (
    DistributedDatasetConfig,
    SampleMetadata,
    build_distributed_dataloader,
    build_distributed_dataset,
)

MODEL_CONFIG = {
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "intermediate_size": 128,
    "mlp_layer_types": ["dense", "dense"],
    "kv_lora_rank": 16,
    "q_lora_rank": None,
    "qk_nope_head_dim": 8,
    "qk_rope_head_dim": 8,
    "v_head_dim": 16,
}


def collate_tokens(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """An ordinary model collator, with no planner or transport knowledge.

    Args:
        samples: Token sequences assigned to one packing bin.
    """
    lengths = torch.tensor([len(sample["input_ids"]) for sample in samples])
    return {
        "input_ids": torch.cat([sample["input_ids"] for sample in samples]),
        "offsets": torch.cat((torch.zeros(1, dtype=torch.int64), lengths.cumsum(0))),
    }


def main() -> None:
    """Bind already-selected steps and consume batches with a normal loop."""
    logging.basicConfig(level=logging.INFO)
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    mesh = init_device_mesh("cpu", (dist.get_world_size(),), mesh_dim_names=("dp",))
    lengths = (100, 90) if rank % 2 == 0 else (10, 20)
    source = [
        [[{
            "input_ids": torch.full((length,), rank + step, dtype=torch.int64),
            "metadata": SampleMetadata(pack_tokens=length, features={"P": length, "D": 0}),
        } for length in lengths]]
        for step in range(3)
    ]
    config = DistributedDatasetConfig(seq_len=300, local_batch_size=1, min_balance_gain=0.0)
    dataset = build_distributed_dataset(
        source, metadata="metadata", collate_fn=collate_tokens,
        cpu_fields=("offsets",), log_fields=("P", "D"),
    )
    try:
        with build_distributed_dataloader(
            dataset, mesh, config, model_config=MODEL_CONFIG, device="cpu", max_steps=3,
        ) as loader:
            for microbatches in loader:
                logging.info("rank=%d step=%d tokens=%d", rank, loader.step, microbatches[0]["input_ids"].numel())
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
