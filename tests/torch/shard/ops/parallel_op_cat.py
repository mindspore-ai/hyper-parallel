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
"""test torch dtensor with distributed cat"""
import numpy as np
import torch
import torch.nn.functional as F
from hyper_parallel import Layout
from tests.torch.utils import init_dist
from tests.torch.shard.utils import local_to_global, global_to_local

# Generate input data using numpy at file header
np.random.seed(42)
vocab_size = 128
embed_dim = 32
batch_size = 8
seq_len = 16

# Input indices: (Batch, Seq)
standalone_input_np = np.random.randint(0, vocab_size, (batch_size, seq_len)).astype(np.int64)
# Embedding Table: (Vocab, Embed_Dim)
standalone_weight_np = np.random.randn(vocab_size, embed_dim).astype(np.float32)

def test_distributed_embedding_layout_inference():
    """
    Feature: dtensor + torch.nn.functional.embedding layout inference
    Description:
        ▪ Test layout inference for embedding op in distributed setting.
        ▪ Scenario: Column Parallel Embedding (splitting embedding dimension).
        ▪ Logic:
            - Input is sharded on Batch dim ("dp").
            - Weight is sharded on Embedding dim ("tp").
            - Output should be sharded on (Batch, Sequence, Embedding) -> ("dp", None, "tp").
    Expectation: Success.
    """
    init_dist()

    # Standalone (single-device) execution for ground truth
    standalone_input = torch.from_numpy(standalone_input_np).npu()
    standalone_weight = torch.from_numpy(standalone_weight_np).npu()

    # torch.nn.functional.embedding(input, weight)
    standalone_output = F.embedding(standalone_input, standalone_weight)

    # Distributed setup
    # Mesh: (2, 4), Alias: ("dp", "tp")
    layout = Layout((2, 4), ("dp", "tp"))

    # 1. Define Input Layout: Shard on Batch (dim 0), Keep Seq (dim 1)
    # Map: ("dp", "None")
    input_layout = layout("dp", "None")

    # 2. Define Weight Layout: Keep Vocab (dim 0), Shard Embed (dim 1)
    # Map: ("None", "tp") -> This simulates Column Parallelism commonly used in LLMs
    weight_layout = layout("None", "tp")

    # 3. Define Expected Output Layout:
    # Logic in parallel_embedding.py: output_map = input_map + (weight_map[-1],)
    # Result: ("dp", "None") + ("tp",) -> ("dp", "None", "tp")
    expected_output_layout = layout("dp", "None", "tp")

    # Convert global view to local shards
    dist_input = global_to_local(standalone_input, input_layout)
    dist_weight = global_to_local(standalone_weight, weight_layout)

    # Execute distributed op
    # Note: Pass strictly as positional args to ensure compatibility with _op_dispatch
    dist_output = F.embedding(dist_input, dist_weight)

    # Layout correctness check
    assert dist_output.layout == expected_output_layout, \
        f"Embedding output layout mismatch.\nExpected: {expected_output_layout}\nGot: {dist_output.layout}"

    # Gather distributed results back to global view
    gathered_output = local_to_global(dist_output)

    # Value correctness check
    assert torch.allclose(
        standalone_output, gathered_output, atol=1e-5
    ), "Embedding output values mismatch between standalone and distributed"
