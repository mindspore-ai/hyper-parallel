# Copyright 2025 Huawei Technologies Co., Ltd
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
"""common net"""
import torch
from torch import nn


class SimpleModel(nn.Module):
    """simple model"""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(8, 8).npu())

    def forward(self, x):
        x = torch.matmul(x, self.weight)
        x = torch.relu(x)
        x = torch.sum(x)
        return x


class DenseNet(nn.Module):
    """Dense model"""
    def __init__(self, in_channels, hidden_size, has_bias=True):
        super().__init__()
        self.has_bias = has_bias
        self.weight = nn.Parameter(torch.ones(in_channels, hidden_size).npu())
        if self.has_bias:
            self.bias = nn.Parameter(torch.zeros(hidden_size).npu())

    def forward(self, x):
        x = torch.matmul(x, self.weight)
        if self.has_bias:
            x = torch.add(x, self.bias)
        return x


class DenseMutiLayerNet(nn.Module):
    """dense net with configurable layer number"""
    def __init__(self, hidden_size, layer_num, has_bias=True):
        super().__init__()
        self.layer_num = layer_num
        self.layers = nn.Sequential()
        for i in range(self.layer_num):
            layer = DenseNet(hidden_size, hidden_size, has_bias)
            self.layers.add_module(f"layer{i}", layer)

    def forward(self, x):
        for i in range(self.layer_num):
            x = self.layers[i](x)
        x = torch.sum(x)
        return x


class TransformerBlock(nn.Module):
    """A simple Transformer block for testing purposes."""

    def __init__(self, dim=256, num_heads=4):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self.norm1(x)
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x


class SimpleTransformer(nn.Module):
    """A simple Transformer model for testing purposes."""

    def __init__(self, vocab_size=1000, dim=2048, depth=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        for block in self.layers:
            x = block(x)
        x = self.norm(x)
        return self.head(x)

class FullyShardTestNet(nn.Module):
    """net for fully_shard test"""
    def __init__(self, dense_hidden, dense_layer_num, has_bias=True):
        super().__init__()
        self.w1 = nn.Parameter(torch.rand(dense_hidden, dense_hidden).npu())
        self.dense_layers = DenseMutiLayerNet(dense_hidden, dense_layer_num, has_bias)

    def forward(self, x):
        w1_out = torch.matmul(x, self.w1)
        layers_out = self.dense_layers(w1_out)
        out = torch.sum(layers_out)
        return out
