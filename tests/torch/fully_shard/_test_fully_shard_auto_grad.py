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
"""Test fully_shard autograd scheduling and standalone parity."""
# pylint: disable=W0611,C0413,C0412
from copy import deepcopy

import torch
import torch_npu
from torch import nn, optim
from torch.nn import functional
from hyper_parallel import DTensor, init_device_mesh, SkipDTensorDispatch
from hyper_parallel.core.fully_shard.api import fully_shard
from hyper_parallel.core.fully_shard.utils import MixedPrecisionPolicy
from tests.torch.common_net import SimpleTransformer
from tests.torch.fully_shard.mem_tracker import MemTrackerWrapper
from tests.torch.utils import init_dist


class TransformerWithChunkLoss(SimpleTransformer):
    """Apply the Transformer output head and cross-entropy one chunk at a time.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        dim: Hidden dimension.
        depth: Number of Transformer blocks.
        num_chunks: Number of sequence chunks passed through the shared head.
    """

    def __init__(self, vocab_size: int, dim: int, depth: int, num_chunks: int = 4) -> None:
        """Initialize the Transformer and its chunked loss configuration."""
        super().__init__(vocab_size=vocab_size, dim=dim, depth=depth)
        self.num_chunks = num_chunks

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return summed cross-entropy over local tokens.

        Args:
            tokens: Input token IDs.
            targets: Target token IDs aligned with inputs.

        Returns:
            The sum of per-token losses across chunks.
        """
        hidden = self.embed(tokens)
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.norm(hidden)
        chunk_losses = []
        for hidden_chunk, target_chunk in zip(
                torch.chunk(hidden, self.num_chunks, dim=1), torch.chunk(targets, self.num_chunks, dim=1)):
            logits = self.head(hidden_chunk)
            chunk_losses.append(functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), target_chunk.reshape(-1), reduction="sum"
            ))
        return torch.stack(chunk_losses).sum()


def _apply_fully_shard_with_prefetch(model, mesh) -> None:
    """Shard the model in-place and configure prefetch in execution order.

    Each transformer block is an independent fully_shard unit. The root owns
    the embedding, final normalization, and output head. Forward and backward
    prefetch follow the same root-to-leaf and leaf-to-root order as autograd.
    """
    mixed_precision_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
        cast_forward_inputs=False,
    )
    for transformer_layer in model.layers:
        fully_shard(transformer_layer, mesh=mesh, mp_policy=mixed_precision_policy)
    fully_shard(model, mesh=mesh, mp_policy=mixed_precision_policy)
    model.set_reduce_op_type("sum")

    model.set_modules_to_forward_prefetch([model.layers[0]])
    for transformer_layer, next_transformer_layer in zip(model.layers[:-1], model.layers[1:]):
        transformer_layer.set_modules_to_forward_prefetch([next_transformer_layer])

    model.set_modules_to_backward_prefetch([model.layers[-1]])
    for transformer_layer, previous_transformer_layer in zip(
            reversed(model.layers[1:]), reversed(model.layers[:-1])):
        transformer_layer.set_modules_to_backward_prefetch([previous_transformer_layer])


def _assert_accumulated_gradients_match(standalone_model, sharded_model, rank: int, training_step: int) -> None:
    """Compare every accumulated full gradient without changing model state."""
    standalone_parameters = dict(standalone_model.named_parameters())
    sharded_parameters = dict(sharded_model.named_parameters())
    assert standalone_parameters.keys() == sharded_parameters.keys(), (
        f"Rank {rank}, step {training_step}: parameter names differ, "
        f"standalone={standalone_parameters.keys()}, sharded={sharded_parameters.keys()}"
    )

    for parameter_name, standalone_parameter in standalone_parameters.items():
        sharded_parameter = sharded_parameters[parameter_name]
        assert standalone_parameter.grad is not None and sharded_parameter.grad is not None, (
            f"Rank {rank}, step {training_step}: missing gradient for {parameter_name}, "
            f"standalone={standalone_parameter.grad}, sharded={sharded_parameter.grad}"
        )
        assert isinstance(sharded_parameter.grad, DTensor), (
            f"Rank {rank}, step {training_step}: expected DTensor gradient for {parameter_name}, "
            f"standalone={type(standalone_parameter.grad)}, sharded={type(sharded_parameter.grad)}"
        )
        sharded_full_gradient = sharded_parameter.grad.full_tensor()
        assert standalone_parameter.grad.shape == sharded_full_gradient.shape, (
            f"Rank {rank}, step {training_step}: gradient shape differs for {parameter_name}, "
            f"standalone={standalone_parameter.grad.shape}, sharded={sharded_full_gradient.shape}"
        )
        assert torch.allclose(standalone_parameter.grad, sharded_full_gradient, rtol=1e-4, atol=1e-5), (
            f"Rank {rank}, step {training_step}: accumulated gradient differs for {parameter_name}, "
            f"standalone={standalone_parameter.grad}, sharded={sharded_full_gradient}"
        )


def _run_fully_shard_autograd_parity(mesh_shape, mesh_dim_names, forward_pass_count: int) -> None:
    """Run a shared standalone-parity flow for FSDP, HSDP, and FFBB."""
    rank, _ = init_dist()
    world_size = torch.distributed.get_world_size()
    mesh = init_device_mesh(device_type="npu", mesh_shape=mesh_shape, mesh_dim_names=mesh_dim_names)

    vocabulary_size = 32
    model_dimension = 16
    transformer_depth = 2
    local_batch_size = 2
    sequence_length = 4
    training_steps = 2
    global_batch_size = world_size * local_batch_size

    torch.manual_seed(2026)
    base_model = SimpleTransformer(
        vocab_size=vocabulary_size,
        dim=model_dimension,
        depth=transformer_depth,
    )
    standalone_model = deepcopy(base_model).npu()
    sharded_model = deepcopy(base_model).npu()
    _apply_fully_shard_with_prefetch(sharded_model, mesh)

    standalone_optimizer = optim.Adam(standalone_model.parameters(), lr=1e-3)
    sharded_optimizer = optim.Adam(sharded_model.parameters(), lr=1e-3)
    input_generator = torch.Generator().manual_seed(84)
    local_batch_start = rank * local_batch_size
    total_samples = global_batch_size * forward_pass_count

    for training_step in range(training_steps):
        global_token_batches = [
            torch.randint(
                vocabulary_size,
                (global_batch_size, sequence_length),
                generator=input_generator,
            ).npu()
            for _ in range(forward_pass_count)
        ]
        local_token_batches = [
            token_batch.narrow(0, local_batch_start, local_batch_size)
            for token_batch in global_token_batches
        ]

        with SkipDTensorDispatch():
            standalone_outputs = [standalone_model(token_batch) for token_batch in global_token_batches]
            sharded_outputs = [sharded_model(token_batch) for token_batch in local_token_batches]

            for forward_index, (standalone_output, sharded_output) in enumerate(
                    zip(standalone_outputs, sharded_outputs)):
                expected_local_output = standalone_output.narrow(0, local_batch_start, local_batch_size)
                assert torch.allclose(expected_local_output, sharded_output, rtol=1e-5, atol=1e-6), (
                    f"Rank {rank}, step {training_step}, forward {forward_index}: output differs, "
                    f"standalone={expected_local_output}, sharded={sharded_output}"
                )

            for standalone_output in standalone_outputs:
                (standalone_output.sum() / total_samples).backward()
            for sharded_output in sharded_outputs:
                (sharded_output.sum() / total_samples).backward()

        _assert_accumulated_gradients_match(standalone_model, sharded_model, rank, training_step)

        with SkipDTensorDispatch():
            standalone_optimizer.step()
            sharded_optimizer.step()
            standalone_optimizer.zero_grad(set_to_none=True)
            sharded_optimizer.zero_grad(set_to_none=True)

    sharded_model.reset_iter_state()


def test_single_rank_fsdp_autograd_parity():
    """Verify a one-rank FSDP mesh matches standalone forward and backward."""
    _run_fully_shard_autograd_parity((1,), ("dp",), forward_pass_count=1)


def test_single_rank_hsdp_autograd_parity():
    """Verify a one-rank HSDP mesh matches standalone forward and backward."""
    _run_fully_shard_autograd_parity((1, 1), ("replicate", "shard"), forward_pass_count=1)


def test_hsdp_ffbb_autograd_parity():
    """Verify two-rank HSDP replica reduction with forward-forward-backward-backward."""
    _run_fully_shard_autograd_parity((2, 1), ("replicate", "shard"), forward_pass_count=2)


def test_fsdp_transformer_chunkloss_ga_mem() -> None:
    """Check Transformer chunk loss, accumulated gradients, and RS storage lifetime.

    Feature: FSDP gradient accumulation.
    Description: Reuse a sharded output head for sequence chunks over accumulated microbatches.
    Expectation: Loss and gradients match dense CE, and previous RS outputs are released promptly.
    """
    rank, _ = init_dist()
    world_size = torch.distributed.get_world_size()
    mesh = init_device_mesh(device_type="npu", mesh_shape=(world_size,), mesh_dim_names=("dp",))
    local_batch_size, microbatches, training_steps, sequence_length = 2, 3, 2, 32
    global_batch_size = world_size * local_batch_size
    model_config = {"vocab_size": 1024, "dim": 2048, "depth": 16}

    torch.manual_seed(2026)
    base_model = SimpleTransformer(**model_config)
    standalone_model = deepcopy(base_model).npu()
    sharded_model = TransformerWithChunkLoss(**model_config).npu()
    sharded_model.load_state_dict(base_model.state_dict())
    del base_model
    fully_shard(sharded_model.head, mesh=mesh)
    _apply_fully_shard_with_prefetch(sharded_model, mesh)
    sharded_model.layers[-1].set_modules_to_forward_prefetch([sharded_model.head])
    sharded_model.set_modules_to_backward_prefetch([sharded_model.head])
    sharded_model.head.set_modules_to_backward_prefetch([sharded_model.layers[-1]])
    standalone_model.layers = nn.ModuleList(
        MemTrackerWrapper(layer, enabled=False) for layer in standalone_model.layers
    )
    sharded_model.layers = nn.ModuleList(MemTrackerWrapper(layer) for layer in sharded_model.layers)
    for layer, next_layer in zip(sharded_model.layers[:-1], sharded_model.layers[1:]):
        layer.set_next_layer(next_layer)

    standalone_optimizer = optim.Adam(standalone_model.parameters(), lr=1e-3)
    sharded_optimizer = optim.Adam(sharded_model.parameters(), lr=1e-3)
    input_generator = torch.Generator().manual_seed(84)
    for training_step in range(training_steps):
        for _ in range(microbatches):
            tokens = torch.randint(
                model_config["vocab_size"], (global_batch_size, sequence_length), generator=input_generator
            ).npu()
            targets = torch.randint(model_config["vocab_size"], tokens.shape, generator=input_generator).npu()
            local_tokens = tokens.narrow(0, rank * local_batch_size, local_batch_size)
            local_targets = targets.narrow(0, rank * local_batch_size, local_batch_size)
            with SkipDTensorDispatch():
                standalone_output = standalone_model(tokens)
                standalone_losses = functional.cross_entropy(
                    standalone_output.transpose(1, 2), targets, reduction="none"
                )
                sharded_loss = sharded_model(local_tokens, local_targets)
                expected = standalone_losses.narrow(0, rank * local_batch_size, local_batch_size).sum()
                torch.testing.assert_close(sharded_loss, expected, rtol=1e-5, atol=1e-6)
                total_tokens = global_batch_size * sequence_length * microbatches
                (standalone_losses.sum() / total_tokens).backward()
                (sharded_loss / total_tokens).backward()
            sharded_model.layers[0].check_pending()
            _assert_accumulated_gradients_match(standalone_model, sharded_model, rank, training_step)

        with SkipDTensorDispatch():
            standalone_optimizer.step()
            sharded_optimizer.step()
            standalone_optimizer.zero_grad(set_to_none=True)
            sharded_optimizer.zero_grad(set_to_none=True)

    for layer_index, (reference_layer, layer) in enumerate(zip(standalone_model.layers, sharded_model.layers)):
        assert layer.checked_count >= training_steps * (microbatches - 1)
        assert layer.output_hook_count == training_steps * microbatches
        assert len(layer.release_history) == training_steps * microbatches
        full_grad_bytes = sum(param.numel() * param.element_size() for param in reference_layer.parameters())
        assert full_grad_bytes % world_size == 0
        sharded_grad_bytes = full_grad_bytes // world_size
        for backward_index, (input_released, output_released, output_retained) in enumerate(layer.release_history):
            first_microbatch = backward_index % microbatches == 0
            expected_release = 0 if first_microbatch else sharded_grad_bytes
            expected_retained = sharded_grad_bytes if first_microbatch else 0
            assert input_released == full_grad_bytes, (
                f"Rank {rank}, layer {layer_index}, backward {backward_index}: "
                f"source gradient release expected={full_grad_bytes}, actual={input_released}"
            )
            assert (output_released, output_retained) == (expected_release, expected_retained), (
                f"Rank {rank}, layer {layer_index}, backward {backward_index}: "
                f"RS release/retention expected={(expected_release, expected_retained)}, "
                f"actual={(output_released, output_retained)}"
            )
        print(
            f"RS_BYTES rank={rank} layer={layer_index} input_expected={full_grad_bytes} "
            f"input_released={[record[0] for record in layer.release_history]} "
            f"output_shard={sharded_grad_bytes} released={[record[1] for record in layer.release_history]} "
            f"retained={[record[2] for record in layer.release_history]}"
        )
    sharded_model.reset_iter_state()
