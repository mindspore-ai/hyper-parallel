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
"""Unit tests for graph-mode shard metadata in checkpoint payloads."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from hyper_parallel.core.distributed_checkpoint.metadata import (
    CHUNK_INFO,
    ChunkInfo,
    ChunkStorageMetadata,
)
from hyper_parallel.core.dtensor.placement_types import StridedShard
from hyper_parallel.core.optimizer.adamw import AdamW
from hyper_parallel.core.optimizer.optimizer import ChainedOptimizer
from hyper_parallel.compile.checkpoint_metadata import annotate_optimizer_state_dict
from hyper_parallel.models._transformers.checkpoint_loader import CheckpointManager
from hyper_parallel.trainer.callbacks.checkpoint_callback import CheckpointerCallback


class _StubCompiler:
    """Minimal compiled-compiler surface used by the callback."""

    is_compiled = True

    def __init__(self, layouts):
        self._layouts = layouts

    @property
    def parameter_layouts(self) -> dict:
        """Return the pre-built parameter layout mapping."""
        return self._layouts


def _chunk_info(global_shape):
    """Build a full-tensor chunk description for a replicated test parameter."""
    return ChunkInfo(
        chunk=ChunkStorageMetadata(
            offsets=(0,) * len(global_shape), sizes=global_shape
        ),
        global_shape=global_shape,
    )


class TestCheckpointGraphMetadata(unittest.TestCase):
    """The callback copies graph layouts onto the tensors DCP and HF read."""

    def _callback(self, model, layouts):
        callback = CheckpointerCallback.__new__(CheckpointerCallback)
        callback.trainer = SimpleNamespace(
            model=model, graph_compiler=_StubCompiler(layouts)
        )
        callback._is_peft = False
        return callback

    def test_model_state_carries_layout_and_chunk_info(self) -> None:
        """Detached state_dict values keep the live parameter's shard metadata."""
        model = nn.Linear(4, 4)
        parameter = model.weight
        layout = SimpleNamespace(placements=())
        parameter._sharding_spec = layout
        parameter._layout = layout
        parameter._placements = ()
        setattr(parameter, CHUNK_INFO, _chunk_info((4, 4)))
        callback = self._callback(model, {parameter: layout})

        state_dict = callback._model_state_dict()

        value = state_dict["weight"]
        self.assertIsNot(value, parameter)
        self.assertIs(value._sharding_spec, layout)
        self.assertEqual(getattr(value, CHUNK_INFO).global_shape, (4, 4))

    def test_tied_alias_state_carries_layout_and_chunk_info(self) -> None:
        """A tied alias key shares its canonical parameter's shard metadata."""
        model = nn.Linear(4, 4)
        parameter = model.weight
        model.weight_alias = parameter
        layout = SimpleNamespace(placements=())
        parameter._sharding_spec = layout
        parameter._layout = layout
        parameter._placements = ()
        setattr(parameter, CHUNK_INFO, _chunk_info((4, 4)))
        callback = self._callback(model, {parameter: layout})

        state_dict = callback._model_state_dict()

        self.assertIs(state_dict["weight"]._sharding_spec, layout)
        alias = state_dict["weight_alias"]
        self.assertIs(alias._sharding_spec, layout)
        self.assertEqual(getattr(alias, CHUNK_INFO).global_shape, (4, 4))

    def test_optimizer_state_carries_layout_and_chunk_info(self) -> None:
        """Dense momentum tensors are annotated before DCP builds its plan."""
        model = nn.Linear(4, 4)
        parameter = model.weight
        layout = SimpleNamespace(placements=())
        parameter._sharding_spec = layout
        parameter._layout = layout
        parameter._placements = ()
        setattr(parameter, CHUNK_INFO, _chunk_info((4, 4)))
        optimizer = ChainedOptimizer(model, {"adamw": AdamW([parameter], lr=1e-2)})
        parameter.grad = torch.randn(4, 4)
        optimizer.step()
        callback = self._callback(model, {parameter: layout})

        payload = optimizer.state_dict()
        annotate_optimizer_state_dict(callback.trainer, payload)

        momentum = payload["state"]["weight"]["exp_avg"]
        self.assertIs(momentum._sharding_spec, layout)
        self.assertEqual(getattr(momentum, CHUNK_INFO).global_shape, (4, 4))

    def test_flattened_optimizer_state_is_annotated(self) -> None:
        """Flattened ``state.<FQN>.<slot>`` keys resolve without a full scan."""
        model = nn.Linear(4, 4)
        parameter = model.weight
        layout = SimpleNamespace(placements=(), tensor_shape=(4, 4))
        parameter._sharding_spec = layout
        parameter._layout = layout
        parameter._placements = ()
        setattr(parameter, CHUNK_INFO, _chunk_info((4, 4)))
        callback = self._callback(model, {parameter: layout})
        payload = {
            "state.weight.exp_avg": torch.zeros(4, 4),
            "state.weight.exp_avg_sq": torch.zeros(4, 4),
            "state.weight.step": torch.tensor(0.0),
        }

        annotate_optimizer_state_dict(callback.trainer, payload)

        self.assertIs(payload["state.weight.exp_avg"]._sharding_spec, layout)
        self.assertIs(payload["state.weight.exp_avg_sq"]._sharding_spec, layout)
        self.assertFalse(hasattr(payload["state.weight.step"], "_sharding_spec"))

    def test_strided_optimizer_state_is_rejected(self) -> None:
        """A strided param with dense state cannot be exported and raises."""
        model = nn.Linear(2, 2, bias=False)
        parameter = model.weight
        layout = SimpleNamespace(placements=(StridedShard(0, 2),), tensor_shape=(4, 2))
        callback = self._callback(model, {parameter: layout})
        payload = {"state.weight.exp_avg": torch.zeros(2, 2)}

        with self.assertRaisesRegex(RuntimeError, "no single rectangular chunk"):
            annotate_optimizer_state_dict(callback.trainer, payload)

    def test_unrelated_strided_param_does_not_block_optimizer_save(self) -> None:
        """A strided param without state must not abort saving other state."""
        model = nn.Linear(4, 4)
        weight, bias = model.weight, model.bias
        strided = SimpleNamespace(placements=(StridedShard(0, 2),), tensor_shape=(8, 4))
        replicated = SimpleNamespace(placements=(), tensor_shape=(4,))
        bias._sharding_spec = replicated
        bias._layout = replicated
        bias._placements = ()
        setattr(bias, CHUNK_INFO, _chunk_info((4,)))
        callback = self._callback(model, {weight: strided, bias: replicated})
        payload = {"state.bias.exp_avg": torch.zeros(4)}

        annotate_optimizer_state_dict(callback.trainer, payload)

        self.assertIs(payload["state.bias.exp_avg"]._sharding_spec, replicated)


class TestCheckpointManagerGraphExport(unittest.TestCase):
    """The manager attaches graph metadata before gathering weights."""

    def test_compiled_compiler_attaches_before_gather(self) -> None:
        """A compiled compiler annotates live parameters for the gather."""
        model = nn.Linear(4, 4)
        calls = []

        class _Compiler:
            is_compiled = True

            def attach_shard_metadata(self) -> None:
                """Record that the manager asked for metadata."""
                calls.append(True)

        manager = CheckpointManager(model, _Compiler())
        manager._gather_full_state_dict(keep_state_dict=True)

        self.assertEqual(calls, [True])

    def test_uncompiled_compiler_attaches_source_metadata(self) -> None:
        """An uncompiled compiler still describes source-local TP/EP storage."""
        model = nn.Linear(4, 4)
        calls = []

        class _Compiler:
            is_compiled = False

            def attach_shard_metadata(self) -> None:
                """Record an unexpected final-layout request."""
                calls.append("final")

            def attach_source_shard_metadata(self) -> None:
                """Record that the manager asked for pre-compile metadata."""
                calls.append("source")

        manager = CheckpointManager(model, _Compiler())
        manager._gather_full_state_dict(keep_state_dict=True)

        self.assertEqual(calls, ["source"])


if __name__ == "__main__":
    unittest.main()
