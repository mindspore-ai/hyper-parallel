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
"""CPU contracts for the isolated KDA parallel style and native boundaries."""
import inspect
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

import hyper_parallel
from hyper_parallel.core.context_parallel import ContextParallel, KDAContextParallel
from hyper_parallel.core.context_parallel import kda_context_parallel as kda_cp
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh, _DEVICE_MESH_MAP
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.custom_ops.experimental import npu_chunk_kda


# White-box contracts intentionally inspect private CP/autograd boundaries.
class TestKDAContextParallel(unittest.TestCase):
    """Exercise real DTensor wrappers without launching a distributed process."""

    # White-box checks of private CP/autograd state are intentional in this class.
    # pylint: disable=protected-access

    def setUp(self) -> None:
        """Patch mesh metadata without creating collective groups."""
        _DEVICE_MESH_MAP.clear()
        self._mesh_platform_patch = patch('hyper_parallel.core.dtensor.device_mesh.platform')
        self._mesh_platform = self._mesh_platform_patch.start()
        self._mesh_platform.get_rank.return_value = 0
        self._mesh_platform.tensor_to_numpy.side_effect = lambda value: value.numpy()
        self.rank_patch = patch.multiple('torch.distributed', get_rank=MagicMock(return_value=0),
                                        get_world_size=MagicMock(return_value=1))
        self.rank_patch.start()

    def tearDown(self) -> None:
        """Restore global mesh metadata after each test."""
        self.rank_patch.stop()
        self._mesh_platform_patch.stop()
        _DEVICE_MESH_MAP.clear()

    def _mesh(self, size):
        self._mesh_platform.get_world_size.return_value = size
        return init_device_mesh('cpu', (size,), mesh_dim_names=('cp',), init_backend=False)

    def test_operator_and_base_have_no_cp_flags(self) -> None:
        """Neither the local operator nor generic attention style exposes KDA policy."""
        self.assertNotIn('context_parallel', inspect.signature(npu_chunk_kda).parameters)
        self.assertNotIn('state_parallel', inspect.signature(ContextParallel).parameters)
        self.assertTrue(issubclass(KDAContextParallel, ContextParallel))

    def test_sequence_saves_only_backward_relays(self) -> None:
        """Forward-only and optional normalization outputs must not remain live."""
        inputs = tuple(torch.ones(1, 3, 2, 128) for _ in range(4))
        beta = torch.ones(1, 3, 2)
        cache = tuple(torch.ones(1) for _ in range(13))
        ctx = SimpleNamespace(save_for_backward=MagicMock())
        transport = MagicMock()
        with patch.object(kda_cp, 'npu_chunk_kda_fwd_prepare', return_value=cache), \
             patch.object(kda_cp, 'npu_chunk_kda_fwd_h', return_value=(cache[0], cache[1], None)), \
             patch.object(kda_cp, 'npu_chunk_kda_fwd_finalize', return_value=inputs[0]):
            kda_cp._KDASequenceFunction.forward(
                ctx, *inputs, beta, torch.zeros(2), torch.zeros(2, 128),
                {'scale': .1, 'chunk_size': 64, 'lower_bound': -5.}, transport)
        retained = ctx.save_for_backward.call_args.args
        for index in (4, 7, 8, 9, 10, 11, 12):
            self.assertFalse(any(value is cache[index] for value in retained), index)

    def test_invalid_inputs_fail_at_parallel_boundary(self) -> None:
        """Diagnose missing hooks and unsupported grouped heads before native dispatch."""
        style = KDAContextParallel(ulysses_degree=1)
        local = torch.ones(1, 3, 2, 128)
        with self.assertRaisesRegex(TypeError, 'DTensor'):
            style.convolve(local, local, local, torch.ones(3, 256, 1, 4), lambda *args: None, 4)
        mesh = self._mesh(1)
        query = DTensor.from_local(local, mesh, (Replicate(),))
        value = DTensor.from_local(local[:, :, :1], mesh, (Replicate(),))
        with self.assertRaisesRegex(ValueError, 'equal.*heads|MHA'):
            style.convolve(query, query, value, torch.ones(3, 256, 1, 4), lambda *args: None, 4)

    def test_head_selection_and_parameter_contract(self) -> None:
        """Rank one selects the second contiguous head block; reject double sharding."""
        mesh = self._mesh(1)
        tensor = DTensor.from_local(torch.ones(1, 3, 2, 4), mesh, (Replicate(),))
        weight = torch.arange(3 * 16 * 4.).reshape(3, 16, 1, 4)
        style = KDAContextParallel(ulysses_degree=2)
        style._head_degree, style._head_rank = 2, 1

        def compute(inputs, selected, size):
            self.assertEqual(size, 4)
            torch.testing.assert_close(selected, weight[:, 8:16])
            return inputs

        style.convolve(tensor, tensor, tensor, weight, compute, 4)
        with self.assertRaisesRegex(ValueError, 'all TP-local heads'):
            style.convolve(tensor, tensor, tensor, weight[:, :8], compute, 4)
        q = DTensor.from_local(torch.ones(1, 3, 2, 128), mesh, (Replicate(),))
        beta = DTensor.from_local(torch.ones(1, 3, 2), mesh, (Replicate(),))
        with self.assertRaisesRegex(ValueError, 'bias'):
            style.attention(q, q, q, q, beta, torch.zeros(4), torch.zeros(512), .1)

    def test_apply_modes_and_mesh_order(self) -> None:
        """Build sequence transport only when the CP group has sequence ranks."""
        mesh = self._mesh(4)
        for degree, sequence_size in ((1, 4), (2, 2), (4, 1)):
            module = torch.nn.Identity()
            style = KDAContextParallel(ulysses_degree=degree)
            with patch.object(ContextParallel, 'apply', return_value=module):
                self.assertIs(style.apply(module, mesh), module)
            self.assertIs(module._kda_parallel, style)
            self.assertEqual(style._head_degree, degree)
            self.assertEqual(style._transport.size if style._transport else 1, sequence_size)
        invalid = SimpleNamespace(mesh=torch.arange(4).reshape(2, 2), ndim=2,
                                  mesh_dim_names=('ds', 'co'))
        with self.assertRaisesRegex(ValueError, 'co, ds'):
            KDAContextParallel(ulysses_degree=2).apply(torch.nn.Identity(), invalid)

    def test_merge_rank_counts_and_return_value(self) -> None:
        """Every rank gathers; prefix/suffix counts and replacement state are preserved."""
        mesh = self._mesh(3)
        summary = torch.ones(2, 1, 2, 4)
        gathered = DTensor.from_local(torch.ones(3, 2, 1, 2, 4), mesh, (Replicate(),))
        transport = kda_cp._KDAStateTransport(mesh)
        for rank in range(3):
            for forward in (False, True):
                transport.rank = rank
                count = rank if forward else 2 - rank
                with patch.object(DTensor, 'redistribute', return_value=gathered) as exchange, \
                     patch.object(kda_cp, 'npu_chunk_kda_merge_fwd_bwd',
                                  side_effect=lambda state, *args, **kwargs: state + 1) as merge:
                    state = transport._merge(summary, forward)
                exchange.assert_called_once()
                self.assertEqual(merge.call_count, 2 if count else 0)
                if count:
                    self.assertEqual(merge.call_args.args[2:4], (count, rank))
                    self.assertEqual(merge.call_args.kwargs, {'forward': forward})
                torch.testing.assert_close(state, torch.full((2, 1, 2, 2), float(bool(count))))

    def test_forward_summaries_pack_dense_batch_once(self) -> None:
        """Use the upstream dense-batch contract instead of B separate launches."""
        transport = kda_cp._KDAStateTransport(self._mesh(2))
        cache = {name: torch.ones(3, 2, 65, 128) for name in ('kg', 'w', 'u', 'gk')}
        summary = torch.ones(3, 2, 128, 256)
        with patch.object(kda_cp, 'npu_chunk_kda_pre_process_fwd', return_value=(summary,)) as prepare, \
             patch.object(transport, '_merge', return_value=summary) as merge:
            transport.forward_boundary(cache)
        prepare.assert_called_once_with(cache['kg'], cache['w'], cache['u'], gk=cache['gk'], cu_seqlens=None)
        merge.assert_called_once_with(summary, True)

    def test_colossal_keeps_five_dtensor_inputs(self) -> None:
        """Sequence-only KDA does not gather the complete token sequence."""
        mesh = self._mesh(2)
        style = KDAContextParallel(ulysses_degree=1)
        inputs = tuple(torch.randn(1, 3, 2, 4) for _ in range(5))
        outputs, _ = style._pre_hook_colossal(torch.nn.Identity(), inputs, {}, mesh)
        for source, output in zip(inputs, outputs):
            self.assertIsInstance(output, DTensor)
            self.assertEqual(output.placements, (Shard(1),))
            torch.testing.assert_close(output.to_local(), source)

    def test_halo_retains_layout_and_future_gradients(self) -> None:
        """Keep real DTensor boundaries around mocked all-gather, including L<W."""
        for length in (1, 5):
            with self.subTest(length=length):
                _DEVICE_MESH_MAP.clear()
                mesh = self._mesh(3)
                source = torch.arange(3 * length * 4, dtype=torch.float32).reshape(2, 3 * length, 2)
                source.requires_grad_()
                shards = source.chunk(3, dim=1)
                tails = torch.cat([part[:, -min(3, length):] for part in shards], dim=1)
                gathered = DTensor.from_local(tails, mesh, (Replicate(),))
                style = KDAContextParallel(ulysses_degree=1)
                transport = kda_cp._KDAStateTransport(mesh)
                style._transport = transport
                outputs = []
                with patch.object(DTensor, 'redistribute', return_value=gathered):
                    for rank, shard in enumerate(shards):
                        transport.rank = rank
                        output = style.sequence_halo(DTensor.from_local(shard, mesh, (Shard(1),)), 3)
                        self.assertIsInstance(output, DTensor)
                        self.assertEqual(output.placements, (Shard(1),))
                        prefix = source[:, max(0, rank * length - 3):rank * length]
                        expected = torch.cat((torch.zeros(2, 3 - prefix.shape[1], 2), prefix, shard), dim=1)
                        torch.testing.assert_close(output.to_local(), expected)
                        outputs.append(output.to_local())
                outputs[-1][:, :3].sum().backward()
                expected_grad = torch.zeros_like(source)
                expected_grad[:, max(0, 2 * length - 3):2 * length] = 1
                torch.testing.assert_close(source.grad, expected_grad)

    def test_convolution_keeps_dtensor_heads(self) -> None:
        """One native convolution boundary restores the original DTensor layouts."""
        mesh = self._mesh(1)
        inputs = tuple(torch.randn(2, 5, 2, 4, requires_grad=True) for _ in range(3))
        values = tuple(DTensor.from_local(value, mesh, (Shard(2),)) for value in inputs)
        weights = torch.ones(3, 8, 1, 4)
        style = KDAContextParallel(ulysses_degree=1)
        outputs = style.convolve(*values, weights, lambda value, weight, size: value * 2, 4)
        for source, output in zip(inputs, outputs):
            self.assertIsInstance(output, DTensor)
            self.assertEqual(output.placements, (Shard(2),))
            torch.testing.assert_close(output.to_local(), source * 2)

    def test_backward_summary_preserves_tensor_contract(self) -> None:
        """A PYBOOST_CALLER(1) summary remains one Tensor, not a tuple of head slices."""
        fake_impl = MagicMock()
        fake_package = SimpleNamespace(custom_op_impl=fake_impl)
        fake_backend = SimpleNamespace(custom_ops=fake_package)
        # CI copies tests separately from the installed package under test.
        path = Path(hyper_parallel.__file__).resolve().parent / "platform/mindspore/custom_ops/custom_ops.py"
        spec = importlib.util.spec_from_file_location("_kda_backend_contract", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {
                "hyper_parallel.platform.mindspore": fake_backend,
                "hyper_parallel.platform.mindspore.custom_ops": fake_package,
                "hyper_parallel.platform.mindspore.custom_ops.custom_op_impl": fake_impl}):
            spec.loader.exec_module(module)
        summary = torch.ones(2, 4, 8)
        fake_impl._custom_ops.npu_chunk_kda_pre_process_bwd.return_value = summary
        with patch.object(module, "_prepare_chunk_kda_stage_args", side_effect=lambda args: args):
            output = module.MindSporeCustomOps.npu_chunk_kda_pre_process_bwd(
                None, None, None, None, None, .1, 64, None, None)
        self.assertIs(output, summary)


if __name__ == '__main__':
    unittest.main()
