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

"""Native attention forward and gradient precision contracts."""

import unittest
from types import SimpleNamespace

import torch

from hyper_parallel.components import ops as operators
from tests.common.op_precision import check_case
from tests.common.op_precision_suite import precision_cases


class TestAttentionPrecision(unittest.TestCase):
    """Verify every variant against independent formulas in all training dtypes."""

    def test_native_forward_and_gradients(self) -> None:
        """Check outputs, all differentiable arguments, discrete metadata and input immutability."""
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for case in precision_cases("attention", dtype):
                with self.subTest(dtype=dtype, variant=case.variant, case=case.name):
                    check_case(case)

    def test_cpu_sdpa_matches_native(self) -> None:
        """Exercise a real alternative backend on CPU, including packed teacher statistics."""
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for case in precision_cases("attention", dtype):
                with self.subTest(dtype=dtype, variant=case.variant, case=case.name):
                    check_case(case, backend="sdpa")


class TestAttentionMasks(unittest.TestCase):
    """Public mask normalization must preserve global and packed document coordinates."""

    def setUp(self) -> None:
        """Restore process selections after each mask regression."""
        self.addCleanup(operators.ops.set_backends, operators.ops.get_backends())
        operators.ops.set_backends({"attention": "native", "rope": "native"})

    def test_context_parallel_pairwise_mask_encodes_global_offset(self) -> None:
        """A rectangular pairwise mask overrides implicit causal alignment and trims padding."""
        query = torch.zeros(1, 1, 2, 2, requires_grad=True)
        key = torch.zeros(1, 1, 4, 2, requires_grad=True)
        value = torch.arange(8.0).reshape(1, 1, 4, 2).requires_grad_()
        mask = torch.zeros(1, 2, 6, dtype=torch.bool)
        mask[0, 0, 3] = True
        mask[0, 1, 0] = True
        actual, statistics = operators.grouped_query_attention(SimpleNamespace(is_causal=True), query, key, value, mask)
        torch.testing.assert_close(actual, value[:, :, [3, 0]].transpose(1, 2))
        self.assertIsNone(statistics)
        actual.sum().backward()
        torch.testing.assert_close(value.grad, torch.tensor([[[[1., 1.], [0., 0.], [0., 0.], [1., 1.]]]]))

    def test_module_window_limits_attention_to_previous_token_and_self(self) -> None:
        """The legacy pre_tockens module option defines the causal window width."""
        query = key = torch.zeros(1, 1, 4, 2)
        value = torch.arange(8.0).reshape(1, 1, 4, 2)
        actual, _ = operators.grouped_query_attention(SimpleNamespace(pre_tockens=1), query, key, value, None)
        expected = torch.stack((value[0, 0, 0], *[(value[0, 0, i - 1] + value[0, 0, i]) / 2 for i in range(1, 4)]))
        torch.testing.assert_close(actual, expected[None, :, None])

    def test_sparse_indices_and_mismatched_packed_segments_are_rejected(self) -> None:
        """Dense contracts reject sparse IDs and ambiguous packed query/key pairings."""
        query = torch.zeros(1, 1, 4, 2)
        options = (({"indices": torch.zeros(1, dtype=torch.int32)}, "sparse attention indices"),
                   ({"actual_seq_qlen": [2, 4], "actual_seq_kvlen": [4]}, "same number of segments"))
        for kwargs, message in options:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                operators.multi_latent_attention(SimpleNamespace(), query, query, query, None, **kwargs)

    def test_rope_qk_broadcasts_shared_positions_and_preserves_gradients(self) -> None:
        """A shared position table expands over the batch without mixing query/key gradients."""
        query = torch.ones(2, 1, 3, 4, requires_grad=True)
        key = torch.full_like(query, 2., requires_grad=True)
        cosine, sine = torch.ones(3, 4), torch.zeros(3, 4)
        rotated_query, rotated_key = operators.rope_qk(query, key, cosine, sine)
        torch.testing.assert_close(rotated_query, query)
        torch.testing.assert_close(rotated_key, key)
        (rotated_query.sum() + 2 * rotated_key.sum()).backward()
        torch.testing.assert_close(query.grad, torch.ones_like(query))
        torch.testing.assert_close(key.grad, torch.full_like(key, 2.))

    def test_packed_padding_masks_output_and_gradient_rows(self) -> None:
        """Padding must neither contribute keys nor produce outputs for padded queries."""
        query = key = torch.zeros(2, 1, 2, 2)
        value = torch.arange(8.0).reshape(2, 1, 2, 2).requires_grad_()
        mask = torch.tensor([[True, False], [True, True]])
        actual, _ = operators.grouped_query_attention(
            SimpleNamespace(), query, key, value, mask, actual_seq_len=[2, 4],
        )
        expected = torch.tensor([[[[0., 1.]], [[0., 0.]]], [[[4., 5.]], [[5., 6.]]]])
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        expected_grad = torch.tensor([[[[1., 1.], [0., 0.]]], [[[1.5, 1.5], [0.5, 0.5]]]])
        torch.testing.assert_close(value.grad, expected_grad)

    def test_packed_pair_masks_support_global_and_batch_local_coordinates(self) -> None:
        """Global masks permit documents crossing batches; local masks select their own batch."""
        query = key = torch.zeros(2, 1, 2, 2)
        value = torch.arange(8.0).reshape(2, 1, 2, 2)
        cases = (([1, 4], torch.ones(1, 1, 4, 4, dtype=torch.bool), [[0., 1.], [4., 5.], [4., 5.], [4., 5.]]),
                 ([2, 4], torch.zeros(2, 1, 2, 2), [[1., 2.], [1., 2.], [5., 6.], [5., 6.]]))
        for ends, mask, expected in cases:
            with self.subTest(ends=ends, mask_shape=mask.shape):
                actual, _ = operators.multi_latent_attention(
                    SimpleNamespace(is_causal=False), query, key, value, mask, actual_seq_len=ends,
                )
                torch.testing.assert_close(actual, torch.tensor(expected).reshape(2, 2, 1, 2))
        with self.assertRaisesRegex(ValueError, "flattened token indices"):
            operators.multi_latent_attention(
                SimpleNamespace(is_causal=False), query, key, value, torch.ones(2, 1, 2, 2, dtype=torch.bool),
                actual_seq_len=[1, 4],
            )
