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

"""CPU mathematical oracle for stream scheduling; collectives and native FA are mocked."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hyper_parallel.distributed.context_parallel import stream_kv as implementation
from hyper_parallel.distributed.context_parallel import _stream_kv_layout as panel_layout


class Mesh:
    """Small rank-order fixture without a real process group."""

    ndim = 1

    def __init__(self, size: int = 1, owner: int = 0) -> None:
        """Bind logical size and rank for mocked collectives."""
        self.degree, self.owner = size, owner

    def size(self) -> int:
        """Return logical group size."""
        return self.degree

    def get_local_rank(self) -> int:
        """Return the contiguous token owner."""
        return self.owner

    def get_group(self) -> Mesh:
        """Use this instance as a mocked group identifier."""
        return self


class DenseNative:
    """Local mathematical FA contract, independently checked against global autograd."""

    @staticmethod
    def _scores(query, key, scale, sparse_mode, next_tockens):
        key = key.repeat_interleave(query.shape[1] // key.shape[1], 1)
        scores = query.double() @ key.double().transpose(-1, -2) * scale
        if sparse_mode == 4:
            rows = torch.arange(query.shape[2])[:, None]
            columns = torch.arange(key.shape[2])[None, :]
            mask = columns > rows + key.shape[2] - query.shape[2] + next_tockens
            scores.masked_fill_(mask, -torch.inf)
        return scores, key.double()

    @classmethod
    def npu_fusion_attention(cls, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                             heads: int, layout: str, **kwargs: object) -> tuple:
        """Return normalized local A and eight-lane softmax statistics."""
        del heads, layout
        scores, _ = cls._scores(query, key, kwargs["scale"], kwargs["sparse_mode"], kwargs["next_tockens"])
        maximum = scores.amax(-1, keepdim=True)
        total = (scores - maximum).exp().sum(-1, keepdim=True)
        expanded = value.repeat_interleave(query.shape[1] // value.shape[1], 1).double()
        output = scores.softmax(-1) @ expanded
        return output.to(query.dtype), maximum.float().expand(*maximum.shape[:-1], 8).contiguous(), (
            total.float().expand(*total.shape[:-1], 8).contiguous())

    @classmethod
    def npu_fusion_attention_grad(cls, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                                  incoming: torch.Tensor, heads: int, layout: str, **kwargs: object) -> tuple:
        """Use supplied global normalization and global output for partial gradients."""
        del heads, layout
        scale = kwargs["scale_value"]
        scores, expanded_k = cls._scores(query, key, scale, kwargs["sparse_mode"], kwargs["next_tockens"])
        probabilities = (scores - kwargs["softmax_max"][..., :1].double()).exp()
        probabilities /= kwargs["softmax_sum"][..., :1].double()
        expanded_v = value.repeat_interleave(query.shape[1] // value.shape[1], 1).double()
        incoming = incoming.double()
        delta = (incoming * kwargs["attention_in"].double()).sum(-1, keepdim=True)
        dscores = probabilities * (incoming @ expanded_v.transpose(-1, -2) - delta)
        dq = dscores @ expanded_k * scale
        dk = dscores.transpose(-1, -2) @ query.double() * scale
        dv = probabilities.transpose(-1, -2) @ incoming
        shape = (1, key.shape[1], query.shape[1] // key.shape[1], key.shape[2], key.shape[3])
        return dq.float(), dk.view(shape).sum(2).float(), dv.view(shape).sum(2).float()


def dense_reference(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, incoming: torch.Tensor) -> tuple:
    """Differentiate one global triangular GQA using ordinary Torch autograd."""
    operands = [tensor.double().requires_grad_(True) for tensor in (query, key, value)]
    q, k, v = operands
    mapping = torch.arange(q.shape[1]) // (q.shape[1] // k.shape[1])
    scores = (q @ k[:, mapping].transpose(-1, -2)) * q.shape[-1]**-0.5
    scores = scores.masked_fill(torch.ones(q.shape[2], k.shape[2], dtype=torch.bool).triu(1), -torch.inf)
    output = scores.softmax(-1) @ v[:, mapping]
    gradients = torch.autograd.grad(output, operands, incoming.double())
    return output.detach(), gradients


class TestStreamKV(unittest.TestCase):
    """Check disjoint panels, causal offsets, global VJP and FP32 replica SUM."""

    @classmethod
    def setUpClass(cls) -> None:
        """Use one CPU thread for these small deterministic matmuls."""
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def test_config_rejects_invalid_widths(self):
        """Check invalid configuration before any collective is reached.

        Feature: Stream KV communication and compute bounds.
        Description: Supply non-positive or non-integer panel/head widths.
        Expectation: Invalid values raise ValueError before communication.
        """
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                implementation.StreamKVConfig(value, 3)
        with self.assertRaises(ValueError):
            implementation.StreamKVConfig(3, 4, 0)

    def test_global_forward_and_vjp(self):
        """Compare the scheduler with dense autograd across owners and tails.

        Feature: Contiguous stream KV global normalization and partial VJP.
        Description: Mock transport and differentiate a separate global FP64 triangle.
        Expectation: Output and all three assembled gradients match the reference.
        """
        generator = torch.Generator().manual_seed(177)
        for pg, local, kv_heads, chunk, panel, block in ((1, 7, 1, 2, 3, 2), (4, 7, 2, 3, 3, 2),
                                                         (3, 5, 1, 1, 99, 2), (4, 7, 2, None, 1, 5)):
            with self.subTest(pg=pg, panel=panel, block=block, chunk=chunk):
                heads, dim, length = 8, 8, pg * local
                query = torch.randn(1, heads, length, dim, generator=generator)
                key, value = [torch.randn(1, kv_heads, length, dim, generator=generator) for _ in range(2)]
                incoming = torch.randn(query.shape, generator=generator)
                expected, gradients = dense_reference(query, key, value, incoming)
                outputs, query_grads, sends = [], [], []
                actual_k, actual_v = torch.empty_like(key), torch.empty_like(value)
                config = implementation.StreamKVConfig(panel, block, chunk)
                for owner in range(pg):
                    selection = slice(owner * local, (owner + 1) * local)
                    owned_k, owned_v = key[:, :, selection], value[:, :, selection]
                    bounds = iter([(start, min(start + panel, local)) for start in range(0, local, panel)] * 2)
                    owner_sends = []

                    def gather(received: torch.Tensor, send: torch.Tensor, group: Mesh) -> None:
                        """Check the caller's own stripe while supplying all owners' actual values."""
                        start, end = next(bounds)
                        packed = torch.stack([torch.stack((key[:, :, rank * local + start:rank * local + end],
                            value[:, :, rank * local + start:rank * local + end]), 2).permute(3, 2, 0, 1, 4)
                            for rank in range(pg)])
                        torch.testing.assert_close(send, packed[group.owner], rtol=0, atol=0)
                        received.copy_(packed.flatten(0, 1))

                    def reduce(received: torch.Tensor, send: torch.Tensor, op: object, group: Mesh) -> None:
                        """Capture real computed dKV for an independent cross-query-owner SUM."""
                        self.assertEqual(op, torch.distributed.ReduceOp.SUM)
                        self.assertEqual(send.dtype, torch.float32)
                        owner_sends.append(send.clone().view(pg, -1, 2, 1, kv_heads, dim))
                        received.copy_(send.view(pg, -1, 2, 1, kv_heads, dim)[group.owner])

                    with patch.object(implementation, "_npu_ops", return_value=DenseNative), (
                            patch.object(panel_layout.dist, "all_gather_into_tensor", side_effect=gather)), (
                            patch.object(panel_layout.dist, "reduce_scatter_tensor", side_effect=reduce)):
                        out, maximum, total = implementation._forward(
                            query[:, :, selection], owned_k, owned_v, Mesh(pg, owner), dim**-0.5, config)
                        dq, dk, dv = implementation._backward(query[:, :, selection], owned_k, owned_v,
                            out, maximum, total, incoming[:, :, selection], Mesh(pg, owner), dim**-0.5, config)
                    outputs.append(out)
                    query_grads.append(dq)
                    if pg == 1:
                        actual_k, actual_v = dk, dv
                    else:
                        sends.append(torch.cat(owner_sends, 1))
                if pg > 1:
                    global_sum = torch.stack(sends).sum(0).flatten(0, 1)
                    actual_k, actual_v = (global_sum[:, component].permute(1, 2, 0, 3) for component in range(2))
                actual = (torch.cat(outputs, 2), torch.cat(query_grads, 2), actual_k, actual_v)
                for observed, reference in zip(actual, (expected, *gradients)):
                    torch.testing.assert_close(observed.double(), reference, rtol=2e-5, atol=3e-6)

    def test_replica_sum_before_bf16_cast(self):
        """Check that replica SUM precedes conversion to the input dtype.

        Feature: FP32 Ulysses and KV-replica gradient return.
        Description: Use values whose BF16 rounding changes a premature replica SUM.
        Expectation: Exchanges and SUM stay FP32 until the final input boundary.
        """
        ctx = SimpleNamespace(ulysses_mesh=Mesh(4), kv_mesh=Mesh(2), scale=1.0,
                              config=implementation.StreamKVConfig(3, 2), replicas=2,
                              kv_heads=2, dtype=torch.bfloat16, saved_tensors=())
        gradients = (torch.ones(1, 8, 3, 8), torch.tensor([1.003, 0.003, 1.003, 0.003]).view(
            1, 4, 1, 1).expand(1, 4, 3, 8).contiguous(), torch.randn(1, 4, 3, 8))
        dtypes = []

        def exchange(tensor: torch.Tensor, *_: object, **_kwargs: object) -> torch.Tensor:
            """Observe dtype at every Ulysses boundary."""
            dtypes.append(tensor.dtype)
            return tensor

        for balanced in (False, True):
            ctx.config = implementation.StreamKVConfig(3, 2, causal_load_balance=balanced)
            dtypes.clear()
            with patch.object(implementation, "_backward", return_value=gradients), (
                    patch.object(implementation, "ulysses_seq_to_head", side_effect=exchange)), (
                    patch.object(implementation, "ulysses_head_to_seq", side_effect=exchange)), (
                    patch.object(implementation, "_redistribute", side_effect=exchange)), (
                    patch.object(implementation.torch, "autocast", return_value=nullcontext())):
                returned = implementation._StreamKV.backward(ctx, torch.ones(1, 8, 3, 8, dtype=torch.bfloat16))
            self.assertEqual(dtypes, [torch.float32] * (8 if balanced else 4))
            for index in (1, 2):
                expected = gradients[index].view(1, 2, 2, 3, 8).sum(2).bfloat16()
                torch.testing.assert_close(returned[index], expected, rtol=0, atol=0)
            prematurely_rounded = gradients[1].bfloat16().view(1, 2, 2, 3, 8).sum(2)
            self.assertFalse(torch.equal(returned[1], prematurely_rounded))

    def test_boundary_rejects_unsupported_arguments(self):
        """Check unsupported masks and dropout without a native call.

        Feature: Explicit first-version attention interface contract.
        Description: Pass unsupported masks, cache arguments and CPU input tensors.
        Expectation: Every unsupported combination raises a descriptive ValueError.
        """
        interface = implementation.StreamKVGQAAttention(Mesh(), Mesh(), implementation.StreamKVConfig(3, 2))
        query = torch.zeros(1, 4, 3, 256, dtype=torch.bfloat16)
        module = SimpleNamespace(is_causal=True)
        for kwargs in ({"attention_mask": torch.ones(3, 3)}, {"dropout": .1},
                       {"actual_seq_len": [3]}, {"sliding_window": 2}, {"past_key_value": object()}):
            with self.subTest(kwargs=tuple(kwargs)), self.assertRaises(ValueError):
                interface(module, query, query, query, **kwargs)
        with self.assertRaisesRegex(ValueError, "Ascend"):
            interface.attention(query, query, query)

    def test_boundary_rejects_invalid_tensors(self):
        """Reject invalid local tensor layouts before communication.

        Feature: Stream KV local tensor input contract.
        Description: Pass non-tensors, empty shards and incompatible BNSD shapes.
        Expectation: Invalid inputs raise ValueError without entering autograd or collectives.
        """
        interface = implementation.StreamKVGQAAttention(Mesh(), Mesh(), implementation.StreamKVConfig(3, 2))
        query = torch.zeros(1, 4, 3, 256, dtype=torch.bfloat16)
        invalid = ((None, query, query), (query.squeeze(0), query, query),
                   (query[:, :, :0], query[:, :, :0], query[:, :, :0]),
                   (query, query[:, :, :2], query[:, :, :2]), (query, query, query.float()))
        with patch.object(implementation._StreamKV, "apply") as apply:
            for tensors in invalid:
                with self.subTest(shapes=[getattr(tensor, "shape", None) for tensor in tensors]):
                    with self.assertRaises(ValueError):
                        interface.attention(*tensors)
            apply.assert_not_called()
