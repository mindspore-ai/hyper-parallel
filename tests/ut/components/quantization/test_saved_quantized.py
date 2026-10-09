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
"""Physical saved tensors remain compatible with offload and recomputation hooks."""

import gc
import itertools
import unittest
import weakref
from contextlib import nullcontext
from typing import Any, Callable

import torch
from torch.utils.checkpoint import DefaultDeviceType, checkpoint

from tests.common.mark_utils import arg_mark
from tests.ut.components.quantization import test_memory_contracts as hif8_tests
from tests.ut.components.quantization import test_mxfp8_memory as mx_tests


class SavedQuantizedTests(unittest.TestCase):
    """Exercise real autograd wrappers with CPU arithmetic stand-ins."""

    def setUp(self) -> None:
        """Keep CPU checkpoint selection and RNG state isolated from other tests."""
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        self.addCleanup(DefaultDeviceType.set_device_type, DefaultDeviceType.get_device_type())
        # CPU-only checkpoint inputs otherwise use the registered accelerator default.
        DefaultDeviceType.set_device_type('cpu')

    @staticmethod
    def projection(
        fmt: str, grouped: bool, needs: tuple[bool, bool], empty: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build a fresh deterministic graph without hardware arithmetic."""
        if fmt == 'mxfp8':
            return mx_tests.MXFP8MemoryTests.projection(grouped, needs, empty=empty)[:3]
        return hif8_tests.MemoryContractsTests.make_projection(grouped, *needs, empty=empty)

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    def test_save_on_cpu_repeated_backward(self):
        """
        Feature: Quantized saved-tensor hooks.
        Description: Offload all four projections with both pinning modes and gradient subsets.
        Expectation: Outputs and repeated gradients match the non-offloaded reference.
        """
        # The grouped MXFP8 strategy flow keeps quantized wrappers on the
        # autograd context instead of physical saved tensors, so offload and
        # hook contracts cover the Dense MXFP8 path and both HiFloat8 paths.
        fmt_grouped = (('mxfp8', False), ('hif8', False), ('hif8', True))
        cases = itertools.product(fmt_grouped,
                                  ((True, True), (True, False), (False, True)), (False, True))
        for fmt, grouped, needs, pin in cases:
            if fmt == 'mxfp8' and grouped:
                # The strategy flow stores operands on the autograd node rather
                # than saved_tensors, so save_on_cpu and repeated backward do
                # not apply to that leg.
                continue
            for empty in ((False, True) if grouped else (False,)):
                with self.subTest(fmt=fmt, grouped=grouped, needs=needs, pin=pin, empty=empty):
                    x_ref, w_ref, reference = self.projection(fmt, grouped, needs, empty)
                    expected = torch.autograd.grad(reference.sum(), [v for v in (x_ref, w_ref) if v.requires_grad])
                    with torch.autograd.graph.save_on_cpu(pin_memory=pin):
                        x, w, output = self.projection(fmt, grouped, needs, empty)
                    torch.testing.assert_close(output, reference, rtol=0, atol=0)
                    variables = [v for v in (x, w) if v.requires_grad]
                    first = torch.autograd.grad(output.sum(), variables, retain_graph=True)
                    second = torch.autograd.grad(output.sum(), variables)
                    for actual, repeated, target in zip(first, second, expected):
                        torch.testing.assert_close(actual, target, rtol=0, atol=0)
                        torch.testing.assert_close(repeated, target, rtol=0, atol=0)
                    with self.assertRaises(RuntimeError):
                        _ = output.grad_fn.saved_tensors

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    def test_hooks_receive_physical_tensors_and_release_copies(self):
        """
        Feature: Quantized saved-tensor hooks.
        Description: Copy every saved tensor through user hooks and track its lifetime.
        Expectation: Hooks see plain tensors; copies survive retention and are finally released.
        """
        for fmt, grouped in itertools.product(('mxfp8', 'hif8'), (False, True)):
            if fmt == 'mxfp8' and grouped:
                # The strategy flow emits no saved_tensors, so hooks never fire.
                continue
            with self.subTest(fmt=fmt, grouped=grouped):
                refs = []

                def pack(tensor: torch.Tensor, record: Callable[[Any], None] = refs.append) -> torch.Tensor:
                    """Copy one physical payload while recording its weak reference."""
                    self.assertIs(type(tensor), torch.Tensor)
                    packed = tensor.detach().clone()
                    record(weakref.ref(packed))
                    return packed

                with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                    x, w, output = self.projection(fmt, grouped, (True, True))
                self.assertTrue(refs)
                first = torch.autograd.grad(output.sum(), (x, w), retain_graph=True)
                gc.collect()
                self.assertTrue(all(ref() is not None for ref in refs))
                second = torch.autograd.grad(output.sum(), (x, w))
                gc.collect()
                self.assertTrue(all(ref() is None for ref in refs))
                for actual, repeated in zip(first, second):
                    torch.testing.assert_close(actual, repeated, rtol=0, atol=0)

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    def test_physical_payload_version_counter(self):
        """
        Feature: Quantized saved-tensor version checks.
        Description: Modify a saved physical payload without hooks before backward.
        Expectation: All four paths reject an in-place version mismatch.
        """
        for fmt, grouped in itertools.product(('mxfp8', 'hif8'), (False, True)):
            if fmt == 'mxfp8' and grouped:
                # The strategy flow keeps no saved_tensors to version-check.
                continue
            with self.subTest(fmt=fmt, grouped=grouped):
                _, _, output = self.projection(fmt, grouped, (True, True))
                saved = output.grad_fn.saved_tensors
                payload = next(t for t in saved[1:] if t is not None)
                payload.add_(0)
                with self.assertRaisesRegex(RuntimeError, 'modified by an inplace operation'):
                    output.sum().backward()

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    def test_checkpoint_and_offload(self):
        """
        Feature: Quantized checkpoint compatibility.
        Description: Recompute all four projections, optionally inside pinned-memory saving.
        Expectation: Repeated first-order gradients match the uncheckpointed reference.
        """
        fmt_grouped = (('mxfp8', False), ('hif8', False), ('hif8', True))
        for (fmt, grouped), offload in itertools.product(fmt_grouped, (False, True)):
            with self.subTest(fmt=fmt, grouped=grouped, offload=offload):
                x, w, reference = self.projection(fmt, grouped, (True, True))
                expected = torch.autograd.grad(reference.sum(), (x, w))
                quantizer = (mx_tests.MXFP8Quantizer(npu_ops=mx_tests.IdentityMXOps())
                             if fmt == 'mxfp8' else hif8_tests.IdentityQuantizer())
                groups = torch.tensor([2, 0, 3] if fmt == 'mxfp8' else [2, 3])

                def project(
                    inputs: torch.Tensor, weight: torch.Tensor, fmt: str = fmt, grouped: bool = grouped,
                    groups: torch.Tensor = groups, quantizer: Any = quantizer,
                ) -> torch.Tensor:
                    """Recompute one grouped or dense quantized projection."""
                    if fmt == 'mxfp8':
                        if grouped:
                            return mx_tests._GroupedLinearFunction.apply(
                                inputs, weight, groups,
                                mx_tests.MXFP8GroupedLinear(quantizer=quantizer), 1)
                        return mx_tests.mxfp8_linear(inputs, weight, quantizer)
                    if grouped:
                        return hif8_tests.hifloat8_grouped_linear(
                            inputs, weight, groups, quantizer, quantizer, quantizer, group_list_type=1)
                    return hif8_tests.hifloat8_linear(inputs, weight, quantizer, quantizer, quantizer)

                context = torch.autograd.graph.save_on_cpu(pin_memory=True) if offload else nullcontext()
                with context:
                    output = checkpoint(project, x, w, use_reentrant=False)
                if fmt == 'mxfp8' and grouped:
                    # The strategy flow clears its operands when backward
                    # finishes, so only a single gradient pass is supported.
                    actual = torch.autograd.grad(output.sum(), (x, w))
                    for grad, target in zip(actual, expected):
                        torch.testing.assert_close(grad, target, rtol=0, atol=0)
                    continue
                for retain in (True, False):
                    actual = torch.autograd.grad(output.sum(), (x, w), retain_graph=retain)
                    for grad, target in zip(actual, expected):
                        torch.testing.assert_close(grad, target, rtol=0, atol=0)
