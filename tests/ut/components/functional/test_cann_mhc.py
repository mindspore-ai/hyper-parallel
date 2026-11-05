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
"""CPU contracts for optional CANN mHC dispatch and residual gradients."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from hyper_parallel.components.functional import _cann_mhc
from hyper_parallel.components.functional.mhc_post import mhc_post
from hyper_parallel.components.functional.mhc_pre import mhc_pre
from hyper_parallel.components.modules.mhc import MhcPostModule, MhcPreModule


class TestCannMhc(unittest.TestCase):
    """Check backend selection without loading an NPU extension on CPU."""

    def tearDown(self) -> None:
        """Keep optional-entry caches isolated between test cases."""
        _cann_mhc._load_cann_mhc_op.cache_clear()

    @staticmethod
    def _inputs():
        return (
            torch.arange(2 * 3 * 512, dtype=torch.float32).reshape(2, 3, 512).div(1024).requires_grad_(),
            torch.arange(24 * 512, dtype=torch.float32).reshape(24, 512).div(8192).requires_grad_(),
            torch.tensor([0.5, 0.75, 1.0], requires_grad=True),
            torch.arange(24, dtype=torch.float32).div(32).requires_grad_(),
        )

    def test_residual_branch_and_coefficient_orientation(self):
        """The pre wrapper must leave the ordinary residual gradient to autograd."""
        x, phi, alpha, bias = self._inputs()

        def pre_op(
            streams: torch.Tensor, projection: torch.Tensor, scales: torch.Tensor, offsets: torch.Tensor,
            mult: int, iters: int, hc_eps: float, norm_eps: float,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Expose distinct coefficient and residual dependencies for the wiring test."""
            self.assertEqual((mult, iters, hc_eps, norm_eps), (4, 20, 1e-6, 1e-6))
            hidden = streams.sum(dim=-2)
            post = (offsets[:4] + scales[0]).expand(*streams.shape[:2], 4)
            combine = (projection[:4, :4] + scales[1]).expand(*streams.shape[:2], 4, 4)
            return hidden, post, combine.flatten(-2)

        def post_op(
            streams: torch.Tensor, combine: torch.Tensor, hidden: torch.Tensor, post: torch.Tensor,
        ) -> torch.Tensor:
            """Stand in for the native post wrapper using differentiable CPU operations."""
            return torch.einsum("bsij,bsid->bsjd", combine, streams) + post.unsqueeze(-1) * hidden.unsqueeze(-2)

        ops = {"mhc_pre_sinkhorn": pre_op, "mhc_post": post_op}
        # Native device/dtype behavior is checked in the NPU integration probe;
        # these differentiable stubs isolate the adapter's two input branches.
        with patch.object(_cann_mhc, "_validate_cann_tensors"), \
             patch.object(_cann_mhc, "_load_cann_mhc_op", side_effect=ops.__getitem__):
            hidden, post, combine, residual = mhc_pre(x, phi, alpha, bias, 4, backend="cann")
            self.assertIs(residual, x)
            output = mhc_post(hidden.square(), residual, post, combine, 4, backend="cann")
            output.sum().backward()

        streams = x.detach().unflatten(-1, (4, 128))
        expected_output = torch.stack([
            sum(combine.detach()[:, :, source, target, None] * streams[:, :, source] for source in range(4))
            + post.detach()[:, :, target, None] * hidden.detach().square()
            for target in range(4)
        ], dim=2).flatten(2)
        expected_grad = (
            (2 * hidden.detach() * post.detach().sum(-1, keepdim=True)).unsqueeze(-2)
            + combine.detach().sum(-1).unsqueeze(-1)
        ).expand_as(streams).flatten(2)
        torch.testing.assert_close(output, expected_output)
        torch.testing.assert_close(x.grad, expected_grad)
        for name, tensor in (("phi", phi), ("alpha", alpha), ("bias", bias)):
            self.assertIsNotNone(tensor.grad, msg=f"Missing coefficient gradient for {name}")
            self.assertGreater(tensor.grad.abs().sum(), 0, msg=f"Empty coefficient gradient for {name}")

    def test_cpu_is_rejected_before_extension_load(self):
        """Explicit CANN execution cannot silently fall back to a CPU formula."""
        x, phi, alpha, bias = self._inputs()
        with patch.object(_cann_mhc, "_load_cann_mhc_op") as loader:
            with self.assertRaisesRegex(ValueError, "same NPU"):
                mhc_pre(x, phi, alpha, bias, 4, backend="cann")
            loader.assert_not_called()

    def test_unsupported_pre_settings_fail_before_extension_load(self):
        """Gamma and unsupported A2 shapes must not be silently ignored."""
        x, phi, alpha, bias = self._inputs()
        cases = ({"gamma": torch.ones(512)}, {"num_stream": 2}, {"sinkhorn_iters": 10}, {"hc_eps": 0})
        with patch.object(_cann_mhc, "_load_cann_mhc_op") as loader:
            for changes in cases:
                kwargs = {"num_stream": 4, "backend": "cann", **changes}
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    mhc_pre(x, phi, alpha, bias, **kwargs)
            loader.assert_not_called()

    def test_dtensor_layout_is_not_silently_discarded(self):
        """Custom CANN operators have no distributed sharding rule here."""
        for tensor_type in (_cann_mhc.DTensor, _cann_mhc.TorchDTensor):
            with self.subTest(tensor_type=tensor_type), self.assertRaisesRegex(ValueError, "DTensor"):
                _cann_mhc._validate_cann_tensors(MagicMock(spec=tensor_type))

    def test_native_dimension_limit_is_checked_before_loading(self):
        """The backward-supported A2 domain has an upper bound as well as alignment."""
        wide = torch.empty(1, 1, 4 * 100096, device="meta")
        with self.assertRaisesRegex(ValueError, "D < 100000"):
            mhc_pre(wide, torch.empty(0), torch.empty(3), torch.empty(24), 4, backend="cann")
        with self.assertRaisesRegex(ValueError, "D < 100000"):
            mhc_post(wide[..., :100096], wide, torch.empty(0), torch.empty(0), 4, backend="cann")

    def test_no_grad_pre_omits_native_backward_caches(self):
        """The optional wrapper must see no grad-requiring inputs in inference."""
        x, phi, alpha, bias = self._inputs()

        def pre_op(*arguments: torch.Tensor | int | float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Record grad flags while returning only the normal three outputs."""
            for value in arguments[:4]:
                self.assertFalse(value.requires_grad)
            return torch.empty(2, 3, 128), torch.empty(2, 3, 4), torch.empty(2, 3, 16)

        with torch.no_grad(), patch.object(_cann_mhc, "_validate_cann_tensors"), \
             patch.object(_cann_mhc, "_load_cann_mhc_op", return_value=pre_op):
            *_, residual = mhc_pre(x, phi, alpha, bias, 4, backend="cann")
        self.assertIs(residual, x)

    def test_loads_explicit_autograd_entry_and_explains_missing_dependency(self):
        """Top-level torch.ops handles do not have the same autograd contract."""
        entry = object()
        module = SimpleNamespace(mhc_pre_sinkhorn=entry)
        with patch.object(_cann_mhc.importlib, "import_module", return_value=module) as importer:
            self.assertIs(_cann_mhc._load_cann_mhc_op("mhc_pre_sinkhorn"), entry)
            importer.assert_called_once_with("cann_ops_transformer.ops.mhc.mhc_pre_sinkhorn")
        _cann_mhc._load_cann_mhc_op.cache_clear()
        with patch.object(_cann_mhc.importlib, "import_module", side_effect=ImportError("absent")):
            with self.assertRaisesRegex(RuntimeError, "backend='cann'.*mhc_post"):
                _cann_mhc._load_cann_mhc_op("mhc_post")

    def test_replacements_keep_parameters_and_select_backend(self):
        """YAML replacement kwargs choose the backend without remapping weights."""
        source = nn.Module()
        source.num_stream = 4
        source.phi = nn.Linear(512, 24, bias=False)
        source.branch_alpha = nn.Parameter(torch.ones(3))
        source.branch_beta = nn.Parameter(torch.zeros(24))
        source.config = SimpleNamespace(mhc_use_gamma=False, mhc_recur_norm=20)
        source.hc_eps = source.norm_eps = 1e-6
        pre = MhcPreModule(module=source, backend="cann")
        post = MhcPostModule(module=source, backend="cann")
        self.assertIs(pre.phi.weight, source.phi.weight)
        self.assertIs(pre.branch_alpha, source.branch_alpha)
        self.assertIs(pre.branch_beta, source.branch_beta)
        with patch("hyper_parallel.components.modules.mhc.mhc_pre") as pre_function:
            pre(torch.empty(2, 3, 512))
            self.assertEqual(pre_function.call_args.kwargs, {"backend": "cann"})
        with patch("hyper_parallel.components.modules.mhc.mhc_post") as post_function:
            post(*(torch.empty(0) for _ in range(4)))
            self.assertEqual(post_function.call_args.kwargs, {"backend": "cann"})
        self.assertEqual(MhcPreModule(module=source).backend, "omni")

    def test_unknown_backend_is_rejected(self):
        """A misspelled backend must not change the chosen implementation."""
        x, phi, alpha, bias = self._inputs()
        with self.assertRaisesRegex(ValueError, "backend"):
            mhc_pre(x, phi, alpha, bias, 4, backend="unknown")
        with self.assertRaisesRegex(ValueError, "backend"):
            mhc_post(x, x, x, x, 4, backend="unknown")
        with self.assertRaisesRegex(ValueError, "backend"):
            MhcPostModule(module=nn.Identity(), backend="unknown")
