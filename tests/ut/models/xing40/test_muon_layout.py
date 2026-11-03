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
"""Logical matrix semantics and native Muon integration without device setup."""

import math
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch

from hyper_parallel.core.optimizer.muon import Muon, compute_muon_slice_scale
from hyper_parallel.models.xing40.adapter import optim
from hyper_parallel.models.xing40.adapter.muon_layout import Xing40MuonLayout
from tests.common.mark_utils import arg_mark


def _config(**overrides):
    """Use asymmetric dimensions to expose incorrect axes and scale grouping."""
    values = {"q_lora_rank": 3, "kv_lora_rank": 2, "qk_nope_head_dim": 2,
              "qk_rope_head_dim": 1, "v_head_dim": 2, "num_attention_heads": 2,
              "hidden_size": 4, "intermediate_size": 5, "moe_intermediate_size": 3,
              "n_shared_experts": 1}
    values.update(overrides)
    return SimpleNamespace(**values)


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
class TestXing40MuonLayout(unittest.TestCase):
    """Verify actual channel identities rather than only reversible reshapes."""

    def setUp(self) -> None:
        """Create one complete two-head layout."""
        self.layout = Xing40MuonLayout.from_config(_config())

    def test_qkv_keeps_three_independent_storage_sections(self) -> None:
        """Q-down, KV-latent and K-RoPE keep their distinct native row ranges."""
        tensor = torch.arange(24).view(6, 4)
        pieces = self.layout.reshape_fn("model.layers.0.self_attn.linear_qkv.weight", tensor)
        self.assertEqual([tuple(piece.shape) for piece in pieces], [(3, 4), (2, 4), (1, 4)])
        self.assertEqual([piece[:, 0].tolist() for piece in pieces], [[0, 4, 8], [12, 16], [20]])
        self.assertTrue(all(piece.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
                            for piece in pieces))
        pieces[1].fill_(-1)
        self.assertEqual(tensor[:, 0].tolist(), [0, 4, 8, -1, -1, 20])

    def test_dense_shared_and_transposed_experts_preserve_gate_up_identities(self) -> None:
        """Dense rows and expert last-axis halves keep each gate/up and expert separate."""
        for suffix, rows in (("linear_fc1.weight", 10), ("shared_experts.linear_fc1.weight", 6)):
            with self.subTest(suffix=suffix):
                tensor = torch.arange(rows * 4).view(rows, 4)
                pieces = self.layout.reshape_fn("model.layers.0.mlp." + suffix, tensor)
                self.assertEqual(pieces[0][:, 0].tolist(), list(range(0, rows * 2, 4)))
                self.assertEqual(pieces[1][:, 0].tolist(), list(range(rows * 2, rows * 4, 4)))
        tensor = torch.arange(48).view(2, 4, 6)
        pieces = self.layout.reshape_fn("model.layers.2.mlp.experts.gate_up_proj", tensor)
        self.assertEqual([tuple(piece.shape) for piece in pieces], [(2, 4, 3), (2, 4, 3)])
        self.assertEqual(pieces[0][:, :, 0].tolist(), [[0, 6, 12, 18], [24, 30, 36, 42]])
        self.assertEqual(pieces[1][:, :, 0].tolist(), [[3, 9, 15, 21], [27, 33, 39, 45]])
        self.assertTrue(all(piece.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
                            for piece in pieces))

    def test_periodic_groups_collect_all_heads_and_scatter_to_original_channels(self) -> None:
        """Q/KV groups aggregate matching channels across heads, rather than per head."""
        cases = (("q_b_proj.weight", (6, 3), [[0, 3, 9, 12], [6, 15]], [7, 7, 9, 7, 7, 9]),
                 ("kv_b_proj.weight", (8, 2), [[0, 2, 8, 10], [4, 6, 12, 14]], [7, 7, 9, 9, 7, 7, 9, 9]))
        for suffix, shape, expected, restored in cases:
            with self.subTest(suffix=suffix):
                tensor = torch.arange(math.prod(shape)).view(shape)
                transform = self.layout.ns_transform_fn("model.layers.0.self_attn." + suffix, tensor)
                self.assertEqual([piece[:, 0].tolist() for piece in transform.tensors], expected)
                output = torch.empty_like(tensor)
                transform.restore(transform.tensors, output)
                self.assertTrue(torch.equal(output, tensor))
                transform.restore([torch.full_like(piece, value)
                                   for piece, value in zip(transform.tensors, (7, 9))], output)
                self.assertEqual(output[:, 0].tolist(), restored)

    def test_native_muon_scales_each_periodic_matrix_with_its_own_shape(self) -> None:
        """Native scale calls use each logical shape without assuming BF16 update rounding."""
        parameter = torch.nn.Parameter(torch.zeros(6, 3))
        parameter.model_name = "model.layers.0.self_attn.q_b_proj.weight"
        optimizer = Muon([parameter], lr=0.1, momentum=0.0, nesterov=False,
                         weight_decay=0.0, matched_adamw_rms=0.2,
                         reshape_fn=self.layout.reshape_fn, ns_transform_fn=self.layout.ns_transform_fn)
        parameter.grad = torch.ones_like(parameter)
        observed = []

        def observe_scale(tensor: torch.Tensor, matched_rms: float, **kwargs: Any) -> float:
            """Observe the real scaling function without changing native arithmetic."""
            scale = compute_muon_slice_scale(tensor, matched_rms, **kwargs)
            observed.append((tuple(tensor.shape), scale))
            return scale

        with patch("hyper_parallel.core.optimizer.muon.compute_muon_slice_scale", side_effect=observe_scale):
            optimizer.step()
        self.assertEqual([shape for shape, _ in observed], [(4, 3), (2, 3)])
        self.assertAlmostEqual(observed[0][1], 0.2 * math.sqrt(4))
        self.assertAlmostEqual(observed[1][1], 0.2 * math.sqrt(3))
        self.assertTrue(torch.isfinite(parameter).all())
        self.assertGreater(parameter.norm().item(), 0)

    def test_incomplete_matrices_and_invalid_dimensions_are_rejected(self) -> None:
        """A partial TP matrix or a different expert packing must not silently change boundaries."""
        cases = (("self_attn.linear_qkv.weight", (6, 2), False),
                 ("self_attn.q_b_proj.weight", (3, 3), True),
                 ("mlp.linear_fc1.weight", (6, 4), False),
                 ("mlp.experts.gate_up_proj", (2, 6, 4), False))
        for suffix, shape, periodic in cases:
            with self.subTest(suffix=suffix), self.assertRaisesRegex(ValueError, "Unexpected Xing40 Muon shape"):
                callback = self.layout.ns_transform_fn if periodic else self.layout.reshape_fn
                callback("model.layers.0." + suffix, torch.ones(shape))
        for value in (0, True, None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "num_attention_heads"):
                Xing40MuonLayout.from_config(_config(num_attention_heads=value))

    def test_unmatched_matrix_retains_native_identity(self) -> None:
        """Down/output projections retain the existing framework matrix boundary."""
        tensor = torch.ones(3, 4)
        name = "model.layers.0.self_attn.o_proj.weight"
        self.assertIs(self.layout.reshape_fn(name, tensor)[0], tensor)
        self.assertIsNone(self.layout.ns_transform_fn(name, tensor))

    def test_packed_and_separate_parameters_match_two_native_updates(self) -> None:
        """Storage packing preserves logical gate/up parameters and FP32 momentum."""
        generator = torch.Generator().manual_seed(317)
        cases = (("linear_fc1.weight", (10, 4), 0),
                 ("shared_experts.linear_fc1.weight", (6, 4), 0),
                 ("experts.gate_up_proj", (2, 4, 6), -1))
        options = {"lr": 1e-3, "momentum": 0.95, "nesterov": True,
                   "weight_decay": 0.1, "matched_adamw_rms": 0.2}
        for suffix, shape, axis in cases:
            with self.subTest(suffix=suffix):
                initial = torch.randn(shape, generator=generator)
                packed = torch.nn.Parameter(initial.clone())
                packed.model_name = "model.layers.0.mlp." + suffix
                separate = [torch.nn.Parameter(piece.clone()) for piece in initial.chunk(2, dim=axis)]
                packed_optimizer = Muon([packed], **options, reshape_fn=self.layout.reshape_fn,
                                        ns_transform_fn=self.layout.ns_transform_fn)
                separate_optimizer = Muon(separate, **options)
                for _ in range(2):
                    gradient = torch.randn(shape, generator=generator)
                    packed.grad = gradient.clone()
                    for parameter, piece in zip(separate, gradient.chunk(2, dim=axis)):
                        parameter.grad = piece.clone()
                    packed_optimizer.step()
                    separate_optimizer.step()
                    torch.testing.assert_close(packed, torch.cat(separate, dim=axis), rtol=0, atol=0)
                    reference = torch.cat([separate_optimizer.state[parameter]["momentum_buffer"]
                                           for parameter in separate], dim=axis)
                    torch.testing.assert_close(packed_optimizer.state[packed]["momentum_buffer"],
                                               reference, rtol=0, atol=0)


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
          card_mark="onecard", essential_mark="essential")
class TestXing40MuonBuilder(unittest.TestCase):
    """Keep the default contract and opt-in callback ownership explicit."""

    def setUp(self) -> None:
        """Attach a small real model to the existing chained optimizer builder."""
        self.model = torch.nn.Linear(4, 4)
        self.model.config = _config()

    def test_logical_layout_binds_both_native_hooks_without_mutating_options(self) -> None:
        """The real chained optimizer preserves model names and one shared layout."""
        options = {"lr": 1e-4}
        builder = optim.build_optimizer(model=self.model, muon_config=options, adamw_config={},
                                        muon_layout="mindformers_pynative")
        leaves = [leaf for leaf in builder.get_optimizer() if getattr(leaf, "is_muon", False)]
        self.assertEqual(len(leaves), 1)
        self.assertIsInstance(leaves[0].ns_transform_fn.__self__, Xing40MuonLayout)
        self.assertIs(leaves[0].reshape_fn.__self__, leaves[0].ns_transform_fn.__self__)
        self.assertTrue(all(getattr(parameter, "model_name", None)
                            for group in leaves[0].param_groups for parameter in group["params"]))
        self.assertEqual(options, {"lr": 1e-4})

    def test_native_default_and_ambiguous_layouts(self) -> None:
        """The default forwards options unchanged; conflicting callbacks fail before construction."""
        with patch.object(optim, "Muon") as factory:
            optim.build_optimizer(model=self.model, muon_config={"lr": 1e-4}, adamw_config={})
        self.assertEqual(factory.call_args.kwargs["muon_config"], {"lr": 1e-4})
        with self.assertRaisesRegex(ValueError, "muon_layout"):
            optim.build_optimizer(model=self.model, muon_config={}, adamw_config={}, muon_layout="other")
        for name in ("reshape_fn", "ns_transform_fn", "muon_reshape_fn", "muon_ns_transform_fn"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "owns"):
                optim.build_optimizer(model=self.model, muon_config={name: lambda *_args: None},
                                      adamw_config={}, muon_layout="mindformers_pynative")

    def test_conflict_guard_matches_by_key_and_names_every_conflict(self) -> None:
        """The bare and ``muon_``-prefixed keys both conflict, named in the error, even when None."""
        with self.assertRaisesRegex(ValueError, r"\['muon_reshape_fn', 'reshape_fn'\]"):
            optim.build_optimizer(model=self.model, muon_config={"reshape_fn": None, "muon_reshape_fn": None},
                                  adamw_config={}, muon_layout="mindformers_pynative")

    def test_unreplaced_attention_rejected_before_construction(self) -> None:
        """A q_b_proj without its fused linear_qkv sibling fails before the optimizer is built."""
        class _Unreplaced(torch.nn.Module):
            """Minimal module tree with the unfused attention's q_b_proj only."""

            def __init__(self) -> None:
                """Nest q_b_proj under self_attn so its name carries that suffix."""
                super().__init__()
                self.layer = torch.nn.Module()
                self.layer.self_attn = torch.nn.Module()
                self.layer.self_attn.q_b_proj = torch.nn.Linear(2, 2, bias=False)

        model = _Unreplaced()
        model.config = _config()
        with patch.object(optim, "Muon") as factory:
            with self.assertRaisesRegex(ValueError, "fused linear_qkv attention"):
                optim.build_optimizer(model=model, muon_config={}, adamw_config={},
                                      muon_layout="mindformers_pynative")
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
