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
"""Unit tests for dry-run runtime replay helpers."""
# pylint: disable=protected-access

import os
import unittest
from collections import OrderedDict, namedtuple
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from hyper_parallel.components.losses import _vocab_parallel_cross_entropy as loss_parallel_ops
from hyper_parallel.distributed.expert_parallel import experts as ep_experts
from hyper_parallel.tools.dryrun.runtime import (
    DryRunBatchIterator,
    DryRunBatchMocker,
    DryRunRuntime,
    FakeHostScalarShim,
    _MoERoutingPlans,
    _MoERoutingRuntime,
    _TPCrossEntropyReplay,
    _is_supported_moe_module,
    derive_tp_target_counts,
    install_moe_replay,
    install_tp_cross_entropy_replay,
)


class _ReadOnlyMapping(Mapping):
    """Minimal immutable mapping used to test fallback reconstruction."""

    def __init__(self, items):
        self._data = dict(items)

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


class _UnreconstructibleMapping(Mapping):
    """Mapping that rejects both clear/update and sequence construction."""

    def __init__(self, a, b):
        self._data = {"a": a, "b": b}

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


@dataclass
class _SampleDataclass:
    """Dataclass with both init and post-init tensor fields."""

    x: torch.Tensor
    tag: str
    derived: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.derived = self.x + 1


class _TensorCarrier:
    """Plain object holding nested tensors."""

    def __init__(self, payload):
        self.payload = payload


_NamedPair = namedtuple("_NamedPair", ["first", "second"])


class _ThreeProjMoE(nn.Module):
    """Minimal three-projection MoE module with optional shared expert."""

    def __init__(self, num_experts=4, top_k=2, hidden=8, inter=6, use_fp32=False):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        self.experts = nn.Module()
        self.experts.w1 = nn.Parameter(torch.empty(num_experts, inter, hidden))
        self.experts.w2 = nn.Parameter(torch.empty(num_experts, hidden, inter))
        self.experts.w3 = nn.Parameter(torch.empty(num_experts, inter, hidden))
        self.experts.local_expert_count = num_experts
        self.shared_expert = nn.Linear(hidden, hidden, bias=False)
        self.shared_expert_gate = nn.Linear(hidden, 1, bias=False)
        self.last_aux_loss = 1.0
        self._hp_moe_ep_fp32_routing = use_fp32


class _PackedGateUpMoE(nn.Module):
    """Minimal packed gate_up_proj MoE module."""

    def __init__(self, num_experts=4, top_k=2, hidden=8, inter=6):
        super().__init__()
        self.config = SimpleNamespace(n_routed_experts=num_experts, num_experts_per_tok=top_k)
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * inter, hidden))
        self.experts.down_proj = nn.Parameter(torch.empty(num_experts, hidden, inter))
        self.experts.local_expert_count = num_experts


class _HFStyleThreeProjMoE(nn.Module):
    """MoE exposing gate_proj/up_proj/down_proj weights."""

    def __init__(self, num_experts=4, top_k=2, hidden=8, inter=6):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        self.experts = nn.Module()
        self.experts.gate_proj = nn.Parameter(torch.empty(num_experts, inter, hidden))
        self.experts.up_proj = nn.Parameter(torch.empty(num_experts, inter, hidden))
        self.experts.down_proj = nn.Parameter(torch.empty(num_experts, hidden, inter))
        self.experts.local_expert_count = num_experts


class TestDryRunRuntimeHelpers(unittest.TestCase):
    """Tests for DryRunRuntime, DryRunBatchMocker, and derive_tp_target_counts."""

    def test_from_torchrun_env_valid_and_errors(self):
        """Validate torchrun env parsing and boundary checks."""
        with patch.dict(os.environ, {"RANK": "2", "WORLD_SIZE": "8", "LOCAL_RANK": "2"}, clear=True):
            rt = DryRunRuntime.from_torchrun_env()
            self.assertEqual(rt, DryRunRuntime(rank=2, world_size=8, local_rank=2))

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "missing environment variables"):
                DryRunRuntime.from_torchrun_env()

        with patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "0", "LOCAL_RANK": "0"}, clear=True):
            with self.assertRaisesRegex(ValueError, "WORLD_SIZE"):
                DryRunRuntime.from_torchrun_env()

        with patch.dict(os.environ, {"RANK": "4", "WORLD_SIZE": "4", "LOCAL_RANK": "0"}, clear=True):
            with self.assertRaisesRegex(ValueError, "RANK"):
                DryRunRuntime.from_torchrun_env()

        with patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "4", "LOCAL_RANK": "-1"}, clear=True):
            with self.assertRaisesRegex(ValueError, "LOCAL_RANK"):
                DryRunRuntime.from_torchrun_env()

    def test_batch_mocker_preserves_containers_and_rejects_custom_carriers(self):
        """Mock tensors across dicts, OrderedDict, immutable mappings, namedtuples, and dataclasses."""
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        mocker = DryRunBatchMocker(fake_mode, torch.device("cpu"))
        sample_dc = _SampleDataclass(x=torch.ones(2, 3), tag="ok")
        payload = {
            "ordered": OrderedDict([("a", torch.zeros(2))]),
            "readonly": _ReadOnlyMapping([("b", torch.ones(3))]),
            "list": [torch.ones(1), "plain"],
            "tuple": (torch.ones(2), 42),
            "named": _NamedPair(torch.ones(2), "keep"),
            "dc": sample_dc,
        }
        mocked = mocker.mock(payload)
        self.assertIsInstance(mocked["ordered"], OrderedDict)
        self.assertTrue(hasattr(mocked["ordered"]["a"], "fake_mode"))
        self.assertIsInstance(mocked["readonly"], _ReadOnlyMapping)
        self.assertTrue(hasattr(mocked["readonly"]["b"], "fake_mode"))
        self.assertIsInstance(mocked["named"], _NamedPair)
        self.assertTrue(hasattr(mocked["named"].first, "fake_mode"))
        self.assertEqual(mocked["named"].second, "keep")
        self.assertIsInstance(mocked["dc"], _SampleDataclass)
        self.assertTrue(hasattr(mocked["dc"].x, "fake_mode"))
        self.assertTrue(hasattr(mocked["dc"].derived, "fake_mode"))
        self.assertEqual(mocked["dc"].tag, "ok")

        with self.assertRaisesRegex(ValueError, "cannot preserve batch mapping carrier"):
            mocker.mock(_UnreconstructibleMapping(torch.ones(1), torch.ones(1)))

        with self.assertRaisesRegex(ValueError, "cannot mock tensor-bearing batch carrier"):
            mocker.mock(_TensorCarrier({"nested": [torch.ones(1)]}))

        plain_obj = _TensorCarrier({"nested": ["no_tensor", (1, 2)]})
        self.assertIs(mocker.mock(plain_obj), plain_obj)

    def test_derive_tp_target_counts_and_errors(self):
        """Derive per-TP-rank label counts with shifted loss_mask and validate errors."""
        labels = torch.tensor([[0, 1, 5, -100]])
        loss_mask = torch.tensor([[1, 1, 1, 0]])
        valid, owned = derive_tp_target_counts(
            {"labels": labels, "loss_mask": loss_mask}, vocab_size=8, tp_size=2
        )
        self.assertEqual(valid, 2)
        self.assertEqual(owned, (1, 1))

        with self.assertRaisesRegex(ValueError, "must contain labels"):
            derive_tp_target_counts({}, vocab_size=8, tp_size=2)
        with self.assertRaisesRegex(ValueError, "must be a torch.Tensor"):
            derive_tp_target_counts({"shift_labels": [1, 2]}, vocab_size=8, tp_size=2)
        with self.assertRaisesRegex(ValueError, "outside the configured vocabulary"):
            derive_tp_target_counts({"shift_labels": torch.tensor([99])}, vocab_size=8, tp_size=2)

    def test_fake_host_scalar_shim_conversions_and_batch_iterator(self):
        """Test __float__, __int__, __bool__, and real tensor passthrough plus DryRunBatchIterator."""
        shim = FakeHostScalarShim()
        real_t = torch.tensor(3.0)
        with FakeTensorMode(), shim:
            fake_t = torch.empty((), dtype=torch.float32)
            shim.register(fake_t, 5.0)
            self.assertEqual(float(fake_t), 5.0)
            self.assertEqual(int(fake_t), 5)
            self.assertTrue(bool(fake_t))
        self.assertEqual(real_t.item(), 3.0)

        it = DryRunBatchIterator({"a": 1}, {"b": 2})
        self.assertIs(iter(it), it)
        self.assertEqual(next(it), ({"a": 1}, {"b": 2}))


class TestMoEAndCrossEntropyReplay(unittest.TestCase):
    """Tests for MoE routing plans, MoE runtime replay, and TP cross-entropy replay."""

    def test_moe_routing_plans_modes_and_validations(self):
        """Exercise balanced, hotspot, explicit modes and error paths."""
        self.assertFalse(_is_supported_moe_module(nn.Linear(2, 2)))
        self.assertTrue(_is_supported_moe_module(_ThreeProjMoE()))
        self.assertTrue(_is_supported_moe_module(_PackedGateUpMoE()))
        self.assertTrue(_is_supported_moe_module(_HFStyleThreeProjMoE()))

        with self.assertRaisesRegex(ValueError, "mode must be one of"):
            _MoERoutingPlans("invalid")
        with self.assertRaisesRegex(ValueError, "source_expert_loads"):
            _MoERoutingPlans("explicit", source_expert_loads=None)
        with self.assertRaisesRegex(ValueError, "non-negative hotspot_expert"):
            _MoERoutingPlans("hotspot", hotspot_expert=-1)

        mod = SimpleNamespace(num_experts=4, top_k=2)
        hotspot_plans = _MoERoutingPlans("hotspot", hotspot_expert=1)
        plan = hotspot_plans.moe_routing_plan("moe", mod, ep_size=2, ep_rank=0, local_tokens=3, local_expert_count=2)
        self.assertEqual(plan.local_expert_loads, (0, 12))
        self.assertEqual(hotspot_plans.metadata()["mode"], "hotspot")

        with self.assertRaisesRegex(ValueError, "hotspot_expert"):
            _MoERoutingPlans("hotspot", hotspot_expert=9).moe_routing_plan("moe", mod, 2, 0, 3, 2)

        explicit_plans = _MoERoutingPlans("explicit", source_expert_loads=[[2, 1, 2, 1], [1, 2, 1, 2]])
        eplan = explicit_plans.moe_routing_plan("moe", mod, ep_size=2, ep_rank=1, local_tokens=3, local_expert_count=2)
        self.assertEqual(eplan.input_split_sizes, (3, 3))
        self.assertEqual(eplan.output_split_sizes, (3, 3))

        with self.assertRaisesRegex(ValueError, "must have 2 rows"):
            _MoERoutingPlans("explicit", source_expert_loads=[[6, 0, 0, 0]]).moe_routing_plan("moe", mod, 2, 0, 3, 2)
        with self.assertRaisesRegex(ValueError, "must have 4 entries"):
            _MoERoutingPlans("explicit", source_expert_loads=[[6, 0], [6, 0]]).moe_routing_plan("moe", mod, 2, 0, 3, 2)
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            _MoERoutingPlans("explicit", source_expert_loads=[[-1, 7, 0, 0], [6, 0, 0, 0]]).moe_routing_plan(
                "moe", mod, 2, 0, 3, 2
            )
        with self.assertRaisesRegex(ValueError, "must sum to"):
            _MoERoutingPlans("explicit", source_expert_loads=[[1, 0, 0, 0], [6, 0, 0, 0]]).moe_routing_plan(
                "moe", mod, 2, 0, 3, 2
            )
        with self.assertRaisesRegex(ValueError, "no local experts"):
            hotspot_plans.moe_routing_plan("moe", mod, 2, 0, 3, 0)
        with self.assertRaisesRegex(ValueError, "incompatible with EP"):
            hotspot_plans.moe_routing_plan("moe", mod, 2, 0, 3, 3)

    def test_moe_runtime_ep1_three_proj_and_packed_forward_backward(self):
        """Run EP=1 MoE replay forward and backward on three-projection and packed layouts."""
        for use_fp32 in (False, True):
            wrapper = nn.Module()
            wrapper.moe = _ThreeProjMoE(use_fp32=use_fp32)
            mesh = SimpleNamespace(ep_size=1, ep_rank=0, fsdp_moe_mesh=None)
            with FakeTensorMode(allow_non_fake_inputs=True):
                wrapper.to_empty(device="cpu")
                install_moe_replay(wrapper, mesh, "balanced")
                x = torch.empty(2, 4, 8, requires_grad=True)
                out = wrapper.moe(x)
                self.assertEqual(out.shape, (2, 4, 8))
                out.sum().backward()
                self.assertIsNotNone(x.grad)
                self.assertIsNone(wrapper.moe.last_aux_loss)

        packed_wrapper = nn.Module()
        packed_wrapper.moe = _PackedGateUpMoE()
        mesh = SimpleNamespace(ep_size=1, ep_rank=0, fsdp_moe_mesh=None)
        with FakeTensorMode(allow_non_fake_inputs=True):
            packed_wrapper.to_empty(device="cpu")
            install_moe_replay(packed_wrapper, mesh, "balanced")
            x = torch.empty(2, 4, 8, requires_grad=True)
            out = packed_wrapper.moe(x)
            out.sum().backward()
            self.assertEqual(x.grad.shape, x.shape)

    def test_moe_runtime_ep_gt1_eager_and_grouped_compute(self):
        """Exercise EP>1 _run_hf_native_ep_compute across eager and grouped GEMM paths."""
        original_ep = ep_experts.ep_routed_forward
        try:
            for moe_cls, grouped, with_gate in (
                (_HFStyleThreeProjMoE, False, False),
                (_PackedGateUpMoE, False, False),
                (_PackedGateUpMoE, False, True),
                (_PackedGateUpMoE, True, False),
            ):
                wrapper = nn.Module()
                wrapper.moe = moe_cls(num_experts=4, top_k=2, hidden=8, inter=6)
                wrapper.moe.experts.local_expert_count = 2
                wrapper.moe.experts._ep_use_grouped_gemm = grouped
                if with_gate:
                    wrapper.moe.experts._ep_apply_gate = lambda t: t.chunk(2, dim=-1)[0]

                ep_group = MagicMock()
                ep_group.size.return_value = 2
                ep_mesh = MagicMock()
                ep_mesh.get_group.return_value = ep_group
                ep_mesh.size.return_value = 2
                ep_mesh.get_local_rank.return_value = 0
                mesh = SimpleNamespace(ep_size=2, ep_rank=0, fsdp_moe_mesh={"ep": ep_mesh})

                def _router(mod, h):
                    del mod
                    idx = torch.zeros(h.shape[0] * h.shape[1], 2, dtype=torch.int64, device=h.device)
                    w = torch.ones(h.shape[0] * h.shape[1], 2, dtype=h.dtype, device=h.device)
                    return idx, w

                with FakeTensorMode(allow_non_fake_inputs=True):
                    wrapper.to_empty(device="cpu")
                    # Slice local expert weights to 2 experts for EP=2
                    for pname, param in list(wrapper.moe.experts.named_parameters()):
                        setattr(wrapper.moe.experts, pname, nn.Parameter(param[:2]))
                    runtime = _MoERoutingRuntime(_MoERoutingPlans("balanced"), wrapper, mesh).install()
                    x = torch.empty(2, 4, 8, requires_grad=True)
                    with (
                        patch("torch.distributed.get_rank", return_value=0),
                        patch(
                            "hyper_parallel.tools.dryrun.runtime.differentiable_all_to_all_single",
                            side_effect=lambda t, _in, _out, group=None: t.clone(),
                        ),
                    ):
                        out = ep_experts.ep_routed_forward(
                            wrapper.moe, x, router_fn=_router, ep_group=ep_group
                        )
                        out.sum().backward()
                    self.assertEqual(out.shape, (2, 4, 8))
                    with self.assertRaisesRegex(ValueError, "unregistered"):
                        runtime._run_hf_native_ep_compute(
                            nn.Linear(2, 2), x, router_fn=_router, ep_group=ep_group
                        )
        finally:
            ep_experts.ep_routed_forward = original_ep

    def test_tp_cross_entropy_replay_all_reductions_and_errors(self):
        """Exercise TP cross-entropy replay for none, mean, sum reductions and error paths."""
        with self.assertRaisesRegex(ValueError, "non-negative and sum"):
            _TPCrossEntropyReplay(-1, (0,))
        replay = _TPCrossEntropyReplay(4, (2, 2))
        with self.assertRaisesRegex(ValueError, "expected 4"):
            replay._local_count(8, 4, 0)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            replay._local_count(2, 2, 0)

        original_fn = loss_parallel_ops.DistributedCrossEntropyFunction
        try:
            install_tp_cross_entropy_replay(4, (2, 2))
            mesh = MagicMock()
            mesh.size.return_value = 2
            mesh.get_local_rank.return_value = 0
            mesh.get_group.return_value = "tp_group"
            with (
                FakeTensorMode(),
                patch.object(
                    loss_parallel_ops,
                    "distributed_log_softmax",
                    side_effect=lambda x, dim=-1, mesh=None, mesh_dim=0: x,
                ),
                patch.object(
                    loss_parallel_ops,
                    "_differentiable_all_reduce",
                    side_effect=lambda x, op="sum", group=None: x,
                ),
            ):
                for reduction in ("none", "mean", "sum"):
                    logits = torch.empty(4, 8, requires_grad=True)
                    target = torch.empty(4, dtype=torch.int64)
                    out = loss_parallel_ops.DistributedCrossEntropyFunction.apply(
                        logits, target, None, -100, reduction, 16, mesh, 0
                    )
                    out.sum().backward()
                    self.assertEqual(logits.grad.shape, logits.shape)
        finally:
            loss_parallel_ops.DistributedCrossEntropyFunction = original_fn


if __name__ == "__main__":
    unittest.main()
