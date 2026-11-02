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
"""CPU-only candidate tests; no collectives or device operations are permitted."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_parallel.distributed.expert_parallel.moonep.fused import ownership
from hyper_parallel.distributed.expert_parallel.moonep.fused.ownership import validate_parameter_ownership
from tests.common.mark_utils import arg_mark


def _owner(*values: torch.Tensor) -> torch.nn.Module:
    module = torch.nn.Module()
    for index, value in enumerate(values):
        parameter = value if isinstance(value, torch.nn.Parameter) else torch.nn.Parameter(value)
        module.register_parameter(f"weight{index}", parameter)
    return module


def _manage(owner: torch.nn.Module, *, shards: tuple | None = None, sharded: bool = True) -> SimpleNamespace:
    parameters = tuple(owner.parameters())
    shards = parameters if shards is None else shards
    managed = [SimpleNamespace(sharded_param=shard, unsharded_param=current)
               for shard, current in zip(shards, parameters)]
    scheduler = SimpleNamespace(modules=[owner], cell=owner,
                                hsdp_state=SimpleNamespace(hsdp_params=managed, is_shard=sharded),
                                restore_with_all_gather_extension=Mock(side_effect=AssertionError("AG invoked")))
    owner.hsdp_scheduler = scheduler
    return scheduler


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
class TestOwnership(unittest.TestCase):
    """Check actual byte ranges and manager identity with CPU storage."""

    def setUp(self) -> None:
        """Forbid accidental payload communication or NPU stream construction."""
        for name in ("all_gather", "all_gather_into_tensor", "all_to_all_single", "all_reduce"):
            self.enterContext(patch.object(torch.distributed, name, side_effect=AssertionError("collective invoked")))
        for name in ("Stream", "Event", "current_stream", "synchronize", "init"):
            self.enterContext(patch.object(torch.npu, name, side_effect=AssertionError("NPU operation invoked")))

    def test_independent_parameters(self) -> None:
        """Separate dense expert tensors admit without communication."""
        owners = [_owner(torch.ones(3, 4), torch.zeros(4, 2)) for _ in range(3)]
        for owner in owners:
            _manage(owner)
        validate_parameter_ownership(owners)
        for owner in owners:
            owner.hsdp_scheduler.restore_with_all_gather_extension.assert_not_called()

    def test_cross_owner_parameter_identity(self) -> None:
        """Tied Parameter objects are rejected even without a manager."""
        parameter = torch.nn.Parameter(torch.ones(4))
        with self.assertRaisesRegex(ValueError, "tied parameter"):
            validate_parameter_ownership([_owner(parameter), _owner(parameter)])

    def test_same_owner_tied_names(self) -> None:
        """Duplicate parameter names cannot bypass duplicate-free traversal."""
        parameter = torch.nn.Parameter(torch.ones(4))
        with self.assertRaisesRegex(ValueError, "tied parameter"):
            validate_parameter_ownership([_owner(parameter, parameter)])

    def test_different_parameters_overlapping_storage(self) -> None:
        """Object identity does not hide partial, exact, or contained byte aliases."""
        base = torch.arange(32, dtype=torch.float32)
        for start, stop in ((0, 8), (4, 12), (2, 6), (0, 16)):
            with self.subTest(start=start, stop=stop):
                with self.assertRaisesRegex(ValueError, "overlapping parameter storage"):
                    validate_parameter_ownership([_owner(base[:8]), _owner(base[start:stop])])

    def test_disjoint_views_of_one_storage(self) -> None:
        """One flat allocation can safely contain adjacent or separated owners."""
        base = torch.arange(64, dtype=torch.float32)
        validate_parameter_ownership([_owner(base[:12].view(3, 4)), _owner(base[12:24].view(3, 4).t()),
                                      _owner(base[32:48].view(2, 2, 4).permute(2, 0, 1))])

    def test_dense_transpose_still_detects_overlap(self) -> None:
        """Transposed physical layouts use the same exact addressed byte span."""
        base = torch.arange(24, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "overlapping parameter storage"):
            validate_parameter_ownership([_owner(base[:12].view(3, 4).t()), _owner(base[8:16])])

    def test_dtype_reinterpretation_uses_byte_ranges(self) -> None:
        """Different element sizes do not conceal an overlapping backing range."""
        base = torch.arange(16, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "overlapping parameter storage"):
            validate_parameter_ownership([_owner(base[:8]), _owner(base.view(torch.bfloat16)[8:24])])

    def test_invalidated_nonempty_storage_rejected(self) -> None:
        """A nonempty Parameter cannot be admitted after its backing storage was freed."""
        owner = _owner(torch.ones(4))
        owner.weight0.untyped_storage().resize_(0)
        with self.assertRaisesRegex(ValueError, "invalid or undersized"):
            validate_parameter_ownership([owner])

    def test_nondense_views_are_unsupported(self) -> None:
        """Sparse strided views cannot be mistaken for a dense byte range."""
        base = torch.arange(32, dtype=torch.float32)
        for view in (base[::2], base[:1].expand(8), base.as_strided((2, 2, 2), (1, 1, 5))):
            with self.subTest(stride=view.stride()):
                with self.assertRaisesRegex(ValueError, "dense non-overlapping"):
                    validate_parameter_ownership([_owner(view)])

    def test_meta_rejected_even_when_empty(self) -> None:
        """Admission cannot infer backing storage from a meta tensor."""
        for count in (0, 8):
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError, "materialized"):
                    validate_parameter_ownership([_owner(torch.empty(count, device="meta"))])

    def test_empty_parameters_have_no_byte_overlap(self) -> None:
        """Independent empty tensors do not alias merely because data_ptr is zero."""
        base = torch.ones(8)
        validate_parameter_ownership([_owner(torch.empty(0)), _owner(base[:0]), _owner(base)])

    def test_empty_tied_parameter_rejected(self) -> None:
        """Zero bytes do not make shared Parameter ownership valid."""
        parameter = torch.nn.Parameter(torch.empty(0))
        with self.assertRaisesRegex(ValueError, "tied parameter"):
            validate_parameter_ownership([_owner(parameter), _owner(parameter)])

    def test_current_unsharded_views_use_stable_shards(self) -> None:
        """Already restored units are checked without reading or restoring full weights."""
        base = torch.arange(32, dtype=torch.float32)
        first, second = _owner(torch.ones(8)), _owner(torch.ones(8))
        _manage(first, shards=(torch.nn.Parameter(base[:8]),), sharded=False)
        _manage(second, shards=(torch.nn.Parameter(base[8:16]),), sharded=False)
        validate_parameter_ownership([first, second])
        second.hsdp_scheduler.hsdp_state.hsdp_params[0].sharded_param = torch.nn.Parameter(base[4:12])
        with self.assertRaisesRegex(ValueError, "overlapping parameter storage"):
            validate_parameter_ownership([first, second])

    def test_shared_stable_shard_identity(self) -> None:
        """Distinct current wrappers cannot hide one shared stable empty Parameter."""
        shard = torch.nn.Parameter(torch.empty(0))
        owners = [_owner(torch.empty(0)) for _ in range(2)]
        for owner in owners:
            _manage(owner, shards=(shard,), sharded=False)
        with self.assertRaisesRegex(ValueError, "tied sharded parameter"):
            validate_parameter_ownership(owners)

    def test_manager_must_cover_every_owner_parameter(self) -> None:
        """FSDP deduplication or ignored parameters cannot leave owner storage unmanaged."""
        for count in (0, 1):
            with self.subTest(count=count):
                owner = _owner(torch.ones(8), torch.ones(4))
                scheduler = _manage(owner)
                scheduler.hsdp_state.hsdp_params = scheduler.hsdp_state.hsdp_params[:count]
                with self.assertRaisesRegex(ValueError, "cover exactly"):
                    validate_parameter_ownership([owner])

    def test_manager_cannot_cover_other_parameters(self) -> None:
        """A same-length manager belonging to different objects is invalid."""
        owner = _owner(torch.ones(8))
        _manage(owner, shards=(torch.nn.Parameter(torch.ones(8)),))
        with self.assertRaisesRegex(ValueError, "cover exactly"):
            validate_parameter_ownership([owner])

    def test_repeated_owner_rejected(self) -> None:
        """One expert module cannot appear twice in the ownership manifest."""
        owner = _owner(torch.ones(2))
        with self.assertRaisesRegex(ValueError, "distinct modules"):
            validate_parameter_ownership([owner, owner])

    def test_non_independent_fsdp_unit_rejected(self) -> None:
        """Grouped and mismatched scheduler roots do not satisfy the lease contract."""
        owner = _owner(torch.ones(2))
        scheduler = _manage(owner)
        scheduler.modules.append(torch.nn.Module())
        with self.assertRaisesRegex(ValueError, "independent FSDP unit"):
            validate_parameter_ownership([owner])
        scheduler.modules = [owner]
        scheduler.cell = torch.nn.Module()
        with self.assertRaisesRegex(ValueError, "independent FSDP unit"):
            validate_parameter_ownership([owner])

    def test_local_dtensor_extraction_does_not_restore(self) -> None:
        """The local adapter is the only operation required from a distributed wrapper."""
        wrapper = Mock()
        wrapper.to_local.return_value = torch.ones(4)
        with patch.object(ownership, "DTensor", type(wrapper)):
            span = ownership._dense_span(wrapper, "test")
        self.assertEqual(span.end - span.begin, 16)
        wrapper.to_local.assert_called_once_with()

    def test_same_pointer_on_different_devices_is_not_overlap(self) -> None:
        """Range addresses are compared only within one device address space."""
        spans = [ownership._Span(torch.device("cpu"), 64, 128, "first"),
                 ownership._Span(torch.device("meta"), 64, 128, "second")]
        with patch.object(ownership, "_dense_span", side_effect=spans):
            validate_parameter_ownership([_owner(torch.ones(2)), _owner(torch.ones(2))])
