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
"""Unit tests for hyper_parallel.core.dtensor.init_weights."""

import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import torch

from hyper_parallel.core.dtensor import init_weights as init_weights_module

_INIT_ON_DEVICE_TARGET = "hyper_parallel.core.dtensor.init_weights._init_on_device"


@contextmanager
def _fake_init_on_device(device, include_buffers=False):
    """Stand-in for the Torch ``init_on_device`` context manager."""
    del device, include_buffers
    yield


class TestInitEmptyWeights(unittest.TestCase):
    """Tests for InitEmptyWeights."""

    def test_deterministic_meta_construction_does_not_fill_discarded_storage(self):
        """Feature: Empty parameter initialization under deterministic algorithms.

        Description: Inspect real factory calls while constructing a meta Linear.
        Expectation: Skip filling discarded CPU storage and preserve deterministic algorithms.
        """
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        fill = torch.utils.deterministic.fill_uninitialized_memory
        original_empty = torch.empty
        observed = []

        def _observed_empty(*args, **kwargs):
            observed.append(torch.utils.deterministic.fill_uninitialized_memory)
            return original_empty(*args, **kwargs)

        try:
            torch.use_deterministic_algorithms(True)
            torch.utils.deterministic.fill_uninitialized_memory = True
            with patch.object(torch, "empty", side_effect=_observed_empty):
                with init_weights_module.init_empty_weights():
                    module = torch.nn.Linear(32, 32)
            self.assertTrue(module.weight.is_meta, "The parameter must retain no physical storage")
            self.assertTrue(observed, "Real tensor factories must execute during construction")
            self.assertFalse(any(observed), "Discarded CPU allocations must not be filled")
            self.assertTrue(torch.are_deterministic_algorithms_enabled(), "Keep deterministic algorithms enabled")
            self.assertTrue(torch.utils.deterministic.fill_uninitialized_memory, "Restore the caller's fill policy")
        finally:
            torch.utils.deterministic.fill_uninitialized_memory = fill
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)

    def test_nested_empty_context_restores_fill_policy_after_exception(self):
        """Feature: Scoped deterministic initialization state.

        Description: Raise inside nested empty initialization for both caller fill policies.
        Expectation: Restore the original fill policy even on exceptional exit.
        """
        original_fill = torch.utils.deterministic.fill_uninitialized_memory
        try:
            for fill in (True, False):
                with self.subTest(fill=fill):
                    torch.utils.deterministic.fill_uninitialized_memory = fill
                    with self.assertRaisesRegex(RuntimeError, "construction failed"):
                        with init_weights_module.init_empty_weights():
                            with init_weights_module.init_empty_weights():
                                raise RuntimeError("construction failed")
                    self.assertEqual(torch.utils.deterministic.fill_uninitialized_memory, fill,
                                     "Nested or exceptional construction must preserve caller state")
        finally:
            torch.utils.deterministic.fill_uninitialized_memory = original_fill

    @patch(_INIT_ON_DEVICE_TARGET, MagicMock(side_effect=_fake_init_on_device))
    def test_preserves_independent_npu_fill_policy(self):
        """Feature: Empty parameter initialization alongside torch_npu's asymmetric fill patch.

        Description: Simulate torch_npu's setter-only property patch under default NPU False,
            explicit NPU True (normal and exception exit), nested contexts, and lazy torch_npu import.
        Expectation: Preserve Torch's fill_uninitialized_memory and NPU's independent fill flag.
        """
        det_mod = torch.utils.deterministic
        det_cls = type(det_mod)
        orig_prop = getattr(det_cls, "fill_uninitialized_memory")
        orig_fget = orig_prop.fget
        orig_fset = orig_prop.fset
        orig_torch_fill = det_mod.fill_uninitialized_memory
        real_npu_c = getattr(sys.modules.get("torch_npu"), "_C", None)
        real_npu_getter = getattr(real_npu_c, "_npu_get_fill_uninitialized_memory", None)
        real_npu_setter = getattr(real_npu_c, "_npu_set_fill_uninitialized_memory", None)
        orig_real_npu_fill = bool(real_npu_getter()) if callable(real_npu_getter) else None
        npu_state = {"fill": False}

        def _get_npu_fill() -> bool:
            return npu_state["fill"]

        def _set_npu_fill(mode: bool) -> None:
            npu_state["fill"] = bool(mode)

        fake_c = types.SimpleNamespace(
            _npu_get_fill_uninitialized_memory=_get_npu_fill,
            _npu_set_fill_uninitialized_memory=_set_npu_fill,
        )
        fake_torch_npu = types.ModuleType("torch_npu")
        fake_torch_npu._C = fake_c

        def _patched_fset(self, mode):
            orig_fset(self, mode)
            fake_c._npu_set_fill_uninitialized_memory(mode)

        try:
            setattr(det_cls, "fill_uninitialized_memory", property(orig_fget, _patched_fset))

            with patch.dict(sys.modules, {"torch_npu": fake_torch_npu}):
                with self.subTest(case="default_npu_false"):
                    orig_fset(det_mod, True)
                    npu_state["fill"] = False
                    with init_weights_module.init_empty_weights():
                        self.assertFalse(
                            det_mod.fill_uninitialized_memory,
                            f"Torch fill policy must be False inside context, got={det_mod.fill_uninitialized_memory}",
                        )
                        self.assertFalse(
                            npu_state["fill"],
                            f"NPU fill flag must be False inside context, got={npu_state['fill']}",
                        )
                    self.assertTrue(
                        det_mod.fill_uninitialized_memory,
                        f"Torch fill policy must restore to True, got={det_mod.fill_uninitialized_memory}",
                    )
                    self.assertFalse(
                        npu_state["fill"],
                        f"Default NPU fill flag must stay False on exit, got={npu_state['fill']}",
                    )

                for raise_in_ctx in (False, True):
                    with self.subTest(case="explicit_npu_true", raise_in_ctx=raise_in_ctx):
                        det_mod.fill_uninitialized_memory = True
                        self.assertTrue(
                            npu_state["fill"],
                            f"Precondition NPU fill flag must be True, got={npu_state['fill']}",
                        )
                        if raise_in_ctx:
                            with self.assertRaisesRegex(RuntimeError, "boom"):
                                with init_weights_module.init_empty_weights():
                                    self.assertFalse(
                                        npu_state["fill"],
                                        f"NPU fill flag must suspend to False, got={npu_state['fill']}",
                                    )
                                    raise RuntimeError("boom")
                        else:
                            with init_weights_module.init_empty_weights():
                                self.assertFalse(
                                    npu_state["fill"],
                                    f"NPU fill flag must suspend to False, got={npu_state['fill']}",
                                )
                        self.assertTrue(
                            det_mod.fill_uninitialized_memory,
                            f"Torch fill policy must restore to True, got={det_mod.fill_uninitialized_memory}",
                        )
                        self.assertTrue(
                            npu_state["fill"],
                            f"Explicit NPU fill flag must restore to True, got={npu_state['fill']}",
                        )

                for initial_npu_fill in (False, True):
                    for raise_in_inner in (False, True):
                        with self.subTest(
                            case="nested_npu_state",
                            initial_npu_fill=initial_npu_fill,
                            raise_in_inner=raise_in_inner,
                        ):
                            orig_fset(det_mod, True)
                            npu_state["fill"] = initial_npu_fill
                            if raise_in_inner:
                                with self.assertRaisesRegex(RuntimeError, "nested boom"):
                                    with init_weights_module.init_empty_weights():
                                        with init_weights_module.init_empty_weights():
                                            self.assertFalse(
                                                det_mod.fill_uninitialized_memory,
                                                "Torch fill policy must be False inside nested context, "
                                                f"got={det_mod.fill_uninitialized_memory}",
                                            )
                                            self.assertFalse(
                                                npu_state["fill"],
                                                "NPU fill flag must be False inside nested context, "
                                                f"got={npu_state['fill']}",
                                            )
                                            raise RuntimeError("nested boom")
                            else:
                                with init_weights_module.init_empty_weights():
                                    with init_weights_module.init_empty_weights():
                                        self.assertFalse(
                                            det_mod.fill_uninitialized_memory,
                                            "Torch fill policy must be False inside nested context, "
                                            f"got={det_mod.fill_uninitialized_memory}",
                                        )
                                        self.assertFalse(
                                            npu_state["fill"],
                                            "NPU fill flag must be False inside nested context, "
                                            f"got={npu_state['fill']}",
                                        )
                            self.assertTrue(
                                det_mod.fill_uninitialized_memory,
                                f"Torch fill policy must restore to True, got={det_mod.fill_uninitialized_memory}",
                            )
                            self.assertEqual(
                                npu_state["fill"],
                                initial_npu_fill,
                                f"Nested context must preserve NPU fill flag: "
                                f"expected={initial_npu_fill}, got={npu_state['fill']}",
                            )

            with self.subTest(case="lazy_torch_npu_import_inside_context"):
                setattr(det_cls, "fill_uninitialized_memory", orig_prop)
                det_mod.fill_uninitialized_memory = True
                npu_state["fill"] = False
                with patch.dict(sys.modules, clear=False):
                    sys.modules.pop("torch_npu", None)
                    with init_weights_module.init_empty_weights():
                        sys.modules["torch_npu"] = fake_torch_npu
                        setattr(det_cls, "fill_uninitialized_memory", property(orig_fget, _patched_fset))
                self.assertTrue(
                    det_mod.fill_uninitialized_memory,
                    f"Torch fill policy must restore to True, got={det_mod.fill_uninitialized_memory}",
                )
                self.assertFalse(
                    npu_state["fill"],
                    f"Lazily imported torch_npu default False flag must stay False, got={npu_state['fill']}",
                )
        finally:
            setattr(det_cls, "fill_uninitialized_memory", orig_prop)
            det_mod.fill_uninitialized_memory = orig_torch_fill
            if orig_real_npu_fill is not None and callable(real_npu_setter):
                real_npu_setter(orig_real_npu_fill)

    @patch(_INIT_ON_DEVICE_TARGET, MagicMock(side_effect=_fake_init_on_device))
    def test_default_include_buffers_false(self):
        """Test default include buffers false."""
        from hyper_parallel.core.dtensor.init_weights import init_empty_weights

        with init_empty_weights():
            pass

        init_weights_module._init_on_device.assert_called_once_with(  # pylint: disable=protected-access
            torch.device("meta"), include_buffers=False
        )

    @patch(_INIT_ON_DEVICE_TARGET, MagicMock(side_effect=_fake_init_on_device))
    def test_include_buffers_true(self):
        """Test include buffers true."""
        from hyper_parallel.core.dtensor.init_weights import init_empty_weights

        with init_empty_weights(include_buffers=True):
            pass

        init_weights_module._init_on_device.assert_called_once_with(  # pylint: disable=protected-access
            torch.device("meta"), include_buffers=True
        )


class TestInitOnDevice(unittest.TestCase):
    """Tests for InitOnDevice."""

    @patch(_INIT_ON_DEVICE_TARGET, MagicMock(side_effect=_fake_init_on_device))
    def test_default(self):
        """Test default."""
        from hyper_parallel.core.dtensor.init_weights import init_on_device

        my_device = "npu:0"
        with init_on_device(my_device):
            pass

        init_weights_module._init_on_device.assert_called_once_with(  # pylint: disable=protected-access
            my_device, include_buffers=False
        )

    @patch(_INIT_ON_DEVICE_TARGET, MagicMock(side_effect=_fake_init_on_device))
    def test_include_buffers_true(self):
        """Test include buffers true."""
        from hyper_parallel.core.dtensor.init_weights import init_on_device

        my_device = "cpu"
        with init_on_device(my_device, include_buffers=True):
            pass

        init_weights_module._init_on_device.assert_called_once_with(  # pylint: disable=protected-access
            my_device, include_buffers=True
        )


if __name__ == "__main__":
    unittest.main()
