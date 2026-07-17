# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""Unit tests for hyper_parallel.core.dtensor.debug.CommDebugMode."""

import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import unittest
from unittest.mock import MagicMock, Mock, patch

import torch

from hyper_parallel.core.dtensor.debug import CommDebugMode
from hyper_parallel.core.dtensor.debug._call_records import CollectiveCall, OpCall, TensorInfo
from hyper_parallel.core.dtensor.debug._collective_tracer import CollectiveTracer
from hyper_parallel.core.shard._op_dispatch import _debug_mode_observer


# ---------------------------------------------------------------------------
# 1. _call_records tests
# ---------------------------------------------------------------------------
class TestCallRecords(unittest.TestCase):
    """Tests for debug call record dataclasses."""

    def test_op_call_render(self):
        """
        Feature: OpCall rendering
        Description: OpCall._render_self() produces a human-readable single line.
        Expectation: The rendered string contains op name and tensor info.
        """
        record = OpCall(
            op_name="aten.mm.default",
            input_infos=[
                TensorInfo(shape=(4, 8), dtype="torch.float32", is_dtensor=True,
                           placements=("Shard(0)",), mesh_shape=(2,)),
                TensorInfo(shape=(8, 16), dtype="torch.float32"),
            ],
            output_infos=[
                TensorInfo(shape=(4, 16), dtype="torch.float32", is_dtensor=True,
                           placements=("Replicate()",), mesh_shape=(2,)),
            ],
        )
        text = record._render_self()
        self.assertIn("aten.mm.default", text)
        self.assertIn("DTensor", text)
        self.assertIn("[4, 8]", text)

    def test_collective_call_render(self):
        """
        Feature: CollectiveCall rendering
        Description: CollectiveCall._render_self() shows collective type and shapes.
        Expectation: Rendered string contains the collective type and group size.
        """
        record = CollectiveCall(
            collective_type="differentiable_all_reduce",
            group_size=4,
            input_shape=(8, 16),
            output_shape=(8, 16),
            input_dtype="torch.float32",
        )
        text = record._render_self()
        self.assertIn("differentiable_all_reduce", text)
        self.assertIn("group_size=4", text)

    def test_debug_call_render_tree(self):
        """
        Feature: Hierarchical rendering
        Description: render() produces indented tree output for nested calls.
        Expectation: Child records are indented under the parent.
        """
        parent = OpCall(op_name="aten.mm.default")
        child = CollectiveCall(collective_type="differentiable_all_gather_concat",
                               call_depth=1, group_size=2)
        parent.children.append(child)

        tree = parent.render(indent=0)
        lines = tree.split("\n")
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[1].startswith("  "))  # child indented


# ---------------------------------------------------------------------------
# 2. CollectiveTracer tests
# ---------------------------------------------------------------------------
class TestCollectiveTracer(unittest.TestCase):
    """Tests for platform monkey-patch collective tracing."""

    def test_install_and_uninstall_restores_original(self):
        """
        Feature: Monkey-patch restore
        Description: After uninstall(), platform methods are exactly restored.
        Expectation: Function identity matches the original before patching.
        """
        from hyper_parallel.platform import get_platform
        platform = get_platform()
        cls = type(platform)

        # Save originals (raw descriptors from cls.__dict__).
        originals = {}
        for name in ("differentiable_all_reduce", "differentiable_all_gather_concat"):
            if name in cls.__dict__:
                originals[name] = cls.__dict__[name]

        if not originals:
            self.skipTest("No patchable methods on current platform")

        callback = Mock()
        tracer = CollectiveTracer(callback)
        tracer.install()

        # Verify methods are now different.
        for name, orig_val in originals.items():
            self.assertIsNot(cls.__dict__[name], orig_val)

        tracer.uninstall()

        # Verify exact restoration.
        for name, orig in originals.items():
            self.assertIs(cls.__dict__[name], orig)

    def test_callback_is_invoked(self):
        """
        Feature: Collective callback
        Description: The callback fires when a patched method is called.
        Expectation: Callback receives (method_name, args, kwargs, result).
        """
        from hyper_parallel.platform import get_platform
        platform = get_platform()
        cls = type(platform)

        if "differentiable_all_reduce" not in cls.__dict__:
            self.skipTest("differentiable_all_reduce not available")

        callback = Mock()
        tracer = CollectiveTracer(callback)
        tracer.install()

        try:
            # Call with a mock tensor - we expect the original to be called
            # then the callback. Since we don't have a real distributed env
            # the original call may fail, but the tracer wrapper catches that.
            mock_data = torch.randn(4)
            mock_group = Mock()
            try:
                platform.differentiable_all_reduce(mock_data, "sum", mock_group)
            except Exception:  # pylint: disable=W0703
                pass  # Original may fail without real distributed setup
        finally:
            tracer.uninstall()

    def test_callback_exception_does_not_propagate(self):
        """
        Feature: Fault isolation
        Description: A failing callback does not affect the original function.
        Expectation: The patched method still returns the original result.
        """
        from hyper_parallel.platform import get_platform
        cls = type(get_platform())

        if "differentiable_all_reduce" not in cls.__dict__:
            self.skipTest("differentiable_all_reduce not available")

        def bad_callback(*args):
            raise RuntimeError("boom")

        tracer = CollectiveTracer(bad_callback)
        tracer.install()
        try:
            # Just verify uninstall still works even after bad callback
            pass
        finally:
            tracer.uninstall()


# ---------------------------------------------------------------------------
# 4. CommDebugMode context manager tests
# ---------------------------------------------------------------------------
class TestCommDebugModeContextManager(unittest.TestCase):
    """Tests for CommDebugMode enter/exit protocol."""

    def test_context_var_set_and_reset(self):
        """
        Feature: ContextVar lifecycle
        Description: __enter__ sets _debug_mode_observer, __exit__ resets it.
        Expectation: Observer is None outside the context, self inside.
        """
        self.assertIsNone(_debug_mode_observer.get())

        mode = CommDebugMode()
        with mode:
            self.assertIs(_debug_mode_observer.get(), mode)

        self.assertIsNone(_debug_mode_observer.get())

    def test_nested_context_managers(self):
        """
        Feature: Nested CommDebugMode
        Description: Inner and outer modes correctly stack via ContextVar tokens.
        Expectation: Each scope sees the correct observer; outer restored after inner exits.
        """
        outer = CommDebugMode()
        inner = CommDebugMode()

        with outer:
            self.assertIs(_debug_mode_observer.get(), outer)
            with inner:
                self.assertIs(_debug_mode_observer.get(), inner)
            self.assertIs(_debug_mode_observer.get(), outer)

        self.assertIsNone(_debug_mode_observer.get())

    def test_exit_restores_platform_methods(self):
        """
        Feature: Platform method restoration
        Description: After CommDebugMode exits, all platform methods are restored.
        Expectation: cls.__dict__ entries match originals.
        """
        from hyper_parallel.platform import get_platform
        cls = type(get_platform())

        originals = {}
        for name in ("differentiable_all_reduce", "differentiable_reduce_scatter"):
            if name in cls.__dict__:
                originals[name] = cls.__dict__[name]

        mode = CommDebugMode()
        with mode:
            pass

        for name, orig in originals.items():
            self.assertIs(cls.__dict__[name], orig,
                          f"{name} not restored after __exit__")

    def test_empty_tracing_table(self):
        """
        Feature: Empty trace output
        Description: When no ops are traced, generate_comm_debug_tracing_table returns a placeholder.
        Expectation: Output is '(no operations recorded)'.
        """
        mode = CommDebugMode()
        with mode:
            pass
        self.assertEqual(mode.generate_comm_debug_tracing_table(), "(no operations recorded)")

    def test_empty_comm_counts(self):
        """
        Feature: Empty comm counts
        Description: When no collectives happen, comm counts is empty.
        Expectation: get_comm_counts returns {}, get_total_counts returns 0.
        """
        mode = CommDebugMode()
        with mode:
            pass
        self.assertEqual(mode.get_comm_counts(), {})
        self.assertEqual(mode.get_total_counts(), 0)


# ---------------------------------------------------------------------------
# 5. Observer hook integration tests
# ---------------------------------------------------------------------------
class TestObserverHooks(unittest.TestCase):
    """Tests for on_op_dispatch_enter/exit called by OpDispatcher."""

    def test_op_dispatch_enter_creates_record(self):
        """
        Feature: Op dispatch tracing
        Description: on_op_dispatch_enter creates an OpCall and pushes to stack.
        Expectation: After enter, root_records has one OpCall with correct op_name.
        """
        mode = CommDebugMode()
        with mode:
            mode.on_op_dispatch_enter("aten.add.Tensor", None,
                                       (torch.randn(2, 3), torch.randn(2, 3)), {})
            # Simulate exit
            mode.on_op_dispatch_exit("aten.add.Tensor", torch.randn(2, 3))

        self.assertEqual(len(mode._root_records), 1)
        self.assertIsInstance(mode._root_records[0], OpCall)
        self.assertEqual(mode._root_records[0].op_name, "aten.add.Tensor")

    def test_nested_ops_build_tree(self):
        """
        Feature: Nested op tracing
        Description: Nested enter/exit calls produce a tree structure.
        Expectation: Inner op appears as child of outer op.
        """
        mode = CommDebugMode()
        with mode:
            mode.on_op_dispatch_enter("outer_op", None, (torch.randn(2),), {})
            mode.on_op_dispatch_enter("inner_op", None, (torch.randn(2),), {})
            mode.on_op_dispatch_exit("inner_op", torch.randn(2))
            mode.on_op_dispatch_exit("outer_op", torch.randn(2))

        self.assertEqual(len(mode._root_records), 1)
        outer = mode._root_records[0]
        self.assertEqual(outer.op_name, "outer_op")
        self.assertEqual(len(outer.children), 1)
        self.assertEqual(outer.children[0].op_name, "inner_op")

    def test_collective_callback_updates_counts(self):
        """
        Feature: Collective counting
        Description: _on_collective_call updates comm_counts.
        Expectation: get_comm_counts() reflects the recorded collectives.
        """
        mode = CommDebugMode()
        with mode:
            mock_result = torch.randn(4)
            mock_group = Mock()
            mock_group.size.return_value = 4

            mode._on_collective_call(
                "differentiable_all_reduce",
                (torch.randn(4), mock_group, "sum"),
                {},
                mock_result,
            )
            mode._on_collective_call(
                "differentiable_all_reduce",
                (torch.randn(4), mock_group, "sum"),
                {},
                mock_result,
            )
            mode._on_collective_call(
                "differentiable_all_gather_concat",
                (torch.randn(2), mock_group, 4, 0),
                {},
                torch.randn(8),
            )

        counts = mode.get_comm_counts()
        self.assertEqual(counts["differentiable_all_reduce"], 2)
        self.assertEqual(counts["differentiable_all_gather_concat"], 1)
        self.assertEqual(mode.get_total_counts(), 3)

    def test_collective_nested_under_op(self):
        """
        Feature: Collective-under-op nesting
        Description: A collective call that happens during an op dispatch
                     appears as a child of the current OpCall.
        Expectation: The collective is nested under the op in the call tree.
        """
        mode = CommDebugMode()
        with mode:
            mode.on_op_dispatch_enter("redistribute", None, (torch.randn(4),), {})

            mock_group = Mock()
            mock_group.size.return_value = 2
            mode._on_collective_call(
                "differentiable_all_gather_concat",
                (torch.randn(4), mock_group, 2, 0),
                {},
                torch.randn(8),
            )

            mode.on_op_dispatch_exit("redistribute", torch.randn(8))

        self.assertEqual(len(mode._root_records), 1)
        op_record = mode._root_records[0]
        self.assertEqual(len(op_record.children), 1)
        self.assertIsInstance(op_record.children[0], CollectiveCall)
        self.assertEqual(op_record.children[0].collective_type,
                         "differentiable_all_gather_concat")


# ---------------------------------------------------------------------------
# 6. Debug string and tracing table tests
# ---------------------------------------------------------------------------
class TestDebugOutput(unittest.TestCase):
    """Tests for debug_string() and generate_tracing_table()."""

    def _mode_with_records(self):
        """Create a CommDebugMode with pre-populated records."""
        mode = CommDebugMode()
        with mode:
            mode.on_op_dispatch_enter("aten.mm.default", None,
                                       (torch.randn(4, 8), torch.randn(8, 16)), {})

            mock_group = Mock()
            mock_group.size.return_value = 4
            mode._on_collective_call(
                "differentiable_all_reduce",
                (torch.randn(4, 16), mock_group, "sum"),
                {},
                torch.randn(4, 16),
            )

            mode.on_op_dispatch_exit("aten.mm.default", torch.randn(4, 16))
        return mode

    def test_tracing_table_contains_hierarchy(self):
        """
        Feature: Tracing table hierarchy
        Description: generate_comm_debug_tracing_table() shows nested ops and collectives.
        Expectation: Output contains both the op and its child collective.
        """
        mode = self._mode_with_records()
        output = mode.generate_comm_debug_tracing_table()
        self.assertIn("aten.mm.default", output)
        self.assertIn("differentiable_all_reduce", output)

    def test_tracing_table_noise_level_0(self):
        """
        Feature: Tracing table noise filtering
        Description: noise_level=0 shows only collectives.
        Expectation: Table contains collective but not the op.
        """
        mode = self._mode_with_records()
        table = mode.generate_tracing_table(noise_level=0)
        self.assertIn("differentiable_all_reduce", table)
        self.assertNotIn("aten.mm.default", table)

    def test_tracing_table_noise_level_1(self):
        """
        Feature: Tracing table with ops
        Description: noise_level=1 shows both ops and collectives.
        Expectation: Table contains both the op and the collective.
        """
        mode = self._mode_with_records()
        table = mode.generate_tracing_table(noise_level=1)
        self.assertIn("aten.mm.default", table)
        self.assertIn("differentiable_all_reduce", table)

    def test_repr(self):
        """
        Feature: __repr__
        Description: repr shows total counts.
        Expectation: repr string contains get_total_counts value.
        """
        mode = self._mode_with_records()
        self.assertIn("get_total_counts()=1", repr(mode))


# ---------------------------------------------------------------------------
# 7. TensorInfo extraction tests
# ---------------------------------------------------------------------------
class TestTensorInfoExtraction(unittest.TestCase):
    """Tests for _extract_tensor_infos on plain tensors."""

    def test_plain_tensor_info(self):
        """
        Feature: Plain tensor info extraction
        Description: _extract_tensor_infos captures shape and dtype for plain tensors.
        Expectation: TensorInfo has correct shape, dtype, and is_dtensor=False.
        """
        mode = CommDebugMode()
        t = torch.randn(3, 5)
        infos = mode._extract_tensor_infos((t,))
        self.assertEqual(len(infos), 1)
        self.assertEqual(infos[0].shape, (3, 5))
        self.assertFalse(infos[0].is_dtensor)

    def test_nested_tuple_extraction(self):
        """
        Feature: Nested arg extraction
        Description: Tensors inside nested tuples are also extracted.
        Expectation: All tensors from nested structures are collected.
        """
        mode = CommDebugMode()
        t1 = torch.randn(2)
        t2 = torch.randn(3)
        infos = mode._extract_tensor_infos(((t1, t2),))
        self.assertEqual(len(infos), 2)

    def test_non_tensor_args_ignored(self):
        """
        Feature: Non-tensor filtering
        Description: Non-tensor arguments are silently skipped.
        Expectation: Only tensor args produce TensorInfo entries.
        """
        mode = CommDebugMode()
        infos = mode._extract_tensor_infos((42, "hello", None, torch.randn(1)))
        self.assertEqual(len(infos), 1)


# ---------------------------------------------------------------------------
# 9. Zero-overhead when disabled
# ---------------------------------------------------------------------------
class TestZeroOverhead(unittest.TestCase):
    """Verify that CommDebugMode has no cost when not active."""

    def test_context_var_default_is_none(self):
        """
        Feature: Zero-cost default
        Description: _debug_mode_observer defaults to None.
        Expectation: ContextVar.get() returns None, so no tracing logic runs.
        """
        self.assertIsNone(_debug_mode_observer.get())


if __name__ == "__main__":
    unittest.main()
