# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""torchrun worker tests that smoke-run every Torch llama3 example end-to-end.

Each function imports the corresponding script under
``examples/torch/llama3/`` and calls its ``main()`` once. The launcher in
``test_llama3_examples.py`` decides the world size for each case so that the
example's hard-coded mesh / sequence-parallel / context-parallel constraints
are satisfied.

Pass criterion: ``main()`` completes without raising. The examples themselves
already check for non-finite losses, so a successful return is sufficient
smoke coverage.
"""
# pylint: disable=C0413,W0611
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest
import torch.distributed as dist
import torch_npu  # noqa: F401  # ensure NPU backend is registered before HCCL init.

os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

# ``examples/torch/llama3`` is not a package, mirror the ``sys.path`` hack the
# example scripts perform so ``import tensor_parallel_example`` resolves their
# colocated ``model.py`` / ``parallelize.py``. Try a few candidate roots so the
# test works both from the local repo (where ``examples/`` sits at
# ``parents[3]``) and from the gate test package (where the repo may be staged
# at a different parent depth, or missing entirely).
def _resolve_examples_dir() -> Path | None:
    """Locate ``examples/torch/llama3`` relative to this test file.

    The launcher tree (``tests/torch/llama3_examples/``) is shipped both inside
    the main HyperParallel repo (where ``parents[3]`` is the repo root) and
    inside an external test package (where the relative depth differs). Probe a
    handful of candidate roots and an env-var override so the test is portable.
    """
    here = Path(__file__).resolve()
    override = os.environ.get("HYPER_PARALLEL_EXAMPLES_DIR")
    candidates = []
    if override:
        candidates.append(Path(override) / "torch" / "llama3")
    # ``parents[3]`` matches the in-repo layout; widen the search a couple of
    # levels in case the gate stages tests under an extra wrapper directory.
    for parent in here.parents[3:6]:
        candidates.append(parent / "examples" / "torch" / "llama3")
    for cand in candidates:
        if cand.is_dir() and (cand / "tensor_parallel_example.py").is_file():
            return cand
    return None


_EXAMPLES_DIR = _resolve_examples_dir()
if _EXAMPLES_DIR is not None and str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))


def _require_world_size(expected: int) -> None:
    """Skip the current test if the launcher world size does not match ``expected``."""
    if not dist.is_initialized():
        # ``main()`` will initialize the process group; defer the check to once
        # the example has set up distributed state.
        return
    actual = dist.get_world_size()
    if actual != expected:
        pytest.skip(f"requires world_size={expected}, got {actual}")


def _run_example_main(module_name: str, expected_world_size: int) -> None:
    """Import ``module_name`` from the examples dir and call its ``main()`` once.

    Args:
        module_name: Python module name under ``examples/torch/llama3``
            (e.g. ``"tensor_parallel_example"``).
        expected_world_size: World size the example was designed for; the test
            skips when the launcher passes a different one.
    """
    if _EXAMPLES_DIR is None:
        pytest.skip(
            "examples/torch/llama3 not found near this test (set "
            "HYPER_PARALLEL_EXAMPLES_DIR to the local examples root)"
        )
    _require_world_size(expected_world_size)
    module = importlib.import_module(module_name)
    if not hasattr(module, "main"):
        raise AssertionError(f"Example module {module_name!r} is missing main().")
    module.main()
    # ``main()`` initializes its own process group; once it returns, double-check
    # the world size matches the launcher's intent.
    if dist.is_initialized():
        actual = dist.get_world_size()
        assert actual == expected_world_size, (
            f"{module_name}: expected world_size={expected_world_size}, got {actual}"
        )


def test_tensor_parallel_example_npu():
    """Feature: examples/torch/llama3/tensor_parallel_example.py — pure TP smoke run.

    Description:
        1. Launcher provides world_size=2 (TP=2).
        2. Calls ``main()`` which builds a 1-D TP mesh and runs two training steps.
    Expectation: ``main()`` returns without raising.
    """
    _run_example_main("tensor_parallel_example", expected_world_size=2)


def test_fsdp_tp_example_npu():
    """Feature: examples/torch/llama3/fsdp_tp_example.py — TP + fully_shard smoke run.

    Description:
        1. Launcher provides world_size=4 with default TP=2 → DP=2.
        2. ``main()`` builds the 2-D ``(dp, tp)`` mesh and trains for two steps.
    Expectation: ``main()`` returns without raising.
    """
    # Fix the TP width so this test does not depend on the ambient env.
    os.environ["LLAMA3_TP_SIZE"] = "2"
    _run_example_main("fsdp_tp_example", expected_world_size=4)


def test_tp_cp_example_npu():
    """Feature: examples/torch/llama3/tp_cp_example.py — TP + ContextParallel smoke run.

    Description:
        1. Launcher provides world_size=4 with TP=2, CP=2.
        2. ``main()`` constructs the 2-D ``(tp, cp)`` mesh, applies
           ``parallelize_llama3`` and ``ContextParallel`` on every layer's
           ``sdpa_core``, and runs one training step.
    Expectation: ``main()`` returns without raising.
    """
    os.environ["LLAMA3_TP_SIZE"] = "2"
    os.environ["LLAMA3_CP_SIZE"] = "2"
    _run_example_main("tp_cp_example", expected_world_size=4)


def test_dp_tp_cp_sp_fsdp_example_npu():
    """Feature: examples/torch/llama3/dp_tp_cp_sp_fsdp_example.py — full 4-D combo smoke run.

    Description:
        1. Launcher provides world_size=8 with default ``(dp,fsdp,cp,tp)=(1,2,2,2)``.
        2. ``main()`` builds the 4-D mesh, applies TP+SP, ContextParallel, and
           ``fully_shard`` over the ``(dp, fsdp)`` HSDP slice, then runs two
           training steps.
    Expectation: ``main()`` returns without raising.
    """
    os.environ["LLAMA3_DP_SIZE"] = "1"
    os.environ["LLAMA3_FSDP_SIZE"] = "2"
    os.environ["LLAMA3_CP_SIZE"] = "2"
    os.environ["LLAMA3_TP_SIZE"] = "2"
    _run_example_main("dp_tp_cp_sp_fsdp_example", expected_world_size=8)
