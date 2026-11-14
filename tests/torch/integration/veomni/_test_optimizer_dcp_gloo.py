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
"""Exercise missing NS metadata through real VeOmni Stateful DCP partial loads."""

import json
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import CheckpointException, DefaultLoadPlanner
from torch.distributed.device_mesh import init_device_mesh
from veomni.checkpoint import dcp_checkpointer as checkpoint
from veomni.optim import optimizer as native_optimizer

from hyper_parallel.integration.veomni.adapter import VeOmniAdapter
from hyper_parallel.integration.veomni.optimizer import (
    _build_hyper_muon_adapter,
    _canonical_muon_option,
    _is_missing_checkpoint_option,
)

OUT = Path(os.environ["HYPER_MUON_DCP_TEST_OUTPUT"])
NativeOptimizerState = checkpoint.OptimizerState
STATE = SimpleNamespace(
    dp_mode="fsdp2", extra_parallel_names=[], extra_parallel_fsdp_device_mesh={}
)


def build(backend):
    """Build a deterministic model and the selected Hyper NS adapter."""
    torch.manual_seed(42)
    model = torch.nn.Linear(8, 8, bias=False)
    optimizer = _build_hyper_muon_adapter(
        model,
        {
            "no_decay_modules": [],
            "no_decay_params": [],
            "lr": 0.003,
            "muon_kwargs": {"ns_implementation": backend},
        },
        native_optimizer,
    )
    return model, optimizer


class LegacyState(NativeOptimizerState):
    def state_dict(self):
        return {
            key: value
            for key, value in super().state_dict().items()
            if not key.endswith((".ns_implementation", ".gram_ns_reset_iterations"))
        }


def check_restored_groups(optimizer, reference):
    """Check effective settings, including fields used only by captured callbacks."""
    keys = (
        "ns_implementation",
        "ns_coefficients",
        "ns_epsilon",
        "ns_steps",
        "ns_variant",
        "gram_ns_reset_iterations",
    )
    for actual, expected in zip(optimizer.param_groups, reference.param_groups):
        for key in keys:
            if _is_missing_checkpoint_option(actual[key]):
                raise AssertionError(
                    f"Unloaded marker reached the live parameter group: {key}"
                )
            if _canonical_muon_option(actual[key]) != _canonical_muon_option(
                expected[key]
            ):
                raise AssertionError(f"Restored parameter group setting differs: {key}")
    for key, value in optimizer.state_dict().items():
        if key.startswith("param_groups.") and _is_missing_checkpoint_option(value):
            raise AssertionError(
                f"Unloaded marker reached saved parameter groups: {key}"
            )


def _check_resave_reload(target, optimizer, source, case, tensors, extra_parallel):
    """Verify that a confirmed legacy restore produces a self-contained save."""
    name, _, backend, _, _ = case
    weights, gradient, expected = tensors
    check_restored_groups(optimizer, source)
    resaved = OUT / f"resaved_{name}_ep{extra_parallel}"
    dcp.save(
        {"optimizer": checkpoint.OptimizerState(target, optimizer, STATE)},
        checkpoint_id=resaved,
    )
    again, again_optimizer = build(backend)
    again.weight.copy_(weights)
    if extra_parallel:
        setattr(again, "_fqn2spec_info", {})
    # A fresh optimizer must restore without another historical confirmation.
    dcp.load(
        {
            "optimizer": checkpoint.OptimizerState(
                again, again_optimizer, STATE, load=True
            )
        },
        checkpoint_id=resaved,
        planner=DefaultLoadPlanner(allow_partial_load=True),
    )
    check_restored_groups(again_optimizer, source)
    target.weight.grad = gradient.clone()
    optimizer.step()
    again.weight.grad = gradient.clone()
    again_optimizer.step()
    torch.testing.assert_close(target.weight, expected, rtol=0, atol=0)
    torch.testing.assert_close(again.weight, expected, rtol=0, atol=0)


def _run_restore_case(case, source, tensors, extra_parallel):
    """Exercise acceptance or rejection without changing live groups on failure."""
    name, saved, backend, confirmation, rejected = case
    target, optimizer = build(backend)
    target.weight.copy_(tensors[0])
    if extra_parallel:
        setattr(target, "_fqn2spec_info", {})
    if confirmation:
        optimizer.confirm_legacy_muon_checkpoint_options(
            ns_implementation=confirmation, gram_ns_reset_iterations=[2]
        )
    groups_before = deepcopy(
        [
            {key: value for key, value in group.items() if key != "params"}
            for group in optimizer.param_groups
        ]
    )
    try:
        dcp.load(
            {
                "optimizer": checkpoint.OptimizerState(
                    target, optimizer, STATE, load=True
                )
            },
            checkpoint_id=OUT / saved,
            planner=DefaultLoadPlanner(allow_partial_load=True),
        )
    except (CheckpointException, ValueError) as error:
        if rejected is None or rejected not in str(error):
            raise
        after = [
            {key: value for key, value in group.items() if key != "params"}
            for group in optimizer.param_groups
        ]
        if groups_before != after:
            raise AssertionError(
                "Rejected restore changed live optimizer groups"
            ) from error
    else:
        if rejected is not None:
            raise AssertionError(f"{name}: incompatible checkpoint accepted")
        _check_resave_reload(target, optimizer, source, case, tensors, extra_parallel)
    return {
        "case": name,
        "extra_parallel_branch": extra_parallel,
        "expected_rejection": rejected,
        "status": "PASS",
    }


@torch.no_grad()
def main():
    dist.init_process_group("gloo")
    torch.set_num_threads(1)
    mesh = init_device_mesh(
        "cpu", (1, dist.get_world_size()), mesh_dim_names=("ep_fsdp", "ep")
    )
    STATE.extra_parallel_names = ["ep"]
    STATE.extra_parallel_fsdp_device_mesh = {"ep": mesh}
    records = []
    try:
        model, source = build("gram")
        model.weight.grad = torch.randn_like(model.weight)
        source.step()
        weights = model.weight.clone()
        gradient = torch.randn_like(model.weight)
        dcp.save(
            {"optimizer": NativeOptimizerState(model, source, STATE)},
            checkpoint_id=OUT / "new",
        )
        dcp.save(
            {"optimizer": LegacyState(model, source, STATE)},
            checkpoint_id=OUT / "legacy",
        )
        keys = dcp.FileSystemReader(OUT / "legacy").read_metadata().state_dict_metadata
        if any(
            key.endswith(("ns_implementation", "gram_ns_reset_iterations"))
            for key in keys
        ):
            raise AssertionError("Legacy fixture unexpectedly contains NS provenance")
        model.weight.grad = gradient.clone()
        source.step()
        expected = model.weight.clone()

        # Negative control: the original Stateful class retains current defaults.
        target, optimizer = build("std")
        dcp.load(
            {"optimizer": NativeOptimizerState(target, optimizer, STATE, load=True)},
            checkpoint_id=OUT / "legacy",
            planner=DefaultLoadPlanner(allow_partial_load=True),
        )
        if optimizer.param_groups[0]["ns_implementation"] != "std":
            raise AssertionError("Negative-control reproduction failed")
        records.append(
            {"case": "unadapted_partial_load_reproduces_bypass", "status": "REPRODUCED"}
        )

        with VeOmniAdapter(fsdp=False, muon=True):
            if checkpoint.OptimizerState is NativeOptimizerState:
                raise AssertionError(
                    "Real VeOmni OptimizerState binding was not installed"
                )
            for extra_parallel in (False, True):
                for case in (
                    ("new_same", "new", "gram", None, None),
                    ("new_backend_mismatch", "new", "std", None, "ns_implementation"),
                    ("legacy_std", "legacy", "std", None, "missing NS configuration"),
                    ("legacy_gram", "legacy", "gram", None, "missing NS configuration"),
                    ("legacy_explicit_gram", "legacy", "gram", "gram", None),
                    (
                        "legacy_false_target",
                        "legacy",
                        "std",
                        "gram",
                        "ns_implementation",
                    ),
                    ("saved_values_override_confirmation", "new", "gram", "std", None),
                ):
                    records.append(
                        _run_restore_case(
                            case, source, (weights, gradient, expected), extra_parallel
                        )
                    )
        result = {
            "resave_reload_cycles": 6,
            "status": "PASS",
            "rank": dist.get_rank(),
            "world": dist.get_world_size(),
            "cases": records,
        }
        (OUT / f"rank{dist.get_rank()}_summary.json").write_text(
            json.dumps(result, indent=2)
        )
        print(
            json.dumps(
                {"rank": dist.get_rank(), "cases": len(records), "status": "PASS"}
            ),
            flush=True,
        )
    finally:
        dist.destroy_process_group()


def test_optimizer_dcp_gloo():
    """Run real Stateful save and partial-load regressions on two ranks."""
    main()
