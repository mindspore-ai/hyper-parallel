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
"""Single-card / CPU unit tests for optimizer state dict (issue240).

Covers:
  U1: FQN key assertion — state keys are FQNs, not integer IDs
  U2: param_groups / initial_lr / empty optimizer state preservation
  U3: AdamW get -> set roundtrip (single-card CPU)
  U4: SGD get -> set roundtrip (single-card CPU)
  U5: strict=False — extra/missing FQNs do not break target param_groups
  U6: flatten roundtrip with FQNs containing dots
  U7: empty param_group in flatten format raises UnsupportedConfigurationError
  U8: ChainedOptimizer rejection raises ValueError
"""
import os

os.environ["HYPER_PARALLEL_PLATFORM"] = "torch"

import pytest  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from hyper_parallel import get_optim_state_dict, set_optim_state_dict  # noqa: E402
from hyper_parallel.platform.torch.fully_shard.optim_state_dict_utils import (  # noqa: E402
    UnsupportedConfigurationError,
    _check_chained_optimizer,
    _flatten_optim_state_dict,
    _unflatten_optim_state_dict,
)


class _DottedFQNNet(nn.Module):
    """Model whose parameter FQNs contain dots (e.g. 'layer.0.weight')."""

    def __init__(self, hidden: int = 8):
        super().__init__()
        self.layer = nn.ModuleDict({"0": nn.Linear(hidden, hidden)})

    def forward(self, x):
        return self.layer["0"](x).sum()


class _SimpleNet(nn.Module):
    """Simple two-layer network for basic roundtrip tests."""

    def __init__(self, hidden: int = 8):
        super().__init__()
        self.linear1 = nn.Linear(hidden, hidden)
        self.linear2 = nn.Linear(hidden, hidden)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        return self.linear2(x).sum()


def _train_step(model, optimizer, x):
    optimizer.zero_grad()
    loss = model(x)
    loss.backward()
    optimizer.step()


# =====================================================================
# U1: FQN key assertion
# =====================================================================
def test_u1_fqn_keys_no_integer_ids():
    """get_optim_state_dict uses FQN keys, not integer optimizer IDs."""
    model = _SimpleNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.randn(2, 8)
    _train_step(model, optimizer, x)

    sd = get_optim_state_dict(model, optimizer)

    for key in sd["state"].keys():
        assert isinstance(key, str), f"state key should be str (FQN), got {type(key)}: {key}"
        assert not isinstance(key, int), f"state key should not be int (optimizer ID): {key}"

    for group in sd["param_groups"]:
        for param_ref in group["params"]:
            assert isinstance(param_ref, str), f"param_group params should be str FQNs, got {type(param_ref)}"

    model_fqns = {name for name, _ in model.named_parameters()}
    state_fqns = set(sd["state"].keys())
    assert state_fqns == model_fqns, f"state FQNs mismatch: {state_fqns} vs {model_fqns}"


# =====================================================================
# U2: param_groups / initial_lr / empty optimizer state
# =====================================================================
def test_u2_param_groups_initial_lr_empty_state():
    """Verify param_groups preservation, initial_lr, and empty optimizer state."""
    model = _SimpleNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    x = torch.randn(2, 8)
    _train_step(model, optimizer, x)
    scheduler.step()

    sd = get_optim_state_dict(model, optimizer)

    for group in sd["param_groups"]:
        assert "initial_lr" in group, f"param_group missing initial_lr: {group}"
        assert group["initial_lr"] == 0.01, f"initial_lr should be 0.01, got {group['initial_lr']}"
        assert "lr" in group, "param_group missing lr"
        assert "betas" in group, "param_group missing betas"
        assert "weight_decay" in group, "param_group missing weight_decay"

    empty_model = _SimpleNet()
    empty_optimizer = torch.optim.AdamW(empty_model.parameters(), lr=0.01)
    empty_sd = get_optim_state_dict(empty_model, empty_optimizer)
    assert len(empty_sd["state"]) == 0, "empty optimizer should yield empty state"
    assert len(empty_sd["param_groups"]) > 0, "param_groups should be present even for empty state"


# =====================================================================
# U3: AdamW roundtrip
# =====================================================================
def test_u3_adamw_roundtrip():
    """get -> set roundtrip with AdamW preserves optimizer state."""
    model = _SimpleNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.randn(2, 8)
    _train_step(model, optimizer, x)
    _train_step(model, optimizer, x)

    sd = get_optim_state_dict(model, optimizer)

    model2 = _SimpleNet()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.01)
    _train_step(model2, optimizer2, x)

    set_optim_state_dict(model2, optimizer2, sd)
    _train_step(model2, optimizer2, x)

    raw_sd1 = optimizer.state_dict()
    raw_sd2 = optimizer2.state_dict()
    for pid in raw_sd1["state"]:
        for key in ("step", "exp_avg", "exp_avg_sq"):
            if key in raw_sd1["state"][pid]:
                assert key in raw_sd2["state"].get(pid, {}), f"Missing {key} in target state[{pid}]"


# =====================================================================
# U4: SGD roundtrip
# =====================================================================
def test_u4_sgd_roundtrip():
    """get -> set roundtrip with SGD (momentum) preserves optimizer state."""
    model = _SimpleNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    x = torch.randn(2, 8)
    _train_step(model, optimizer, x)
    _train_step(model, optimizer, x)

    sd = get_optim_state_dict(model, optimizer)

    assert len(sd["state"]) > 0, "SGD state should not be empty after training"
    for fqn, state in sd["state"].items():
        assert "momentum_buffer" in state, f"SGD state.{fqn} missing momentum_buffer"

    model2 = _SimpleNet()
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01, momentum=0.9)
    _train_step(model2, optimizer2, x)

    set_optim_state_dict(model2, optimizer2, sd)
    _train_step(model2, optimizer2, x)


# =====================================================================
# U5: strict=False
# =====================================================================
def test_u5_strict_false():
    """strict=False allows extra/missing FQNs without breaking param_groups."""
    model = _SimpleNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.randn(2, 8)
    _train_step(model, optimizer, x)

    sd = get_optim_state_dict(model, optimizer)

    sd["state"]["nonexistent.param.xyz"] = {"step": torch.tensor(1.0)}

    model2 = _SimpleNet()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.01)
    _train_step(model2, optimizer2, x)

    original_pg_count = len(optimizer2.param_groups)
    original_params_per_group = [len(g["params"]) for g in optimizer2.param_groups]

    class _Opts:
        strict = False

    set_optim_state_dict(model2, optimizer2, sd, options=_Opts())

    assert len(optimizer2.param_groups) == original_pg_count, (
        "strict=False should not change param_groups count"
    )
    for i, g in enumerate(optimizer2.param_groups):
        assert len(g["params"]) == original_params_per_group[i], (
            f"strict=False should not change param_groups[{i}] params count"
        )


# =====================================================================
# U6: flatten roundtrip with dotted FQNs
# =====================================================================
def test_u6_flatten_roundtrip_dotted_fqn():
    """Flatten -> unflatten roundtrip with FQNs containing dots."""
    model = _DottedFQNNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.randn(2, 8)
    _train_step(model, optimizer, x)

    sd = get_optim_state_dict(model, optimizer)

    flat = _flatten_optim_state_dict(sd)

    for key in flat.keys():
        assert key.startswith("state.") or key.startswith("param_group."), (
            f"flat key '{key}' should start with 'state.' or 'param_group.'"
        )

    has_dotted_fqn_key = any("layer.0" in k for k in flat.keys())
    assert has_dotted_fqn_key, "Expected flat keys containing 'layer.0' (dotted FQN)"

    unflat = _unflatten_optim_state_dict(flat, model)

    original_fqns = set(sd["state"].keys())
    roundtrip_fqns = set(unflat["state"].keys())
    assert original_fqns == roundtrip_fqns, (
        f"FQN mismatch after flatten/unflatten: {original_fqns} vs {roundtrip_fqns}"
    )

    model2 = _DottedFQNNet()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.01)
    _train_step(model2, optimizer2, x)

    class _Opts:
        flatten_optimizer_state_dict = True

    set_optim_state_dict(model2, optimizer2, flat, options=_Opts())
    _train_step(model2, optimizer2, x)


# =====================================================================
# U7: empty param_group in flatten raises UnsupportedConfigurationError
# =====================================================================
def test_u7_empty_param_group_flatten_error():
    """Flatten with optimizer that has an empty param_group raises error.

    When an optimizer has a param_group with no parameters (empty 'params'
    list), there are no FQNs to prefix the param_group fields in flatten
    format. This makes it impossible to reconstruct the empty group during
    unflatten, so UnsupportedConfigurationError is raised.
    """
    model = _SimpleNet()

    nested_sd = {
        "state": {
            "linear1.weight": {
                "step": torch.tensor(1.0),
                "exp_avg": torch.zeros(8, 8),
                "exp_avg_sq": torch.zeros(8, 8),
            },
            "linear1.bias": {
                "step": torch.tensor(1.0),
                "exp_avg": torch.zeros(8),
                "exp_avg_sq": torch.zeros(8),
            },
            "linear2.weight": {
                "step": torch.tensor(1.0),
                "exp_avg": torch.zeros(8, 8),
                "exp_avg_sq": torch.zeros(8, 8),
            },
            "linear2.bias": {
                "step": torch.tensor(1.0),
                "exp_avg": torch.zeros(8),
                "exp_avg_sq": torch.zeros(8),
            },
        },
        "param_groups": [
            {"params": [], "lr": 0.001, "initial_lr": 0.001},
            {"params": ["linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias"], "lr": 0.01, "initial_lr": 0.01},
        ],
    }

    with pytest.raises(UnsupportedConfigurationError):
        _flatten_optim_state_dict(nested_sd)


# =====================================================================
# U8: ChainedOptimizer rejection
# =====================================================================
def test_u8_chained_optimizer_rejection():
    """get/set_optim_state_dict raises ValueError for ChainedOptimizer."""

    class ChainedOptimizer:
        """Fake ChainedOptimizer for testing rejection."""
        pass

    model = _SimpleNet()
    fake_optimizer = ChainedOptimizer()

    with pytest.raises(ValueError, match="ChainedOptimizer"):
        _check_chained_optimizer(fake_optimizer)

    with pytest.raises(ValueError, match="ChainedOptimizer"):
        get_optim_state_dict(model, fake_optimizer)

    with pytest.raises(ValueError, match="ChainedOptimizer"):
        set_optim_state_dict(model, fake_optimizer, {"state": {}, "param_groups": []})
