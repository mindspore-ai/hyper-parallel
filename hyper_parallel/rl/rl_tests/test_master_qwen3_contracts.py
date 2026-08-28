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
"""CPU contracts for Qwen3 RL changes that touch master infrastructure."""

from contextlib import AbstractContextManager
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

import hyper_parallel.auto_models._transformers.checkpoint_loader as checkpoint_loader_module
import hyper_parallel.auto_models._transformers.infrastructure as infrastructure_module
import hyper_parallel.auto_models.components.distributed.infrastructure as distributed_infrastructure_module
import hyper_parallel.auto_models.components.distributed.sharding.apply as sharding_apply_module
import hyper_parallel.core.optimizer.adamw as adamw_module
import hyper_parallel.core.optimizer.optimizer as optimizer_module
import hyper_parallel.platform.torch.clip_grad as clip_grad_module
from hyper_parallel.auto_models.components.distributed.param_role import ParameterClassifier, ParamRole
from hyper_parallel.core.dtensor.placement_types import Shard


class _TiedQwen3Model(nn.Module):
    """Minimal Qwen3-style model with tied embedding and output weights."""

    def __init__(self, *, device: str = "cpu") -> None:
        """Create both public aliases and the Transformers tie mapping."""
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(8, 4, device=device)
        self.lm_head = nn.Linear(4, 8, bias=False, device=device)
        self.lm_head.weight = self.model.embed_tokens.weight
        self.config = SimpleNamespace(tie_word_embeddings=True)
        self.all_tied_weights_keys = {
            "lm_head.weight": "model.embed_tokens.weight",
        }


def test_checkpoint_loader_uses_transformers_prefix_contract(
    monkeypatch: Any,
) -> None:
    """TP checkpoint loading passes the pinned Transformers rename keyword."""
    model = nn.Linear(2, 2, bias=False)
    model.base_model_prefix = "model"
    renaming = checkpoint_loader_module.WeightRenaming(
        source_patterns="source",
        target_patterns="weight",
    )
    seen_prefixes = []
    rename_source_key = checkpoint_loader_module.rename_source_key

    def _rename_source_key(
        source_key: str,
        renamings: Any,
        converters: Any,
        *,
        prefix: Any,
        meta_state_dict: Any,
    ) -> tuple[str, Any]:
        assert meta_state_dict == {"weight": model.weight}
        seen_prefixes.append(prefix)
        return rename_source_key(  # pylint: disable=unexpected-keyword-arg
            source_key,
            renamings,
            converters,
            prefix=prefix,
            meta_state_dict=meta_state_dict,
        )

    checkpoint_index = SimpleNamespace(keys=lambda: ("source",))
    monkeypatch.setattr(
        checkpoint_loader_module,
        "rename_source_key",
        _rename_source_key,
    )

    groups, unexpected, used = checkpoint_loader_module._build_load_groups(  # pylint: disable=protected-access
        model,
        checkpoint_index,
        {"weight": model.weight},
        weights_mapping=[renaming],
    )

    assert len(groups) == 1
    assert not unexpected
    assert seen_prefixes == ["model"]
    assert used == [renaming]


def test_pure_tp_runtime_uses_size_one_fsdp_for_replicated_gradients(
    monkeypatch: Any,
) -> None:
    """TP2 keeps size-one FSDP active to synchronize replicated gradients."""

    class _FakeDeviceMesh:
        mesh_dim_names = ("tp",)

        def __getitem__(self, dim_name: str) -> "_FakeDeviceMesh":
            """Return the requested TP mesh dimension."""
            assert dim_name == "tp"
            return self

        @staticmethod
        def get_local_rank(dim_name: str) -> int:
            """Return the only local rank in the fake mesh."""
            assert dim_name == "tp"
            return 0

    mesh_context = SimpleNamespace(
        device_mesh=_FakeDeviceMesh(),
        fsdp_moe_mesh=None,
        dp_size=1,
        dp_replicate_size=1,
        dp_shard_size=1,
        edp_shard_size=1,
        tp_size=2,
        cp_size=1,
        pp_size=1,
        ep_size=1,
    )
    fsdp_config = SimpleNamespace(dp_shard_size=1, edp_shard_size=1)
    config = SimpleNamespace(
        accelerator=SimpleNamespace(
            tp_size=2,
            cp_size=1,
            pp_size=1,
            ep_size=1,
            sequence_parallel=False,
            loss_parallel=False,
        ),
        fsdp_config=fsdp_config,
        plan_overrides=None,
        training=SimpleNamespace(low_precision=None),
    )
    monkeypatch.setattr(distributed_infrastructure_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed_infrastructure_module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        distributed_infrastructure_module,
        "_build_device_mesh_from_accelerator",
        lambda *args, **kwargs: (mesh_context, ("tp",)),
    )

    setup = distributed_infrastructure_module.create_distributed_setup_from_config(config)

    assert setup.strategy_config is fsdp_config


def test_tied_qwen3_classifier_preserves_both_parameter_boundaries() -> None:
    """Tied storage does not hide the lm_head boundary from the TP planner."""
    roles = ParameterClassifier().classify(_TiedQwen3Model())

    assert roles["model.embed_tokens.weight"] is ParamRole.EMBED
    assert roles["lm_head.weight"] is ParamRole.LM_HEAD


def test_production_tp_unwrap_retains_checkpoint_layout(
    monkeypatch: Any,
) -> None:
    """A plain TP Parameter keeps the layout needed by the checkpoint loader."""

    class _FakeDTensorParameter(nn.Parameter):
        """Parameter subclass exposing the DTensor attributes used by unwrap."""

        @staticmethod
        def __new__(  # pylint: disable=signature-differs
            cls,
            data: torch.Tensor,
            layout: Any,
        ) -> "_FakeDTensorParameter":
            parameter = nn.Parameter._make_subclass(cls, data, True)
            parameter._test_layout = layout  # pylint: disable=protected-access
            parameter.placements = layout.placements
            return parameter

        @property
        def layout(self) -> Any:
            """Return the layout carried by this test DTensor."""
            return self._test_layout  # pylint: disable=protected-access

        def to_local(self) -> torch.Tensor:
            """Return the zero-copy local tensor view."""
            return torch.Tensor.detach(self)

    model = nn.Linear(4, 4, bias=False)
    layout = SimpleNamespace(
        placements=(Shard(0),),
        alias_placements=(Shard(0),),
        mesh=object(),
    )
    original_parameter = _FakeDTensorParameter(model.weight.detach(), layout)
    model.weight = original_parameter
    monkeypatch.setattr(sharding_apply_module, "DTensor", _FakeDTensorParameter)

    records = sharding_apply_module._local_params_context(model)  # pylint: disable=protected-access

    assert records == {"weight": (Shard(0),)}
    assert model.weight._sharding_spec is layout  # pylint: disable=protected-access
    assert model.weight.untyped_storage().data_ptr() == original_parameter.untyped_storage().data_ptr()



def test_meta_materialization_preserves_tp_layout_and_tied_identity() -> None:
    """Materialization keeps both pure-TP metadata and one tied Parameter."""
    model = _TiedQwen3Model(device="meta")
    layout = object()
    model.model.embed_tokens.weight._sharding_spec = layout  # pylint: disable=protected-access
    model.lm_head.weight._sharding_spec = layout  # pylint: disable=protected-access

    result = infrastructure_module._move_model_to_device(  # pylint: disable=protected-access
        model,
        is_meta_device=True,
        device=torch.device("cpu"),
    )

    assert result.model.embed_tokens.weight._sharding_spec is layout  # pylint: disable=protected-access
    assert result.lm_head.weight._sharding_spec is layout  # pylint: disable=protected-access
    assert result.lm_head.weight is result.model.embed_tokens.weight


def test_clip_grad_uses_plain_tp_parameter_layout() -> None:
    """Global-norm grouping sees the TP shard axis after production unwrap."""
    mesh = object()
    parameter = nn.Parameter(torch.ones(4))
    parameter._sharding_spec = SimpleNamespace(  # pylint: disable=protected-access
        mesh=mesh,
        placements=(Shard(0),),
        alias_placements=(("tp", "fsdp_shard"), "None"),
    )

    actual_mesh, shard_dims, partial_info = clip_grad_module._get_param_mesh_info(  # pylint: disable=protected-access
        parameter
    )

    assert actual_mesh is mesh
    assert shard_dims == (0,)
    assert not partial_info


def test_optimizer_restore_runs_inside_skip_dtensor_dispatch(monkeypatch: Any) -> None:
    """Synthetic optimizer-slot initialization follows the local tensor path."""
    active = []
    calls = []

    class _RecordingSkipDispatch(AbstractContextManager):
        """Record entry around the patched optimizer state loader."""

        def __enter__(self) -> None:
            """Record entry into the dispatch-suppression context."""
            active.append(True)

        def __exit__(self, *exc_info: Any) -> None:
            """Record exit from the dispatch-suppression context."""
            del exc_info
            active.pop()

    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    chained = optimizer_module.ChainedOptimizer(model, {"sgd": optimizer})
    monkeypatch.setattr(optimizer_module, "SkipDTensorDispatch", _RecordingSkipDispatch)

    def _set_optimizer_state_dict(*args: Any, **kwargs: Any) -> None:
        """Require the state loader to run while dispatch is disabled."""
        assert active == [True]
        calls.append((args, kwargs))

    monkeypatch.setattr(optimizer_module, "set_optimizer_state_dict", _set_optimizer_state_dict)
    monkeypatch.setattr(chained, "_synchronize_steps", lambda: None)

    state_dict = {"state": {}, "param_groups": []}
    chained.load_state_dict(state_dict)

    assert len(calls) == 1
    assert calls[0][1]["optim_state_dict"] is state_dict


def test_adamw_step_tensor_follows_offloaded_parameter_device(monkeypatch: Any) -> None:
    """CPU-offloaded AdamW state does not create its step tensor on the NPU."""
    parameter = torch.ones(2)
    gradient = torch.ones(2)
    exp_avg = torch.zeros(2)
    exp_avg_sq = torch.zeros(2)
    calls = []

    def _fused_adamw(*args: Any, **kwargs: Any) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr(adamw_module.torch, "_fused_adamw_", _fused_adamw)

    adamw_module.adamw(
        [parameter],
        [gradient],
        [exp_avg],
        [exp_avg_sq],
        [],
        1,
        amsgrad=False,
        beta1=0.9,
        beta2=0.999,
        lr=1.0e-3,
        weight_decay=0.0,
        eps=1.0e-8,
        maximize=False,
    )

    state_steps = calls[0][0][5]
    assert len(state_steps) == 1
    assert state_steps[0].device == parameter.device
