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
"""minimal plan-cache verification API case"""
from pathlib import Path
import shutil
from typing import Any

import torch
import torch.distributed as dist

from hyper_parallel.core.distributed_checkpoint.standard_planner import StandardSavePlanner
from tests.torch.utils import init_dist


class TraceMinimalCacheSavePlanner(StandardSavePlanner):
    """Planner that traces cache behavior without depending on plan scheduling details.

    Defined at module scope so it can be pickled for :func:`async_save` worker processes.
    """

    def __init__(self) -> None:
        """Enable plan caching for the test planner."""
        super().__init__(enable_plan_caching=True)
        self.cache_miss_count = 0
        self.cache_hit_count = 0
        self.cache_write_keys: list[str] = []
        self.cache_hit_keys: list[str] = []

    def get_cached(self) -> Any:
        """Trace whether the planner cache is missed or hit for the current key."""
        cached = super().get_cached()
        if cached is None:
            self.cache_miss_count += 1
        else:
            self.cache_hit_count += 1
            self.cache_hit_keys.append(self._cached_plans_key)
        return cached

    def cache_result(self, final_plan: Any, metadata: Any) -> None:
        """Trace which cache key is written when a plan is materialized."""
        self.cache_write_keys.append(self._cached_plans_key)
        super().cache_result(final_plan, metadata)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear class-level cache for test isolation."""
        cls._cached_save_result.clear()


def _init_dist_for_case() -> tuple[int, int]:
    """Reuse the repository's standard torch distributed initialization helper."""
    return init_dist()


def _runtime_imports() -> dict[str, Any]:
    """Import hyper_parallel modules after NPU device binding is complete."""
    # pylint: disable=import-outside-toplevel
    from hyper_parallel import DTensor as dtensor_cls
    from hyper_parallel.core.distributed_checkpoint import async_save as async_save_fn
    from hyper_parallel.core.distributed_checkpoint import load as load_fn
    from hyper_parallel.core.distributed_checkpoint import save as save_fn
    from hyper_parallel.core.dtensor.device_mesh import init_device_mesh as init_device_mesh_fn
    from hyper_parallel.core.dtensor.placement_types import Replicate as replicate_cls
    from hyper_parallel.core.dtensor.placement_types import Shard as shard_cls

    return {
        "dtensor_cls": dtensor_cls,
        "save_fn": save_fn,
        "async_save_fn": async_save_fn,
        "load_fn": load_fn,
        "planner_cls": TraceMinimalCacheSavePlanner,
        "init_device_mesh_fn": init_device_mesh_fn,
        "shard_cls": shard_cls,
        "replicate_cls": replicate_cls,
    }


def _build_model_state(step: int, device: torch.device, mesh, runtime: dict[str, Any]) -> dict[str, Any]:
    """Build a mixed model-like state dict."""
    dense = torch.full((64, 32), float(step), dtype=torch.float32, device=device)
    local_sharded = torch.arange(0, 32 * 32, dtype=torch.float32, device=device).reshape(32, 32) + step
    local_replicated = torch.full((16, 24), float(step), dtype=torch.float32, device=device)
    io_payload = {"step": step, "tag": f"model-{step}"}
    return {
        "dense": dense,
        "dt_sharded": runtime["dtensor_cls"].from_local(local_sharded, mesh, [runtime["shard_cls"](0)]),
        "dt_replicated": runtime["dtensor_cls"].from_local(local_replicated, mesh, [runtime["replicate_cls"]()]),
        "io_payload": io_payload,
    }


def _mutate_model_state(
        state_dict: dict[str, Any], step: int, device: torch.device, mesh, runtime: dict[str, Any]
) -> None:
    """Mutate the same model state_dict object in place while preserving schema."""
    updated = _build_model_state(
        step=step,
        device=device,
        mesh=mesh,
        runtime=runtime,
    )
    state_dict.clear()
    state_dict.update(updated)


def _clear_plan_cache(planner_cls: type) -> None:
    """Clear class-level plan cache to isolate test cases."""
    planner_cls.clear_cache()


def _async_save_wait(runtime: dict[str, Any], state_dict: dict[str, Any], **kwargs: Any) -> None:
    """Run ``async_save`` and block until persistence completes (staging is synchronous in the caller)."""
    resp = runtime["async_save_fn"](state_dict, **kwargs)
    resp.persist_completion.result()


def _build_optimizer_state(step: int, device: torch.device) -> dict[str, Any]:
    """Build an optimizer-like state dict with a different schema."""
    exp_avg = torch.full((64, 32), float(step), dtype=torch.float32, device=device)
    exp_avg_sq = torch.full((64, 32), float(step + 1), dtype=torch.float32, device=device)
    return {
        "exp_avg": exp_avg,
        "exp_avg_sq": exp_avg_sq,
        "optim_state": {"step": step, "tag": f"optim-{step}"},
    }


def _mutate_optimizer_state(state_dict: dict[str, Any], step: int, device: torch.device) -> None:
    """Mutate the same optimizer state_dict object in place while preserving schema."""
    updated = _build_optimizer_state(step=step, device=device)
    state_dict.clear()
    state_dict.update(updated)


def test_dcp_minimal_plan_cache_hit() -> None:
    """
    Feature: minimal save-plan cache hit path.
    Description:
        1) Save model state once.
        2) Save the same state_dict object again after mutating values in place.
        3) Verify the second save hits the planner cache.
        4) Load the second checkpoint and verify all values are the latest ones.
    Expectation: Repeated save with the same schema hits cache and preserves correctness of the saved data.
    """
    rank, device_id = _init_dist_for_case()
    runtime = _runtime_imports()
    world = dist.get_world_size()
    device = torch.device("npu", device_id)
    mesh = runtime["init_device_mesh_fn"](
        device_type="npu",
        mesh_shape=(world,),
        mesh_dim_names=("dp",),
    )

    _clear_plan_cache(runtime["planner_cls"])
    planner = runtime["planner_cls"]()
    base = Path("./test_dcp_minimal_plan_cache_hit")
    ckpt1 = base / "ckpt1"
    ckpt2 = base / "ckpt2"

    if rank == 0:
        if base.exists():
            shutil.rmtree(base)
        base.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    model_state = _build_model_state(
        step=1,
        device=device,
        mesh=mesh,
        runtime=runtime,
    )
    runtime["save_fn"](model_state, checkpoint_id=ckpt1, planner=planner, use_collectives=True)
    dist.barrier()
    assert planner.cache_miss_count == 1
    assert planner.cache_hit_count == 0
    assert len(planner.cache_write_keys) == 1

    _mutate_model_state(
        model_state,
        step=2,
        device=device,
        mesh=mesh,
        runtime=runtime,
    )
    runtime["save_fn"](model_state, checkpoint_id=ckpt2, planner=planner, use_collectives=True)
    dist.barrier()
    assert planner.cache_miss_count == 1
    assert planner.cache_hit_count == 1
    assert len(planner.cache_write_keys) == 1
    assert planner.cache_hit_keys[0] == planner.cache_write_keys[0]

    load_state = {
        "dense": torch.zeros((64, 32), dtype=torch.float32, device=device),
        "dt_sharded": runtime["dtensor_cls"].from_local(
            torch.zeros((32, 32), dtype=torch.float32, device=device), mesh, [runtime["shard_cls"](0)]
        ),
        "dt_replicated": runtime["dtensor_cls"].from_local(
            torch.zeros((16, 24), dtype=torch.float32, device=device), mesh, [runtime["replicate_cls"]()]
        ),
        "io_payload": {"step": 0, "tag": ""},
    }
    runtime["load_fn"](load_state, checkpoint_id=ckpt2, use_collectives=True)
    dist.barrier()

    assert torch.allclose(load_state["dense"], model_state["dense"])
    assert torch.allclose(load_state["dt_sharded"].to_local(), model_state["dt_sharded"].to_local())
    assert torch.allclose(load_state["dt_replicated"].to_local(), model_state["dt_replicated"].to_local())
    io_payload = load_state["io_payload"]
    assert isinstance(io_payload, dict)
    # pylint: disable=unsubscriptable-object
    assert io_payload["step"] == 2


def test_dcp_minimal_plan_cache_hit_async() -> None:
    """
    Feature: minimal save-plan cache hit path with ``async_save`` (process staging + persistence).

    Same steps as ``test_dcp_minimal_plan_cache_hit`` but checkpoints are written via
    ``async_save`` and futures are waited explicitly.
    """
    rank, device_id = _init_dist_for_case()
    runtime = _runtime_imports()
    world = dist.get_world_size()
    device = torch.device("npu", device_id)
    mesh = runtime["init_device_mesh_fn"](
        device_type="npu",
        mesh_shape=(world,),
        mesh_dim_names=("dp",),
    )

    _clear_plan_cache(runtime["planner_cls"])
    planner = runtime["planner_cls"]()
    base = Path("./test_dcp_minimal_plan_cache_hit_async")
    ckpt1 = base / "ckpt1"
    ckpt2 = base / "ckpt2"

    if rank == 0:
        if base.exists():
            shutil.rmtree(base)
        base.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    model_state = _build_model_state(
        step=1,
        device=device,
        mesh=mesh,
        runtime=runtime,
    )
    _async_save_wait(
        runtime, model_state, checkpoint_id=ckpt1, planner=planner, use_collectives=True
    )
    dist.barrier()
    assert planner.cache_miss_count == 1
    assert planner.cache_hit_count == 0
    assert len(planner.cache_write_keys) == 1

    _mutate_model_state(
        model_state,
        step=2,
        device=device,
        mesh=mesh,
        runtime=runtime,
    )
    _async_save_wait(
        runtime, model_state, checkpoint_id=ckpt2, planner=planner, use_collectives=True
    )
    dist.barrier()
    assert planner.cache_miss_count == 1
    assert planner.cache_hit_count == 1
    assert len(planner.cache_write_keys) == 1
    assert planner.cache_hit_keys[0] == planner.cache_write_keys[0]

    load_state = {
        "dense": torch.zeros((64, 32), dtype=torch.float32, device=device),
        "dt_sharded": runtime["dtensor_cls"].from_local(
            torch.zeros((32, 32), dtype=torch.float32, device=device), mesh, [runtime["shard_cls"](0)]
        ),
        "dt_replicated": runtime["dtensor_cls"].from_local(
            torch.zeros((16, 24), dtype=torch.float32, device=device), mesh, [runtime["replicate_cls"]()]
        ),
        "io_payload": {"step": 0, "tag": ""},
    }
    runtime["load_fn"](load_state, checkpoint_id=ckpt2, use_collectives=True)
    dist.barrier()

    assert torch.allclose(load_state["dense"], model_state["dense"])
    assert torch.allclose(load_state["dt_sharded"].to_local(), model_state["dt_sharded"].to_local())
    assert torch.allclose(load_state["dt_replicated"].to_local(), model_state["dt_replicated"].to_local())
    io_payload = load_state["io_payload"]
    assert isinstance(io_payload, dict)
    # pylint: disable=unsubscriptable-object
    assert io_payload["step"] == 2


def test_dcp_minimal_plan_cache_model_optimizer_isolation() -> None:
    """
    Feature: minimal save-plan cache namespace isolation.
    Description:
        1) Save a model-like state twice with the same planner.
        2) Save an optimizer-like state twice with the same planner.
        3) Verify model and optimizer use different cache entries through the same planner.
        4) Load the latest model and optimizer checkpoints.
    Expectation: Different schemas do not share one cache entry and still preserve their own latest data.
    """
    rank, device_id = _init_dist_for_case()
    runtime = _runtime_imports()
    world = dist.get_world_size()
    device = torch.device("npu", device_id)
    mesh = runtime["init_device_mesh_fn"](
        device_type="npu",
        mesh_shape=(world,),
        mesh_dim_names=("dp",),
    )

    _clear_plan_cache(runtime["planner_cls"])
    planner = runtime["planner_cls"]()
    base = Path("./test_dcp_minimal_plan_cache_namespace")
    model_ckpt = base / "model_ckpt"
    model_ckpt_2 = base / "model_ckpt_2"
    optim_ckpt = base / "optim_ckpt"
    optim_ckpt_2 = base / "optim_ckpt_2"

    if rank == 0:
        if base.exists():
            shutil.rmtree(base)
        base.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    model_state = _build_model_state(
        step=10,
        device=device,
        mesh=mesh,
        runtime=runtime,
    )
    runtime["save_fn"](model_state, checkpoint_id=model_ckpt, planner=planner, use_collectives=True)
    dist.barrier()
    assert planner.cache_miss_count == 1
    assert planner.cache_hit_count == 0
    assert len(planner.cache_write_keys) == 1

    _mutate_model_state(
        model_state,
        step=11,
        device=device,
        mesh=mesh,
        runtime=runtime,
    )
    runtime["save_fn"](model_state, checkpoint_id=model_ckpt_2, planner=planner, use_collectives=True)
    dist.barrier()
    assert planner.cache_miss_count == 1
    assert planner.cache_hit_count == 1
    assert len(planner.cache_write_keys) == 1
    assert planner.cache_hit_keys[0] == planner.cache_write_keys[0]

    optimizer_state = _build_optimizer_state(step=20, device=device)
    runtime["save_fn"](optimizer_state, checkpoint_id=optim_ckpt, planner=planner, use_collectives=True)
    dist.barrier()
    assert planner.cache_miss_count == 2
    assert planner.cache_hit_count == 1
    assert len(planner.cache_write_keys) == 2
    assert planner.cache_write_keys[1] != planner.cache_write_keys[0]

    _mutate_optimizer_state(optimizer_state, step=21, device=device)
    runtime["save_fn"](optimizer_state, checkpoint_id=optim_ckpt_2, planner=planner, use_collectives=True)
    dist.barrier()
    assert planner.cache_miss_count == 2
    assert planner.cache_hit_count == 2
    assert len(planner.cache_write_keys) == 2
    assert planner.cache_hit_keys[1] == planner.cache_write_keys[1]
    assert planner.cache_hit_keys[1] != planner.cache_write_keys[0]

    model_load_state = {
        "dense": torch.zeros((64, 32), dtype=torch.float32, device=device),
        "dt_sharded": runtime["dtensor_cls"].from_local(
            torch.zeros((32, 32), dtype=torch.float32, device=device), mesh, [runtime["shard_cls"](0)]
        ),
        "dt_replicated": runtime["dtensor_cls"].from_local(
            torch.zeros((16, 24), dtype=torch.float32, device=device), mesh, [runtime["replicate_cls"]()]
        ),
        "io_payload": {"step": 0, "tag": ""},
    }
    runtime["load_fn"](model_load_state, checkpoint_id=model_ckpt_2, use_collectives=True)
    dist.barrier()

    assert torch.allclose(model_load_state["dense"], model_state["dense"])
    assert torch.allclose(model_load_state["dt_sharded"].to_local(), model_state["dt_sharded"].to_local())
    assert torch.allclose(model_load_state["dt_replicated"].to_local(), model_state["dt_replicated"].to_local())
    model_io_payload = model_load_state["io_payload"]
    assert isinstance(model_io_payload, dict)
    # pylint: disable=unsubscriptable-object
    assert model_io_payload["step"] == 11

    load_state = {
        "exp_avg": torch.zeros((64, 32), dtype=torch.float32, device=device),
        "exp_avg_sq": torch.zeros((64, 32), dtype=torch.float32, device=device),
        "optim_state": {"step": 0, "tag": ""},
    }
    runtime["load_fn"](load_state, checkpoint_id=optim_ckpt_2, use_collectives=True)
    dist.barrier()

    assert torch.allclose(load_state["exp_avg"], optimizer_state["exp_avg"])
    assert torch.allclose(load_state["exp_avg_sq"], optimizer_state["exp_avg_sq"])
    optim_state = load_state["optim_state"]
    assert isinstance(optim_state, dict)
    # pylint: disable=unsubscriptable-object
    assert optim_state["step"] == 21


def test_dcp_minimal_plan_cache_model_optimizer_isolation_async() -> None:
    """
    Feature: minimal save-plan cache namespace isolation with ``async_save``.

    Same as ``test_dcp_minimal_plan_cache_model_optimizer_isolation`` but all saves
    use ``async_save`` with explicit future completion.
    """
    rank, device_id = _init_dist_for_case()
    runtime = _runtime_imports()
    world = dist.get_world_size()
    device = torch.device("npu", device_id)
    mesh = runtime["init_device_mesh_fn"](
        device_type="npu",
        mesh_shape=(world,),
        mesh_dim_names=("dp",),
    )

    _clear_plan_cache(runtime["planner_cls"])
    planner = runtime["planner_cls"]()
    base = Path("./test_dcp_minimal_plan_cache_namespace_async")
    model_ckpt = base / "model_ckpt"
    model_ckpt_2 = base / "model_ckpt_2"
    optim_ckpt = base / "optim_ckpt"
    optim_ckpt_2 = base / "optim_ckpt_2"

    if rank == 0:
        if base.exists():
            shutil.rmtree(base)
        base.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    model_state = _build_model_state(
        step=10,
        device=device,
        mesh=mesh,
        runtime=runtime,
    )
    _async_save_wait(
        runtime, model_state, checkpoint_id=model_ckpt, planner=planner, use_collectives=True
    )
    dist.barrier()
    assert planner.cache_miss_count == 1
    assert planner.cache_hit_count == 0
    assert len(planner.cache_write_keys) == 1

    _mutate_model_state(
        model_state,
        step=11,
        device=device,
        mesh=mesh,
        runtime=runtime,
    )
    _async_save_wait(
        runtime, model_state, checkpoint_id=model_ckpt_2, planner=planner, use_collectives=True
    )
    dist.barrier()
    assert planner.cache_miss_count == 1
    assert planner.cache_hit_count == 1
    assert len(planner.cache_write_keys) == 1
    assert planner.cache_hit_keys[0] == planner.cache_write_keys[0]

    optimizer_state = _build_optimizer_state(step=20, device=device)
    _async_save_wait(
        runtime, optimizer_state, checkpoint_id=optim_ckpt, planner=planner, use_collectives=True
    )
    dist.barrier()
    assert planner.cache_miss_count == 2
    assert planner.cache_hit_count == 1
    assert len(planner.cache_write_keys) == 2
    assert planner.cache_write_keys[1] != planner.cache_write_keys[0]

    _mutate_optimizer_state(optimizer_state, step=21, device=device)
    _async_save_wait(
        runtime, optimizer_state, checkpoint_id=optim_ckpt_2, planner=planner, use_collectives=True
    )
    dist.barrier()
    assert planner.cache_miss_count == 2
    assert planner.cache_hit_count == 2
    assert len(planner.cache_write_keys) == 2
    assert planner.cache_hit_keys[1] == planner.cache_write_keys[1]
    assert planner.cache_hit_keys[1] != planner.cache_write_keys[0]

    model_load_state = {
        "dense": torch.zeros((64, 32), dtype=torch.float32, device=device),
        "dt_sharded": runtime["dtensor_cls"].from_local(
            torch.zeros((32, 32), dtype=torch.float32, device=device), mesh, [runtime["shard_cls"](0)]
        ),
        "dt_replicated": runtime["dtensor_cls"].from_local(
            torch.zeros((16, 24), dtype=torch.float32, device=device), mesh, [runtime["replicate_cls"]()]
        ),
        "io_payload": {"step": 0, "tag": ""},
    }
    runtime["load_fn"](model_load_state, checkpoint_id=model_ckpt_2, use_collectives=True)
    dist.barrier()

    assert torch.allclose(model_load_state["dense"], model_state["dense"])
    assert torch.allclose(model_load_state["dt_sharded"].to_local(), model_state["dt_sharded"].to_local())
    assert torch.allclose(model_load_state["dt_replicated"].to_local(), model_state["dt_replicated"].to_local())
    model_io_payload = model_load_state["io_payload"]
    assert isinstance(model_io_payload, dict)
    # pylint: disable=unsubscriptable-object
    assert model_io_payload["step"] == 11

    load_state = {
        "exp_avg": torch.zeros((64, 32), dtype=torch.float32, device=device),
        "exp_avg_sq": torch.zeros((64, 32), dtype=torch.float32, device=device),
        "optim_state": {"step": 0, "tag": ""},
    }
    runtime["load_fn"](load_state, checkpoint_id=optim_ckpt_2, use_collectives=True)
    dist.barrier()

    assert torch.allclose(load_state["exp_avg"], optimizer_state["exp_avg"])
    assert torch.allclose(load_state["exp_avg_sq"], optimizer_state["exp_avg_sq"])
    optim_state = load_state["optim_state"]
    assert isinstance(optim_state, dict)
    # pylint: disable=unsubscriptable-object
    assert optim_state["step"] == 21
