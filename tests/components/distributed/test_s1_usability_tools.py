# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0
# ============================================================================
"""易用性工具测试（2026-08-10）：
- ShardingPlan.explain() 内省报告器（1.1）
- insert 模式报错附建议 spec 草稿（1.1）
- apply 期注入选择结果 INFO（1.2）
- check_dispatchable() region_dispatch 判定工具（1.2）
"""

import logging

import pytest
import torch
import torch.distributed as dist

from hyper_parallel.auto_models.components.distributed.dispatch_probe import check_dispatchable
from hyper_parallel.auto_models.components.distributed.sharding_applier import (
    _log_injection_choice,
)
from hyper_parallel.auto_models.components.distributed.sharding_config import (
    MeshAxisName,
    ModuleShardingSpec,
)
from hyper_parallel.auto_models.components.distributed.sharding_planner import ShardingPlanner
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh

from tests.components.distributed.conftest import (
    TinyConfig,
    TinyLlamaForCausalLM,
)

TP = MeshAxisName.TP


@pytest.fixture(scope="module")
def mesh():
    if not dist.is_initialized():
        dist.init_process_group(
            "gloo", init_method="tcp://127.0.0.1:29751", rank=0, world_size=1)
    return init_device_mesh("cpu", (1,), mesh_dim_names=("tp",))


@pytest.fixture(scope="module")
def plan(mesh):
    torch.manual_seed(0)
    model = TinyLlamaForCausalLM(TinyConfig())
    return ShardingPlanner().plan(model, mesh, tp_size=1)


class TestPlanExplain:
    def test_report_contains_boundary_and_params(self, plan):
        report = plan.explain()
        assert "ShardingPlan introspection report" in report
        assert "[model.layers.0.self_attn]" in report
        assert "q_proj.weight" in report
        assert "injection: none (ordinary boundary" in report

    def test_fqn_filter(self, plan):
        report = plan.explain(fqn="model.layers.0.mlp")
        assert "[model.layers.0.mlp]" in report
        assert "[model.layers.1.mlp]" not in report

    def test_unknown_fqn_noted(self, plan):
        report = plan.explain(fqn="no.such.module")
        assert "is not a boundary of this plan" in report

    def test_injection_section_rendered(self, mesh):
        spec = ModuleShardingSpec(
            local_compute_fn=lambda m, x: x, region_dispatch=False)
        torch.manual_seed(0)
        model = TinyLlamaForCausalLM(TinyConfig())
        p = ShardingPlanner(
            plan_overrides={"*.mlp": spec}).plan(model, mesh, tp_size=1)
        report = p.explain(fqn="model.layers.0.mlp")
        assert "local_compute_fn=" in report
        assert "region_dispatch=False -> black-box managed" in report

    def test_planner_explain_flag_logs(self, mesh, caplog):
        torch.manual_seed(0)
        model = TinyLlamaForCausalLM(TinyConfig())
        with caplog.at_level(logging.INFO):
            ShardingPlanner().plan(model, mesh, tp_size=1, explain=True)
        assert any("ShardingPlan explain" in r.message for r in caplog.records)


class TestInsertSkeletonHint:
    def test_all_empty_insert_error_carries_skeleton(self, mesh):
        torch.manual_seed(0)
        model = TinyLlamaForCausalLM(TinyConfig())
        with pytest.raises(ValueError, match="Suggested draft") as exc_info:
            ShardingPlanner(plan_overrides={
                "model.layers.0": ModuleShardingSpec()}).plan(
                    model, mesh, tp_size=1)
        msg = str(exc_info.value)
        assert '- match: "model.layers.0"' in msg
        assert "in_src:" in msg and "out_src:" in msg

    def test_skeleton_uses_forward_required_params(self, mesh):
        skeleton = ShardingPlanner._suggest_insert_skeleton(
            TinyLlamaForCausalLM(TinyConfig()), "model.layers.0")
        assert "hidden_states" in skeleton   # forward 必填入参被采纳


class TestInjectionChoiceInfo:
    def test_false_blackbox_logged(self, caplog):
        spec = ModuleShardingSpec(
            local_compute_fn=lambda m, x: x, region_dispatch=False)
        with caplog.at_level(logging.INFO):
            _log_injection_choice("model.layers.0.mlp", spec)
        assert any("black-box hosting" in r.message for r in caplog.records)

    def test_true_penetration_logged(self, caplog):
        spec = ModuleShardingSpec(
            inner_target="self", inner_wrapper="sdpa_hf", region_dispatch=True)
        with caplog.at_level(logging.INFO):
            _log_injection_choice("model.layers.0.self_attn", spec)
        assert any("dispatch-through true validation enabled" in r.message for r in caplog.records)

    def test_plain_boundary_silent(self, caplog):
        with caplog.at_level(logging.INFO):
            _log_injection_choice("model.layers.0.mlp", ModuleShardingSpec())
        assert not caplog.records


class TestCheckDispatchable:
    def test_pure_ops_dispatchable(self, mesh):
        def pure_fn(x, w):
            return torch.relu(torch.nn.functional.linear(x, w))
        report = check_dispatchable(
            pure_fn, [torch.randn(2, 4), torch.randn(8, 4)], mesh)
        assert report.dispatchable is True
        assert "region_dispatch=True" in report.recommendation
        assert report.ops   # dispatch 轨迹非空

    def test_comm_op_not_dispatchable(self, mesh):
        def comm_fn(x):
            out = [torch.empty_like(x)]
            dist.all_gather(out, x)
            return out[0]
        report = check_dispatchable(comm_fn, [torch.randn(2, 4)], mesh)
        assert report.dispatchable is False
        assert "all_gather" in (report.failed_op or "") + (report.error or "")
        assert "region_dispatch=False" in report.recommendation

    def test_module_input_and_str(self, mesh):
        module = torch.nn.Linear(4, 4)
        report = check_dispatchable(module, [torch.randn(2, 4)], mesh)
        assert report.dispatchable is True
        text = str(report)
        assert "dispatchable: True" in text and "recommendation" in text
