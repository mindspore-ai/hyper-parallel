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
"""Opt-in MoonEP placement with the original HP MegaMoE execution schedule."""
from __future__ import annotations

from typing import Any
import weakref

import torch


from hyper_parallel.core.multicore.modules.mega_moe.function import execute_mega_moe_with_permutation
from hyper_parallel.core.multicore.modules.mega_moe.heap_manager import _workspace_layout
from hyper_parallel.core.multicore.modules.mega_moe.module import MegaMoeExperts
from hyper_parallel.core.multicore.modules.mega_moe.route import prepare_topk_route, restore_topk_output
from hyper_parallel.core.multicore.modules.mega_moe.spec import _align_capacity
from .placement import build_placement, placement_is_profitable

from .shmem_transfer import execute_shared, get_expert_pool


_VIRTUAL_GROUPS: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()


class MoonEPMegaMoeExperts(MegaMoeExperts):
    """Share placement with ordinary EP while keeping MegaMoE's native schedule.

    Route selection precedes symmetric growth and source publication. Each
    invocation owns its placement; weights remain parameters of the original
    module and the shared bank is leased only while an executor uses its views.
    """

    def __init__(self, **configuration: Any) -> None:
        """Retain the original expert configuration and defer optional resources."""
        super().__init__(**configuration)
        self._virtual_executor = None
        self._virtual_configuration = {
            **configuration, "num_experts": 2 * self.num_experts, "create_parameters": False,
        }
        self._transfer_pool = None
        self.placement_statistics = {
            "native_balanced_calls": 0, "native_cost_skip_calls": 0,
            "native_no_copy_calls": 0, "moonep_calls": 0,
        }

    def _get_virtual_executor(
        self, hidden: torch.Tensor, plan_tokens: int, maximum_received: int,
    ) -> MegaMoeExperts:
        """Reserve the cold virtual workspace and expert bank in one transaction."""
        if self._virtual_executor is not None:
            return self._virtual_executor
        with torch.inference_mode(False), torch.no_grad():
            executor = MegaMoeExperts(**self._virtual_configuration)
        target = _VIRTUAL_GROUPS.get(self._resource_group)
        if target is None or target.closed or target.closing:
            target = executor._resource_group
            target.shared = self._resource_group.shared
            _VIRTUAL_GROUPS[self._resource_group] = target
        else:
            if target.compatibility_key != executor._resource_group.compatibility_key:
                raise ValueError("Shared MoonEP virtual plans require matching HP configuration")
            executor._move_to_resource_group(target)
        reservations = ()
        if target.resources is None:
            specification = target.specification
            layout = _workspace_layout(
                specification, hidden.dtype, capacity=_align_capacity(max(1, maximum_received)),
                token_capacity=plan_tokens if specification.get("auto_tokens", False) else None,
            )
            reservations = ((specification, layout),)
        with torch.inference_mode(False), torch.no_grad():
            self._transfer_pool = get_expert_pool(
                hidden, self._resource_group.resources, self.local_experts, self, reservations,
            )
        self._virtual_executor = executor
        return executor

    def _persistent_engine(self, flat: torch.Tensor, executor: MegaMoeExperts | None = None) -> Any:
        """Keep persistent state writable after an inference-only first call."""
        if executor is None:
            executor = self
        if executor._resource_group.resources is not None:
            return executor._get_execution_resources(flat)
        with torch.inference_mode(False), torch.no_grad():
            return executor._get_execution_resources(flat)

    def _prepare_native_route(
        self, flat: torch.Tensor, ids: torch.Tensor, probs: torch.Tensor,
        counts: torch.Tensor | None, weights: tuple[torch.Tensor, torch.Tensor],
        executor: MegaMoeExperts | None = None, select_route: Any = None, resources: Any = None,
    ) -> tuple[Any, Any, bool]:
        """Allocate only the route chosen after the existing count agreement."""
        engine = self._persistent_engine(flat, executor) if resources is None else resources
        pull = self.dispatch_mode == "pull"
        claimed = False

        def prepare_workspace(tokens: int, received: int) -> None:
            """Commit the chosen route before a pull view can reference SHMEM."""
            nonlocal claimed
            if engine.spec.auto_tokens:
                growth = (tokens > engine._token_capacity
                          or received > engine.workspace.capacity_floor and not pull
                          or engine.workspace.forward_event_counters is None)
                if growth:
                    with torch.inference_mode(False), torch.no_grad():
                        engine.prepare_auto_workspace(tokens, received, flat)
                else:
                    engine.prepare_auto_workspace(tokens, received, flat)
            elif pull:
                with torch.inference_mode(False), torch.no_grad():
                    engine.workspace.ensure(engine.spec, flat.dtype, flat.device)
                engine.workspace.claim()
            claimed = pull

        options = {"autograd_mask": (
            int(torch.is_grad_enabled()) | (int(flat.requires_grad) << 1)
            | (int(weights[0].requires_grad) << 2) | (int(weights[1].requires_grad) << 3)
            | (int(probs.requires_grad) << 4)
        )}
        if engine.spec.auto_tokens or select_route is not None:
            options["prepare_workspace"] = prepare_workspace
        elif pull:
            prepare_workspace(engine.spec.plan_tokens, engine.spec.receive_capacity)
        try:
            with torch.no_grad():
                route = prepare_topk_route(
                    flat, ids, probs, engine.spec, counts, workspace=engine.workspace,
                    select_route=select_route, **options,
                )
        except BaseException:
            if claimed:
                engine.workspace.release()
            raise
        return engine, route, claimed

    def _execute_native_route(
        self, flat: torch.Tensor, ids: torch.Tensor, probs: torch.Tensor,
        weights: tuple[torch.Tensor, torch.Tensor], engine: Any, route: Any,
        pull: bool, output_shape: torch.Size,
    ) -> torch.Tensor:
        """Use the original plan and kernels after selecting the ordinary route."""
        if not pull and not engine.spec.auto_tokens:
            with torch.inference_mode(False), torch.no_grad():
                engine.ensure_capacity(route.maximum_received_slots)
                engine.workspace.ensure(engine.spec, flat.dtype, flat.device)
        output = execute_mega_moe_with_permutation(
            flat, ids, *weights, route, engine.plan_for_tokens(route.metadata.plan_tokens, flat.device),
            engine.workspace, topk_weights=probs if pull else None, workspace_claimed=pull,
        )
        if not pull:
            output = restore_topk_output(output, route.unpermute_mapping, probs)
        return output.reshape(output_shape)

    def _placement_is_profitable(
        self, engine: Any, maximum: int, target: int, weights: tuple[torch.Tensor, torch.Tensor],
    ) -> bool:
        return placement_is_profitable(
            maximum, target, self.hidden_size, self.intermediate_size, engine.spec.num_cube_cores,
            sum(weight.numel() * weight.element_size() for weight in weights),
        )

    def forward(
        self, hidden_states: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor, *,
        tokens_per_expert: torch.Tensor | None = None,
        expert_weights: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Borrow only when rank imbalance is estimated to repay its transfer cost."""
        weights = (self.gate_up_weight, self.down_weight) if expert_weights is None else expert_weights
        flat = self._validate_forward_inputs(
            hidden_states, topk_ids, topk_weights, tokens_per_expert, weights,
        )
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("MegaMoe push heap growth does not support graph capture")
        if self.ep_size == 1:
            with torch.inference_mode(False), torch.no_grad():
                engine = self._persistent_engine(flat)
                engine.workspace.ensure(engine.spec, flat.dtype, flat.device)
            return super().forward(hidden_states, topk_ids, topk_weights,
                                   tokens_per_expert=tokens_per_expert, expert_weights=weights)
        placement = None
        selected_tokens = 0
        engine = self._persistent_engine(flat)

        def select_route(gathered: torch.Tensor, tokens: int, maximum: int, total: int | None) -> bool:
            """Choose placement using the existing validated count table."""
            nonlocal placement, selected_tokens
            selected_tokens = tokens
            target = (total + self.ep_size - 1) // self.ep_size if total is not None else tokens * self.top_k
            if maximum <= target:
                self.placement_statistics["native_balanced_calls"] += 1
            elif not self._placement_is_profitable(engine, maximum, target, weights):
                self.placement_statistics["native_cost_skip_calls"] += 1
            else:
                placement = build_placement(flat, topk_ids, gathered, max(1, target), engine.spec.rank_id)
                if placement is not None:
                    return False
                self.placement_statistics["native_no_copy_calls"] += 1
            return True

        engine, route, claimed = self._prepare_native_route(
            flat, topk_ids, topk_weights, tokens_per_expert, weights, select_route=select_route, resources=engine,
        )
        try:
            if placement is None:
                return self._execute_native_route(
                    flat, topk_ids, topk_weights, weights, engine, route,
                    self.dispatch_mode == "pull", hidden_states.shape,
                )
        finally:
            if claimed:
                engine.workspace.release()
        self.placement_statistics["moonep_calls"] += 1
        counts = placement.counts_by_source
        maximum_received = 0
        if self._virtual_executor is None:
            maximum_received = int(counts.reshape(self.ep_size, self.ep_size, -1).sum((0, 2)).max().item())
        return execute_shared(
            self, hidden_states, placement.virtual_ids, topk_weights, weights,
            counts[engine.spec.rank_id], placement.copies, selected_tokens, maximum_received,
        )

    def close(self) -> None:
        """Release HP plans and close the bank after its last invocation."""
        pool = self._transfer_pool
        if pool is not None:
            pool.owners.discard(self)
            if not pool.owners and not pool.graphs:
                pool.manager.close_workspace(pool)
        if self._virtual_executor is not None:
            self._virtual_executor.close()
        super().close()
