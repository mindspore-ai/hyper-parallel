# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Backward overhead module"""

from typing import Tuple

from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.body import EvalBody


class _BackwardOverhead:
    """Backward overhead class"""

    def __init__(self, backbone, ccfg, ctx, inner_dyn_fun):
        self.backbone = backbone
        self._ccfg = ccfg
        self._ctx = ctx
        self._inner_dynamic_mem = inner_dyn_fun

    def _fetch_node_and_switch_env(
        self, stages, record_lay_types, stage_id, chunk_id, lay_id
    ):
        """Fetch one node and restore the evaluation context for it."""
        # if negative index access, convert to positive index
        if stage_id < 0:
            stage_id = len(stages) + stage_id
        if chunk_id < 0:
            chunk_id = len(stages[stage_id]) + chunk_id
        if lay_id < 0:
            lay_id = len(stages[stage_id][chunk_id]) + lay_id
        ccfg, ctx, kind, switches = record_lay_types[(stage_id, chunk_id, lay_id)]
        # print("here-",id(ccfg))
        # print("here0",id(self._ccfg))
        # print("here00",id(self.backbone._ccfg))
        self._ccfg = ccfg
        self._ctx = ctx
        self._ctx.switches = switches
        self._ctx.current_stage_id = stage_id
        self._ctx.current_chunk_id = chunk_id
        self._ctx.current_lay_id = lay_id
        # print("here1",id(self._ccfg))
        self.backbone.apply_kind(kind, ccfg=self._ccfg, ctx=self._ctx)
        # print("here11",id(self.backbone._ccfg))
        return stages[stage_id][chunk_id][lay_id]

    def _working_set(self, gathered: int = 2, on_saved: bool = False) -> float:
        """The dynamic memory of the node set up, as the working set of its backward.

        Under FSDP that reshards, a layer's backward holds *gathered* layers'
        parameters: its own and the next layer's, prefetched.  *on_saved*
        says the stage's dynamic memory counts what the node keeps for this
        backward already, as it does where the node does not recompute.
        """
        self._ctx.working_set = gathered
        self._ctx.working_on_saved = on_saved
        try:
            return sum(self._inner_dynamic_mem(default_micro_factor=1))
        finally:
            self._ctx.working_set = 0
            self._ctx.working_on_saved = False

    def _last_backward_lay_ids(self, stages: list, stage_id: int, record_lay_types: dict) -> list:
        """The stage's regular layers from the one whose backward a micro-batch's ends on, in stage order.

        A stage runs its layers' backward in reverse, so the last is its
        first layer.  Where a vision tower shares the stage with the
        language model the search drives, the tower's layers come first and
        run last, and their working set is the smaller of the two; but the
        moment the language model's own first layer runs its backward, every
        activation of the tower is still held, and that is the moment the
        stage has to fit (F2, F54).  A stage of one module, or one holding
        no layer of the main module, keeps its own first layers.
        """
        regular = [lay_id for lay_id, node in enumerate(stages[stage_id][0])
                   if self.backbone.is_regular_layer(node)]
        main = getattr(self.backbone, "main_module", None)
        if main is None:
            return regular
        of_main = [lay_id for lay_id in regular
                   if record_lay_types[(stage_id, 0, lay_id)][0].model_name == main]
        return of_main or regular

    def first_layer_working_set(self, stages: list, stage_id: int, record_lay_types: dict) -> Tuple[float, float]:
        """The working set of the backward a micro-batch's ends on, the stage's first layer's, unlogged.

        It holds its own gathered parameters, with no layer left to prefetch.
        FSDP that reduces a layer's gradients while the next layer's backward
        runs holds its whole gradients, and the second layer's, whose
        reduction is still in flight.  Also returns that layer's
        reduce-scatter output, which does not exist yet.  Where submodules
        share the stage, it is the main module's first layer
        (:meth:`_last_backward_lay_ids`).
        """
        lay_ids = self._last_backward_lay_ids(stages, stage_id, record_lay_types)
        working, pending = 0, 0
        for rank, lay_id in enumerate(lay_ids[:2]):
            node = self._fetch_node_and_switch_env(stages, record_lay_types, stage_id, 0, lay_id)
            ctx = self._ctx
            saved = (ctx.current_node, ctx.current_lay_id, ctx.enable_node_log, dict(ctx.accu_mem_type))
            ctx.current_node = LayerType.NOT_REC_LAYER if node == LayerType.FULL_REC_LAYER else node
            ctx.enable_node_log = False
            try:
                if rank == 0:
                    working += self._working_set(gathered=1)
                if self._ccfg.overlaps_grad_reduce:
                    whole, sharded = EvalBody.reduced_grad_layer(self._ccfg, ctx)
                    working += whole
                    pending = pending if rank else sharded
            finally:
                ctx.current_node, ctx.current_lay_id, ctx.enable_node_log, ctx.accu_mem_type = saved
        return working, pending

    def estimate(
        self, stages: list, stage_id: int, record_lay_types: dict
    ) -> float:
        """estimate stage's end-of-warmup overhead"""
        # self._ctx.enable_node_log = False
        res = 0
        if self._ccfg.pp_sched in ["1f1b", "seqpipe", "seqsmartvpp"]:
            res = self.__stage_bwd_overhead_1f1b(
                stages, stage_id, record_lay_types
            )
        if self._ccfg.pp_sched == "zero_bubble_v":
            res = self.__stage_bwd_overhead_zbv(
                stages, stage_id, record_lay_types
            )
        return res

    def vpp_1f1b_steady_overhead(self, stage_id, dyn_mem_i):
        """potential overhead due to imbalanced chunks"""

        def dyn(chunk_id):
            """chunk total mem"""
            return sum(dyn_mem_i[chunk_id])

        # less mem
        micro_left = self._ccfg.m - self._ccfg.p
        vpp = self._ccfg.vp
        max_overhead = 0
        if self._ctx.vpp_less_mem:
            top_triangles_chunks = [(vpp - 1 - v, v) for v in range(vpp // 2)]
            bot_triangles_chunks = [
                (vpp - 2 - v, v) for v in range((vpp - 1) // 2)
            ]
            print("top_triangles_chunks", top_triangles_chunks)
            print("bot_triangles_chunks", bot_triangles_chunks)
            for c0, c1 in top_triangles_chunks:
                overhead = (micro_left - stage_id) * abs(dyn(c0) - dyn(c1))
                max_overhead = max(max_overhead, overhead)
            for c0, c1 in bot_triangles_chunks:
                overhead = stage_id * abs(dyn(c0) - dyn(c1))
                max_overhead = max(max_overhead, overhead)
        # bigmem
        else:
            top_triangles_chunks = [
                (vpp - 1 - v, v + 1) for v in range((vpp - 1) // 2)
            ]
            diamond_chunks = [(vpp - 1 - v, v) for v in range(vpp // 2)]
            bot_triangles_chunks = [
                (vpp - 2 - v, v) for v in range((vpp - 1) // 2)
            ]
            print("top_triangles_chunks", top_triangles_chunks)
            print("diamond_chunks", diamond_chunks)
            print("bot_triangles_chunks", bot_triangles_chunks)
            for c0, c1 in top_triangles_chunks:
                overhead = (micro_left - stage_id) * abs(dyn(c0) - dyn(c1))
                max_overhead = max(max_overhead, overhead)
            for c0, c1 in diamond_chunks:
                overhead = (micro_left - stage_id) * abs(dyn(c0) - dyn(c1))
                max_overhead = max(max_overhead, overhead)
            for c0, c1 in bot_triangles_chunks:
                overhead = stage_id * abs(dyn(c0) - dyn(c1))
                max_overhead = max(max_overhead, overhead)
        return max_overhead

    def __stage_bwd_overhead_1f1b(
        self, stages: list, stage_id: int, record_lay_types: dict
    ) -> float:
        """1f1b end of warmup"""
        res = 0
        if stages[stage_id][self._ccfg.vp - 1]:
            last_node = self._fetch_node_and_switch_env(
                stages, record_lay_types, stage_id, -1, -1
            )
            # full rec -> not rec + grad
            # not rec -> grad
            if last_node == LayerType.FULL_REC_LAYER:
                self._ctx.current_lay_id = f"rec_{self._ctx.current_lay_id}"
                self._ctx.current_node = LayerType.NOT_REC_LAYER
                res = self._working_set()
            else:
                self._ctx.current_node = last_node
                self._ctx.current_lay_id = f"G_{self._ctx.current_lay_id}"
                res = self._working_set(on_saved=True)
                if (
                    last_node == LayerType.OUTPUT_LAYER
                    and self._ccfg.n_mtp > 0
                ):
                    last_mtp = self._fetch_node_and_switch_env(
                        stages, record_lay_types, stage_id, -1, -2
                    )
                    if last_mtp == LayerType.FULL_REC_LAYER:
                        self._ctx.current_lay_id = (
                            f"rec_{self._ctx.current_lay_id}"
                        )
                        self._ctx.current_node = LayerType.NOT_REC_LAYER
                    else:
                        self._ctx.current_lay_id = (
                            f"G_{self._ctx.current_lay_id}"
                        )
                        self._ctx.current_node = last_mtp
                    res += self._working_set(on_saved=last_mtp != LayerType.FULL_REC_LAYER)
        return res

    def __stage_bwd_overhead_zbv(
        self, stages: list, stage_id: int, record_lay_types: dict
    ) -> float:
        """dualpipeV end of warmup"""
        res = 0
        # Overlapping first/last chunks FWD/BWD
        # fwd first + bwd last
        fwd_first, bwd_last = 0, 0
        for lay_id, lay in enumerate(stages[stage_id][0]):
            self._ctx.current_node = lay
            result = self._fetch_node_and_switch_env(
                stages, record_lay_types, stage_id, 0, lay_id
            )
            if result is None:
                raise RuntimeError("_fetch_node_and_switch_env returned None.")
            fwd_first += sum(self._inner_dynamic_mem(default_micro_factor=1))
        for lay_id, lay in enumerate(stages[stage_id][1]):
            self._ctx.current_node = LayerType.NOT_REC_LAYER
            result = self._fetch_node_and_switch_env(
                stages, record_lay_types, stage_id, 1, lay_id
            )
            if result is None:
                raise RuntimeError("_fetch_node_and_switch_env returned None.")
            if lay == LayerType.FULL_REC_LAYER:
                bwd_last = max(bwd_last, self._working_set())
        # fwd last + bwd first
        fwd_last, bwd_first = 0, 0
        for lay_id, _ in enumerate(stages[stage_id][1]):
            self._ctx.current_node = LayerType.NOT_REC_LAYER
            result = self._fetch_node_and_switch_env(
                stages, record_lay_types, stage_id, 1, lay_id
            )
            if result is None:
                raise RuntimeError("_fetch_node_and_switch_env returned None.")
            fwd_last = sum(self._inner_dynamic_mem(default_micro_factor=1))
        for lay_id, lay in enumerate(stages[stage_id][0]):
            self._ctx.current_node = LayerType.NOT_REC_LAYER
            result = self._fetch_node_and_switch_env(
                stages, record_lay_types, stage_id, 0, lay_id
            )
            if result is None:
                raise RuntimeError("_fetch_node_and_switch_env returned None.")
            if lay == LayerType.FULL_REC_LAYER:
                bwd_first = max(bwd_first, self._working_set())

        # print(self.mb(overlap1),self.mb(overlap2))
        res = max(fwd_first + bwd_last, fwd_last + bwd_first)
        # res = 0
        return res
