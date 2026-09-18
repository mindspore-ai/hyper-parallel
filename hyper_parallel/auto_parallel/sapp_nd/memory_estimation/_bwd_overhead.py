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

from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType


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
        ccfg, ctx, hook = record_lay_types[(stage_id, chunk_id, lay_id)]
        # print("here-",id(ccfg))
        # print("here0",id(self._ccfg))
        # print("here00",id(self.backbone._ccfg))
        self._ccfg = ccfg
        self._ctx = ctx
        self._ctx.current_stage_id = stage_id
        self._ctx.current_chunk_id = chunk_id
        self._ctx.current_lay_id = lay_id
        # print("here1",id(self._ccfg))
        self.backbone.apply_hook(hook, ccfg=self._ccfg, ctx=self._ctx)
        # print("here11",id(self.backbone._ccfg))
        return stages[stage_id][chunk_id][lay_id]

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
                res = sum(self._inner_dynamic_mem(default_micro_factor=1))
            else:
                self._ctx.current_node = last_node
                self._ctx.current_lay_id = f"G_{self._ctx.current_lay_id}"
                res = sum(self._inner_dynamic_mem(default_micro_factor=1))
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
                    res += sum(self._inner_dynamic_mem(default_micro_factor=1))
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
                bwd_last = max(
                    bwd_last,
                    sum(self._inner_dynamic_mem(default_micro_factor=1)),
                )
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
                bwd_first = max(
                    bwd_first,
                    sum(self._inner_dynamic_mem(default_micro_factor=1)),
                )

        # print(self.mb(overlap1),self.mb(overlap2))
        res = max(fwd_first + bwd_last, fwd_last + bwd_first)
        # res = 0
        return res
