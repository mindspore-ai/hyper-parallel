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
"""generate pipeline partitions"""
from paradise.common._cost_model_variables import _CostModVar
from paradise.common.layer_type import LayerType


class PartitionGenerator(_CostModVar):
    """partition generator class"""

    def combine_partition_multimodal(self, mm_stages: dict):
        """Call after generate_partitions
        Assuming same PP and VPP degree for each mod
        Assuming offsets are well set and compatible
        (otherwise need to preprocess offsets for module compatibility)
        """
        combined = []
        for idx in range(self.p):
            combined_stages = []
            for v_idx in range(self.vp):
                combined_chunks = []
                for m in self.mm_order:
                    combined_chunks += mm_stages[m][idx][v_idx]
                combined_stages += [combined_chunks]
            combined += [combined_stages]
        return combined

    def generate_partitions_vpp(self):
        """infer layers arrangement"""
        if not self.multimodal:
            return self.generate_partitions_vpp_unimodal()
        lpartitions = {}
        for k, v in self.mm_ccfgs.items():
            lpartitions[k] = v.generate_partitions_vpp_unimodal()
        return lpartitions

    def __process_select_recompute(self, *args):
        """Build layer types for selective recompute partitions."""
        args = dict(
            zip(
                [
                    "s_idx",
                    "v_idx",
                    "num_assigned",
                    "num_layer_per_stage",
                ],
                args,
            )
        )
        chunk = []
        sel_rec = self.sel_rec
        if isinstance(sel_rec, list):
            if isinstance(sel_rec[0], str):
                sel_rec = [
                    self.n_lay // self.p
                    for _ in range(self.p)
                ]  # Operator names list treated as True
            if self.vp > 1:
                if isinstance(sel_rec[0], int):
                    # Preprocess input: Even recomputed layers
                    # distribution throughout chunks
                    sel_rec = [
                        [
                            (
                                s // self.vp
                                + (1 if v_idx < s % self.vp else 0)
                            )
                            for s in sel_rec
                        ]
                        for v_idx in range(self.vp)
                    ]
                if isinstance(sel_rec[0][0], str):
                    sel_rec = [
                        [
                            args["num_layer_per_stage"]
                            for _ in range(self.p)
                        ]
                        for _ in range(self.vp)
                    ]
                    # Operator names list treated
                    # as True (chunk level)
                chunk = [
                    (
                        LayerType.SEL_REC_LAYER
                        if j < sel_rec[args["v_idx"]][args["s_idx"]]
                        else LayerType.NOT_REC_LAYER
                    )
                    for j in range(args["num_assigned"])
                ]  # Assuming list of list
            else:
                chunk = [
                    (
                        LayerType.SEL_REC_LAYER
                        if j < sel_rec[args["s_idx"]]
                        else LayerType.NOT_REC_LAYER
                    )
                    for j in range(args["num_assigned"])
                ]
        else:
            chunk = [
                LayerType.SEL_REC_LAYER for _ in range(args["num_assigned"])
            ]
        return chunk

    def __process_full_recompute(self, *args):
        """Build layer types for full recompute partitions."""
        args = dict(
            zip(["s_idx", "v_idx", "num_assigned"], args)
        )
        chunk = []
        full_rec = self.full_rec
        if isinstance(full_rec, list):
            if self.vp > 1:
                if isinstance(full_rec[0], int):
                    # Preprocess input: Even recomputed layers
                    # distribution throughout chunks
                    full_rec = [
                        [
                            (
                                f // self.vp
                                + (1 if v_idx < f % self.vp else 0)
                            )
                            for f in full_rec
                        ]
                        for v_idx in range(self.vp)
                    ]
                chunk = [
                    (
                        LayerType.FULL_REC_LAYER
                        if j < full_rec[args["v_idx"]][args["s_idx"]]
                        else LayerType.NOT_REC_LAYER
                    )
                    for j in range(args["num_assigned"])
                ]  # Assuming list of list
            else:
                chunk = [
                    (
                        LayerType.FULL_REC_LAYER
                        if j < full_rec[args["s_idx"]]
                        else LayerType.NOT_REC_LAYER
                    )
                    for j in range(args["num_assigned"])
                ]
        else:
            chunk = [
                LayerType.FULL_REC_LAYER for _ in range(args["num_assigned"])
            ]
        return chunk

    def __process_offset(self, *args):
        """Compute assigned layer count after applying PP offsets."""
        args = dict(
            zip(
                [
                    "s_idx",
                    "v_idx",
                    "lay",
                    "num_layer_per_stage",
                ],
                args,
            )
        )
        num_assigned = 0
        # try:
        if isinstance(self.offset, list):
            if isinstance(self.offset[0], list):  # VPP
                num_assigned = min(
                    args["lay"],
                    args["num_layer_per_stage"]
                    + self.offset[args["v_idx"]][args["s_idx"]],
                )  # Assuming list of list
            else:
                # add extra to last chunk
                if args["v_idx"] < self.vp - 1:
                    num_assigned = min(
                        args["lay"],
                        args["num_layer_per_stage"]
                        + self.offset[args["s_idx"]] // self.vp,
                    )
                else:
                    num_assigned = min(
                        args["lay"],
                        args["num_layer_per_stage"]
                        + self.offset[args["s_idx"]] // self.vp
                        + (self.offset[args["s_idx"]] % self.vp),
                    )
        # print(args["num_layer_per_stage"], args["lay"])
        # print(f"num_assigned for stage {args['s_idx']} chunk {args['v_idx']}: {num_assigned}")
        return num_assigned

    def generate_partitions_vpp_unimodal(self):
        """infer layers arrangement for a module"""
        lay = self.n_lay
        if self.is_mtp_in_offset:
            lay += self.n_mtp
        if self.emb_out_in_offset:
            lay += 2
        num_layer_per_stage = lay // self.p // self.vp
        # print("lay",lay,"num_layer_per_stage",num_layer_per_stage, self.emb_out_in_offset)
        num_layer_per_stage = max(
            int(self.config_format == "json"), num_layer_per_stage
        )
        partitions = []
        for idx in range(self.p):
            partitions.append([])
            for v_idx in range(self.vp):
                # Process offset
                if self.offset:
                    num_assigned = self.__process_offset(
                        idx, v_idx, lay, num_layer_per_stage
                    )
                else:
                    num_assigned = min(lay, num_layer_per_stage)

                # Process full recompute
                if self.full_rec:
                    partitions[-1] += [
                        self.__process_full_recompute(
                            idx, v_idx, num_assigned
                        )
                    ]
                # Process select recompute
                elif self.sel_rec:
                    partitions[-1] += [
                        self.__process_select_recompute(
                            idx,
                            v_idx,
                            num_assigned,
                            num_layer_per_stage,
                        )
                    ]
                else:
                    partitions[-1] += [
                        [LayerType.NOT_REC_LAYER for _ in range(num_assigned)]
                    ]
                lay -= num_assigned

        partitions = self.insert_emb_out_partitions(partitions)
        return partitions

    def insert_emb_out_partitions(self, partitions):
        """end of generation"""
        sched = self.pp_sched
        if sched == "zero_bubble_v":
            return self.insert_emb_out_partitions_zbv(partitions)
        return self.insert_emb_out_partitions_1f1b(partitions)

    def insert_emb_out_partitions_zbv(self, partitions):
        """Emb and Out"""
        first, last = self.first_and_last_non_empty_stage(partitions)
        if not self.is_mtp_in_offset:
            last_lay = partitions[last[0]][last[1]][-1]
            partitions[last[0]][last[1]] += [last_lay] * self.n_mtp
        if not self.emb_out_in_offset:
            partitions[first[0]][first[1]].insert(0, LayerType.EMBEDDING_LAYER)
            partitions[last[0]][last[1]].append(LayerType.OUTPUT_LAYER)
        else:
            partitions[first[0]][first[1]][0] = LayerType.EMBEDDING_LAYER
            partitions[last[0]][last[1]][-1] = LayerType.OUTPUT_LAYER
        return partitions

    def insert_emb_out_partitions_1f1b(self, partitions):
        """Insert embedding, output, and MTP layers for 1F1B schedules."""
        first, last = self.first_and_last_non_empty_stage(partitions)
        if not self.is_mtp_in_offset:
            last_lay = partitions[last[0]][last[1]][-1]
            partitions[last[0]][last[1]] += [last_lay] * self.n_mtp
        if not self.emb_out_in_offset:
            partitions[first[0]][first[1]].insert(0, LayerType.EMBEDDING_LAYER)
            partitions[last[0]][last[1]].append(LayerType.OUTPUT_LAYER)
        else:
            partitions[first[0]][first[1]][0] = LayerType.EMBEDDING_LAYER
            partitions[last[0]][last[1]][-1] = LayerType.OUTPUT_LAYER
        return partitions

    def first_and_last_non_empty_stage(self, partitions):
        """Return the first and last non-empty stage/chunk indexes."""
        # Useful for submodule partitioning in multimodals
        first, last = None, None
        for chunk_id in range(self.vp):
            for stage_id in range(self.p):
                if partitions[stage_id][chunk_id]:
                    last = (stage_id, chunk_id)
                    if not first:
                        first = (stage_id, chunk_id)
                elif first and last:
                    return first, last
        if not first:
            first = (0, 0)
        if not last:
            last = (self.p - 1, self.vp - 1)
        return first, last

    # def insert_emb_out_partitions_1f1b(self, partitions):
    #     """Emb and Out"""
    #     # Insert emb in the first non empty chunk
    #     put_emb = None
    #     for chunk_id in range(self.vp):
    #         for stage_id in range(self.p):
    #             if partitions[stage_id][chunk_id]:
    #                 partitions[stage_id][chunk_id].insert(
    #                     0, LayerType.EMBEDDING_LAYER
    #                 )
    #                 put_emb = (stage_id, chunk_id)
    #                 break
    #         if put_emb:
    #             break
    #     # Insert out in the last non empty chunk
    #     if self.n_lay > 1:
    #         last_layer = None
    #         for chunk_id in range(self.vp - 1, -1, -1):
    #             for stage_id in range(self.p - 1, -1, -1):
    #                 if partitions[stage_id][chunk_id]:
    #                     last_layer = (stage_id, chunk_id)
    #                     break
    #             if last_layer:
    #                 break
    #         if not last_layer:
    #             last_layer = (self.p - 1, self.vp - 1)
    #         if not self.is_mtp_in_offset:
    #             # duplicate last_layer for mtp
    #             partitions[self.p - 1][self.vp - 1] += [
    #                 partitions[last_layer[0]][last_layer[1]][-1]
    #             ] * self.n_mtp
    #             last_layer = (self.p - 1, self.vp - 1)
    #         partitions[last_layer[0]][last_layer[1]].append(
    #             LayerType.OUTPUT_LAYER
    #         )
    #         return partitions
    #     if not self.is_mtp_in_offset:
    #         partitions[put_emb[0]][put_emb[1]] += [
    #             partitions[put_emb[0]][put_emb[1]][-1]
    #         ] * self.n_mtp
    #     partitions[put_emb[0]][put_emb[1]].append(LayerType.OUTPUT_LAYER)
    #     return partitions
