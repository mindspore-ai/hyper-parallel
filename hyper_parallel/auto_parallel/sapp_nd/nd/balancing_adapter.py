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
"""Utilities for adapting balancing for a given pipeline configuration"""
from dataclasses import dataclass
import copy

from hyper_parallel.auto_parallel.sapp_nd.nd.logger import logger


@dataclass
class Pipeline:
    """Class storing a pipeline configuration"""

    pp: int
    vpp: int

    def __init__(self, pp, vpp):
        self.pp = pp
        self.vpp = vpp

    def __str__(self):
        return "PP: " + str(self.pp) + ", VPP: " + str(self.vpp)

    def chunk_stage(self):
        """Product of chunks and stages"""
        return self.pp * self.vpp


def infer_pp_and_vpp(offset):
    """Return a pipeline configuration inferred from an offset"""
    if is_zero_d(offset):
        return Pipeline(1, 1)
    if is_one_d(offset):
        return Pipeline(len(offset), 1)
    if is_two_d(offset):
        return Pipeline(len(offset[0]), len(offset))
    raise TypeError(f"Offset {offset} has a wrong type!")


class BalancingAdapter:
    """Adapt pipeline balancing to a given 'new' pipeline configuration"""

    layers: int
    prev_pip: Pipeline

    def __init__(self, layers, offset, recompute, manual_ppb):
        self.layers = layers
        self.prev_offset = offset
        self.prev_recompute = recompute
        self.from_config = manual_ppb
        logger.debug(
            "init: layers = %d, from_config = %s, offset = %s, recompute = %s",
            layers,
            str(self.from_config),
            str(offset),
            str(recompute),
        )
        self.prev_pip = infer_pp_and_vpp(offset)

    def treat_pp_list(self, new_pip, stages):
        """Treat 1D offset or recompute config"""
        current_pp_len = len(stages)
        logger.debug(
            "new pp (%d) = prev pp (%d)? l = %s, len(l) = %d",
            new_pip.pp,
            self.prev_pip.pp,
            str(stages),
            len(stages),
        )
        if new_pip.pp == current_pp_len:
            return stages
        new_l = []
        if new_pip.pp > current_pp_len:
            while new_pip.pp % len(stages) != 0:
                stages.insert(len(stages) // 2, 0)
            logger.debug("stages: %s", str(stages))
            factor = new_pip.pp // current_pp_len
            for s in stages:
                rest = s % factor
                for _ in range(factor):
                    if rest > 0:
                        new_l.append(s // factor + 1)
                        rest -= 1
                    else:
                        new_l.append(s // factor)
        if new_pip.pp < current_pp_len:
            for _ in range(current_pp_len % new_pip.pp):
                stages.insert(len(stages) // 2, 0)
            factor = current_pp_len // new_pip.pp
            for i in range(new_pip.pp):
                total_rec_layers = 0
                for j in range(factor):
                    total_rec_layers += stages[i * factor + j]
                new_l.append(total_rec_layers)
        return new_l

    def treat_vpp_list(self, new_pip, ll):
        """Treat 2D offset or recompute config"""
        logger.debug("treat_vpp_list start: ll = %s", str(ll))
        prev_vpp = len(ll)
        if prev_vpp < new_pip.vpp:
            if prev_vpp == 1:
                ll.append([0] * self.prev_pip.pp)
                ll[1][-1] = ll[0][-1]
                ll[0][-1] = 0
                prev_vpp = 2
            for _ in range(new_pip.vpp - prev_vpp):
                ll.insert(prev_vpp // 2, [0] * self.prev_pip.pp)
            logger.debug("B: ll = %s", str(ll))

            prev_layer_per_stage = [
                prev_vpp * (self.layers // self.prev_pip.chunk_stage())
                + sum(stages[p] for stages in make_two_d(self.prev_offset))
                for p in range(self.prev_pip.pp)
            ]
            new_layer_per_stage = [
                new_pip.vpp * (self.layers // self.prev_pip.chunk_stage())
                + sum(stages[p] for stages in ll)
                for p in range(self.prev_pip.pp)
            ]
            logger.debug(
                "prev_layer_per_stage = %s", str(prev_layer_per_stage)
            )
            logger.debug("new_layer_per_stage = %s", str(new_layer_per_stage))
            for s in range(self.prev_pip.pp):
                for v in range(new_pip.vpp):
                    if (
                        prev_layer_per_stage[s] - new_layer_per_stage[s] > 0
                        and ll[v][s] < 1
                    ):
                        ll[v][s] += 1
                        new_layer_per_stage[s] += 1

        elif prev_vpp > new_pip.vpp:
            for _ in range(prev_vpp - new_pip.vpp):
                ll[-2] = [sum(x) for x in zip(ll[-1], ll[-2])]
                del ll[-1]
        logger.debug("C: ll = %s", str(ll))
        new_vpp_list = []
        for stages in ll:
            new_vpp_list.append(self.treat_pp_list(new_pip, stages))
        logger.debug("D: new_vpp_list = %s", str(new_vpp_list))
        return new_vpp_list

    def treat_recompute_list(self, new_pip, recompute):
        """Treat recompute config recursively"""
        if all(isinstance(x, int) for x in recompute):
            if all(isinstance(x, int) for x in recompute):
                if new_pip.vpp == 1:
                    return self.treat_pp_list(new_pip, recompute)
                return self.treat_vpp_list(new_pip, recompute)
        elif all(isinstance(x, int) for stages in recompute for x in stages):
            if new_pip.vpp == 1:
                return self.treat_vpp_list(new_pip, recompute)[0]
            return self.treat_vpp_list(new_pip, recompute)
        return recompute

    def treat_recompute(self, new_pp, new_vpp):
        """Treat recompute config for the new given Pipeline config"""
        new_pip = Pipeline(new_pp, new_vpp)
        if not self.from_config:
            return self.default_recompute(
                new_pip, copy.deepcopy(self.prev_recompute)
            )
        if new_pip == self.prev_pip or isinstance(self.prev_recompute, bool):
            return copy.deepcopy(self.prev_recompute)
        return self.treat_recompute_list(
            new_pip, copy.deepcopy(self.prev_recompute)
        )

    def treat_offset(self, new_pp, new_vpp):
        """Treat offset for the new given Pipeline config"""
        new_pip = Pipeline(new_pp, new_vpp)
        if not self.from_config:
            return self.make_valid(new_pip, self.default_offset(new_pip))
        offset = copy_offset(self.prev_offset)
        if new_pip == self.prev_pip:
            logger.debug("Same pipeline, no offset change")
            return offset
        logger.debug(
            "change offset %s from PP = %d, VPP = %d, to PP = %d, VPP = %d",
            str(offset),
            self.prev_pip.pp,
            self.prev_pip.vpp,
            new_pip.pp,
            new_pip.vpp,
        )

        if is_zero_d(offset):
            offset = []
            for _ in range(new_pip.vpp):
                offset.append([])
                for _ in range(new_pip.pp):
                    offset[-1].append(0)
            return self.make_valid(new_pip, offset)
        if is_one_d(offset):
            if new_pip.vpp == 1:
                return self.make_valid(
                    new_pip, self.treat_pp_list(new_pip, offset)
                )
            return self.make_valid(
                new_pip, self.treat_vpp_list(new_pip, [offset])
            )
        if is_two_d(offset):
            return self.make_valid(
                new_pip, self.treat_vpp_list(new_pip, offset)
            )
        raise TypeError(f"Offset {offset} has a wrong type!")

    def check_offset(self, new_pip, offset):
        """Check offset validity"""
        if is_zero_d(offset):
            return (new_pip.pp == 1) and (new_pip.vpp == 1)
        if is_one_d(offset):
            return (
                len(offset) == new_pip.pp
                and sum(offset) == self.layers % new_pip.pp
            )
        if is_two_d(offset):
            return (
                len(offset) == new_pip.vpp
                and all(len(stages) == new_pip.pp for stages in offset)
                and sum(sum(stages) for stages in offset)
                == self.layers % (new_pip.chunk_stage())
            )
        return False

    def offset_checker(self, new_pp, new_vpp, offset):
        """Log an error message if offset is invalid"""
        new_pip = Pipeline(new_pp, new_vpp)
        if not self.check_offset(new_pip, offset):
            logger.error(
                "offset %s is wrong!! pp = %d, vpp = %d, L = %d",
                str(offset),
                new_pip.pp,
                new_pip.vpp,
                self.layers,
            )
            return False
        return True

    def make_valid(self, new_pip, offset):
        """Transform an invalid offset into a valid one"""
        logger.debug("offset to make valid = %s", str(offset))
        flat = make_one_d(offset)

        delta = self.layers % (new_pip.chunk_stage()) - sum(flat)
        logger.debug("delta = %s", str(delta))
        if delta < 0:
            for _ in range(-delta):
                top = max(flat)
                for i, _ in enumerate(flat):
                    if flat[i] == top:
                        flat[i] = flat[i] - 1
                        break
        elif delta > 0:
            for _ in range(delta):
                bot = min(flat)
                for i, _ in enumerate(flat):
                    # for i, _ in reversed(list(enumerate(flat))):
                    if flat[i] == bot:
                        flat[i] = flat[i] + 1
                        break
        logger.debug("valid flat offset = %s", str(flat))

        # cases where sum > (new_pip.vpp * new_pip.pp)
        delta = sum(flat) // (new_pip.chunk_stage())
        for i, _ in enumerate(flat):
            flat[i] -= delta

        logger.debug("new flat offset = %s", str(flat))
        if new_pip.vpp == 1:
            return flat

        for v in range(new_pip.vpp):
            for s in range(new_pip.pp):
                offset[v][s] = flat[v * new_pip.pp + s]
        return offset

    def default_offset(self, new_pip):
        """Construct default offset"""
        rest = self.layers % (new_pip.chunk_stage())
        offset = []
        for v in range(new_pip.vpp):
            offset.insert(0, [])
            for s in range(new_pip.pp):
                if rest > 0 and v > 0 and s > 0:
                    offset[0].insert(0, 1)
                    rest -= 1
                else:
                    offset[0].insert(0, 0)
        if offset[0][0] == 1:
            offset[0][0] = 0
            offset[-1][-1] = 1
        return offset

    def default_recompute(self, _, recompute_cfg):
        """Construct Default recompute config when not taken from config"""
        logger.debug(
            "recompute config has type %s and value %s",
            type(recompute_cfg),
            str(recompute_cfg),
        )
        return True


def copy_offset(offset):
    """Copy an offset"""
    if is_zero_d(offset):
        return offset
    return copy.deepcopy(offset)


def is_zero_d(offset):
    """Check if offset is an int"""
    return isinstance(offset, int)


def is_one_d(offset):
    """Check if offset is an int list"""
    return all(isinstance(x, int) for x in offset)


def is_two_d(offset):
    """Check if offset is an int list list"""
    return all(isinstance(x, int) for stages in offset for x in stages)


def make_one_d(offset):
    """Transform offset is an int list"""
    if is_zero_d(offset):
        return [0]
    if is_one_d(offset):
        return offset
    if is_two_d(offset):
        return [x for stages in offset for x in stages]
    raise TypeError(f"Offset {offset} has a wrong type!")


def make_two_d(offset):
    """Transform offset is an int list list"""
    if is_zero_d(offset):
        return [[0]]
    if is_one_d(offset):
        return [offset]
    if is_two_d(offset):
        return offset
    raise TypeError(f"Offset {offset} has a wrong type!")
