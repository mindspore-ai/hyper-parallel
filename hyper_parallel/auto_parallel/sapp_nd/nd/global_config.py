# Copyright 2024 Huawei Technologies Co., Ltd
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
"""One configuration interface for parallelization"""

import copy
from typing import Any, Dict, Optional

from hyper_parallel.auto_parallel.sapp_nd.nd.common.arch_hooks import CWrap, check_and_apply_custom_hook
from hyper_parallel.auto_parallel.sapp_nd.nd.logger import logger
import hyper_parallel.auto_parallel.sapp_nd.nd.dimensions as Dim
import hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware as Hard
import hyper_parallel.auto_parallel.sapp_nd.nd.balancing_adapter as BA


class GlobalConfig:
    """Union of cost model & parallel config"""

    def __init__(self, config, dimensions=None, mppb=False, parent=None):

        self.wrap = CWrap(config)
        self.ccfg = self.wrap.ccfg
        # The other submodules of a multimodal model, which run on the same
        # stages as the one the search drives: each takes the candidate's
        # degrees, with its own layers placed and its own recompute adapted.
        self.siblings = self._pipeline_siblings(config, parent, mppb)

        if dimensions is not None:
            logger.debug("dimensions = %s", str(dimensions))
            self.dimensions = dimensions
        else:
            logger.debug("dimensions = %s", str(Dim.ALL_DIMS))
            self.dimensions = Dim.ALL_DIMS.copy()
        logger.debug("self.dimensions = %s", str(self.dimensions))
        logger.debug("layer_num_for_offset = %d", self.layer_num_for_offset())
        logger.debug("total layer num = %d", self.total_layer_num())
        self.balancing = BA.BalancingAdapter(
            self.layer_num_for_offset(),
            copy.deepcopy(self.ccfg.offset),
            copy.deepcopy(self.ccfg.full_rec),
            mppb,
        )

    def dim_val(self, dim, parallel_config):
        """Get the value of a parallel dimension"""
        if parallel_config.has_dim(dim):
            return parallel_config.val(dim)
        return dim.from_config(self.ccfg)

    def accumulates_grads(self) -> bool:
        """Whether the run accumulates gradients over micro-batches without pipeline parallelism."""
        return bool(getattr(self.ccfg, "accumulates_grads", False))

    def global_batch_size(self, parallel_config):
        """Compute global batch size from hyperparameters.

        Micro-batches multiply it under pipeline parallelism, and without it
        in a run that accumulates gradients over them.
        """
        dp = self.dim_val(Dim.DP, parallel_config)
        pp = self.dim_val(Dim.PP, parallel_config)
        mb = self.dim_val(Dim.MBN, parallel_config)
        bs = self.dim_val(Dim.MBS, parallel_config)
        if pp > 1 or self.accumulates_grads():
            logger.info("GBS = %dDP * %dMB * %dBS", dp, mb, bs)
            return dp * mb * bs
        logger.info("GBS = %dDP * %dBS", dp, bs)
        return dp * bs

    def layer_num_for_offset(self):
        """Compute layer number including MTP when necessary for offset"""
        layer_num = self.ccfg.n_lay
        if self.ccfg.emb_out_in_offset:
            layer_num += 2
        if self.ccfg.is_mtp_in_offset:
            layer_num += self.ccfg.n_mtp
        return layer_num

    def total_layer_num(self):
        """Compute total layer number, always including MTP"""
        layer_num = self.ccfg.n_lay + self.ccfg.n_mtp
        return layer_num

    def adapt_config_balancing(self, new_pp, new_vpp):
        """Adapt the layer-to-stage assignment to different PP"""
        logger.debug("new_pp=%d, new_vpp=%d", new_pp, new_vpp)

        new_recompute_config = self.balancing.treat_recompute(new_pp, new_vpp)
        logger.debug("adapted recompute config: %s", str(new_recompute_config))
        new_offset = self.balancing.treat_offset(new_pp, new_vpp)
        logger.debug("adapted offset: %s", str(new_offset))
        ok = self.balancing.offset_checker(new_pp, new_vpp, new_offset)
        if not ok:
            logger.error("Offset {%s} NOT VALID", str(new_offset))
        return new_offset, new_recompute_config

    def adapt_config(self, pp, vpp):
        """Adapt configuration to different parallel config"""
        return self.adapt_config_balancing(pp, vpp)

    def state_recompute(self, full: Optional[bool]) -> None:
        """Give every candidate *full* as its full recompute, on every config of its pipeline.

        Args:
            full: The full recompute a recompute dimension's mode states, or
                None for the recompute each candidate's balancing derives.
        """
        self.balancing.stated_recompute = full
        for _, balancing in self.siblings:
            if balancing is not None:
                balancing.stated_recompute = full

    def write(self, folder, parallel_config):
        """Dump config into a yaml file"""
        if folder:
            file_name = parallel_config.unique_name()
            self.ccfg.config.dump(file_name, folder)

    def moe_valid(self, parallel_config):
        """Whether a MoE model's EP fits its experts and the ranks they spread over (:meth:`max_ep`)."""
        expert_num = self.ccfg.n_exp
        if expert_num > 1:
            ep = self.dim_val(Dim.EP, parallel_config)
            dp = self.dim_val(Dim.DP, parallel_config)
            mp = self.dim_val(Dim.TP, parallel_config)
            cp = self.dim_val(Dim.CP, parallel_config)
            logger.debug(
                "moe valid ? EP %d <= E %d & EP %d <= %d, DP %d MP %d CP %d",
                ep,
                expert_num,
                ep,
                self.max_ep(dp, mp, cp),
                dp,
                mp,
                cp,
            )
            return ep <= min(expert_num, self.max_ep(dp, mp, cp))
        return True

    def max_ep(self, dp: int, tp: int, cp: int = 1) -> int:
        """Compute bound for dimension EP: the ranks a stage spreads its experts over.

        They are its DP x TP ranks, and under HyperParallel its DP x CP x TP
        ranks: it builds its expert mesh ``(edp_replicate, edp_shard, ep)``
        over the whole ``(dp, cp, tp)`` device mesh
        (``MeshContext._build_expert_parallel_mesh``), the domain its FSDP
        shard spans as well (``shard_spans_cp``).  Bounded by DP x TP, a
        strategy under CP could not spread its experts over CP's ranks,
        though the runtime does.
        """
        return dp * tp * (cp if getattr(self.ccfg, "shard_spans_cp", False) else 1)

    def ep_constraints_valid(self, parallel_config):
        """Check EP-specific divisibility constraints.

        Runs only for MoE models (n_exp > 1).  C1 ensures experts can be
        evenly partitioned across EP ranks; C2 ensures the expert FFN hidden
        dim can be evenly sharded by the expert TP degree.  Both checks use
        architecture constants from ``self.ccfg`` and the candidate values
        from ``parallel_config``.

        C3 (device count) is intentionally skipped here because the search
        loop borrows EP from the dp*tp budget, so dp*tp*pp*cp already
        equals total_devices by construction.

        Args:
            parallel_config: candidate ``Dim.Dimensions``.

        Returns:
            bool: True if all applicable EP constraints pass (or the model
            is dense), False otherwise.
        """
        if self.ccfg.n_exp <= 1:
            return True
        # pylint: disable=C0415
        from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.validators.ep_constraints import EpConstraints
        ep = self.dim_val(Dim.EP, parallel_config)
        r1 = EpConstraints.check_ep_divisibility(self.ccfg.n_exp, ep)
        if not r1:
            logger.warning("EP constraint C1 failed: %s", r1.message)
            return False
        tp = self.dim_val(Dim.TP, parallel_config)
        etp = max(getattr(self.ccfg, "etp", 0), 0)
        t_exp = max(etp, 1) if etp > 1 else max(tp, 1)
        hff_exp = max(getattr(self.ccfg, "hff_exp", 0), 0)
        r2 = EpConstraints.check_expert_hidden_divisibility(hff_exp, t_exp)
        if not r2:
            logger.warning("EP constraint C2 failed: %s", r2.message)
            return False
        # C4: an expert shard the run states divides its expert data-parallel
        # group, the stage's ranks over EP, as HyperParallel requires.  A run
        # that shards over the whole group takes each strategy's own.
        shard = getattr(self.ccfg, "expert_shard", None) or 1
        if ep > 1 and shard > 1 and not getattr(self.ccfg, "expert_shard_group", False):
            group = self.dim_val(Dim.DP, parallel_config) * self.dim_val(Dim.CP, parallel_config) * tp // ep
            if group % shard:
                logger.warning("EP constraint C4 failed: an expert shard of %d in a group of %d", shard, group)
                return False
        return True

    def make_parallel_config_args(self, **kwargs):
        """Create a parallel config from parallel values"""
        logger.debug("dimensions considered: %s", str(self.dimensions))

        dims = []
        # dims.append((Dim.DP, dp))
        for dim in self.dimensions:
            dims.append((dim, kwargs.get(dim.lname())))

        # The micro-batch count is the search's own, the global batch over DP
        # and MBS, whatever -l names. Kept from the yaml where -l left it out,
        # every DP but the yaml's own made another batch, so the search
        # refused every candidate or kept a single DP.
        if Dim.MBN not in self.dimensions:
            dims.append((Dim.MBN, kwargs.get(Dim.MBN.lname())))
            self.dimensions.append(Dim.MBN)
        return Dim.Dimensions(dims, all_dims=self.dimensions, accumulates=self.accumulates_grads())

    def make_parallel_config(self, dtpc_p, mbsn, evos_p):
        """Create a parallel config from parallel values"""
        logger.debug("dimensions considered: %s", str(self.dimensions))
        (dp, mp, pp, cp) = dtpc_p
        (mbs, mbn) = mbsn
        (ep, vpp, op, sp) = evos_p
        return self.make_parallel_config_args(
            dp=dp,
            mp=mp,
            pp=pp,
            cp=cp,
            mbs=mbs,
            mb=mbn,
            ep=ep,
            vpp=vpp,
            op=op,
            sp=sp,
        )

    def set_parallel_config(self, parallel_config):
        """Set a given parallel configuration in the config"""
        kwargs = {}
        ok = True
        new_pp = self.dim_val(Dim.PP, parallel_config)
        new_vp = self.dim_val(Dim.VPP, parallel_config)
        new_offset, new_recompute = self.adapt_config(new_pp, new_vp)
        kwargs["offset"] = new_offset
        kwargs["full_rec"] = new_recompute
        # kwargs["sel_rec"] = sel_rec
        for dim, value in parallel_config.dims_val.items():
            kwargs[dim.name.lower()] = value

        self.ccfg.set_strategy(**kwargs)
        self.siblings_take(kwargs)
        if not self.ccfg.multimodal:
            if not self.ccfg.hooks_dict:
                logger.info(
                    "'hook_cls' not specified,"
                    "search in predefined arch_hooks"
                )
                check_and_apply_custom_hook(self.ccfg)
            else:
                logger.info("Apply hooks")
                hook = list(self.ccfg.hooks_dict.values())[0]
                hook(self.wrap)

        return ok

    @staticmethod
    def _pipeline_siblings(config, parent, mppb):
        """Every other config of *config*'s pipeline: a multimodal parent's other submodules, and the parent.

        Each submodule comes with the balancing of its own layers, which
        adapts its recompute to a candidate's pipeline; the parent has no
        layers of its own and takes the degrees alone.
        """
        if parent is None or not parent.multimodal:
            return []
        siblings = []
        for name in parent.mm_order:
            sub = parent.mm_ccfgs[name]
            if sub is not config:
                siblings.append((sub, BA.BalancingAdapter(
                    sub.n_lay + sub.n_mtp,
                    copy.deepcopy(sub.offset),
                    copy.deepcopy(sub.full_rec),
                    mppb,
                )))
        return siblings + [(parent, None)]

    def siblings_take(self, kwargs: Dict[str, Any]) -> None:
        """Give every other config of a shared pipeline the degrees of *kwargs*.

        A vision tower is trained on the same devices as the language model
        the search drives, so it takes the candidate's degrees; its layers
        stay on the first stages, where the parser places them, and its
        recompute is adapted to the candidate's pipeline rather than the
        language model's, which counts other layers.  The parent, which
        holds the strategy its submodules share and combines their
        partitions, takes the degrees and no layer of its own.
        """
        degrees = {key: value for key, value in kwargs.items() if key not in ("offset", "full_rec")}
        for sibling, balancing in self.siblings:
            pipeline = (degrees.get("pp", sibling.p), degrees.get("vpp", sibling.vp))
            stated = {"offset": BA.front_loaded_offset(balancing.layers if balancing else 0, *pipeline)}
            if balancing is not None:
                stated["full_rec"] = balancing.treat_recompute(*pipeline)
            sibling.set_shared_strategy(**degrees, **stated)

    def space(self, dim, divide, reverse=False):
        """Generate the space for a given dimension"""
        if dim in self.dimensions:
            if dim.get_bound() is not None:
                logger.debug(
                    "Space of bounded dim %s is %s",
                    str(dim),
                    str(
                        Hard.all_divisors(
                            divide, reverse=reverse, max_bound=dim.get_bound()
                        )
                    ),
                )
                return Hard.all_divisors(
                    divide, reverse=reverse, max_bound=dim.get_bound()
                )
            logger.debug(
                "Space of dim %s is %s",
                str(dim),
                str(Hard.all_divisors(divide, reverse=reverse)),
            )
            return Hard.all_divisors(divide, reverse=reverse)
        logger.debug(
            "Space of original dim %s is [%s]",
            str(dim),
            str(dim.from_config(self.ccfg)),
        )
        return [dim.from_config(self.ccfg)]

    def range_space(self, dim, bound):
        """Generate the space for a given dimension"""
        if dim in self.dimensions:
            return range(1, bound + 1)
        return [dim.from_config(self.ccfg)]

    def bool_space(self, dim):
        """Generate the space for a given boolean dimension"""
        if dim in self.dimensions:
            return [False, True]
        return [dim.from_config(self.ccfg)]

    def max_op(self, dp, tp, ep, cp=1):
        """Compute bound for dimension OP.

        OP is the runtime's ``dp_shard_size``, and all the runtime asks of it
        is that the data-parallel group divide into a replicate axis and a
        shard axis (``distributed/mesh.py``, ``MeshContext.build_meshs``), so
        every divisor of DP is reachable.  HyperParallel's group is DP times
        CP, its FSDP domain, where a shard spans CP's ranks
        (``shard_spans_cp``): bounded by DP alone, a strategy under CP could
        not be sharded wider than DP, though the runtime shards it over the
        whole domain.

        Under Muon this used to narrow to a greatest common divisor over the
        expert count and the attention widths.  Nothing in the runtime asks
        for that: Newton-Schulz runs on a matrix's last two dimensions and the
        optimizer all-gathers them whenever the shard falls there, running
        locally when it does not (``core/optimizer/muon.py``,
        ``_classify_parameters_for_step``; ``core/optimizer/
        sharding_category.py``, ``is_last2d_sharded``), and a width that does
        not divide a parameter is only logged, never refused.  The cap came in
        with the original search import and cost the strategies engineers run:
        at 64 devices with EP 16 it allowed no shard wider than 4, and at 16
        devices with EP 16 none at all.  What Muon does change, one momentum
        per matrix in place of two moments, the parser carries as
        ``optimizer_states``.
        """
        del tp, ep  # the shard is bounded by the data-parallel group alone
        return dp * cp if getattr(self.ccfg, "shard_spans_cp", False) else dp
