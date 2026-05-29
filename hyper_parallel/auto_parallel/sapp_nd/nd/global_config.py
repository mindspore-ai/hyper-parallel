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
from math import gcd

from hyper_parallel.auto_parallel.sapp_nd.nd.common.arch_hooks import CWrap, check_and_apply_custom_hook
from hyper_parallel.auto_parallel.sapp_nd.nd.logger import logger
import hyper_parallel.auto_parallel.sapp_nd.nd.dimensions as Dim
import hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware as Hard
import hyper_parallel.auto_parallel.sapp_nd.nd.balancing_adapter as BA


class GlobalConfig:
    """Union of cost model & parallel config"""

    def __init__(self, config, dimensions=None, mppb=False):

        self.wrap = CWrap(config)
        self.ccfg = self.wrap.ccfg

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

    def global_batch_size(self, parallel_config):
        """Compute global batch size from hyperparameters"""
        dp = self.dim_val(Dim.DP, parallel_config)
        pp = self.dim_val(Dim.PP, parallel_config)
        mb = self.dim_val(Dim.MBN, parallel_config)
        bs = self.dim_val(Dim.MBS, parallel_config)
        if pp > 1:
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

    def write(self, folder, parallel_config):
        """Dump config into a yaml file"""
        if folder:
            file_name = parallel_config.unique_name()
            self.ccfg.config.dump(file_name, folder)

    def moe_valid(self, parallel_config):
        """Check whether  the model is MoE"""
        expert_num = self.ccfg.n_exp
        if expert_num > 1:
            ep = self.dim_val(Dim.EP, parallel_config)
            dp = self.dim_val(Dim.DP, parallel_config)
            mp = self.dim_val(Dim.TP, parallel_config)
            logger.debug(
                "moe valid ? EP %d <= E %d & EP %d <= DP %d * MP %d",
                ep,
                expert_num,
                ep,
                dp,
                mp,
            )
            return ep <= min(expert_num, dp * mp)
        return True

    def make_parallel_config_args(self, **kwargs):
        """Create a parallel config from parallel values"""
        logger.debug("dimensions considered: %s", str(self.dimensions))

        dims = []
        # dims.append((Dim.DP, dp))
        for dim in self.dimensions:
            dims.append((dim, kwargs.get(dim.lname())))

        if (
            Dim.MBN not in self.dimensions
            and Dim.PP in self.dimensions
            and (Dim.DP in self.dimensions or Dim.MBS in self.dimensions)
        ):
            dims.append((Dim.MBN, kwargs.get(Dim.MBN.lname())))
            self.dimensions.append(Dim.MBN)
        return Dim.Dimensions(dims, all_dims=self.dimensions)

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

    def max_op(self, dp, tp, ep):
        """Compute bound for dimension OP"""
        if (
            isinstance(self.ccfg.optimizer, str)
            and "muon" not in self.ccfg.optimizer.lower()
        ):
            return dp
        if self.ccfg.n_exp and self.ccfg.n_exp > 1:
            exp_gcd = gcd(dp * tp // max(tp, ep), self.ccfg.n_exp)
        else:
            exp_gcd = dp

        if (
            self.ccfg.dc_kv
            and self.ccfg.dc_kv > 1
            and self.ccfg.dhr
            and self.ccfg.dhr > 1
        ):
            att_gcd = gcd(self.ccfg.h, self.ccfg.dc_kv + self.ccfg.dhr)
        else:
            att_gcd = self.ccfg.h
        return gcd(exp_gcd, att_gcd)
