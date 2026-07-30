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
"""EP constraint validators for auto-parallel strategy search.

Four constraints that any valid EP strategy must satisfy:
  C1: n_experts % ep_degree == 0  (expert divisibility)
  C2: hff_exp % t_exp == 0       (expert hidden dim divisibility, t_exp = etp or tp)
  C3: dp*tp*pp*cp <= total_devices  (device limit, EP borrows from DP)
  C4: EP+PP stage expert memory <= device_capacity  (stage feasibility)
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ConstraintResult:
    """Structured result for a constraint check."""

    name: str
    passed: bool
    message: str

    def __bool__(self):
        return self.passed


class EpConstraints:
    """EP-specific constraint checks for auto-parallel strategy search."""

    @staticmethod
    def check_ep_divisibility(n_exp: int, ep: int) -> ConstraintResult:
        """C1: n_experts must be divisible by ep_degree."""
        if ep < 1:
            return ConstraintResult(
                "ep_divisibility", False,
                f"ep={ep} must be >= 1")
        if n_exp < 1:
            return ConstraintResult(
                "ep_divisibility", False,
                f"n_exp={n_exp} must be >= 1")
        if n_exp % ep != 0:
            return ConstraintResult(
                "ep_divisibility", False,
                f"n_exp={n_exp} not divisible by ep={ep}, "
                f"remainder={n_exp % ep}")
        return ConstraintResult(
            "ep_divisibility", True,
            f"n_exp={n_exp}/ep={ep} = {n_exp // ep} experts/rank")

    @staticmethod
    def check_expert_hidden_divisibility(
        hff_exp: int, t_exp: int
    ) -> ConstraintResult:
        """C2: hff_exp must be divisible by expert TP degree.

        Args:
            hff_exp: Expert FFN hidden dimension.
            t_exp: Expert TP degree (= etp if etp > 1, else tp).
                   etp and tp are alternatives (not multiplicative).
        """
        if t_exp < 1:
            return ConstraintResult(
                "expert_hidden_divisibility", False,
                f"t_exp={t_exp} must be >= 1")
        if hff_exp <= 0:
            return ConstraintResult(
                "expert_hidden_divisibility", True,
                f"hff_exp={hff_exp} (dense FFN, no expert TP needed)")
        if hff_exp % t_exp != 0:
            return ConstraintResult(
                "expert_hidden_divisibility", False,
                f"hff_exp={hff_exp} not divisible by t_exp={t_exp}, "
                f"remainder={hff_exp % t_exp}")
        return ConstraintResult(
            "expert_hidden_divisibility", True,
            f"hff_exp={hff_exp}/t_exp={t_exp} = {hff_exp // t_exp}")

    @staticmethod
    def check_device_limit(
        dp: int, tp: int, pp: int, cp: int, total_devices: int
    ) -> ConstraintResult:
        """C3: dp*tp*pp*cp must not exceed total_devices.

        EP borrows devices from DP and does not occupy independent
        device slots, so it is not included in the device count.
        """
        required = dp * tp * pp * cp
        if required > total_devices:
            return ConstraintResult(
                "device_limit", False,
                f"dp={dp}*tp={tp}*pp={pp}*cp={cp}="
                f"{required} > total_devices={total_devices}")
        return ConstraintResult(
            "device_limit", True,
            f"{required}/{total_devices} devices used")

    @staticmethod
    def check_ep_pp_stage_feasibility(
        n_moe_layers: int,
        n_exp: int,
        ep: int,
        dp: int,
        h: int,
        hff_exp: int,
        bytes_p: int,
        device_capacity_gb: float,
        zero_level: int = 2,
        t_exp: int = 1,
        n_ffMM: int = 3,  # pylint: disable=invalid-name
    ) -> ConstraintResult:
        """C4: EP+PP stage expert memory must fit within device capacity.

        Args:
            n_moe_layers: Number of MoE layers in this PP stage.
            n_exp: Number of experts per MoE layer.
            ep: Expert parallelism degree.
            dp: Data parallelism degree (for ZeRO sharding).
            h: Model hidden dimension.
            hff_exp: Expert FFN hidden dimension (per expert).
            bytes_p: Bytes per parameter.
            device_capacity_gb: Device memory capacity in GB.
            zero_level: ZeRO optimizer sharding level (2 or 3).
            t_exp: Expert TP degree (= etp if etp > 1, else tp).
            n_ffMM: Number of feedforward linear layers per expert
                (SwiGLU: 3, standard MLP: 2).
        """
        if n_moe_layers <= 0:
            return ConstraintResult(
                "ep_pp_stage_feasibility", True,
                "Dense-only stage, no expert memory")
        if ep < 1:
            return ConstraintResult(
                "ep_pp_stage_feasibility", False,
                f"ep={ep} must be >= 1")
        experts_per_rank = n_exp / ep
        params_per_expert = n_ffMM * h * hff_exp / max(t_exp, 1)
        param_bytes = n_moe_layers * experts_per_rank * params_per_expert * bytes_p
        os_bytes = param_bytes * 2
        grad_bytes = param_bytes
        if zero_level >= 3:
            total = (param_bytes + os_bytes + grad_bytes) / max(dp, 1)
        elif zero_level == 2:
            total = param_bytes + os_bytes / max(dp, 1) + grad_bytes / max(dp, 1)
        else:
            total = param_bytes + os_bytes + grad_bytes
        total_gb = total / 1e9
        if total_gb > device_capacity_gb:
            return ConstraintResult(
                "ep_pp_stage_feasibility", False,
                f"Stage expert mem={total_gb:.1f}GB > capacity="
                f"{device_capacity_gb:.1f}GB "
                f"(exp/rank={experts_per_rank:.0f}, ZeRO-{zero_level})")
        return ConstraintResult(
            "ep_pp_stage_feasibility", True,
            f"Stage expert mem={total_gb:.1f}GB <= capacity="
            f"{device_capacity_gb:.1f}GB")

    @classmethod
    def validate_all(cls, ccfg, total_devices: int,
                     device_capacity_gb: float) -> list:
        """Run all EP constraint checks, return list of ConstraintResult."""
        t_exp = max(ccfg.etp, 1) if ccfg.etp > 1 else max(ccfg.t, 1)
        results = [
            cls.check_ep_divisibility(ccfg.n_exp, ccfg.ep),
            cls.check_expert_hidden_divisibility(ccfg.hff_exp, t_exp),
            cls.check_device_limit(
                ccfg.d, ccfg.t, ccfg.p, ccfg.cp, total_devices),
        ]
        if ccfg.p > 1 and ccfg.n_exp > 1:
            n_moe_per_stage = getattr(ccfg, 'n_lay', 1) // ccfg.p
            results.append(cls.check_ep_pp_stage_feasibility(
                n_moe_layers=n_moe_per_stage,
                n_exp=ccfg.n_exp,
                ep=ccfg.ep, dp=ccfg.d, h=ccfg.h,
                hff_exp=ccfg.hff_exp, bytes_p=ccfg.bytes_p,
                device_capacity_gb=device_capacity_gb,
                zero_level=int(ccfg.comm_d_exp), t_exp=t_exp,
                n_ffMM=max(getattr(ccfg, 'n_ffMM', 3), 1),
            ))
        return results
