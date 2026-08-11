# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
# Standalone MHC Module - Extracted from Sophon-Pytorch
# This module contains the core MHC (Manifold-constrained Hyper Connection) algorithms
# with hardware acceleration features preserved, but with all distributed/memory
# optimization logic removed.

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Conditional NPU imports
try:
    import torch_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False

try:
    import omni_training_custom_ops  # noqa: F401
    HAS_CUSTOM_OPS = True
except (ImportError, Exception):
    HAS_CUSTOM_OPS = False

if TYPE_CHECKING:
    from hyper_parallel.models.vl_moe.model import (
        VLTextConfig,
    )


# ============================================================================
# NPU Custom Autograd Functions (AscendC backends)
# ============================================================================

class MhcPreCustomOp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        phi: torch.Tensor,
        alpha: torch.Tensor,
        bias: torch.Tensor,
        gamma: torch.Tensor = None,
        out_flag: bool = True,
        norm_eps: float = 1e-6,
        hc_eps: float = 1e-6
    ):
        hin, h_post, h_comb_before, inv_rms, h_mix, h_pre = \
            torch.ops.custom.npu_manifold_constrained_hyper_connection_pre(x, phi, alpha, bias, gamma=gamma,
            out_flag=out_flag, norm_eps=norm_eps, hc_eps=hc_eps)
        ctx.save_for_backward(
            x, phi, alpha, gamma, h_post, inv_rms, h_mix, h_pre
        )
        ctx.hc_eps = hc_eps
        return hin, h_pre, h_post, h_comb_before, x

    @staticmethod
    def backward(ctx, dh_in, dh_pre, dh_post, dh_res, dh_x):
        x, phi, alpha, gamma, h_post, inv_rms, h_mix, h_pre = ctx.saved_tensors
        hc_eps = ctx.hc_eps
        dx, dphi, dalpha, dbias, dgamma = torch.ops.custom.npu_manifold_constrained_hyper_connection_pre_grad(
            x, phi, alpha, dh_in, dh_post, dh_res,
            inv_rms, h_mix, h_pre, h_post, gamma=gamma, hc_eps=hc_eps, grad_x_post=dh_x
        )
        grads = [dx, dphi, dalpha, dbias, dgamma, None, None, None]
        return tuple(grads)


class MhcPostCustomOp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        h_res,
        h_out,
        h_post
    ):

        ctx.save_for_backward(x, h_res, h_out, h_post)
        output = torch.ops.custom.npu_ai_infra_manifold_constrained_hyper_connection_post(
            x, h_res, h_out, h_post
        )
        return output

    @staticmethod
    def backward(
        ctx,
        grad_output
    ):
        x, h_res, h_out, h_post = ctx.saved_tensors

        grad_x, grad_h_res, grad_h_out, grad_h_post = \
            torch.ops.custom.npu_ai_infra_mhc_post_grad(
            grad_output.contiguous(), x, h_res, h_out, h_post
        )
        return grad_x, grad_h_res, grad_h_out, grad_h_post


class SinkhornCustomop(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        h_res: torch.Tensor,
        sinkhorn_iters: int = 20,
        eps: float = 1e-6
    ):
        output, norm_out, sum_out = torch.ops.custom.npu_sinkhorn(h_res, out_flag=1, eps=eps, num_iters=sinkhorn_iters)

        ctx.save_for_backward(
            norm_out, sum_out
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        norm_out, sum_out = ctx.saved_tensors
        grad_h_res = torch.ops.custom.npu_sinkhorn_grad(grad_output, norm_out, sum_out)
        grads = [grad_h_res, None, None]
        return tuple(grads)


# ============================================================================
# Core Algorithm Functions
# ============================================================================

def sinkhorn_knopps(h_res, sinkhorn_iters, eps):
    h_res = h_res.softmax(-1) + eps
    col_sum = h_res.sum(-2, keepdim=True)
    h_res = h_res / (col_sum + eps)
    for _ in range(sinkhorn_iters - 1):
        row_sum = h_res.sum(-1, keepdim=True)
        h_res = h_res / (row_sum + eps)
        col_sum = h_res.sum(-2, keepdim=True)
        h_res = h_res / (col_sum + eps)
    return h_res


def hc_split_sinkhorn_torch(
    weight: torch.Tensor,
    branch_alpha: torch.Tensor,
    branch_beta: torch.Tensor,
    num_stream: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple:
    h_pre, h_post, h_res = weight.split([num_stream, num_stream, num_stream * num_stream], dim=-1)
    h_res = h_res.unflatten(-1, (num_stream, num_stream))

    hpre_input_alpha = h_pre * branch_alpha[0]
    hpre_input_beta = branch_beta[:num_stream].unsqueeze(0).unsqueeze(0)
    h_pre = F.sigmoid(hpre_input_alpha + hpre_input_beta) + eps

    hpost_input_alpha = h_post * branch_alpha[1]
    hpost_input_beta = branch_beta[num_stream:2 * num_stream].unsqueeze(0).unsqueeze(0)
    h_post = 2 * F.sigmoid(hpost_input_alpha + hpost_input_beta)

    hres_input_alpha = h_res * branch_alpha[2]
    hres_input_beta = branch_beta[2 * num_stream:].view(num_stream, num_stream).unsqueeze(0).unsqueeze(0)
    h_res = hres_input_alpha + hres_input_beta

    h_res = sinkhorn_knopps(h_res, sinkhorn_iters, eps)

    return h_pre, h_post, h_res


# ============================================================================
# Standalone MHC Modules
# ============================================================================

class MhcPreModule(nn.Module):
    def __init__(
        self,
        config: VLTextConfig,
        layer_number: int = 1,
    ):
        super().__init__()
        self.config = config
        self.num_stream = config.mhc_num_stream
        self.layer_number = layer_number

        self.phi = nn.Linear(
            config.hidden_size * self.num_stream,
            (self.num_stream + 2) * self.num_stream,
            bias=False,
        )
        self.phi._init_role = "input"
        if config.perform_initialization:
            config._standalone_init_weights(self.phi)
        self.branch_alpha = nn.Parameter(torch.ones(3) * config.mhc_init_alpha)

        init_residual_index = self.layer_number % self.num_stream
        pre_branch_beta = torch.zeros(self.num_stream)
        pre_branch_beta[init_residual_index] = 1
        post_branch_beta = torch.zeros(self.num_stream)
        residual_beta = torch.eye(self.num_stream)
        branch_beta = torch.concat((pre_branch_beta, post_branch_beta, residual_beta.reshape(-1)))
        self.branch_beta = nn.Parameter(branch_beta)

        if config.mhc_use_gamma:
            self.norm_gamma = nn.Parameter(torch.ones(config.hidden_size * self.num_stream) *
                config.mhc_init_gamma)

        self.hc_eps = config.hc_eps
        self.norm_eps = self.hc_eps

    def forward(self, x):
        if self.config.use_mhc_ascendc_pre:
            x_shape, dtype = x.size(), x.dtype
            x = x.reshape(x_shape[0], x_shape[1], self.num_stream, -1)
            gamma = None
            if self.config.mhc_use_gamma:
                gamma = self.norm_gamma.reshape(self.num_stream, -1).float()

            y, h_pre, h_post, h_comb_before, residual = MhcPreCustomOp.apply(
                x, self.phi.weight.float(), self.branch_alpha.float(),
                self.branch_beta.float(), gamma, True, self.norm_eps, self.hc_eps)
            h_res = SinkhornCustomop.apply(h_comb_before, self.config.mhc_recur_norm, self.hc_eps)

            residual = residual.reshape(x_shape[0], x_shape[1], -1)

        else:
            shape, dtype = x.size(), x.dtype
            x = x.float()

            rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
            if self.config.mhc_use_gamma:
                weight = F.linear(x * rsqrt * self.norm_gamma, self.phi.weight)
            else:
                weight = F.linear(x, self.phi.weight) * rsqrt
            h_pre, h_post, h_res = hc_split_sinkhorn_torch(
                weight, self.branch_alpha, self.branch_beta,
                self.num_stream, self.config.mhc_recur_norm, self.hc_eps)

            if self.config.mhc_hpre_renorm:
                eps_cache = torch.full(
                    (),
                    1e-30,
                    dtype=h_pre.dtype,
                    device=h_pre.device
                )
                h_pre = h_pre / h_pre.sum(dim=-1, keepdim=True).maximum(eps_cache)

            y = torch.sum(h_pre.unsqueeze(-1) * x.unflatten(dim=-1, sizes=(self.num_stream, -1)), dim=2).to(dtype)

        if self.config.use_mhc_ascendc_pre:
            return y, h_post, h_res, residual
        else:
            return y, h_post, h_res, None


class MhcPostModule(nn.Module):
    def __init__(self, config: VLTextConfig):
        super().__init__()
        self.config = config
        self.num_stream = config.mhc_num_stream

    def forward(self, x, residual, h_post, h_res):
        if self.config.use_mhc_ascendc_post:
            x_shape = x.size()
            residual_reshape = residual.reshape(x_shape[0], x_shape[1], self.num_stream, -1)
            y_flat = MhcPostCustomOp.apply(residual_reshape, h_res, x, h_post)
            y_flat = y_flat.flatten(2)
            return y_flat
        y = (h_post.unsqueeze(-1) * x.unsqueeze(-2) +
            torch.sum(h_res.unsqueeze(-1) *
            residual.unflatten(dim=-1, sizes=(self.num_stream, -1)).unsqueeze(-2), dim=2)
            ).flatten(2)
        return y.type_as(x)


class MhcPostProcessModule(nn.Module):
    def __init__(
        self,
        config: VLTextConfig,
        layer_number: int = 1,
    ):
        super().__init__()
        self.config = config
        self.num_stream = config.mhc_num_stream
        self.layer_number = layer_number

        self.phi = nn.Linear(
            config.hidden_size * self.num_stream,
            self.num_stream,
            bias=False,
        )
        self.phi._init_role = "input"
        if config.perform_initialization:
            config._standalone_init_weights(self.phi)

        self.branch_alpha = nn.Parameter(torch.ones(1) * config.mhc_init_alpha)

        branch_beta = torch.zeros(self.num_stream)
        self.branch_beta = nn.Parameter(branch_beta)

        if config.mhc_use_gamma:
            self.norm_gamma = nn.Parameter(torch.ones(config.hidden_size * self.num_stream) *
                config.mhc_init_gamma)

        self.hc_eps = config.hc_eps
        self.norm_eps = self.hc_eps

    def forward(self, x: torch.Tensor):
        shape, dtype = x.size(), x.dtype

        x = x.float()

        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        if self.config.mhc_use_gamma:
            weight = F.linear(x * rsqrt * self.norm_gamma, self.phi.weight)
        else:
            weight = F.linear(x, self.phi.weight) * rsqrt
        h_pre = F.sigmoid(weight * self.branch_alpha + self.branch_beta.unsqueeze(0).unsqueeze(0)) + self.hc_eps

        if self.config.mhc_hpre_renorm:
            eps_cache = torch.full(
                (),
                1e-30,
                dtype=h_pre.dtype,
                device=h_pre.device
            )
            h_pre = h_pre / h_pre.sum(dim=-1, keepdim=True).maximum(eps_cache)

        y = torch.sum(h_pre.unsqueeze(-1) * x.unflatten(dim=-1, sizes=(self.num_stream, -1)), dim=2).to(dtype)

        return y
