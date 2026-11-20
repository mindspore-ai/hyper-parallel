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
"""Optional Ascend Triton kernels for the V4.1 eager training path."""

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl


@triton.jit
def _clamped_swiglu_fwd_kernel(gate_ptr, up_ptr, out_ptr, n_elements, limit: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    step = num_progs * BLOCK
    for base in range(pid * BLOCK, n_elements, step):
        offs = base + tl.arange(0, BLOCK)
        mask = offs < n_elements
        g = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        g_c = tl.minimum(g, limit)
        u_c = tl.maximum(tl.minimum(u, limit), -limit)
        sig = 1.0 / (1.0 + tl.exp(-g_c))
        silu = g_c * sig
        out = (silu * u_c).to(tl.bfloat16)
        tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def _clamped_swiglu_bwd_kernel(dy_ptr, gate_ptr, up_ptr, dg_ptr, du_ptr, n_elements, limit: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    step = num_progs * BLOCK
    for base in range(pid * BLOCK, n_elements, step):
        offs = base + tl.arange(0, BLOCK)
        mask = offs < n_elements
        dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        g_c = tl.minimum(g, limit)
        u_c = tl.maximum(tl.minimum(u, limit), -limit)
        sig = 1.0 / (1.0 + tl.exp(-g_c))
        silu = g_c * sig
        du = tl.where((u >= -limit) & (u <= limit), dy * silu, 0.0)
        dsilu = sig * (1.0 + g_c * (1.0 - sig))
        dg = tl.where(g <= limit, (dy * u_c) * dsilu, 0.0)
        tl.store(dg_ptr + offs, dg.to(tl.bfloat16), mask=mask)
        tl.store(du_ptr + offs, du.to(tl.bfloat16), mask=mask)


class FusedClampedSwiGLU(torch.autograd.Function):
    """Fuse BF16 shared-expert activation with FP32 clamp and SwiGLU math."""
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
        """Apply the fused activation and save the input projections."""
        gate = gate.contiguous()
        up = up.contiguous()
        ctx.save_for_backward(gate, up)
        ctx.limit = float(limit)
        out = torch.empty_like(gate)
        n = gate.numel()
        _clamped_swiglu_fwd_kernel[(48,)](gate, up, out, n, limit=ctx.limit, BLOCK=4096)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Apply the derivative including the inclusive clamp boundaries."""
        gate, up = ctx.saved_tensors
        dy = dy.contiguous()
        dg = torch.empty_like(gate)
        du = torch.empty_like(up)
        n = gate.numel()
        _clamped_swiglu_bwd_kernel[(48,)](dy, gate, up, dg, du, n, limit=ctx.limit, BLOCK=4096)
        return dg, du, None



@triton.jit
def _stream_collapse_fwd_kernel(rows_ptr, w_ptr, out_ptr, R, H: tl.constexpr, BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    for r in range(pid, R, num_progs):
        w0 = tl.load(w_ptr + r * 4 + 0).to(tl.float32)
        w1 = tl.load(w_ptr + r * 4 + 1).to(tl.float32)
        w2 = tl.load(w_ptr + r * 4 + 2).to(tl.float32)
        w3 = tl.load(w_ptr + r * 4 + 3).to(tl.float32)
        for c_start in range(0, H, BLOCK_H):
            cols = c_start + tl.arange(0, BLOCK_H)
            mask = cols < H
            base = r * (4 * H) + cols
            r0 = tl.load(rows_ptr + base + 0 * H, mask=mask, other=0.0).to(tl.float32)
            r1 = tl.load(rows_ptr + base + 1 * H, mask=mask, other=0.0).to(tl.float32)
            r2 = tl.load(rows_ptr + base + 2 * H, mask=mask, other=0.0).to(tl.float32)
            r3 = tl.load(rows_ptr + base + 3 * H, mask=mask, other=0.0).to(tl.float32)
            out = (w0 * r0 + w1 * r1 + w2 * r2 + w3 * r3).to(tl.bfloat16)
            tl.store(out_ptr + r * H + cols, out, mask=mask)


@triton.jit
def _stream_collapse_bwd_kernel(grad_ptr, rows_ptr, w_ptr, grows_ptr, gw_ptr, R, H: tl.constexpr, BLOCK_H: tl.constexpr):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    for r in range(pid, R, num_progs):
        w0 = tl.load(w_ptr + r * 4 + 0).to(tl.float32)
        w1 = tl.load(w_ptr + r * 4 + 1).to(tl.float32)
        w2 = tl.load(w_ptr + r * 4 + 2).to(tl.float32)
        w3 = tl.load(w_ptr + r * 4 + 3).to(tl.float32)
        acc0 = tl.zeros((BLOCK_H,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_H,), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_H,), dtype=tl.float32)
        acc3 = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for c_start in range(0, H, BLOCK_H):
            cols = c_start + tl.arange(0, BLOCK_H)
            mask = cols < H
            dy = tl.load(grad_ptr + r * H + cols, mask=mask, other=0.0).to(tl.float32)
            base = r * (4 * H) + cols
            r0 = tl.load(rows_ptr + base + 0 * H, mask=mask, other=0.0).to(tl.float32)
            r1 = tl.load(rows_ptr + base + 1 * H, mask=mask, other=0.0).to(tl.float32)
            r2 = tl.load(rows_ptr + base + 2 * H, mask=mask, other=0.0).to(tl.float32)
            r3 = tl.load(rows_ptr + base + 3 * H, mask=mask, other=0.0).to(tl.float32)
            tl.store(grows_ptr + base + 0 * H, (dy * w0).to(tl.bfloat16), mask=mask)
            tl.store(grows_ptr + base + 1 * H, (dy * w1).to(tl.bfloat16), mask=mask)
            tl.store(grows_ptr + base + 2 * H, (dy * w2).to(tl.bfloat16), mask=mask)
            tl.store(grows_ptr + base + 3 * H, (dy * w3).to(tl.bfloat16), mask=mask)
            acc0 += dy * r0
            acc1 += dy * r1
            acc2 += dy * r2
            acc3 += dy * r3
        tl.store(gw_ptr + r * 4 + 0, tl.sum(acc0, axis=0))
        tl.store(gw_ptr + r * 4 + 1, tl.sum(acc1, axis=0))
        tl.store(gw_ptr + r * 4 + 2, tl.sum(acc2, axis=0))
        tl.store(gw_ptr + r * 4 + 3, tl.sum(acc3, axis=0))



class FusedStreamCollapse(torch.autograd.Function):
    """Fuse the four mHC stream weights and residual rows."""

    @staticmethod
    def forward(ctx, hidden_streams: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        """Reduce the four streams in FP32 and restore the input dtype."""
        hidden = hidden_streams.shape[-1]
        rows = hidden_streams.reshape(-1, 4, hidden)
        weights = pre_mix.reshape(-1, 4)
        out = rows.new_empty(rows.shape[0], hidden)
        _stream_collapse_fwd_kernel[(48,)](rows, weights, out, rows.shape[0], H=hidden, BLOCK_H=1024)
        ctx.save_for_backward(rows, weights)
        ctx.input_shapes = hidden_streams.shape, pre_mix.shape
        return out.view(*hidden_streams.shape[:-2], hidden)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute gradients for both the residual rows and the mixing weights."""
        rows, weights = ctx.saved_tensors
        hidden = rows.shape[-1]
        grad_rows, grad_weights = torch.empty_like(rows), torch.empty_like(weights)
        _stream_collapse_bwd_kernel[(48,)](
            grad_output.contiguous(), rows, weights, grad_rows, grad_weights, rows.shape[0],
            H=hidden, BLOCK_H=1024,
        )
        return grad_rows.view(ctx.input_shapes[0]), grad_weights.view(ctx.input_shapes[1])
