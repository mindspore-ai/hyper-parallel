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
"""KDA-specific sequence parallelism above distribution-agnostic local kernels."""
from typing import Any, Callable, Optional

from hyper_parallel.core.context_parallel.context_parallel import (
    ContextParallel, _build_hybrid_cp_mesh, _ensure_1d, _is_tensor_or_dtensor, _to_cp_dtensor,
)
from hyper_parallel.core.dtensor.device_mesh import DeviceMesh
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard
from hyper_parallel.core.shard.custom_shard import custom_shard
from hyper_parallel.core.shard.dfunction import DFunction
from hyper_parallel.custom_ops.experimental import npu_chunk_kda
from hyper_parallel.custom_ops.experimental import (
    npu_chunk_kda_fwd_prepare, npu_chunk_kda_fwd_h, npu_chunk_kda_fwd_finalize,
    npu_chunk_kda_bwd_prepare, npu_chunk_kda_bwd_dhu, npu_chunk_kda_bwd_finalize,
    npu_chunk_kda_pre_process_fwd, npu_chunk_kda_pre_process_bwd, npu_chunk_kda_merge_fwd_bwd,
)
from hyper_parallel.platform import get_platform

platform = get_platform()
Tensor = platform.Tensor
Module = platform.Module

# Output order of experimental.npu_chunk_kda_fwd_prepare.
_PREPARE_OUTPUTS = ('gk', 'aqk', 'akk', 'w', 'u', 'qg', 'kg', 'qg_scaled',
                    'q_hat', 'k_hat', 'q_rstd', 'k_rstd', 'beta_eff')
_BACKWARD_RELAYS = ('gk', 'aqk', 'akk', 'w', 'qg', 'kg')


# The ctx-style autograd protocol does not use the optional setup_context hook.
class _KDASequenceFunction(DFunction):  # pylint: disable=abstract-method
    """Autograd at the CP/kernel boundary; local operators receive only tensors."""

    # Autograd maps these explicit input slots to the positional gradients below.
    @staticmethod
    def forward(  # pylint: disable=arguments-differ
            ctx: Any, query: Tensor, key: Tensor, value: Tensor, gate: Tensor,
            beta: Tensor, a_log: Tensor, bias: Tensor, options: dict, transport: Any,
    ) -> Tensor:
        """Prepare local relays, exchange summaries and evaluate this sequence shard."""
        inputs = tuple(tensor.swapaxes(1, 2).contiguous() for tensor in (query, key, value, gate, beta))
        query, key, value, gate, beta = inputs
        cache = dict(zip(_PREPARE_OUTPUTS, npu_chunk_kda_fwd_prepare(
            query, key, value, gate, beta, options['scale'], options['chunk_size'],
            a_log=a_log, dt_bias=bias.reshape(-1).contiguous(), backward_mode='save',
            use_gate_in_kernel=True, safe_gate=True, lower_bound=options['lower_bound'])))
        initial = transport.forward_boundary(cache)
        h_block, updated_value, _ = npu_chunk_kda_fwd_h(
            cache['kg'], cache['w'], cache['u'], gk=cache['gk'], initial_state=initial,
            output_final_state=False, chunk_size=options['chunk_size'])
        output = npu_chunk_kda_fwd_finalize(
            cache['qg_scaled'], cache['aqk'], updated_value, h_block, output_layout='BNSD')
        saved = dict(zip(('q', 'k', 'v', 'g', 'beta'), inputs))
        saved.update(a=a_log, bias=bias, initial=initial, h_block=h_block, updated_value=updated_value)
        saved.update({name: cache[name] for name in _BACKWARD_RELAYS})
        ctx.names = tuple(saved)
        ctx.save_for_backward(*saved.values())
        ctx.options, ctx.transport = options, transport
        return output.swapaxes(1, 2)

    @staticmethod
    def backward(  # pylint: disable=arguments-differ
            ctx: Any, grad_output: Tensor,
    ) -> tuple:
        """Exchange suffix summaries before local Dhu and return local input gradients."""
        saved = dict(zip(ctx.names, ctx.saved_tensors))
        options = ctx.options
        query, key, value, gate, beta = (saved[name] for name in ('q', 'k', 'v', 'g', 'beta'))
        cache = saved
        grad_output = grad_output.swapaxes(1, 2).contiguous()
        h_block, updated_value = saved['h_block'], saved['updated_value']
        da, dv_local, dq_raw = npu_chunk_kda_bwd_prepare(
            cache['aqk'], updated_value, grad_output, h_block,
            scale=options['scale'], chunk_size=options['chunk_size'])
        terminal = ctx.transport.backward_boundary(cache, grad_output, dv_local, options['scale'])
        dh, _, dv_scan = npu_chunk_kda_bwd_dhu(
            cache['qg'], cache['kg'], cache['w'], grad_output, dv_local, options['scale'],
            gk=cache['gk'], h0=saved['initial'].to(query.dtype),
            dht=terminal.to(query.dtype).contiguous(), chunk_size=options['chunk_size'])
        dq, dk, dv, dbeta, dg, grad_a, grad_bias = npu_chunk_kda_bwd_finalize(
            query, key, value, cache['gk'], gate, beta, saved['a'], saved['bias'].reshape(value.shape[1], -1),
            cache['akk'], updated_value, h_block, dh, dv_scan, da, dq_raw,
            scale=options['scale'], lower_bound=options['lower_bound'], chunk_size=options['chunk_size'])
        gradients = tuple(tensor.swapaxes(1, 2) for tensor in (dq, dk, dv, dg, dbeta))
        return gradients + (grad_a, grad_bias.reshape(saved['bias'].shape), None, None)


class _KDAStateTransport:
    """Sequence summary transport owned by the parallelization layer."""

    def __init__(self, mesh: DeviceMesh) -> None:
        """Bind the chronological sequence mesh."""
        self.mesh = mesh
        self.rank, self.size = mesh.get_local_rank(), mesh.size()

    def _merge(self, summary: Tensor, forward: bool) -> Tensor:
        """Gather summaries and compose this rank's chronological prefix or suffix."""
        gathered = DTensor.from_local(summary.unsqueeze(0), self.mesh, (Shard(0),))
        gathered = gathered.redistribute(self.mesh, (Replicate(),))
        # Merge is a local native kernel: localize only after DTensor communication.
        full = gathered.to_local()
        count = self.rank if forward else self.size - self.rank - 1
        states = []
        for batch in range(summary.shape[0]):
            state = platform.zeros(summary.shape[1:-1] + (summary.shape[-1] - summary.shape[-2],),
                                   dtype=summary.dtype)
            if count:
                state = npu_chunk_kda_merge_fwd_bwd(
                    state, full[:, batch].contiguous(), count, self.rank, forward=forward)
            states.append(state.unsqueeze(0))
        return platform.cat(states, dim=0)

    def forward_boundary(self, cache: dict) -> Tensor:
        """Build all dense-batch summaries in one launch; the backend packs B>1."""
        summary = npu_chunk_kda_pre_process_fwd(
            cache['kg'], cache['w'], cache['u'], gk=cache['gk'], cu_seqlens=None)[0]
        return self._merge(summary, True)

    def backward_boundary(self, cache: dict, grad_output: Tensor, dv_local: Tensor, scale: float) -> Tensor:
        """Construct the suffix gradient using gated Q (qg), as in local Dhu."""
        summaries = [npu_chunk_kda_pre_process_bwd(
            cache['qg'][i:i + 1], cache['kg'][i:i + 1], cache['w'][i:i + 1],
            grad_output[i:i + 1], dv_local[i:i + 1], scale,
            gk=cache['gk'][i:i + 1].to(grad_output.dtype),
            cu_seqlens=[0, grad_output.shape[2]]).unsqueeze(0)
                     for i in range(grad_output.shape[0])]
        return self._merge(platform.cat(summaries, dim=0), False).to(grad_output.dtype)


class KDAContextParallel(ContextParallel):
    """KDA CP style retaining DTensor layouts until convolution/native-stage boundaries.

    Args:
        ulysses_degree: Head subgroup size; None (default) selects pure Ulysses, one selects sequence-only CP.
        use_local_output: Restore local tensors at the model output boundary.

    The model calls ``convolve`` and ``attention`` on its attached ``_kda_parallel``
    adapter. Neither local KDA operators nor the generic ContextParallel need CP flags.
    Sequence shards must be nonempty, equal-length and in natural order. This dense
    training path uses BSND, BF16, equal Q/K/V heads (MHA), head128, chunk64 and
    zero initial state. Pass Q/K/V/gate/beta positionally to the hooked module.
    Q/K L2 normalization and beta sigmoid belong to the caller: kernel options are
    qk_norm=False, beta_sigmoid=False, allow_neg_eigval=False, epsilon=1e-6,
    use_exp2=True and safe_gate=True. Other mathematical options are not exposed.
    No activation recompute is provided: local block states and six relays are saved.
    Parameters must contain all TP-local heads after FSDP unsharding, before the
    Ulysses head selection: weight [3, H*D, 1, K], a_log [H], bias [H, D]. DTensor
    parameters are supported if their local tensors obey this contract. Parameter
    gradient reductions across CP replicas belong to the caller's DP/FSDP layer.
    State summaries remain FP32: each direction gathers B*H*D*(D+D)*4*R bytes per
    rank (R sequence ranks); no reduced-precision summary transport is provided.
    """

    def __init__(self, ulysses_degree: Optional[int] = None, use_local_output: bool = True) -> None:
        """Configure the KDA head subgroup and output policy."""
        super().__init__(seq_dim=1, head_dim=2, ulysses_degree=ulysses_degree,
                         qkv_indices=(0, 1, 2, 3, 4), use_local_output=use_local_output)
        self._transport = None
        self._head_rank = 0
        self._head_degree = 1

    def apply(self, module: Module, device_mesh: DeviceMesh) -> Module:
        """Install inherited output hooks and KDA-specific input redistribution."""
        count = device_mesh.mesh.numel()
        degree = self.ulysses_degree if self.ulysses_degree is not None else count
        if degree < 1 or count % degree:
            raise ValueError('KDA Ulysses degree must be positive and divide CP size.')
        sequence = count // degree
        if device_mesh.ndim == 2 and (tuple(device_mesh.mesh_dim_names or ()) != ('co', 'ds')
                                      or tuple(device_mesh.mesh.shape) != (sequence, degree)):
            raise ValueError('KDA Hybrid mesh must have dimensions (co, ds) with shape (sequence, heads).')
        self._head_degree = degree
        self._transport, self._head_rank = None, 0
        if sequence > 1:
            if degree == 1:
                self._transport = _KDAStateTransport(_ensure_1d(device_mesh))
            else:
                mesh = _build_hybrid_cp_mesh(device_mesh, degree, sequence)
                self._transport = _KDAStateTransport(mesh[mesh.mesh_dim_names[0]])
                self._head_rank = mesh[mesh.mesh_dim_names[1]].get_local_rank()
        else:
            self._head_rank = _ensure_1d(device_mesh).get_local_rank()
        # Private integration slot consumed by the model core, outside the math op API.
        module._kda_parallel = self  # pylint: disable=protected-access
        return super().apply(module, device_mesh)

    def _pre_hook_colossal(self, module, args, kwargs, co_submesh):
        """Retain five positional sequence shards for the local KDA stages."""
        if len(args) < 5:
            raise TypeError('KDA CP requires positional query/key/value/gate/beta inputs.')
        self._record_q_output_tp_layout(module, args, kwargs, co_submesh)
        new_args = list(args)
        for index in self.qkv_indices:
            if index < len(new_args) and _is_tensor_or_dtensor(new_args[index]):
                new_args[index] = _to_cp_dtensor(new_args[index], co_submesh, (Shard(1),), (Shard(1),), 1)
        return tuple(new_args), dict(kwargs)

    def _pre_hook_hybrid(self, module, args, kwargs, hybrid_cp_mesh, ds_submesh, ds_size):
        """Redistribute positional inputs onto the ordered sequence/head mesh."""
        if len(args) < 5:
            raise TypeError('KDA CP requires positional query/key/value/gate/beta inputs.')
        self._record_q_output_tp_layout(module, args, kwargs, hybrid_cp_mesh)
        new_args = list(args)
        for index in self.qkv_indices:
            if index < len(new_args) and self._needs_hybrid_ata(new_args[index], hybrid_cp_mesh):
                new_args[index] = self._ata_scatter_to_hybrid(
                    new_args[index], ds_submesh, hybrid_cp_mesh, ds_size)
        return tuple(new_args), dict(kwargs)

    @staticmethod
    def _native(function: Callable, arguments: tuple, output: DTensor) -> DTensor:
        """Use HP's standard local-kernel boundary instead of manual to_local calls."""
        placements = tuple(arg.placements if isinstance(arg, DTensor) else None for arg in arguments)
        return custom_shard(function, output.device_mesh, (output.placements,),
                            placements, redistribute_inputs=False)(*arguments)

    def sequence_halo(self, tensor: DTensor, width: int) -> DTensor:
        """Exchange short tails as DTensors and retain the sequence/head placements."""
        if width <= 0 or self._transport is None:
            return tensor

        def _tail(value):
            length = min(width, value.shape[1])
            if length == 0:
                raise ValueError('KDA convolution requires nonempty sequence shards.')
            return value[:, -length:]

        tails = self._native(_tail, (tensor,), tensor)
        replicated = tuple(Replicate() if placement.is_shard(1) else placement for placement in tails.placements)
        tails = tails.redistribute(tails.device_mesh, replicated)

        def _prepend(value, all_tails):
            length = min(width, value.shape[1])
            end = self._transport.rank * length
            available = min(width, end)
            prefix = all_tails[:, end - available:end]
            if available < width:
                padding = platform.zeros((value.shape[0], width - available) + tuple(value.shape[2:]),
                                         dtype=value.dtype, device=getattr(value, 'device', None))
                prefix = platform.cat((padding + all_tails.sum() * 0, prefix), dim=1)
            return platform.cat((prefix, value), dim=1)

        return self._native(_prepend, (tensor, tails), tensor)

    @staticmethod
    def _check_qkv(query: DTensor, key: DTensor, value: DTensor) -> None:
        """Reject missing hooks and unequal heads before native operations."""
        if not all(isinstance(tensor, DTensor) for tensor in (query, key, value)):
            raise TypeError('KDA CP inputs must be DTensors produced by KDAContextParallel hooks.')
        if query.ndim != 4 or query.shape != key.shape or query.shape != value.shape:
            raise ValueError('KDA CP supports MHA with equal Q/K/V heads and dimensions in BSND layout.')
        if key.placements != query.placements or value.placements != query.placements:
            raise ValueError('KDA CP Q/K/V must share placements.')

    def convolve(self, query: DTensor, key: DTensor, value: DTensor, weight: Tensor,
                 compute: Callable, kernel_size: int) -> tuple:
        """Preserve distributed layouts around one local depthwise-convolution boundary."""
        self._check_qkv(query, key, value)
        projected = platform.cat(tuple(tensor.unsqueeze(2) for tensor in (query, key, value)), dim=2)
        projected = self.sequence_halo(projected, kernel_size - 1)

        def _convolution(inputs, conv_weight):
            batch, length, _, heads, dim = inputs.shape
            channels = heads * dim
            start = self._head_rank * channels
            conv_weight = conv_weight.reshape(3, -1, 1, kernel_size)
            if conv_weight.shape[1] != channels * self._head_degree:
                raise ValueError('KDA convolution weight must contain all TP-local heads before Ulysses slicing.')
            conv_weight = conv_weight[:, start:start + channels]
            output = compute(inputs.reshape(batch, length, -1), conv_weight, kernel_size)
            if self._transport is not None:
                output = output[:, kernel_size - 1:]
            output = output.reshape(batch, output.shape[1], 3, heads, dim)
            return output[:, :, 0], output[:, :, 1], output[:, :, 2]

        placements = (projected.placements, weight.placements if isinstance(weight, DTensor) else None)
        return custom_shard(_convolution, query.device_mesh, (query.placements,) * 3,
                            placements, redistribute_inputs=False)(projected, weight)

    def attention(self, query: DTensor, key: DTensor, value: DTensor, gate: DTensor,
                  beta: DTensor, a_log: Tensor, bias: Tensor, scale: float,
                  chunk_size: int = 64, lower_bound: float = -5.) -> DTensor:
        """Execute parallel attention above the unchanged local mathematical operator API."""
        self._check_qkv(query, key, value)
        if not all(isinstance(tensor, DTensor) for tensor in (gate, beta)):
            raise TypeError('KDA CP gate/beta must be DTensors produced by KDAContextParallel hooks.')
        if gate.shape != query.shape or beta.shape != query.shape[:-1]:
            raise ValueError('KDA gate/beta must match the BSND query dimensions.')
        if chunk_size != 64 or query.shape[-1] != 128 or value.shape[-1] != 128:
            raise ValueError('KDA CP requires chunk64 and head128.')

        def _kernel(q_local, k_local, v_local, g_local, beta_local, a_local, bias_local):
            heads = v_local.shape[2]
            expected_heads = heads * self._head_degree
            if a_local.shape != (expected_heads,) or bias_local.shape != (expected_heads, v_local.shape[-1]):
                raise ValueError('KDA a_log [H] and bias [H,D] must contain all TP-local heads before slicing.')
            start = self._head_rank * heads
            a_local = a_local[start:start + heads]
            bias_local = bias_local[start:start + heads]
            options = {'scale': scale, 'chunk_size': chunk_size, 'lower_bound': lower_bound}
            if self._transport is not None:
                return _KDASequenceFunction.apply(q_local, k_local, v_local, g_local, beta_local,
                                                  a_local, bias_local, options, self._transport)
            return npu_chunk_kda(q_local, k_local, v_local, g_local, beta_local,
                                 layout='BSND', safe_gate=True, use_gate_in_kernel=True,
                                 a_log=a_local, dt_bias=bias_local, disable_recompute=True, **options)[0]

        return self._native(_kernel, (query, key, value, gate, beta, a_log, bias), query)
