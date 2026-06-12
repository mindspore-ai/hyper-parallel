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
"""Example: PP + EP + comm/compute overlap on an Attention+MoE model.

Putting three pieces together:

1. **Model** — each pipeline chunk is a toy ``Attention + MoE`` block; MoE
   experts are sharded across ranks via :class:`ExpertParallel`.
2. **Pipeline schedule** — :class:`ScheduleInterleaved1F1B` with
   ``overlap_b_f=True`` emits ``OVERLAP_B_F`` composite steps in the
   1F1B steady-state, each pairing a BWD from iteration ``i`` with the
   FWD from iteration ``i+1`` of a *different* virtual chunk.
3. **Comm/compute overlap** — :class:`CommComputeOverlap` runs the
   paired BWD + FWD on two threads with deterministic comm-first
   ordering at the A/B/C/D hooks around EP dispatch / combine.
   Registered as the ``OVERLAP_B_F`` callback so the schedule drives it
   automatically.

Mesh layout (4 ranks)::

          ep →
    pp ↓  rank 0   rank 1
          rank 2   rank 3

``pp`` splits virtual chunks across rows; ``ep`` shards experts across
columns.  With ``PP=2`` and ``2`` interleaved chunks per rank, there are
4 virtual stages in total — exactly what the interleaved schedule expects.

Launch::

    torchrun --nproc_per_node=4 pp_overlap_moe_example.py
"""
# pylint: disable=C0413
import os
os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "torch")

import torch
from torch import nn
import torch.distributed as dist

from hyper_parallel import PipelineStage, init_device_mesh
from hyper_parallel.core.expert_parallel.expert_parallel import (  # pylint: disable=C0412
    ExpertParallel,
    _permute,
    _unpermute,
)
from hyper_parallel.core.pipeline_parallel import (
    CommComputeOverlap,
    MetaStepType,
    ScheduleInterleaved1F1B,
)
from hyper_parallel.platform import get_platform
from hyper_parallel.platform.torch.common import MoE

platform = get_platform()


# =========================================================================
# Step 1: EP strategy that keeps A→B / C→D windows minimal
# =========================================================================

class OverlapExpertParallel(ExpertParallel):
    """Expert Parallel with comm/compute overlap sync hooks wrapping ONLY the token a2a.

    The baseline ``ExpertParallel._token_dispatch`` performs counts-a2a →
    ``int()`` D2H sync on counts → token-a2a → rank-major → expert-major
    permutation.  Wrapping A/B around the whole sequence puts the ``int()``
    sync inside the overlap window, which delays
    :meth:`HookCoordinator.notify_dispatched` at B and therefore delays the
    paired COMPUTE side's kernel launch, shrinking GPU overlap.

    This subclass restructures hook placement so A→B / C→D only contain the
    main token / combine a2a launches:

    * counts-a2a + ``tolist()`` run **before** A (inside the previous
      Attention's COMPUTE window, host wait does not delay notify).
    * ``_permute`` runs **after** B (COMPUTE region, overlaps peer comm).
    * ``_unpermute`` runs **before** C (same idea, symmetrically).

    Applies to HCCL-style backends where device-side split_sizes are
    unsupported and ``int()`` cannot be avoided.

    Args:
        overlap: The :class:`CommComputeOverlap` whose coordinator drives
            the rendezvous.
    """

    def __init__(self, overlap: CommComputeOverlap, is_last_layer: bool = False) -> None:
        super().__init__()
        self._overlap = overlap
        # Tag the closing D hook as ``"D_LAST"`` for the last MoE layer in
        # a chunk: in forward there is no Attention after it, in backward
        # it is the first BWD hook to fire (combine.bwd has already
        # dispatched freely).  Tagging skips the rendezvous on both sides
        # and replaces the runtime cycle counter / BWD-D-skip mechanisms
        # that earlier versions of HookCoordinator had to maintain.
        self._d_hook = "D_LAST" if is_last_layer else "D"

    def _token_dispatch(self, module, inputs, device_mesh):
        """Dispatch tokens to expert ranks with A/B sync hooks bracketing the a2a.

        Counts a2a + ``tolist()`` happen inside A→B; ``_permute`` runs after B
        in the COMPUTE region.  See class docstring for the full hook layout.
        """
        del module
        coord = self._overlap.coordinator
        routed_input, num_tokens_per_expert = inputs[0], inputs[1]
        ep_group = device_mesh.get_group()
        ep_size = device_mesh.size()
        num_local_experts = num_tokens_per_expert.shape[0] // ep_size

        # ---- Hook A: rendezvous with BWD thread BEFORE any HCCL call. ----
        # Putting counts a2a before hook A made it race with the peer
        # thread's combine.bwd / dispatch.bwd HCCL calls on the same group,
        # producing garbage ``counts_out`` values and 1EB-sized
        # ``torch.empty`` allocations.  Keep counts a2a inside A→B so the
        # coordinator serialises cross-thread HCCL traffic.
        routed_input = platform.differentiable_sync_hook(routed_input, "A", coord)

        # ---- Inside A→B: counts a2a + splits + token a2a. ----
        # Wrap counts a2a in ``no_grad`` to guarantee the autograd graph
        # never captures it; otherwise a stray ``requires_grad=True`` on
        # ``num_tokens_per_expert`` would make backward replay a phantom
        # counts collective and inflate the per-OVERLAP_B_F a2a count.
        with torch.no_grad():
            counts_out, handle = platform.all_to_all_single(
                num_tokens_per_expert,
                output_shape=[num_tokens_per_expert.shape[0]],
                group=ep_group,
                async_op=True,
            )
            if handle is not None:
                handle.wait()
            input_splits = num_tokens_per_expert.view(ep_size, num_local_experts).sum(dim=1).tolist()
            output_splits = counts_out.view(ep_size, num_local_experts).sum(dim=1).tolist()
        self._input_splits = input_splits
        self._output_splits = output_splits

        # Async a2a: host returns immediately after launching the HCCL
        # kernel, so ``notify_dispatched(B)`` fires while the collective is
        # still in flight and the paired COMPUTE side can start its kernel
        # concurrently — this is what actually produces the overlap.
        dispatched = platform.differentiable_all_to_all_single_async(
            routed_input, input_splits, output_splits, group=ep_group,
        )
        dispatched = platform.differentiable_sync_hook(dispatched, "B", coord)

        # ---- AFTER hook B (COMPUTE region): rank-major -> expert-major. ----
        self._input_shape, permuted, self._permuted_indices, local_counts = _permute(
            dispatched, counts_out, ep_size, num_local_experts,
        )
        return permuted, local_counts

    def _token_combine(self, module, routed_output, device_mesh):
        """Combine expert outputs back to original ranks with C/D sync hooks.

        ``_unpermute`` runs before C in the COMPUTE region; the combine a2a
        is launched between C and D so the peer thread can race ahead while
        the collective is in flight.  The closing D hook is tagged
        ``"D_LAST"`` for the last MoE layer (set at construction time).
        """
        del module
        coord = self._overlap.coordinator
        ep_group = device_mesh.get_group()

        # ---- BEFORE hook C (COMPUTE region): expert-major -> rank-major. ----
        unpermuted = _unpermute(routed_output, self._input_shape, self._permuted_indices)

        # ---- C → combine a2a launch → D. ----
        # Same async rationale as dispatch: host returns right after the
        # HCCL kernel is queued so the peer thread can race ahead.
        unpermuted = platform.differentiable_sync_hook(unpermuted, "C", coord)
        combined = platform.differentiable_all_to_all_single_async(
            unpermuted,
            self._output_splits,
            self._input_splits,
            group=ep_group,
        )
        combined = platform.differentiable_sync_hook(combined, self._d_hook, coord)
        return combined


# =========================================================================
# Step 2: Per-chunk model — N stacked Attention + MoE blocks
# =========================================================================

class AttentionMoEBlock(nn.Module):
    """Single transformer block: self-attention followed by one MoE layer.

    Args:
        dim:          Token embedding dimension.
        hidden_dim:   Expert hidden dimension.
        num_experts:  Total experts in the MoE layer (shared across EP group).
        top_k:        Experts selected per token.
    """

    def __init__(self, dim: int, hidden_dim: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.moe = MoE(dim=dim, hidden_dim=hidden_dim, num_experts=num_experts, top_k=top_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: self-attention with residual connection followed by MoE feed-forward."""
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        attn = torch.softmax(q @ k.transpose(-1, -2) / (k.size(-1) ** 0.5), dim=-1)
        x = x + self.proj(attn @ v)
        return self.moe(x)


class AttentionThenMoE(nn.Module):
    """One pipeline chunk: ``num_layers`` stacked :class:`AttentionMoEBlock`.

    Stacking >1 MoE layer per chunk is what unlocks the **cross-layer**
    overlap window (combine of layer L ↔ attention of layer L+1 on the peer
    thread).  With a single layer per chunk, only intra-layer windows
    (A→B / B→C / C→D) exist and the D→A_next boundary never fires, so most
    of dual-pipe's payoff is missed.
    """

    def __init__(self, dim: int, hidden_dim: int, num_experts: int,
                 top_k: int, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionMoEBlock(dim, hidden_dim, num_experts, top_k)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run all stacked Attention+MoE blocks sequentially and return the final output."""
        for layer in self.layers:
            x = layer(x)
        return x


# =========================================================================
# Step 3: OVERLAP_B_F callback driven by CommComputeOverlap
# =========================================================================

def make_overlap_b_f_callback(overlap: CommComputeOverlap, device: torch.device):
    """Build a callback that executes paired ``(BWD, FWD)`` sub-steps concurrently.

    Layer-boundary handling is encoded statically: the last MoE layer in
    each chunk has its closing D hook tagged ``"D_LAST"`` at wrap time
    (see :class:`OverlapExpertParallel`), so the coordinator does not
    need a per-call layer count.

    Args:
        overlap: Shared :class:`CommComputeOverlap` with A/B/C/D hooks
            already installed on every chunk's MoE experts.
        device:  The rank's device.  Must be passed through because PyTorch's
            current device is thread-local and the spawned BWD thread would
            otherwise default to card 0, sending HCCL collectives on the
            wrong channel and hanging.

    Returns:
        A callable ``(step, ctx) -> None`` suitable for
        :meth:`PipelineScheduleRuntime.register_custom_function`.
    """

    def _callback(step, ctx):
        bwd_step, fwd_step = step.sub_steps
        schedule = ctx.schedule
        fwd_stage = schedule._stage_dict[fwd_step.stage_index]  # pylint: disable=W0212
        bwd_stage = schedule._stage_dict[bwd_step.stage_index]  # pylint: disable=W0212
        fwd_mi, bwd_mi = fwd_step.micro_index, bwd_step.micro_index

        # When overlap_p2p=True the scheduler issues FWD_RECV / BWD_RECV
        # before this OVERLAP_B_F step and stashes the handles in
        # schedule.fwd_handle_cache / schedule.bwd_handle_cache keyed by
        # (stage_index, micro_index).  Pop them here so each side waits on
        # its own compute stream, letting PP-RECV stream overlap with the
        # paired-side compute.  Empty when overlap_p2p=False.  (The
        # ``schedule.wait_fwd_recv`` / ``wait_bwd_recv`` helpers pop *and*
        # wait in one call; this callback splits the two so the wait can
        # land on the worker thread's stream — see fwd_fn / bwd_fn below.)
        fwd_recv_handles = schedule.fwd_handle_cache.pop(
            (fwd_stage.stage_index, fwd_mi), None,
        )
        bwd_recv_handles = schedule.bwd_handle_cache.pop(
            (bwd_stage.stage_index, bwd_mi), None,
        )

        def fwd_fn() -> None:
            # Wait happens on the FWD thread's current (default) stream so
            # the event sync goes PP-RECV stream -> default stream; FWD
            # Attention/MoE kernels then enqueue after the wait fires.
            if fwd_recv_handles:
                schedule._wait_p2p(fwd_recv_handles)  # pylint: disable=W0212
            out = fwd_stage.forward_one_chunk(
                fwd_mi, ctx.arg_mbs[fwd_mi], ctx.kwarg_mbs[fwd_mi],
            )
            schedule.update_losses(fwd_stage, out, ctx.losses)

        def bwd_fn() -> None:
            # Pin the spawned BWD thread to this rank's device — current
            # device is thread-local; without this HCCL on the BWD thread
            # targets device 0 and hangs waiting for a peer that never
            # arrives on that channel.
            #
            # BWD runs on the BWD thread's default stream (same physical
            # stream as the main thread's default).  Using a dedicated
            # side stream broke recv-A2A serialization: ``recv_buffer`` is
            # allocated on the main default stream, so PyTorch's tensor
            # allocator-stream tracking has the buffer pinned to the main
            # default stream.  When the BWD ``combine.bwd`` A2A reads it,
            # HCCL inserts cross-stream sync from the buffer's owning
            # stream (main default) — not from the side stream — so the
            # ``handle.wait()`` event landing on the side stream never
            # propagates to the EP HCCL stream and the A2A starts before
            # PP-RECV finishes.  Running BWD on the BWD thread's default
            # stream aligns the buffer's owning stream with the consumer
            # chain so the wait event propagates correctly.
            if device.type == "npu" and hasattr(torch, "npu"):
                torch.npu.set_device(device.index)
            elif device.type == "cuda":
                torch.cuda.set_device(device.index)
            if bwd_recv_handles:
                schedule._wait_p2p(bwd_recv_handles)  # pylint: disable=W0212
            bwd_stage.backward_one_chunk(bwd_mi)

        overlap.run(fwd_fn=fwd_fn, bwd_fn=bwd_fn)

    return _callback


# =========================================================================
# Step 4: Build per-rank chunks, wire up PP + EP, launch schedule
# =========================================================================

def _init_dist():
    """Initialize the process group and pick the local device + type."""
    if not dist.is_initialized():
        dist.init_process_group()
    rank = dist.get_rank()
    if hasattr(torch, "npu") and torch.npu.is_available():
        local = rank % torch.npu.device_count()
        torch.npu.set_device(local)
        return rank, torch.device(f"npu:{local}"), "npu"
    if torch.cuda.is_available():
        local = rank % torch.cuda.device_count()
        torch.cuda.set_device(local)
        return rank, torch.device(f"cuda:{local}"), "cuda"
    return rank, torch.device("cpu"), "cpu"


def main() -> None:
    """Entry point: set up PP+EP mesh, build interleaved MoE chunks with overlap, and run the schedule."""
    rank, device, device_type = _init_dist()
    world_size = dist.get_world_size()

    # ---- Parallel topology ----
    pp_size, ep_size = 2, 2
    if world_size != pp_size * ep_size:
        raise ValueError(
            f"This example expects world_size={pp_size * ep_size} "
            f"(pp={pp_size} x ep={ep_size}), got {world_size}."
        )
    mesh = init_device_mesh(
        device_type=device_type,
        mesh_shape=(pp_size, ep_size),
        mesh_dim_names=("pp", "ep"),
    )
    pp_mesh, ep_mesh = mesh["pp"], mesh["ep"]

    # ---- Model / pipeline config ----
    dim, hidden_dim = 2048, 8192
    num_experts = ep_size                 # one expert per EP rank
    top_k = 2
    chunks_per_rank = 2                   # interleaved virtual chunks per rank
    moe_layers_per_chunk = 2              # ≥2 unlocks cross-layer (D→A_next) overlap
    virtual_stages = pp_size * chunks_per_rank
    micro_batch_num = 8
    bs, seq_len = 8, 4096

    pp_rank = pp_mesh.get_local_rank()

    # ---- Shared comm/compute overlap orchestrator (one coordinator per rank) ----
    overlap = CommComputeOverlap()

    # ---- Build interleaved chunks; apply EP to EVERY MoE layer's experts ----
    # Seed by pp_rank (NOT global rank) so all ranks in the same EP group
    # share identical model weights — including the router, which is
    # replicated (only ``experts`` is EP-sharded).  Different routers across
    # EP ranks would produce inconsistent routing decisions, mismatched
    # all-to-all splits, and zero-token corner cases that break grad tracking
    # inside ``_permute`` / ``_unpermute``.
    torch.manual_seed(42 + pp_rank)
    chunks = []
    stage_indices = []
    for chunk_id in range(chunks_per_rank):
        block = AttentionThenMoE(
            dim, hidden_dim, num_experts, top_k, num_layers=moe_layers_per_chunk,
        ).to(device)
        # Each MoE layer's ``experts`` needs its own EP strategy instance so
        # per-layer state (``_input_splits`` etc.) does not get clobbered;
        # all share the same ``overlap`` and thus the same coordinator.
        # The last MoE layer in each chunk gets ``is_last_layer=True`` so
        # its closing D hook is tagged ``"D_LAST"`` and the rendezvous is
        # skipped (no Attention follows in forward; combine.bwd has
        # already free-run before any rendezvous in backward).
        last_idx = len(block.layers) - 1
        for layer_idx, layer in enumerate(block.layers):
            OverlapExpertParallel(
                overlap, is_last_layer=(layer_idx == last_idx),
            ).apply(layer.moe.experts, ep_mesh)
        chunks.append(block)
        # Interleaved layout: chunk 0 -> stage pp_rank, chunk 1 -> stage pp_rank + pp_size
        stage_indices.append(pp_rank + chunk_id * pp_size)

    stages = [
        PipelineStage(block, stage_index=si, stage_num=virtual_stages,
                      device=device, mesh=pp_mesh)
        for block, si in zip(chunks, stage_indices)
    ]

    # ---- Schedule: interleaved 1F1B with B/F overlap + P2P overlap. ----
    # ``overlap_b_f=True`` emits OVERLAP_B_F composite steps in the 1F1B
    # steady state; ``overlap_p2p=True`` defers PP recv waits to the
    # callback so RECV overlaps with the prior OVERLAP_B_F's compute.
    schedule = ScheduleInterleaved1F1B(
        stages, micro_batch_num, overlap_p2p=True, overlap_b_f=True,
    )
    schedule.register_custom_function(
        MetaStepType.OVERLAP_B_F, make_overlap_b_f_callback(overlap, device),
    )

    # ---- Input only at stage 0; other ranks call run() with no args ----
    # Re-seed so the input is identical across an EP group (same pp_rank).
    torch.manual_seed(100 + pp_rank)
    x = torch.randn(bs, seq_len, dim, device=device, requires_grad=True)
    losses = schedule.run(x) if pp_rank == 0 else schedule.run()

    if pp_rank == pp_size - 1 and ep_mesh.get_local_rank() == 0:
        print(f"[rank{rank}] PP+EP+overlap done. losses[0]={losses[0].mean().item():.4f}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
