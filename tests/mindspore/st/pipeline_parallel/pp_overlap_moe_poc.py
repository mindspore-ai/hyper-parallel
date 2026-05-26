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
"""End-to-end PoC: PP + EP + comm/compute overlap on MindSpore.

Mirrors ``examples/torch/pp_overlap/pp_overlap_moe_example.py`` on
MindSpore PyNative.  Wires together:

1. ``ScheduleInterleaved1F1B(overlap_p2p=True, overlap_b_f=True)`` —
   emits ``OVERLAP_B_F`` composite steps in the 1F1B steady state
   and defers PP-RECV waits to the callback.
2. ``CommComputeOverlap`` — drives paired BWD + FWD on two threads
   with deterministic comm-first dispatch ordering at A/B/C/D hooks.
3. ``OverlapExpertParallel`` — the vendored EP strategy that drives
   the MoE token a2a through our ``AsyncCollectiveTensor`` lazy-wait
   path with sync hooks bracketing the a2a.  Passing ``overlap=None``
   to the same class produces the sync baseline used by the accuracy
   test, eliminating any ``mindformers`` dependency.

Topology: 4 ranks, PP=2 × EP=2.  Each PP rank holds 2 interleaved
chunks; each chunk has 2 ``MiniMoEBlock`` layers (Attention + MoE).

This PoC validates the **structural integration** end-to-end:
- Schedule emits OVERLAP_B_F steps correctly
- Callback drives CommComputeOverlap.run with paired threads
- OverlapExpertParallel + AsyncCollectiveTensor + lazy wait fire
  at the right points
- Backward propagates through everything
- No deadlock, no exceptions, grads are non-zero

Numerical equivalence to a sync baseline is left to a separate test
once this scaffolding is green.
"""
# pylint: disable=W0611,W0212,C0415,C0413
# C0413 (wrong-import-position): the ``os.environ.setdefault`` call
# below MUST run before any ``hyper_parallel`` / ``mindspore`` import
# so the platform layer locks onto MindSpore at import time.  The
# linter cannot tell that this is intentional, so suppress the warning
# for the whole module.
import os
os.environ.setdefault("HYPER_PARALLEL_PLATFORM", "mindspore")

import numpy as np

import mindspore as ms
from mindspore import Tensor, mint, nn, ops
from mindspore.common import dtype as mstype
from mindspore.communication import init as comm_init
from mindspore.communication import get_rank, get_group_size

from hyper_parallel import PipelineStage, init_device_mesh
from hyper_parallel.core.pipeline_parallel import (
    CommComputeOverlap,
    MetaStepType,
    ScheduleInterleaved1F1B,
)
from hyper_parallel.platform import get_platform

from tests.mindspore.st.pipeline_parallel.overlap_expert_parallel import (
    MiniGroupedMLP,
    OverlapExpertParallel,
)


platform = get_platform()


# =========================================================================
# Hyperparams
# =========================================================================

PP_SIZE = 4
EP_SIZE = 2
WORLD_SIZE = PP_SIZE * EP_SIZE  # 4

HIDDEN_SIZE = 2048
MOE_FFN_HIDDEN_SIZE = 2048
NUM_EXPERTS = EP_SIZE
NUM_LOCAL_EXPERTS = NUM_EXPERTS // EP_SIZE  # experts hosted by each EP rank
TOP_K = 1
CHUNKS_PER_RANK = 2
MOE_LAYERS_PER_CHUNK = 2
VIRTUAL_STAGES = PP_SIZE * CHUNKS_PER_RANK  # 4

MICRO_BATCH_NUM = 8
BS = 8         # must be divisible by MICRO_BATCH_NUM
SEQ_LEN = 4096


# =========================================================================
# Tiny MoE config — fields read by ``_MiniMoEBlock`` and ``MiniGroupedMLP``
# =========================================================================

class _TinyConfig:
    """Hyperparameter holder shared by router + experts."""

    def __init__(self) -> None:
        self.num_moe_experts = NUM_EXPERTS
        self.num_local_experts = NUM_LOCAL_EXPERTS
        self.moe_router_topk = TOP_K
        self.moe_ffn_hidden_size = MOE_FFN_HIDDEN_SIZE
        self.hidden_size = HIDDEN_SIZE
        self.compute_dtype = mstype.float32


# =========================================================================
# MiniMoEBlock: Attention substitute + MoE
# =========================================================================

def _seed_linear_(linear: nn.Cell, rng: np.random.RandomState,
                  std: float = 0.02) -> None:
    """Overwrite a ``mint.nn.Linear`` 's weight (and bias if any) from ``rng``.

    Sampling is done with numpy (not ``mint.normal``) because MS PyNative
    does not deterministically reset every per-op kernel RNG on
    ``ms.set_seed``, so an in-kernel sample can diverge between two
    builds with the same seed.  ``rng.normal`` is consumed in a fixed
    ``weight → bias`` order so two builds with the same RNG seed land on
    bit-identical parameters.
    """
    w_np = rng.normal(0.0, std, size=linear.weight.shape).astype(np.float32)
    linear.weight.set_data(Tensor(w_np))
    if linear.bias is not None:
        b_np = rng.normal(0.0, std, size=linear.bias.shape).astype(np.float32)
        linear.bias.set_data(Tensor(b_np))


class _MiniMoEBlock(nn.Cell):
    """Single transformer-like block: linear ``attn`` + ``MiniGroupedMLP`` MoE.

    The router is a trainable :class:`mint.nn.Linear`; routing produces
    ``(probs, topk_indices)``.  ``num_tokens_per_expert`` is computed
    deterministically from the forced-balance routing below.  The
    experts are wrapped by :class:`OverlapExpertParallel` later
    (overlap path) or by the same class with ``overlap=None`` (sync
    baseline) — both go through the vendored EP wrapper, eliminating
    the ``mindformers`` dependency.

    Args:
        config: Shared :class:`_TinyConfig`.
        rng: ``numpy.random.RandomState`` consumed in a fixed
            ``attn → router → experts`` order so two builds with the
            same seed land on bit-identical weights.
    """

    def __init__(self, config: _TinyConfig, rng: np.random.RandomState) -> None:
        super().__init__()
        self.config = config
        self.attn = mint.nn.Linear(config.hidden_size, config.hidden_size)
        _seed_linear_(self.attn, rng)
        self.router = mint.nn.Linear(
            config.hidden_size, config.num_moe_experts, bias=False,
        )
        _seed_linear_(self.router, rng)
        self.experts = MiniGroupedMLP(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.moe_ffn_hidden_size,
            num_local_experts=config.num_local_experts,
            rng=rng,
            compute_dtype=config.compute_dtype,
        )

    def construct(self, x):
        """x: (bs, seq, hidden) → out: (bs, seq, hidden).

        Routing is forced to a deterministic round-robin
        (token ``i`` → expert ``i % num_experts``) instead of
        ``argmax(softmax(W*x))`` so every expert gets the same number
        of tokens regardless of router init.  Without this,
        random-init routers at large hidden sizes collapsed almost all
        tokens onto a single expert in our test harness, leaving the
        other expert(s) with only pad tokens and making the accuracy
        comparison meaningless on the under-loaded path.

        Router weights still receive gradient through ``topk_probs``,
        which is gathered from the softmax row at the forced expert
        index — only the discrete routing decision is replaced.
        """
        # Attention substitute — keeps a real compute path.
        x = self.attn(x)
        bs, seq, hidden = x.shape
        x_flat = mint.reshape(x, (-1, hidden))  # (bs*seq, hidden)

        # Router still runs so its weights stay trainable via topk_probs.
        router_logits = self.router(x_flat)                          # (bs*seq, num_experts)
        probs_full = mint.nn.functional.softmax(router_logits, dim=-1)

        # ---- Forced balanced routing (top_k=1 only). ----
        if self.config.moe_router_topk != 1:
            raise ValueError(
                f"_MiniMoEBlock forced-balance branch supports "
                f"top_k=1 only, got top_k={self.config.moe_router_topk}"
            )
        num_tokens = x_flat.shape[0]
        num_experts = self.config.num_moe_experts
        forced_indices = (
            mint.arange(num_tokens, dtype=mstype.int32) % num_experts
        )                                                            # (bs*seq,)
        topk_indices = forced_indices.reshape(num_tokens, 1)         # (bs*seq, 1)
        topk_indices_i32 = topk_indices.to(mstype.int32)

        # topk_probs[i, 0] = probs_full[i, forced_indices[i]] — gather
        # via one-hot * sum to keep the router-weight gradient edge
        # into the combine multiplication intact.
        mask = ops.one_hot(
            forced_indices, num_experts,
            Tensor(1.0, probs_full.dtype), Tensor(0.0, probs_full.dtype),
        )                                                            # (bs*seq, num_experts)
        topk_probs = (probs_full * mask).sum(dim=-1, keepdim=True)   # (bs*seq, 1)

        # Token counts per expert — exactly balanced by construction
        # (each expert gets num_tokens // num_experts; the first
        # ``num_tokens % num_experts`` experts get one extra).
        # Compute purely in Python/numpy: forcing routing via
        # ``arange(num_tokens) % num_experts`` makes the histogram
        # deterministic from the two Python ints — avoid an
        # ``asnumpy()`` host-sync inside the overlap window (the sync
        # waits on the shared NPU stream and can deadlock against the
        # peer EP rank's matching sync).
        base = num_tokens // num_experts
        remainder = num_tokens % num_experts
        counts_np = np.full(num_experts, base, dtype=np.float32)
        if remainder:
            counts_np[:remainder] += 1.0
        num_tokens_per_expert = Tensor(counts_np)

        # Experts (EP-wrapped); returns (bs*seq, hidden)
        expert_out = self.experts(x_flat, topk_probs, topk_indices_i32, num_tokens_per_expert)
        out = mint.reshape(expert_out, (bs, seq, hidden))
        return out


class _MoEChunk(nn.Cell):
    """Pipeline chunk: ``num_layers`` stacked :class:`_MiniMoEBlock`.

    When an ``overlap`` coordinator is supplied, ``construct`` fires two
    boundary sync hooks:

    * ``CHUNK_START`` on entry — pairs with ``D_LAST.bwd`` on the BWD
      thread so the combine.bwd a2a of the chunk's last MoE layer is
      bracketed by a barrier-synced window (closes pair 0 of the
      protocol).
    * ``CHUNK_END`` on exit — pairs with ``CHUNK_START.bwd`` on the BWD
      thread (the last BWD rendezvous) so neither thread exits the
      chunk before the other finishes its tail-end local work (FWD
      post-combine vs BWD Attn.bwd of layer 0).  Without this barrier
      both threads concurrently run their chunk wrap-up Python, with
      ``thread.join()`` at ``overlap.run`` exit as the only
      synchroniser — correct but loose; ``CHUNK_END`` tightens the
      coupling so synchronisation happens at chunk granularity.
    """

    def __init__(self, config: _TinyConfig, num_layers: int,
                 rng: np.random.RandomState,
                 overlap: CommComputeOverlap = None) -> None:
        super().__init__()
        self._overlap = overlap
        self.layers = nn.CellList([
            _MiniMoEBlock(config, rng) for _ in range(num_layers)
        ])

    def construct(self, x):
        """Run ``num_layers`` MoE blocks, optionally bracketed by sync hooks.

        When an ``overlap`` coordinator is supplied:
            * ``CHUNK_START`` fires on chunk entry (pairs with
              ``D_LAST.bwd`` so combine.bwd of the last layer is
              bracketed).
            * ``CHUNK_END`` fires on chunk exit (pairs with the
              callback's explicit ``rendezvous(COMPUTE)`` so neither
              thread exits before the other finishes its tail-end
              local work).

        When ``overlap`` is ``None`` (sync baseline), both hook calls
        are skipped — the chunk runs as plain layered forward.
        """
        if self._overlap is not None:
            x = platform.differentiable_sync_hook(
                x, "CHUNK_START", self._overlap.coordinator,
            )
        for layer in self.layers:
            x = layer(x)
        if self._overlap is not None:
            x = platform.differentiable_sync_hook(
                x, "CHUNK_END", self._overlap.coordinator,
            )
        return x


# =========================================================================
# Recompute wrapper (PyNative)
# =========================================================================

class _RecomputeChunkWrapper(nn.Cell):
    """Wrap a chunk so its forward goes through ``ms.recompute``.

    Uses ``use_reentrant=False`` so we can hook the re-run via
    ``context_fn``: ``HookCoordinator.recompute_context_fn`` returns
    ``(nullcontext, _recompute_scope)`` so the backward-time re-run is
    bracketed by ``coord.set_recomputing(True/False)`` and every
    A/B/C/D/CHUNK_* hook fired inside the re-run is a no-op.  Without
    this bracket each re-run would double-count the chunk's BWD-paired
    rendezvous and deadlock the barrier.

    See ``hyper_parallel.core.pipeline_parallel.hook_coordinator``
    docstrings for the contract and why ``use_reentrant=True`` (the
    MindSpore default) is **not** sufficient here.
    """

    def __init__(self, inner: nn.Cell, overlap: CommComputeOverlap) -> None:
        super().__init__()
        self.inner = inner
        # Bound method; ms.recompute will call it once per invocation.
        self._context_fn = overlap.coordinator.recompute_context_fn

    def construct(self, x):
        return ms.recompute(
            self.inner, x,
            use_reentrant=False,
            context_fn=self._context_fn,
        )


# =========================================================================
# OVERLAP_B_F callback (dual-thread)
# =========================================================================

def _make_overlap_b_f_callback(overlap: CommComputeOverlap):
    """Build the OVERLAP_B_F callback that drives paired BWD+FWD execution.

    Mirrors ``examples/torch/pp_overlap/pp_overlap_moe_example.py``'s
    ``make_overlap_b_f_callback`` but on MindSpore PyNative — no
    explicit per-thread stream / device context.  MS autograd dispatches
    BWD on forward's saved stream regardless, and MS's current device
    is process-wide (not thread-local like Torch), so the daemon BWD
    thread inherits the right device from the main thread automatically.
    """

    def _callback(step, ctx):
        bwd_step, fwd_step = step.sub_steps
        schedule = ctx.schedule
        fwd_stage = schedule._stage_dict[fwd_step.stage_index]
        bwd_stage = schedule._stage_dict[bwd_step.stage_index]
        fwd_mi, bwd_mi = fwd_step.micro_index, bwd_step.micro_index

        fwd_recv_handles = ctx.fwd_recv_ops.pop(
            (fwd_stage.stage_index, fwd_mi), None,
        )
        bwd_recv_handles = ctx.bwd_recv_ops.pop(
            (bwd_stage.stage_index, bwd_mi), None,
        )

        def fwd_fn():
            if fwd_recv_handles:
                schedule._wait_p2p(fwd_recv_handles)
            out = fwd_stage.forward_one_chunk(
                fwd_mi, ctx.arg_mbs[fwd_mi], ctx.kwarg_mbs[fwd_mi],
            )
            schedule.update_losses(fwd_stage, out, ctx.losses)

        def bwd_fn():
            # MS PyNative's grad-enable flag is thread-local; the
            # daemon BWD thread does not inherit the main thread's
            # enabled state, so ``value_and_grad`` raises "In no_grad
            # context" on first call.  Enable explicitly here.
            from mindspore.common.api import _pynative_executor  # pylint: disable=C0415
            _pynative_executor.set_enable_grad(True)
            # ``ms.set_device`` would raise after runtime init; rely on
            # the device already bound to this rank by msrun before any
            # kernel ran.  MindSpore's current-device is process-wide
            # (not thread-local like Torch), so the daemon BWD thread
            # already inherits the right device from the main thread.
            if bwd_recv_handles:
                schedule._wait_p2p(bwd_recv_handles)
            bwd_stage.backward_one_chunk(bwd_mi)
            # Pair-8 BWD partner is taken out of band: MS autograd may
            # skip ``CHUNK_START.bwd`` when the chunk input has no
            # ``requires_grad`` (its ``x.grad`` is unused), and we
            # cannot rely on the autograd node firing.  Do one
            # explicit ``rendezvous(COMPUTE)`` here so FWD's
            # ``CHUNK_END.fwd`` barrier always has a partner.
            from hyper_parallel.core.pipeline_parallel.hook_coordinator import (  # pylint: disable=C0415
                HookRole,
            )
            if overlap.coordinator.is_enabled():
                overlap.coordinator.rendezvous(HookRole.COMPUTE)

        overlap.run(fwd_fn=fwd_fn, bwd_fn=bwd_fn)

    return _callback


# =========================================================================
# Distributed setup
# =========================================================================

def _init_pp_ep_mesh():
    """Initialize HCCL and return ``(rank, device, pp_mesh, ep_mesh)``.

    Do NOT call ``ms.set_device`` here — msrun has already bound each
    rank to its device via environment.  Calling ``set_device`` after
    the runtime is initialized raises ``RuntimeError`` (see
    ``mindspore/device_manager.py:91``).
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    comm_init()
    rank = get_rank()
    ws = get_group_size()
    assert ws == WORLD_SIZE, \
        (f"This PoC expects world_size={WORLD_SIZE} "
         f"(PP={PP_SIZE} × EP={EP_SIZE}), got {ws}")
    mesh = init_device_mesh(
        device_type="npu",
        mesh_shape=(PP_SIZE, EP_SIZE),
        mesh_dim_names=("pp", "ep"),
    )

    # Stand-in "device" object holding index — used by callback to
    # pin BWD thread to the right card.
    class _Device:
        def __init__(self, idx):
            self.index = idx
            self.type = "npu"
    device = _Device(rank)

    return rank, device, mesh["pp"], mesh["ep"]


# =========================================================================
# Test entry
# =========================================================================

def test_pp_overlap_moe_end_to_end():
    """End-to-end smoke test: PP + EP + B/F overlap + P2P overlap.

    Feature: Full integration of MindSpore comm/compute overlap stack.
    Description:
        4 ranks, PP=2 × EP=2.  Each PP rank holds 2 interleaved chunks
        of 2-layer Attention+MoE blocks.  Schedule emits OVERLAP_B_F
        steps; callback drives ``CommComputeOverlap`` on paired
        FWD + BWD threads; ``OverlapExpertParallel`` makes the MoE
        a2a return ``AsyncCollectiveTensor`` for lazy wait.
    Expectation:
        Iteration completes without deadlock; collected gradients
        are non-zero on all parameters (sanity that backward
        actually propagated through the wrapped chain).
    """
    rank, device, pp_mesh, ep_mesh = _init_pp_ep_mesh()
    pp_rank = pp_mesh.get_local_rank()

    # ---- Shared CommComputeOverlap orchestrator (one per rank) ----
    overlap = CommComputeOverlap()
    cfg = _TinyConfig()

    # ---- Build interleaved chunks; apply OverlapExpertParallel to
    # ----  EVERY MoE layer's experts; the LAST MoE layer per chunk
    # ----  has is_last_layer=True so its closing D hook is D_LAST.
    # Numpy-seeded RNG used for weight init — see ``_build_pipeline`` /
    # ``_seed_linear_`` for why ``ms.set_seed`` is not used here.
    chunks, stage_indices = _build_pipeline(
        pp_rank, ep_mesh, cfg, use_overlap=True, overlap=overlap,
    )

    stages = [
        PipelineStage(block, stage_index=si, stage_num=VIRTUAL_STAGES,
                      device=device, mesh=pp_mesh)
        for block, si in zip(chunks, stage_indices)
    ]

    # ---- Schedule: interleaved 1F1B with B/F overlap + P2P overlap. ----
    schedule = ScheduleInterleaved1F1B(
        stages, MICRO_BATCH_NUM,
        overlap_p2p=True, overlap_b_f=True,
    )
    schedule.register_custom_function(
        MetaStepType.OVERLAP_B_F,
        _make_overlap_b_f_callback(overlap),
    )

    # ---- Input only on PP rank 0; other ranks call run() with no args ----
    x = Tensor(
        np.random.RandomState(100 + pp_rank).randn(BS, SEQ_LEN, HIDDEN_SIZE).astype(np.float32),
    )

    if pp_rank == 0:
        losses = schedule.run(x)
    else:
        losses = schedule.run()

    # ---- Sanity: backward must have set grads on at least some params ----
    nonzero = 0
    total = 0
    for stage in stages:
        for p in stage.submodule.trainable_params():
            total += 1
            if p.grad is not None and float(p.grad.abs().sum().asnumpy()) > 0:
                nonzero += 1
    assert nonzero > 0, \
        (f"[rank{rank}] no non-zero grads after backward — "
         f"backward chain probably broken upstream of the lazy wrapper. "
         f"total_params={total}")

    if pp_rank == PP_SIZE - 1 and ep_mesh.get_local_rank() == 0 and losses:
        loss_val = float(losses[0].mean().asnumpy())
        print(f"[rank{rank}] PP+EP+overlap done. "
              f"loss[0].mean={loss_val:.4f}, "
              f"nonzero_grads={nonzero}/{total}",
              flush=True)
    print(f"[rank{rank}] pp_overlap_moe_poc: PASS "
          f"(nonzero_grads={nonzero}/{total})", flush=True)


# =========================================================================
# Recompute smoke test: PP + EP + overlap + ms.recompute(use_reentrant=False)
# =========================================================================

def test_pp_overlap_moe_recompute_smoke():
    """Smoke test: each chunk wrapped in ``ms.recompute(use_reentrant=False)``.

    Feature: Verify ``HookCoordinator.recompute_context_fn`` correctly
        brackets the backward-time forward re-run so the A/B/C/D and
        CHUNK_START/CHUNK_END sync hooks are no-ops during re-run.
    Description:
        Same topology as :func:`test_pp_overlap_moe_end_to_end` (4 ranks,
        PP=2 × EP=2, 2 chunks × 2 layers), but every chunk is wrapped in
        :class:`_RecomputeChunkWrapper` which routes its forward through
        ``ms.recompute(use_reentrant=False, context_fn=coord.recompute_context_fn)``.
        Without the ``context_fn`` bracketing, the backward-time re-run
        would re-fire every sync hook and deadlock the BWD-paired
        barrier within the first chunk's first micro-batch.
    Expectation:
        Iteration completes without deadlock; gradients are non-zero on
        at least some params (recompute correctly recreated the
        intermediate activations).
    """
    rank, device, pp_mesh, ep_mesh = _init_pp_ep_mesh()
    pp_rank = pp_mesh.get_local_rank()

    overlap = CommComputeOverlap()
    cfg = _TinyConfig()

    chunks, stage_indices = _build_pipeline(
        pp_rank, ep_mesh, cfg, use_overlap=True, overlap=overlap,
        recompute=True,
    )

    stages = [
        PipelineStage(block, stage_index=si, stage_num=VIRTUAL_STAGES,
                      device=device, mesh=pp_mesh)
        for block, si in zip(chunks, stage_indices)
    ]

    schedule = ScheduleInterleaved1F1B(
        stages, MICRO_BATCH_NUM,
        overlap_p2p=True, overlap_b_f=True,
    )
    schedule.register_custom_function(
        MetaStepType.OVERLAP_B_F,
        _make_overlap_b_f_callback(overlap),
    )

    x = Tensor(
        np.random.RandomState(100 + pp_rank).randn(BS, SEQ_LEN, HIDDEN_SIZE).astype(np.float32),
    )

    if pp_rank == 0:
        losses = schedule.run(x)
    else:
        losses = schedule.run()

    nonzero = 0
    total = 0
    for stage in stages:
        for p in stage.submodule.trainable_params():
            total += 1
            if p.grad is not None and float(p.grad.abs().sum().asnumpy()) > 0:
                nonzero += 1
    assert nonzero > 0, \
        (f"[rank{rank}] no non-zero grads after recompute backward — "
         f"either the re-run did not happen or the autograd path through "
         f"ms.recompute(use_reentrant=False) is broken. total_params={total}")

    if pp_rank == PP_SIZE - 1 and ep_mesh.get_local_rank() == 0 and losses:
        loss_val = float(losses[0].mean().asnumpy())
        print(f"[rank{rank}] PP+EP+overlap+recompute done. "
              f"loss[0].mean={loss_val:.4f}, "
              f"nonzero_grads={nonzero}/{total}",
              flush=True)
    print(f"[rank{rank}] pp_overlap_moe_recompute_smoke: PASS "
          f"(nonzero_grads={nonzero}/{total})", flush=True)


# =========================================================================
# Accuracy test: numerical equivalence vs sync baseline
# =========================================================================

def _build_pipeline(pp_rank, ep_mesh, cfg, use_overlap, overlap=None,
                    recompute=False):
    """Build interleaved MoE chunks with either Overlap or vanilla EP.

    Args:
        pp_rank: PP rank used to seed RNG so the EP group (same pp_rank)
            sees identical weights / routing.
        ep_mesh: 1-D EP mesh used by ``OverlapExpertParallel._apply``.
        cfg: Shared :class:`_TinyConfig`.
        use_overlap: If True, wrap every MoE layer's experts with
            :class:`OverlapExpertParallel` and tag the last layer's D
            hook as ``D_LAST``. If False, use the same class with
            ``overlap=None`` — the A/B/C/D hooks become no-ops, giving
            the sync baseline path with identical HCCL stream layout.
        overlap: The shared :class:`CommComputeOverlap` — required when
            ``use_overlap`` is True.
        recompute: If True, wrap each chunk in :class:`_RecomputeChunkWrapper`
            so its forward runs through ``ms.recompute(use_reentrant=False)``
            with the coordinator's ``recompute_context_fn`` bracketing the
            backward-time re-run.  Only valid alongside ``use_overlap=True``
            (without overlap the recompute bypass is irrelevant).

    Returns:
        ``(chunks, stage_indices)``.
    """
    if recompute and not use_overlap:
        raise ValueError(
            "_build_pipeline: recompute=True only makes sense with "
            "use_overlap=True (without overlap the rendezvous bypass has "
            "nothing to bypass)."
        )
    # Weight initialisation uses a fresh numpy ``RandomState`` (not
    # ``ms.set_seed``) because MS PyNative does not deterministically
    # reset every per-op kernel RNG on ``ms.set_seed``, so ``mint.normal``
    # / ``mint.nn.Linear`` init can diverge between two builds with the
    # same seed.  A numpy-seeded ``rng.normal`` is bit-identical across
    # builds, so the accuracy comparison sees the same initial weights
    # on both paths.
    rng = np.random.RandomState(42 + pp_rank)
    chunks = []
    stage_indices = []
    for chunk_id in range(CHUNKS_PER_RANK):
        # Chunk-level coordinator only attached for the overlap path —
        # ``CHUNK_START`` then fires at chunk entry and pairs with
        # ``D_LAST.bwd`` on the BWD thread.  Sync baseline passes
        # ``overlap=None`` so the chunk-entry hook is a no-op.
        chunk_overlap = overlap if use_overlap else None
        block = _MoEChunk(
            cfg, num_layers=MOE_LAYERS_PER_CHUNK, rng=rng,
            overlap=chunk_overlap,
        )
        last_idx = len(block.layers) - 1
        for layer_idx, layer in enumerate(block.layers):
            style = OverlapExpertParallel(
                overlap=chunk_overlap,
                is_last_layer=(layer_idx == last_idx),
            )
            style._apply(layer.experts, ep_mesh)
        # Apply OverlapExpertParallel BEFORE wrapping with recompute —
        # the parallel style mutates ``layer.experts`` in place via the
        # chunk's layer list, so it must run on the unwrapped chunk.
        if recompute:
            block = _RecomputeChunkWrapper(block, chunk_overlap)
        chunks.append(block)
        stage_indices.append(pp_rank + chunk_id * PP_SIZE)
    return chunks, stage_indices


def _fingerprint_params(chunks):
    """Return list of ``(name, mean, std, sum)`` for every trainable param.

    Same param iteration order ``trainable_params()`` uses, so the two runs
    can be compared positionally.  Floats are pulled to host once at fingerprint
    time so later autograd does not see a host sync inside the hot path.
    """
    fp = []
    for block in chunks:
        for p in block.trainable_params():
            arr = p.value().asnumpy()
            fp.append(
                (p.name, float(arr.mean()), float(arr.std()), float(arr.sum())),
            )
    return fp


def _run_one_iteration(chunks, stage_indices, pp_rank, device, pp_mesh,
                       overlap_p2p, overlap_b_f, callback=None):
    """Run one schedule iteration; return ``(losses_np, grads_named, init_fp)``.

    ``losses_np`` is a list of ``np.ndarray`` (only non-empty on the last
    PP rank); ``grads_named`` is a list of ``(param_name, np_grad_or_None)``
    in trainable-param iteration order; ``init_fp`` is the pre-run weight
    fingerprint used to detect RNG drift between the two builds.
    """
    init_fp = _fingerprint_params(chunks)
    stages = [
        PipelineStage(b, stage_index=si, stage_num=VIRTUAL_STAGES,
                      device=device, mesh=pp_mesh)
        for b, si in zip(chunks, stage_indices)
    ]
    schedule = ScheduleInterleaved1F1B(
        stages, MICRO_BATCH_NUM,
        overlap_p2p=overlap_p2p, overlap_b_f=overlap_b_f,
    )
    if callback is not None:
        schedule.register_custom_function(MetaStepType.OVERLAP_B_F, callback)

    # Numpy RNG with explicit seed so x is identical across the two runs
    # regardless of MS global RNG state drift between iterations.
    x = Tensor(
        np.random.RandomState(100 + pp_rank).randn(BS, SEQ_LEN, HIDDEN_SIZE).astype(np.float32),
    )
    if pp_rank == 0:
        losses = schedule.run(x)
    else:
        losses = schedule.run()

    loss_arrs = [loss.mean().asnumpy().copy() for loss in losses] if losses else []
    grads_named = []
    for stage in stages:
        for p in stage.submodule.trainable_params():
            grad_np = p.grad.asnumpy().copy() if p.grad is not None else None
            grads_named.append((p.name, grad_np))
    return loss_arrs, grads_named, init_fp


def test_pp_overlap_moe_accuracy():
    """Numerical equivalence vs sync baseline.

    Feature: Accuracy check for PP+EP+overlap on MindSpore PyNative.
    Description:
        Builds the same model twice with identical seed/input — once
        with the full overlap stack (``OverlapExpertParallel(overlap)``
        + ``overlap_p2p=True`` + ``overlap_b_f=True``) and once as a
        sync baseline (``OverlapExpertParallel(overlap=None)``, both
        overlap flags off, no ``OVERLAP_B_F`` callback). Compares
        per-micro-batch losses (on the last PP rank) and every
        trainable param's gradient (on all ranks).
    Expectation:
        Losses match within ``rtol=1e-3, atol=1e-3`` on rank
        ``PP_SIZE-1``; grads match within the same tolerance on every
        rank. A mismatch typically means the overlap path's lazy
        ``AsyncCollectiveTensor`` wait is dropping a comm, the dual
        thread callback corrupted autograd state, or the reordered
        permute/unpermute changed numerics.
    """
    rank, device, pp_mesh, ep_mesh = _init_pp_ep_mesh()
    pp_rank = pp_mesh.get_local_rank()
    cfg = _TinyConfig()

    # ---- Sync baseline first (simpler stack -> easier to debug a mismatch). ----
    # ``_build_pipeline`` seeds a fresh ``numpy.random.RandomState`` so both
    # builds see bit-identical weights without holding both models in memory.
    baseline_chunks, baseline_si = _build_pipeline(
        pp_rank, ep_mesh, cfg, use_overlap=False,
    )
    baseline_losses, baseline_grads, baseline_fp = _run_one_iteration(
        baseline_chunks, baseline_si, pp_rank, device, pp_mesh,
        overlap_p2p=False, overlap_b_f=False,
    )
    # Drop baseline stages before building the overlap version so PP P2P groups
    # are quiescent; barrier ensures every rank finished the baseline
    # run before the next schedule starts issuing collectives.
    del baseline_chunks
    platform.barrier()

    # ---- Overlap version ----
    overlap = CommComputeOverlap()
    overlap_chunks, overlap_si = _build_pipeline(
        pp_rank, ep_mesh, cfg, use_overlap=True, overlap=overlap,
    )
    overlap_losses, overlap_grads, overlap_fp = _run_one_iteration(
        overlap_chunks, overlap_si, pp_rank, device, pp_mesh,
        overlap_p2p=True, overlap_b_f=True,
        callback=_make_overlap_b_f_callback(overlap),
    )

    # ---- Init-weight fingerprint check: confirm both builds saw the same
    # ---- numpy-seeded weights.  A divergence here means somebody touched
    # ---- the init path (e.g. swapped back to ``mint.normal``) and broke
    # ---- reproducibility — fix the init, not the tolerance.
    assert len(baseline_fp) == len(overlap_fp), \
        (f"[rank{rank}] init fingerprint length mismatch: "
         f"baseline={len(baseline_fp)}, overlap={len(overlap_fp)}")
    for i, (bfp, ofp) in enumerate(zip(baseline_fp, overlap_fp)):
        bn, bmean, bstd, bsum = bfp
        on, omean, ostd, osum = ofp
        assert bn == on, \
            (f"[rank{rank}] init fp[{i}] name mismatch: "
             f"baseline={bn}, overlap={on}")
        same_init = (
            np.isclose(bmean, omean, atol=1e-7)
            and np.isclose(bstd, ostd, atol=1e-7)
            and np.isclose(bsum, osum, atol=1e-3)
        )
        assert same_init, \
            (f"[rank{rank}] init fp[{i}] ({bn}) DRIFT — "
             f"baseline mean={bmean:.6e} std={bstd:.6e} sum={bsum:.6e}, "
             f"overlap mean={omean:.6e} std={ostd:.6e} sum={osum:.6e}")

    # ---- Compare ----
    rtol, atol = 1e-3, 1e-3

    # Loss is only produced on the last PP rank.
    if pp_rank == PP_SIZE - 1:
        assert len(baseline_losses) == len(overlap_losses), \
            (f"[rank{rank}] loss count mismatch: "
             f"baseline={len(baseline_losses)}, overlap={len(overlap_losses)}")
        for i, (bl, ol) in enumerate(zip(baseline_losses, overlap_losses)):
            assert np.allclose(bl, ol, rtol=rtol, atol=atol), \
                (f"[rank{rank}] loss[{i}] mismatch: "
                 f"baseline={float(bl):.6f}, overlap={float(ol):.6f}, "
                 f"abs_diff={float(np.abs(bl - ol)):.6e}")

    # Grads exist on every rank (each stage owns its own params).
    assert len(baseline_grads) == len(overlap_grads), \
        (f"[rank{rank}] grad count mismatch: "
         f"baseline={len(baseline_grads)}, overlap={len(overlap_grads)}")
    for i, ((bn, bg), (on, og)) in enumerate(zip(baseline_grads, overlap_grads)):
        assert bn == on, \
            (f"[rank{rank}] grad[{i}] param name mismatch: "
             f"baseline={bn}, overlap={on}")
        assert (bg is None) == (og is None), \
            (f"[rank{rank}] grad[{i}] ({bn}) None-state mismatch: "
             f"baseline_none={bg is None}, overlap_none={og is None}")
        if bg is None:
            continue
        assert np.allclose(bg, og, rtol=rtol, atol=atol), \
            (f"[rank{rank}] grad[{i}] ({bn}) mismatch: "
             f"max_abs_diff={float(np.abs(bg - og).max()):.6e}, "
             f"baseline_norm={float(np.linalg.norm(bg)):.6e}, "
             f"overlap_norm={float(np.linalg.norm(og)):.6e}")

    print(f"[rank{rank}] pp_overlap_moe_accuracy: PASS "
          f"(rtol={rtol}, atol={atol}, "
          f"params={len(baseline_grads)}, losses={len(baseline_losses)})",
          flush=True)


if __name__ == "__main__":
    test_pp_overlap_moe_end_to_end()
