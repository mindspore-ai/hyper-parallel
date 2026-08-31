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
from mindspore.profiler import (
    ProfilerActivity,
    ProfilerLevel,
    _ExperimentalConfig,
    profile,
    tensorboard_trace_handler,
)
from mindspore.profiler import schedule as profiler_schedule

from hyper_parallel import PipelineStage, init_device_mesh
from hyper_parallel.core.pipeline_parallel import (
    CommComputeOverlap,
    MetaStepType,
    ScheduleInterleaved1F1B,
)
from hyper_parallel.platform import get_platform
from hyper_parallel.core.activation_checkpoint.activation_checkpoint import (
    checkpoint, checkpoint_wrapper, CheckpointPolicy,
)

from tests.mindspore.st.pipeline_parallel.overlap_expert_parallel import (
    MiniGroupedMLP,
    OverlapExpertParallel,
)


platform = get_platform()


# =========================================================================
# Hyperparams
# =========================================================================

PP_SIZE = int(os.environ.get("PP_OVERLAP_PP_SIZE", "4"))
EP_SIZE = int(os.environ.get("PP_OVERLAP_EP_SIZE", "2"))
WORLD_SIZE = PP_SIZE * EP_SIZE  # default 8 (PP4xEP2); env-overridable for 4-card runs

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

# dx/dw accuracy + profiling loop (env-overridable so hardware runs can tune
# the wall-clock / timeout trade-off without editing the file).
NUM_STEPS = int(os.environ.get("PP_OVERLAP_NUM_STEPS", "100"))
PROFILE_STEP = int(os.environ.get("PP_OVERLAP_PROFILE_STEP", "5"))


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


class _MiniAttnSubstitute(nn.Cell):
    """Attention-shaped compute placeholder (NOT real attention).

    Replaces the single ``Linear(h, h)`` stand-in with an MLP-shaped
    block (``h → 4h → GELU → h``) so per-layer compute mass is large
    enough for the ``OVERLAP_B_F`` window to actually have compute to
    overlap against the EP a2a — the original single Linear was
    ``~bs*seq*h^2`` FLOPs (e.g. ~17 GFLOPs at ``h=2048, seq=4096, bs=1``),
    well below a single dispatch/combine a2a in wall time, so the
    paired FWD thread finished its compute almost immediately and
    spent the rest of the window waiting on comm.

    Why MLP shape, not real multi-head attention: a faithful MHA would
    materialise an ``(bs, num_heads, seq, seq)`` attention matrix in
    fp32 which at ``seq=4096`` is ~1 GB per layer per micro-batch and
    OOMs the test card.  Fused SDPA / FlashAttention would fix the
    memory but adds an Ascend-specific op dependency the PoC has been
    careful to avoid.  The MLP shape gives ~``8*bs*seq*h^2`` FLOPs at
    bounded memory and matches the original code's intent that this
    module is a stand-in for compute mass, not a faithful reproduction
    of an attention block.  Bias terms are kept (mirroring the original
    ``mint.nn.Linear`` default) so seeding stays self-explanatory.
    """

    def __init__(self, config: '_TinyConfig', rng: np.random.RandomState) -> None:
        super().__init__()
        hidden = config.hidden_size
        intermediate = 4 * hidden
        self.up = mint.nn.Linear(hidden, intermediate)
        _seed_linear_(self.up, rng)
        self.down = mint.nn.Linear(intermediate, hidden)
        _seed_linear_(self.down, rng)

    def construct(self, x):
        x = self.up(x)
        x = mint.nn.functional.gelu(x)
        return self.down(x)


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
        # Use the MLP-shaped attention substitute (see _MiniAttnSubstitute
        # docstring for why MHA isn't used).  Seeding lives inside the
        # cell, so the rng-consumption order here is
        # ``attn.up → attn.down → router → experts``.
        self.attn = _MiniAttnSubstitute(config, rng)
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

        # topk_probs[i, 0] = probs_full[i, forced_indices[i]].
        # ``mint.gather`` (Ascend → ``aclnnGatherV2`` single async kernel)
        # replaces the previous ``one_hot * sum(-1)`` chain: gather is itself
        # differentiable (backward is scatter) with identical gradient
        # semantics on ``probs_full``, so the router-weight edge into the
        # combine multiplication is preserved.  Eliminates two
        # ``Tensor(scalar)`` ctors per call and the
        # ``[bs*seq, num_experts]`` one-hot intermediate; collapses 5
        # dispatches (Tensor + Tensor + one_hot + mul + sum) into 1.
        topk_probs = mint.gather(probs_full, dim=1, index=topk_indices_i32)  # (bs*seq, 1)

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
        # When per-layer recompute is enabled, holds one
        # ``checkpoint_wrapper(layer)`` callable per layer so each layer
        # becomes its OWN checkpoint segment (multi-segment).  None means no
        # per-layer checkpoint (the layer runs directly).
        self._per_layer_calls = None

    def enable_per_layer_recompute(self, recompute_layers=None) -> None:
        """Wrap selected layers' forward in their own ``checkpoint`` segment.

        Multi-segment activation checkpoint with **per-layer granularity**:
        each selected layer re-runs independently during backward (instead of
        one chunk-level re-run), and non-selected layers run directly (their
        activations are kept, not recomputed).  This is exactly the
        "recompute some layers, keep others" pattern that SAC's op-granularity
        ``policy_fn`` cannot express.  The ``CHUNK_START`` / ``CHUNK_END``
        hooks stay OUTSIDE the per-layer checkpoints, and each selected
        layer's forward re-run is fired serially before the paired backward
        by :meth:`PipelineStage.recompute_one_chunk` and reused during
        backward.  Must be called after :class:`OverlapExpertParallel` has
        been applied to each layer's experts so the wrapped call includes the
        EP sync hooks.

        Args:
            recompute_layers: Iterable of layer indices to checkpoint.  When
                ``None`` (default) every layer is checkpointed.  A subset
                (e.g. ``{0}``) leaves the other layers running directly —
                the mixed recompute case.
        """
        if recompute_layers is None:
            recompute_layers = set(range(len(self.layers)))
        else:
            recompute_layers = set(recompute_layers)
        calls = []
        for idx, layer in enumerate(self.layers):
            if idx in recompute_layers:
                calls.append(checkpoint_wrapper(layer))
            else:
                # Non-recomputed layer: run the cell directly (its activations
                # are kept by autograd, no backward-time re-run).
                calls.append(layer)
        self._per_layer_calls = calls

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
        if self._per_layer_calls is not None:
            # Per-layer (multi-segment) checkpoint: each layer is its own
            # re-run segment.
            for call in self._per_layer_calls:
                x = call(x)
        else:
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
    """Wrap a chunk so its forward is checkpointed under overlap_b_f.

    Routes the chunk's forward through
    ``hyper_parallel.core.activation_checkpoint.checkpoint_wrapper`` (plain
    ``ms.recompute``).  The chunk's forward re-run is fired serially before
    the paired backward by :meth:`PipelineStage.recompute_one_chunk` and
    reused during backward, so the re-run never races the FWD thread's
    forward record on the MS PyNative autograd executor.  Exercises exactly
    the path real users hit when combining ``overlap_b_f`` with activation
    checkpoint.
    """

    def __init__(self, inner: nn.Cell) -> None:
        # auto_prefix=False: Cell.__setattr__ would otherwise rename every
        # wrapped param to "inner.<name>", breaking grad-name comparison
        # against the unwrapped baseline (same convention as TrainOneStepCell).
        super().__init__(auto_prefix=False)
        self.inner = inner
        self._wrapped_call = checkpoint_wrapper(self.inner)

    def construct(self, x):
        return self._wrapped_call(x)


# Records (is_recompute,) every time the save-a2a policy classifies an
# all-to-all op, so a test can prove the EP a2a went through SAC's
# save (forward) + restore (recompute) path instead of being re-communicated.
# Per-process (each rank is its own msrun worker).
_A2A_SAC_SEEN: list = []


def _save_a2a_policy(ctx, func, *args, **kwargs):  # pylint: disable=unused-argument
    """SAC policy: keep (MUST_SAVE) the EP all-to-all output, recompute the rest.

    Op-granularity selective activation checkpoint.  The chunk's dispatch /
    combine ``inner_comm_all_to_all_v`` (matched by normalized name) is saved
    in the forward pass and **restored from storage during the backward
    re-run**, so ``recompute_one_chunk`` re-runs only the local compute and
    re-issues no EP HCCL.  Every other op returns MUST_RECOMPUTE.

    Note: this MS SAC's forward (caching) pass accepts only MUST_SAVE /
    PREFER_SAVE / MUST_SWAP / MUST_RECOMPUTE and raises on PREFER_RECOMPUTE
    (``sac.py``), so the non-a2a default must be MUST_RECOMPUTE, not
    PREFER_RECOMPUTE.

    Layer-granularity recompute (``enable_per_layer_recompute``) cannot express
    this — it can only keep/recompute whole layers, not the a2a inside one.
    """
    name = func.name.lower().replace("_", "")
    if "alltoall" in name:
        _A2A_SAC_SEEN.append(bool(getattr(ctx, "is_recompute", False)))
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.MUST_RECOMPUTE


class _RecomputeChunkWrapperSaveA2A(nn.Cell):
    """Chunk recompute that SAVES the EP all-to-all (op-granularity SAC).

    Routes the chunk forward through ``checkpoint(..., policy_fn=_save_a2a_policy)``
    instead of plain ``checkpoint_wrapper``.  The serial re-forward fired by
    :meth:`PipelineStage.recompute_one_chunk` reruns local compute but restores
    the dispatch/combine a2a outputs from storage — no EP HCCL is re-issued
    during recompute.  Exercises the "recompute compute, keep comm" path.
    """

    def __init__(self, inner: nn.Cell) -> None:
        # auto_prefix=False keeps wrapped param names unchanged (no "inner."
        # prefix) so the grad-name comparison against the baseline holds.
        super().__init__(auto_prefix=False)
        self.inner = inner

    def construct(self, x):
        return checkpoint(self.inner, x, policy_fn=_save_a2a_policy)


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

    Dispatches on the bwd sub-step's type:

    * ``BWD`` — ``bwd_fn`` runs the unified ``backward_one_chunk`` and
      the scheduler's ``BWD_SEND`` step issues the gradient send.
    * ``BWD_INPUT`` (``enable_dxdw_split=True``: the schedule rewrote the
      pair via ``split_overlap_dxdw``) — ``bwd_fn`` runs only
      ``backward_input_one_chunk``; the overlap joins at ``max(dx, fwd)``,
      the gap's ``BWD_SEND`` picks up the dx grad from ``bwd_cache``, and
      the matching ``BWD_WEIGHT`` step then runs dw on the main thread
      while that P2P is in flight.
    """

    def _callback(step, ctx):
        bwd_step, fwd_step = step.sub_steps
        schedule = ctx.schedule
        bwd_stage = schedule._stage_dict[bwd_step.stage_index]
        bwd_mi = bwd_step.micro_index

        def fwd_fn():
            # ``forward_one_chunk`` fires the schedule's after-forward hook,
            # which issues the fwd-boundary P2P (p2p_transport="boundary")
            # while the backward is still running — no callback cooperation
            # needed here.  No-op under the default duplex transport.
            schedule.execute_fwd_leaf(
                fwd_step, ctx.arg_mbs, ctx.kwarg_mbs, ctx.losses,
            )

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
            # Top-level span over the whole bwd_fn body so the trace shows the
            # daemon BWD thread actually executing (it brackets wait_bwd_recv,
            # dx/send/dw, and the rendezvous).  If this span is absent from a
            # rank's trace, bwd_fn did not run for that (stage, micro).
            with platform.profiler_record(
                    f"dxdw/bwd_fn/stage_{bwd_stage.stage_index}/mi_{bwd_mi}"):
                # Under enable_dxdw_split the schedule emits (BWD_INPUT, FWD)
                # pairs (stage-0 pairs stay unified BWD: dx would be a no-op
                # there), so dispatch on the sub-step type.  dx writes
                # ``bwd_cache``; the gap's BWD_SEND and the standalone
                # BWD_WEIGHT step that follow the overlap consume it on the
                # main thread.
                schedule.wait_bwd_recv(bwd_stage.stage_index, bwd_mi)
                if bwd_step.type == MetaStepType.BWD_INPUT:
                    bwd_stage.backward_input_one_chunk(bwd_mi)
                else:
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

        # Fire the BWD chunk's recompute serially on the main thread BEFORE
        # submitting to the backward worker, so the forward re-run never
        # races fwd_fn's forward record.  ``backward_one_chunk`` then reuses
        # the cached activations instead of re-running on the daemon thread.
        bwd_stage.recompute_one_chunk(bwd_mi)
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
# Recompute test: PP + EP + overlap + activation checkpoint (serialized)
# =========================================================================

def test_pp_overlap_moe_recompute():
    """Production-path checkpoint integration with overlap_b_f.

    Feature: ``checkpoint_wrapper`` chunk recompute composing with overlap_b_f,
        with the re-run fired serially by ``PipelineStage.recompute_one_chunk``.
    Description:
        Same topology as :func:`test_pp_overlap_moe_end_to_end` (4 ranks,
        PP=2 × EP=2, 2 chunks × 2 layers), but each chunk is wrapped in
        :class:`_RecomputeChunkWrapper`, which uses ``checkpoint_wrapper``.
        The chunk's forward re-run is fired serially before the paired
        backward by :meth:`PipelineStage.recompute_one_chunk` and reused
        during backward, so the re-run never races the FWD thread's forward
        record.  Serializing the re-run is required on MS PyNative, whose
        autograd executor does not support concurrent FWD-record + BWD-replay.
    Expectation:
        Iteration completes without deadlock and produces non-zero grads.
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
        np.random.RandomState(200 + pp_rank).randn(BS, SEQ_LEN, HIDDEN_SIZE).astype(np.float32),
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
         f"recompute_one_chunk did not rebuild/reuse the checkpointed "
         f"activations. total_params={total}, nonzero_grads={nonzero}")

    if pp_rank == PP_SIZE - 1 and ep_mesh.get_local_rank() == 0 and losses:
        loss_val = float(losses[0].mean().asnumpy())
        print(f"[rank{rank}] PP+EP+overlap+recompute done. "
              f"loss[0].mean={loss_val:.4f}, "
              f"nonzero_grads={nonzero}/{total}",
              flush=True)
    print(f"[rank{rank}] pp_overlap_moe_recompute: PASS "
          f"(nonzero_grads={nonzero}/{total})", flush=True)


# =========================================================================
# Accuracy test: numerical equivalence vs sync baseline
# =========================================================================

def _build_pipeline(pp_rank, ep_mesh, cfg, use_overlap, overlap=None,
                    recompute=False, layers_per_chunk=None,
                    recompute_granularity="chunk", recompute_layers=None):
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
            so its forward is checkpointed via ``checkpoint_wrapper``; the
            re-run is fired serially before the paired backward by
            ``PipelineStage.recompute_one_chunk``.  Only exercised alongside
            ``use_overlap=True`` in this PoC.
        layers_per_chunk: Optional list of per-chunk MoE layer counts, one
            entry per interleaved chunk.  When ``None`` every chunk uses
            ``MOE_LAYERS_PER_CHUNK``.  A heterogeneous list (e.g. ``[3, 2]``)
            makes the rank's chunks differ in depth, so each ``OVERLAP_B_F``
            step pairs a BWD and FWD of different layer counts and exercises
            :meth:`HookCoordinator.depart`.
        recompute_granularity: ``"chunk"`` (default) wraps the whole chunk in
            one checkpoint segment via :class:`_RecomputeChunkWrapper`;
            ``"layer"`` wraps each layer in its own segment via
            :meth:`_MoEChunk.enable_per_layer_recompute` (multi-segment).
            Only consulted when ``recompute=True``.
        recompute_layers: Which layers to recompute (``recompute_granularity
            ="layer"`` only).  ``None`` = all layers in every chunk; a single
            set = the same selection for all chunks; a list/tuple of sets =
            one selection per chunk (e.g. ``[{0, 1}, {0}]``).

    Returns:
        ``(chunks, stage_indices)``.
    """
    if recompute and not use_overlap:
        raise ValueError(
            "_build_pipeline: recompute=True is only exercised with "
            "use_overlap=True in this PoC."
        )
    if recompute_granularity not in ("chunk", "layer", "chunk_save_a2a"):
        raise ValueError(
            f"_build_pipeline: recompute_granularity must be 'chunk', 'layer', "
            f"or 'chunk_save_a2a', got {recompute_granularity!r}"
        )
    if layers_per_chunk is None:
        layers_per_chunk = [MOE_LAYERS_PER_CHUNK] * CHUNKS_PER_RANK
    if len(layers_per_chunk) != CHUNKS_PER_RANK:
        raise ValueError(
            f"_build_pipeline: layers_per_chunk must have length "
            f"CHUNKS_PER_RANK={CHUNKS_PER_RANK}, got {len(layers_per_chunk)} "
            f"({layers_per_chunk})"
        )
    # Resolve recompute_layers into one set per chunk.  Accept:
    #   None                  -> every layer recomputed in every chunk
    #   a set / iterable      -> the same selection for all chunks
    #   a list/tuple of sets  -> one entry per chunk (per-chunk selection,
    #                            e.g. [{0, 1}, {0}] for the combined config)
    if isinstance(recompute_layers, (list, tuple)):
        if len(recompute_layers) != CHUNKS_PER_RANK:
            raise ValueError(
                f"_build_pipeline: per-chunk recompute_layers must have length "
                f"CHUNKS_PER_RANK={CHUNKS_PER_RANK}, got {len(recompute_layers)} "
                f"({recompute_layers})"
            )
        per_chunk_recompute = list(recompute_layers)
    else:
        per_chunk_recompute = [recompute_layers] * CHUNKS_PER_RANK
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
            cfg, num_layers=layers_per_chunk[chunk_id], rng=rng,
            overlap=chunk_overlap,
        )
        last_idx = len(block.layers) - 1
        for layer_idx, layer in enumerate(block.layers):
            style = OverlapExpertParallel(
                overlap=chunk_overlap,
                is_last_layer=(layer_idx == last_idx),
                # Enable the ``ops.moe_token_permute`` /
                # ``ops.moe_token_unpermute`` fused fast path — folds the
                # manual sort + fmod + index_select chain into one kernel
                # call to keep per-layer host dispatch out of the
                # OVERLAP_B_F window.  Mirrors mindformers' ExpertParallel
                # invocation with ``moe_permute_fusion=True``.
                moe_permute_fusion=True,
            )
            style._apply(layer.experts, ep_mesh)
        # Apply OverlapExpertParallel BEFORE wrapping with recompute —
        # the parallel style mutates ``layer.experts`` in place via the
        # chunk's layer list, so it must run on the unwrapped chunk.
        if recompute:
            if recompute_granularity == "layer":
                # Multi-segment: selected layers each get their own
                # checkpoint, CHUNK_* hooks stay outside.  ``recompute_layers``
                # selects which layers recompute (None = all); a subset is the
                # mixed "recompute some, keep others" case.
                block.enable_per_layer_recompute(
                    recompute_layers=per_chunk_recompute[chunk_id],
                )
            elif recompute_granularity == "chunk_save_a2a":
                # Op-granularity: recompute the chunk but SAVE the EP a2a so the
                # backward re-run restores it instead of re-communicating.
                block = _RecomputeChunkWrapperSaveA2A(block)
            else:
                block = _RecomputeChunkWrapper(block)
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
                       overlap_p2p, overlap_b_f, callback=None,
                       enable_dxdw_split=False, x_input=None, p2p_transport="auto"):
    """Run one schedule iteration; return ``(losses_np, grads_named, init_fp)``.

    ``losses_np`` is a list of ``np.ndarray`` (only non-empty on the last
    PP rank); ``grads_named`` is a list of ``(param_name, np_grad_or_None)``
    in trainable-param iteration order; ``init_fp`` is the pre-run weight
    fingerprint used to detect RNG drift between the two builds.

    ``x_input`` overrides the stage-0 input tensor; pass the *same* tensor to
    the baseline and split runs of a step so their numerics are comparable.
    When ``None`` a fixed per-rank seed is used (single-iteration callers).
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
        enable_dxdw_split=enable_dxdw_split, p2p_transport=p2p_transport,
    )
    if callback is not None:
        schedule.register_custom_function(MetaStepType.OVERLAP_B_F, callback)

    # Numpy RNG with explicit seed so x is identical across the two runs
    # regardless of MS global RNG state drift between iterations.
    if x_input is None:
        x = Tensor(
            np.random.RandomState(100 + pp_rank).randn(BS, SEQ_LEN, HIDDEN_SIZE).astype(np.float32),
        )
    else:
        x = x_input
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


# =========================================================================
# Variable-layer test: heterogeneous per-chunk depth under overlap_b_f
# =========================================================================

# The rank's two interleaved chunks deliberately differ in depth (3 vs 2
# MoE layers) so every OVERLAP_B_F step in the 1F1B steady state pairs a
# BWD chunk with a FWD chunk of a DIFFERENT layer count.  The dual-thread
# rendezvous counts then differ by a full layer (4 hooks), which without
# ``HookCoordinator.depart`` deadlocks: the shorter chunk's thread returns
# and the longer chunk's thread blocks forever on the 2-party barrier.
VARIABLE_LAYERS_PER_CHUNK = [3, 2]


def test_pp_overlap_moe_variable_layers():
    """Heterogeneous per-chunk layer counts under overlap_b_f (depart path).

    Feature: ``HookCoordinator.depart`` — graceful one-party-left drain
        when paired forward / backward chunks have different layer counts.
    Description:
        8 ranks, PP=4 × EP=2.  Each PP rank holds 2 interleaved chunks with
        ``[3, 2]`` MoE layers instead of the usual uniform ``[2, 2]``, so
        every steady-state ``OVERLAP_B_F`` pairing has ``layers(fwd) !=
        layers(bwd)`` and the two threads fire rendezvous counts that differ
        by a full layer.  Without ``depart`` this deadlocks.  Builds the
        model twice from identical numpy-seeded weights — sync baseline vs
        full overlap stack — and compares every trainable parameter's
        gradient.
    Expectation:
        Iteration completes without deadlock; per-parameter gradients (and
        per-micro-batch losses on the last PP rank) match the sync baseline
        within ``rtol=1e-3, atol=1e-3`` on every rank — proving the
        solo-drained tail of the longer chunk is numerically identical to
        running the two chunks sequentially.
    """
    rank, device, pp_mesh, ep_mesh = _init_pp_ep_mesh()
    pp_rank = pp_mesh.get_local_rank()
    cfg = _TinyConfig()

    # ---- Sync baseline (overlap off), heterogeneous depth. ----
    baseline_chunks, baseline_si = _build_pipeline(
        pp_rank, ep_mesh, cfg, use_overlap=False,
        layers_per_chunk=VARIABLE_LAYERS_PER_CHUNK,
    )
    baseline_losses, baseline_grads, _ = _run_one_iteration(
        baseline_chunks, baseline_si, pp_rank, device, pp_mesh,
        overlap_p2p=False, overlap_b_f=False,
    )
    del baseline_chunks
    platform.barrier()

    # ---- Overlap version with the SAME heterogeneous depth. ----
    overlap = CommComputeOverlap()
    overlap_chunks, overlap_si = _build_pipeline(
        pp_rank, ep_mesh, cfg, use_overlap=True, overlap=overlap,
        layers_per_chunk=VARIABLE_LAYERS_PER_CHUNK,
    )
    overlap_losses, overlap_grads, _ = _run_one_iteration(
        overlap_chunks, overlap_si, pp_rank, device, pp_mesh,
        overlap_p2p=True, overlap_b_f=True,
        callback=_make_overlap_b_f_callback(overlap),
    )

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

    print(f"[rank{rank}] pp_overlap_moe_variable_layers: PASS "
          f"(layers_per_chunk={VARIABLE_LAYERS_PER_CHUNK}, "
          f"rtol={rtol}, atol={atol}, params={len(baseline_grads)})",
          flush=True)


# =========================================================================
# Per-layer (multi-segment) recompute under overlap_b_f.
# =========================================================================

def test_pp_overlap_moe_recompute_per_layer():
    """Per-layer (multi-segment) checkpoint under overlap_b_f — correctness.

    Feature: multi-segment activation checkpoint under ``overlap_b_f``.
    Description:
        8 ranks, PP=4 × EP=2.  Each chunk wraps EACH layer in its OWN
        ``checkpoint`` segment (:meth:`_MoEChunk.enable_per_layer_recompute`).
        All segments' forward re-runs are fired serially before the paired
        backward by :meth:`PipelineStage.recompute_one_chunk` and reused during
        backward, so no re-run record runs concurrently with the FWD thread's
        forward record.  Builds the model twice from identical numpy-seeded
        weights — sync baseline (no overlap, no recompute) vs per-layer
        recompute under the full overlap stack — and compares every trainable
        parameter's gradient.
    Expectation:
        No deadlock; per-parameter gradients (and last-rank per-micro-batch
        losses) match the sync baseline within ``rtol=1e-3, atol=1e-3`` on
        every rank.
    """
    rank, device, pp_mesh, ep_mesh = _init_pp_ep_mesh()
    pp_rank = pp_mesh.get_local_rank()
    cfg = _TinyConfig()

    # ---- Sync baseline: no overlap, no recompute (ground truth). ----
    baseline_chunks, baseline_si = _build_pipeline(
        pp_rank, ep_mesh, cfg, use_overlap=False,
    )
    baseline_losses, baseline_grads, _ = _run_one_iteration(
        baseline_chunks, baseline_si, pp_rank, device, pp_mesh,
        overlap_p2p=False, overlap_b_f=False,
    )
    del baseline_chunks
    platform.barrier()

    # ---- Per-layer (multi-segment) recompute under the full overlap stack. ----
    overlap = CommComputeOverlap()
    overlap_chunks, overlap_si = _build_pipeline(
        pp_rank, ep_mesh, cfg, use_overlap=True, overlap=overlap,
        recompute=True, recompute_granularity="layer",
    )
    overlap_losses, overlap_grads, _ = _run_one_iteration(
        overlap_chunks, overlap_si, pp_rank, device, pp_mesh,
        overlap_p2p=True, overlap_b_f=True,
        callback=_make_overlap_b_f_callback(overlap),
    )

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

    print(f"[rank{rank}] pp_overlap_moe_recompute_per_layer: PASS "
          f"(rtol={rtol}, atol={atol}, params={len(baseline_grads)})",
          flush=True)


# Recompute only the FIRST layer of each chunk, keep the rest.  The first
# layer (forward order) is the LAST processed in backward; its re-run is
# fired serially before the backward by recompute_one_chunk and reused.
# This is the "recompute some layers, keep others" case the user flagged.
MIXED_RECOMPUTE_LAYERS = {0}


def test_pp_overlap_moe_recompute_mixed():
    """Mixed per-layer recompute (some layers recompute, some don't) under overlap_b_f.

    Feature: per-layer checkpoint with a SUBSET of layers recomputed.
    Description:
        8 ranks, PP=4 × EP=2, 2 layers per chunk.  Only layer 0 of each chunk
        is wrapped in a ``checkpoint`` segment (recomputed); layer 1 runs
        directly (activations kept).  This is the asymmetric mixed case that
        op-granularity SAC cannot express.  Builds the model twice from
        identical numpy-seeded weights — sync baseline (no overlap, no
        recompute) vs mixed per-layer recompute under the full overlap stack —
        and compares every trainable parameter's gradient.
    Expectation:
        No deadlock; per-parameter gradients (and last-rank losses) match the
        sync baseline within ``rtol=1e-3, atol=1e-3`` on every rank.
    """
    rank, device, pp_mesh, ep_mesh = _init_pp_ep_mesh()
    pp_rank = pp_mesh.get_local_rank()
    cfg = _TinyConfig()

    # ---- Sync baseline: no overlap, no recompute (ground truth). ----
    baseline_chunks, baseline_si = _build_pipeline(
        pp_rank, ep_mesh, cfg, use_overlap=False,
    )
    baseline_losses, baseline_grads, _ = _run_one_iteration(
        baseline_chunks, baseline_si, pp_rank, device, pp_mesh,
        overlap_p2p=False, overlap_b_f=False,
    )
    del baseline_chunks
    platform.barrier()

    # ---- Mixed recompute (layer 0 only) under the full overlap stack. ----
    overlap = CommComputeOverlap()
    overlap_chunks, overlap_si = _build_pipeline(
        pp_rank, ep_mesh, cfg, use_overlap=True, overlap=overlap,
        recompute=True, recompute_granularity="layer",
        recompute_layers=MIXED_RECOMPUTE_LAYERS,
    )
    overlap_losses, overlap_grads, _ = _run_one_iteration(
        overlap_chunks, overlap_si, pp_rank, device, pp_mesh,
        overlap_p2p=True, overlap_b_f=True,
        callback=_make_overlap_b_f_callback(overlap),
    )

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

    print(f"[rank{rank}] pp_overlap_moe_recompute_mixed: PASS "
          f"(recompute_layers={sorted(MIXED_RECOMPUTE_LAYERS)}, "
          f"rtol={rtol}, atol={atol}, params={len(baseline_grads)})",
          flush=True)


def _assert_overlap_matches_baseline(rank, pp_rank, baseline_losses, baseline_grads,
                                     overlap_losses, overlap_grads, label,
                                     rtol=1e-3, atol=1e-3):
    """Assert overlap losses (last PP rank) and grads (every rank) match baseline.

    Shared comparison used by the stress configs.  ``label`` is interpolated
    into every assertion message so a failure says which config broke.
    """
    if pp_rank == PP_SIZE - 1:
        assert len(baseline_losses) == len(overlap_losses), \
            (f"[rank{rank}] {label} loss count mismatch: "
             f"baseline={len(baseline_losses)}, overlap={len(overlap_losses)}")
        for i, (bl, ol) in enumerate(zip(baseline_losses, overlap_losses)):
            assert np.allclose(bl, ol, rtol=rtol, atol=atol), \
                (f"[rank{rank}] {label} loss[{i}] mismatch: "
                 f"baseline={float(bl):.6f}, overlap={float(ol):.6f}, "
                 f"abs_diff={float(np.abs(bl - ol)):.6e}")
    assert len(baseline_grads) == len(overlap_grads), \
        (f"[rank{rank}] {label} grad count mismatch: "
         f"baseline={len(baseline_grads)}, overlap={len(overlap_grads)}")
    for i, ((bn, bg), (on, og)) in enumerate(zip(baseline_grads, overlap_grads)):
        assert bn == on, \
            (f"[rank{rank}] {label} grad[{i}] name mismatch: "
             f"baseline={bn}, overlap={on}")
        assert (bg is None) == (og is None), \
            (f"[rank{rank}] {label} grad[{i}] ({bn}) None-state mismatch: "
             f"baseline_none={bg is None}, overlap_none={og is None}")
        if bg is None:
            continue
        assert np.allclose(bg, og, rtol=rtol, atol=atol), \
            (f"[rank{rank}] {label} grad[{i}] ({bn}) mismatch: "
             f"max_abs_diff={float(np.abs(bg - og).max()):.6e}, "
             f"baseline_norm={float(np.linalg.norm(bg)):.6e}, "
             f"overlap_norm={float(np.linalg.norm(og)):.6e}")


# 3-layer-per-chunk stress config for mixed recompute: recompute ONLY layer 0,
# keep layers 1 and 2.  Exercises mixed per-layer recompute at a deeper layer
# count than the 2-layer mixed test, with the recomputed layer last in backward.
MIXED_3LAYER_PER_CHUNK = [3, 3]
MIXED_3LAYER_RECOMPUTE_LAYERS = {0}


def test_pp_overlap_moe_recompute_mixed_3layer():
    """Mixed per-layer recompute at a 3-layer-per-chunk config (stress).

    Feature: mixed per-layer recompute robustness at a non-default depth.
    Description:
        8 ranks, PP=4 × EP=2, 3 layers per chunk, only layer 0 recomputed
        (layers 1 and 2 kept).  The recomputed layer's re-run is fired serially
        before the paired backward by ``PipelineStage.recompute_one_chunk`` and
        reused during backward — exercised here at a deeper layer count than the
        2-layer mixed test.  Compares grads against a sync baseline.
    Expectation:
        No deadlock; grads and last-rank losses match the sync baseline within
        ``rtol=1e-3, atol=1e-3``.
    """
    rank, device, pp_mesh, ep_mesh = _init_pp_ep_mesh()
    pp_rank = pp_mesh.get_local_rank()
    cfg = _TinyConfig()

    # ---- Sync baseline: no overlap, no recompute, 3 layers per chunk. ----
    baseline_chunks, baseline_si = _build_pipeline(
        pp_rank, ep_mesh, cfg, use_overlap=False,
        layers_per_chunk=MIXED_3LAYER_PER_CHUNK,
    )
    baseline_losses, baseline_grads, _ = _run_one_iteration(
        baseline_chunks, baseline_si, pp_rank, device, pp_mesh,
        overlap_p2p=False, overlap_b_f=False,
    )
    del baseline_chunks
    platform.barrier()

    # ---- Mixed recompute (layer 0 only) under the full overlap stack. ----
    overlap = CommComputeOverlap()
    overlap_chunks, overlap_si = _build_pipeline(
        pp_rank, ep_mesh, cfg, use_overlap=True, overlap=overlap,
        recompute=True, recompute_granularity="layer",
        recompute_layers=MIXED_3LAYER_RECOMPUTE_LAYERS,
        layers_per_chunk=MIXED_3LAYER_PER_CHUNK,
    )
    overlap_losses, overlap_grads, _ = _run_one_iteration(
        overlap_chunks, overlap_si, pp_rank, device, pp_mesh,
        overlap_p2p=True, overlap_b_f=True,
        callback=_make_overlap_b_f_callback(overlap),
    )

    _assert_overlap_matches_baseline(
        rank, pp_rank, baseline_losses, baseline_grads,
        overlap_losses, overlap_grads,
        label=f"mixed_3layer(recompute={sorted(MIXED_3LAYER_RECOMPUTE_LAYERS)})",
    )
    print(f"[rank{rank}] pp_overlap_moe_recompute_mixed_3layer: PASS "
          f"(layers_per_chunk={MIXED_3LAYER_PER_CHUNK}, "
          f"recompute_layers={sorted(MIXED_3LAYER_RECOMPUTE_LAYERS)}, "
          f"params={len(baseline_grads)})", flush=True)


# Two stressors composed in one config:
#   chunk0: 4 layers, recompute {0, 1}, keep {2, 3}  (kept layers at the end)
#   chunk1: 3 layers, recompute {0},    keep {1, 2}
# so a single OVERLAP_B_F step pairs a 4-layer chunk (17 rendezvous) with a
# 3-layer chunk (13) -> depart drains the 4-rendezvous mismatch;  every
# recomputed layer keeps its EP a2a (MUST_SAVE) so the serial re-run does not
# re-issue the collective.
COMBINED_LAYERS_PER_CHUNK = [4, 3]
COMBINED_RECOMPUTE_LAYERS = [{0, 1}, {0}]


def test_pp_overlap_moe_recompute_combined():
    """Combined stressors: variable layers + mixed recompute + save a2a.

    Feature: ``HookCoordinator.depart`` (variable layers) × mixed per-layer
        recompute, composed in a single config.
    Description:
        8 ranks, PP=4 × EP=2.  ``chunk0`` has 4 layers (recompute ``{0, 1}``,
        keep ``{2, 3}``); ``chunk1`` has 3 layers (recompute ``{0}``, keep
        ``{1, 2}``).  Every recomputed layer keeps its EP a2a output
        (``MUST_SAVE``) and recomputes only the compute ops; the re-runs are
        fired serially before each backward by
        ``PipelineStage.recompute_one_chunk``.  This is the strongest combined
        stressor: the depart drain (17-vs-13 rendezvous) and the
        save-collectives policy (no racing re-run a2a) fire together with mixed
        recompute.  Compares grads against a sync baseline.
    Expectation:
        No deadlock; per-parameter gradients (and last-rank losses) match the
        sync baseline within ``rtol=1e-3, atol=1e-3`` on every rank.
    """
    rank, device, pp_mesh, ep_mesh = _init_pp_ep_mesh()
    pp_rank = pp_mesh.get_local_rank()
    cfg = _TinyConfig()

    # ---- Sync baseline: no overlap, no recompute, same [4, 3] layout. ----
    baseline_chunks, baseline_si = _build_pipeline(
        pp_rank, ep_mesh, cfg, use_overlap=False,
        layers_per_chunk=COMBINED_LAYERS_PER_CHUNK,
    )
    baseline_losses, baseline_grads, _ = _run_one_iteration(
        baseline_chunks, baseline_si, pp_rank, device, pp_mesh,
        overlap_p2p=False, overlap_b_f=False,
    )
    del baseline_chunks
    platform.barrier()

    # ---- Overlap + per-chunk mixed recompute + saved a2a. ----
    overlap = CommComputeOverlap()
    overlap_chunks, overlap_si = _build_pipeline(
        pp_rank, ep_mesh, cfg, use_overlap=True, overlap=overlap,
        recompute=True, recompute_granularity="layer",
        recompute_layers=COMBINED_RECOMPUTE_LAYERS,
        layers_per_chunk=COMBINED_LAYERS_PER_CHUNK,
    )
    overlap_losses, overlap_grads, _ = _run_one_iteration(
        overlap_chunks, overlap_si, pp_rank, device, pp_mesh,
        overlap_p2p=True, overlap_b_f=True,
        callback=_make_overlap_b_f_callback(overlap),
    )

    _assert_overlap_matches_baseline(
        rank, pp_rank, baseline_losses, baseline_grads,
        overlap_losses, overlap_grads,
        label="combined(var_layers+mixed+save_a2a)",
    )
    print(f"[rank{rank}] pp_overlap_moe_recompute_combined: PASS "
          f"(layers_per_chunk={COMBINED_LAYERS_PER_CHUNK}, "
          f"recompute_layers={[sorted(s) for s in COMBINED_RECOMPUTE_LAYERS]}, "
          f"params={len(baseline_grads)})", flush=True)


# =========================================================================
# dx/dw split accuracy + profiling test (OVERLAP_B_F backward split)
# =========================================================================

def _zero_grads(chunks):
    """Reset accumulated grads on every trainable param (per-step zero_grad).

    The pipeline backward accumulates into ``param.grad`` (``+=``); zeroing
    between steps keeps each step's grads independent so the split-vs-baseline
    comparison is per-step rather than over a growing sum.
    """
    for chunk in chunks:
        for p in chunk.trainable_params():
            p.grad = None


def _compare_step(rank, pp_rank, step, base, split, rtol, atol):
    """Assert one step's split-vs-baseline equivalence; return (loss, grad) max diff."""
    base_losses, base_grads = base
    split_losses, split_grads = split
    loss_diff = 0.0
    grad_diff = 0.0
    if pp_rank == PP_SIZE - 1:
        assert len(base_losses) == len(split_losses), \
            (f"[rank{rank}] step{step} loss count mismatch: "
             f"base={len(base_losses)}, split={len(split_losses)}")
        for i, (bl, sl) in enumerate(zip(base_losses, split_losses)):
            loss_diff = max(loss_diff, float(np.abs(bl - sl)))
            assert np.allclose(bl, sl, rtol=rtol, atol=atol), \
                (f"[rank{rank}] step{step} loss[{i}] mismatch: "
                 f"base={float(bl):.6f}, split={float(sl):.6f}, "
                 f"abs_diff={float(np.abs(bl - sl)):.6e}")
    assert len(base_grads) == len(split_grads), \
        (f"[rank{rank}] step{step} grad count mismatch: "
         f"base={len(base_grads)}, split={len(split_grads)}")
    for i, ((bn, bg), (sn, sg)) in enumerate(zip(base_grads, split_grads)):
        assert bn == sn, \
            (f"[rank{rank}] step{step} grad[{i}] param name mismatch: "
             f"base={bn}, split={sn}")
        assert (bg is None) == (sg is None), \
            (f"[rank{rank}] step{step} grad[{i}] ({bn}) None-state mismatch: "
             f"base_none={bg is None}, split_none={sg is None}")
        if bg is None:
            continue
        grad_diff = max(grad_diff, float(np.abs(bg - sg).max()))
        assert np.allclose(bg, sg, rtol=rtol, atol=atol), \
            (f"[rank{rank}] step{step} grad[{i}] ({bn}) mismatch: "
             f"max_abs_diff={float(np.abs(bg - sg).max()):.6e}, "
             f"base_norm={float(np.linalg.norm(bg)):.6e}, "
             f"split_norm={float(np.linalg.norm(sg)):.6e}")
    return loss_diff, grad_diff


def test_pp_overlap_moe_dxdw_accuracy():
    """Numerical equivalence over ``NUM_STEPS`` steps, profiling step ``PROFILE_STEP``.

    Feature: Accuracy check + profiling for the ``OVERLAP_B_F`` dx/dw split
        on MindSpore PyNative.
    Description:
        Builds two models from identical seeds: a **sync baseline** with the
        overlap stack OFF (``use_overlap=False``, ``overlap_p2p=False``,
        ``overlap_b_f=False``, no callback) — the same ground truth as
        :func:`test_pp_overlap_moe_accuracy` — and the dx/dw split path
        (``enable_dxdw_split=True``: the schedule pairs ``(BWD_INPUT, FWD)``
        in each ``OVERLAP_B_F`` and runs the matching ``BWD_WEIGHT`` after the
        pair's P2P gap, so the grad send issues once dx and the paired forward
        finish, before dw).  Runs ``NUM_STEPS`` steps; each step feeds
        the *same* fresh input to both paths, zeroes grads, runs both, and
        asserts equivalence — the baseline comparison is mandatory, there is no
        skip path.  A ``mindspore.profiler.profile`` with
        ``schedule(wait=PROFILE_STEP-1, active=1)`` captures only the
        ``PROFILE_STEP``-th step; dx / dw carry ``profiler_record`` tags so the
        split shows up as distinct spans.  Both ``NUM_STEPS`` and
        ``PROFILE_STEP`` are env-overridable.
    Expectation:
        Per step, losses match within ``rtol=1e-3, atol=1e-3`` on rank
        ``PP_SIZE-1`` and grads match within the same tolerance on every rank.
        A mismatch typically means: dx forgot to write ``bwd_cache`` (so the
        scheduler's ``BWD_SEND`` sends a stale grad), dw's grad_fn lost
        intermediates after dx, or the overlap path itself diverged from the
        sync baseline (same failure modes as ``test_pp_overlap_moe_accuracy``).
    """
    rank, device, pp_mesh, ep_mesh = _init_pp_ep_mesh()
    pp_rank = pp_mesh.get_local_rank()
    cfg = _TinyConfig()

    rtol, atol = 1e-3, 1e-3

    # dx/dw split pipeline (always built — this is the path under profile).
    split_overlap = CommComputeOverlap()
    split_chunks, split_si = _build_pipeline(
        pp_rank, ep_mesh, cfg, use_overlap=True, overlap=split_overlap,
    )
    split_cb = _make_overlap_b_f_callback(split_overlap)

    # Sync baseline (overlap off) — same ground truth as
    # ``test_pp_overlap_moe_accuracy``: ``use_overlap=False`` with both overlap
    # flags off and no OVERLAP_B_F callback.  Built from the same seeded init in
    # ``_build_pipeline`` so its weights are bit-identical to the split build,
    # and kept alive across the loop to avoid re-seeding drift.
    base_chunks, base_si = _build_pipeline(
        pp_rank, ep_mesh, cfg, use_overlap=False,
    )

    # on_trace_ready=None -> profiler writes to its default output_path "./data"
    # (each rank under its own ascend_ms subdir).
    prof_dir = "./data"
    prof_sched = profiler_schedule(
        wait=1, warmup=2, active=1, repeat=1, skip_first=0,
    )
    exp_cfg = _ExperimentalConfig(profiler_level=ProfilerLevel.Level1)
    max_loss_diff = 0.0
    max_grad_diff = 0.0

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
        schedule=prof_sched,
        on_trace_ready=None,
        experimental_config=exp_cfg,
        with_stack=True,
    ) as prof:
        for step in range(1, NUM_STEPS + 1):
            # Vary input per step so the loop exercises distinct activations;
            # when comparing, the same tensor feeds both paths.
            x = Tensor(
                np.random.RandomState(100 + pp_rank + step * 1000)
                .randn(BS, SEQ_LEN, HIDDEN_SIZE).astype(np.float32),
            )
            # Sync baseline (overlap off) is the ground truth; the dx/dw split
            # must match it every step — there is no skip path.
            _zero_grads(base_chunks)
            base_losses, base_grads, _ = _run_one_iteration(
                base_chunks, base_si, pp_rank, device, pp_mesh,
                overlap_p2p=False, overlap_b_f=False, x_input=x,
            )
            _zero_grads(split_chunks)
            split_losses, split_grads, _ = _run_one_iteration(
                split_chunks, split_si, pp_rank, device, pp_mesh,
                overlap_p2p=True, overlap_b_f=True, enable_dxdw_split=True,
                callback=split_cb, x_input=x,
            )
            loss_diff, grad_diff = _compare_step(
                rank, pp_rank, step,
                (base_losses, base_grads), (split_losses, split_grads),
                rtol, atol,
            )
            max_loss_diff = max(max_loss_diff, loss_diff)
            max_grad_diff = max(max_grad_diff, grad_diff)
            prof.step()

    print(f"[rank{rank}] pp_overlap_moe_dxdw_accuracy: PASS "
          f"(steps={NUM_STEPS}, rtol={rtol}, atol={atol}, "
          f"max_loss_diff={max_loss_diff:.3e}, max_grad_diff={max_grad_diff:.3e}, "
          f"profile_dir={prof_dir})",
          flush=True)


def test_pp_overlap_moe_recompute_save_a2a():
    """Chunk recompute that KEEPS (does not recompute) the EP all-to-all.

    Feature: op-granularity selective recompute under overlap_b_f — recompute
        the chunk's local compute but SAVE/restore the dispatch+combine a2a.
    Description:
        8 ranks, PP=4 x EP=2.  Each chunk is wrapped with
        ``_RecomputeChunkWrapperSaveA2A`` (``checkpoint(policy_fn=_save_a2a_policy)``):
        the forward saves every EP all-to-all output, and the serial re-forward
        fired by ``PipelineStage.recompute_one_chunk`` RESTORES them from storage
        instead of re-issuing the HCCL all-to-all, while every other op is
        recomputed.  Unlike ``recompute_granularity="layer"`` (which keeps/
        recomputes whole layers and re-runs the a2a), this is op-granularity and
        is the only way to recompute a layer's compute while keeping its a2a.
        Compares against a sync baseline and checks the a2a actually went through
        SAC's save (forward) + restore (recompute) path.
    Expectation:
        No deadlock; the save-a2a policy classifies the a2a in BOTH the forward
        (cache) and the recompute (restore) phase — proving it was not
        re-communicated; per-parameter grads and last-rank losses match the sync
        baseline within ``rtol=1e-3, atol=1e-3`` on every rank.
    """
    rank, device, pp_mesh, ep_mesh = _init_pp_ep_mesh()
    pp_rank = pp_mesh.get_local_rank()
    cfg = _TinyConfig()

    # ---- Sync baseline: no overlap, no recompute (ground truth). ----
    baseline_chunks, baseline_si = _build_pipeline(
        pp_rank, ep_mesh, cfg, use_overlap=False,
    )
    baseline_losses, baseline_grads, _ = _run_one_iteration(
        baseline_chunks, baseline_si, pp_rank, device, pp_mesh,
        overlap_p2p=False, overlap_b_f=False,
    )
    del baseline_chunks
    platform.barrier()

    # ---- Chunk recompute that SAVES the EP a2a, under the full overlap stack. ----
    overlap = CommComputeOverlap()
    overlap_chunks, overlap_si = _build_pipeline(
        pp_rank, ep_mesh, cfg, use_overlap=True, overlap=overlap,
        recompute=True, recompute_granularity="chunk_save_a2a",
    )
    _A2A_SAC_SEEN.clear()
    overlap_losses, overlap_grads, _ = _run_one_iteration(
        overlap_chunks, overlap_si, pp_rank, device, pp_mesh,
        overlap_p2p=True, overlap_b_f=True,
        callback=_make_overlap_b_f_callback(overlap),
    )

    # ---- Verify the a2a went through SAC save (forward) + restore (recompute),
    # ---- i.e. it was NOT re-communicated during the backward re-run. ----
    fwd_saves = sum(1 for is_rec in _A2A_SAC_SEEN if not is_rec)
    rec_restores = sum(1 for is_rec in _A2A_SAC_SEEN if is_rec)
    assert fwd_saves > 0, (
        f"[rank{rank}] save-a2a policy never matched the a2a op in the forward "
        f"pass — SAC did not intercept it. Check the op-name match in "
        f"_save_a2a_policy (seen={_A2A_SAC_SEEN[:8]}).")
    assert rec_restores > 0, (
        f"[rank{rank}] a2a never classified during recompute — the re-run did "
        f"not hit the SAC cached path, so the a2a may have been re-communicated "
        f"(fwd_saves={fwd_saves}, rec_restores={rec_restores}).")

    _assert_overlap_matches_baseline(
        rank, pp_rank, baseline_losses, baseline_grads,
        overlap_losses, overlap_grads, label="recompute_save_a2a",
    )
    print(f"[rank{rank}] pp_overlap_moe_recompute_save_a2a: PASS "
          f"(a2a fwd_saves={fwd_saves}, rec_restores={rec_restores}, "
          f"params={len(baseline_grads)})", flush=True)


def test_pp_overlap_moe_recompute_save_a2a_dxdw():
    """Save-a2a chunk recompute combined with the dx/dw split.

    Feature: ``enable_dxdw_split`` x activation checkpoint — the split backward
        halves must reuse the chunk's pre-fired recompute session.
    Description:
        Same topology and SAC policy as
        :func:`test_pp_overlap_moe_recompute_save_a2a`, with the overlap run
        built under ``enable_dxdw_split=True``.  MS keeps the *current*
        recompute session in a ContextVar (thread-local) over a global cache,
        so ``backward_input_one_chunk`` / ``backward_weight_one_chunk`` must
        each enter ``recompute_session_ctx`` on their own thread (dx retains
        for dw; dw is the terminal consumer).  Without that, dx's unpack
        misses the cache ``recompute_one_chunk`` pre-fired and lazily re-runs
        the chunk forward on the BWD thread — its a2a hooks rendezvous against
        the paired forward's hooks and both threads deadlock.
    Expectation:
        No hang; the a2a is SAC-saved in the forward and restored during the
        single pre-fired recompute (never re-communicated, and dx/dw trigger
        no second re-run); losses and grads match the sync baseline within
        ``rtol=1e-3, atol=1e-3`` on every rank.
    """
    rank, device, pp_mesh, ep_mesh = _init_pp_ep_mesh()
    pp_rank = pp_mesh.get_local_rank()
    cfg = _TinyConfig()

    # ---- Sync baseline: no overlap, no recompute (ground truth). ----
    baseline_chunks, baseline_si = _build_pipeline(
        pp_rank, ep_mesh, cfg, use_overlap=False,
    )
    baseline_losses, baseline_grads, _ = _run_one_iteration(
        baseline_chunks, baseline_si, pp_rank, device, pp_mesh,
        overlap_p2p=False, overlap_b_f=False,
    )
    del baseline_chunks
    platform.barrier()

    # ---- Save-a2a recompute + dx/dw split under the full overlap stack. ----
    overlap = CommComputeOverlap()
    overlap_chunks, overlap_si = _build_pipeline(
        pp_rank, ep_mesh, cfg, use_overlap=True, overlap=overlap,
        recompute=True, recompute_granularity="chunk_save_a2a",
    )
    _A2A_SAC_SEEN.clear()
    overlap_losses, overlap_grads, _ = _run_one_iteration(
        overlap_chunks, overlap_si, pp_rank, device, pp_mesh,
        overlap_p2p=True, overlap_b_f=True, enable_dxdw_split=True,
        callback=_make_overlap_b_f_callback(overlap),
    )

    fwd_saves = sum(1 for is_rec in _A2A_SAC_SEEN if not is_rec)
    rec_restores = sum(1 for is_rec in _A2A_SAC_SEEN if is_rec)
    assert fwd_saves > 0, (
        f"[rank{rank}] save-a2a policy never matched the a2a op in the forward "
        f"pass — SAC did not intercept it (seen={_A2A_SAC_SEEN[:8]}).")
    # Equality, not just >0: a split half missing its session ctx re-runs the
    # chunk forward a second time and inflates the recompute-side count.
    assert rec_restores == fwd_saves, (
        f"[rank{rank}] a2a classified {rec_restores} times during recompute vs "
        f"{fwd_saves} forward saves — expected exactly one pre-fired re-run; "
        f"a mismatch means dx/dw triggered an extra lazy re-run (or the SAC "
        f"cached path was missed).")

    _assert_overlap_matches_baseline(
        rank, pp_rank, baseline_losses, baseline_grads,
        overlap_losses, overlap_grads, label="recompute_save_a2a_dxdw",
    )
    print(f"[rank{rank}] pp_overlap_moe_recompute_save_a2a_dxdw: PASS "
          f"(a2a fwd_saves={fwd_saves}, rec_restores={rec_restores}, "
          f"params={len(baseline_grads)})", flush=True)


def _test_pp_overlap_moe_accuracy_batched_p2p(p2p_transport: str, label: str) -> None:
    """Compare one batched P2P transport against the synchronous baseline."""
    rank, device, pp_mesh, ep_mesh = _init_pp_ep_mesh()
    pp_rank = pp_mesh.get_local_rank()
    cfg = _TinyConfig()

    # ---- Sync baseline (plain P2P, no overlap). ----
    baseline_chunks, baseline_si = _build_pipeline(
        pp_rank, ep_mesh, cfg, use_overlap=False,
    )
    baseline_losses, baseline_grads, _ = _run_one_iteration(
        baseline_chunks, baseline_si, pp_rank, device, pp_mesh,
        overlap_p2p=False, overlap_b_f=False,
    )
    del baseline_chunks
    platform.barrier()

    # ---- Overlap stack with ALL PP P2P batched. ----
    overlap = CommComputeOverlap()
    overlap_chunks, overlap_si = _build_pipeline(
        pp_rank, ep_mesh, cfg, use_overlap=True, overlap=overlap,
    )
    overlap_losses, overlap_grads, _ = _run_one_iteration(
        overlap_chunks, overlap_si, pp_rank, device, pp_mesh,
        overlap_p2p=True, overlap_b_f=True,
        callback=_make_overlap_b_f_callback(overlap),
        p2p_transport=p2p_transport,
    )

    _assert_overlap_matches_baseline(
        rank, pp_rank, baseline_losses, baseline_grads,
        overlap_losses, overlap_grads, label=label,
    )
    print(f"[rank{rank}] pp_overlap_moe_accuracy_{label}: PASS "
          f"(transport={p2p_transport}, params={len(baseline_grads)})",
          flush=True)


def test_pp_overlap_moe_accuracy_batch_p2p() -> None:
    """Numerical equivalence vs sync baseline with same-peer duplex P2P batching.

    Feature: ``p2p_transport="batch"`` (what the ``"auto"`` default resolves
        to under overlap_b_f) — ``build_exec_order`` runs ``coalesce_p2p``,
        turning each contiguous P2P run into a ``BATCH_SEND_RECV`` step that the
        runtime groups by peer and issues as one ``batch_isend_irecv`` per peer
        (same-peer send+recv -> TX||RX duplex); leftover singletons are batched
        too, so every transfer is batch-vs-batch, matched per-peer FIFO.
    Expectation:
        No ``HcclBatchISendIRecv`` EI0005, no deadlock; per-micro-batch losses
        and per-parameter grads match the sync baseline.
    """
    _test_pp_overlap_moe_accuracy_batched_p2p("batch", "batch_p2p")


def test_pp_overlap_moe_accuracy_multi_stream_p2p() -> None:
    """Numerical equivalence with independent communication streams for adjacent PP peers.

    Feature: ``p2p_transport="multi_stream"`` keeps the same coalesced per-peer batch
        sequence as ``"batch"`` while assigning previous/next peers distinct
        communication groups, including the interleaved last-to-first edge.
    Description:
        Run the overlap schedule with peer-specific communication groups and compare
        its losses and gradients with the synchronous baseline.
    Expectation:
        Group creation and P2P matching do not deadlock; losses and gradients
        match the synchronous baseline.
    """
    _test_pp_overlap_moe_accuracy_batched_p2p("multi_stream", "multi_stream_p2p")


def test_pp_overlap_moe_accuracy_boundary():
    """fwd-boundary batching (EXPERIMENTAL p2p_transport="boundary") vs baseline.

    Feature: the combined mode — ``attach_fwd_boundary_p2p`` hangs each steady
        gap's ``F_SEND`` (payload ready when the overlap's forward finishes;
        the backward is the long pole) plus the next slot's recvs on the
        OVERLAP_B_F step, and the callback's ``fwd_fn`` issues them mid-overlap
        via ``exec_boundary_p2p`` — so the activation send leaves ~half a slot
        early and the peer's next recv-wait is largely hidden.  Every op is a
        per-op solo batch (no ``coalesce_p2p``): per-pair batch sequences stay
        complementary ``[F_SEND, B_RECV]`` vs ``[F_RECV, B_SEND]``, safe under
        both candidate HCCL batch-pairing semantics (the earlier naive
        hoist+coalesce composition violated shape mirroring and hung here).
    Description:
        Same topology as :func:`test_pp_overlap_moe_accuracy`.  The overlap run
        is built with ``overlap_p2p=True, overlap_b_f=True,
        p2p_transport="boundary"`` (explicit opt-in — the auto default under
        overlap_b_f is the measured-beneficial duplex "batch"); the baseline is
        the plain sync stack.  Passing here is the gate for ever promoting
        boundary to the auto default.  The mode
        only changes *when* and *how grouped* the P2P launches are (same ops,
        same per-direction FIFO), so numerics must be unchanged.
    Expectation:
        No EI0005, no hang; per-micro-batch losses (last PP rank) and
        per-parameter grads (every rank) match the sync baseline within
        ``rtol=1e-3, atol=1e-3``.
    """
    rank, device, pp_mesh, ep_mesh = _init_pp_ep_mesh()
    pp_rank = pp_mesh.get_local_rank()
    cfg = _TinyConfig()

    # ---- Sync baseline (plain P2P, no overlap). ----
    baseline_chunks, baseline_si = _build_pipeline(
        pp_rank, ep_mesh, cfg, use_overlap=False,
    )
    baseline_losses, baseline_grads, _ = _run_one_iteration(
        baseline_chunks, baseline_si, pp_rank, device, pp_mesh,
        overlap_p2p=False, overlap_b_f=False,
    )
    del baseline_chunks
    platform.barrier()

    # ---- Overlap stack with fwd-boundary batching. ----
    overlap = CommComputeOverlap()
    overlap_chunks, overlap_si = _build_pipeline(
        pp_rank, ep_mesh, cfg, use_overlap=True, overlap=overlap,
    )
    overlap_losses, overlap_grads, _ = _run_one_iteration(
        overlap_chunks, overlap_si, pp_rank, device, pp_mesh,
        overlap_p2p=True, overlap_b_f=True,
        callback=_make_overlap_b_f_callback(overlap),
        p2p_transport="boundary",
    )

    _assert_overlap_matches_baseline(
        rank, pp_rank, baseline_losses, baseline_grads,
        overlap_losses, overlap_grads, label="boundary",
    )
    print(f"[rank{rank}] pp_overlap_moe_accuracy_boundary: PASS "
          f"(fwd-boundary batching, params={len(baseline_grads)})", flush=True)


if __name__ == "__main__":
    test_pp_overlap_moe_end_to_end()
