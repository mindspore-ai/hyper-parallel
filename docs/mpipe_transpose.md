# MPipe Transpose — Design

## 1. Motivation

In standard (interleaved) 1F1B pipeline parallelism, the **warmup** phase leaves
downstream ranks idle while stage 0 sequentially computes the forward of every
micro-batch. MPipe Transpose recovers part of that bubble by **transposing** the
forward of a model's first `T` layers — the *preprocess* block.

Notation:

- `PP` — number of pipeline stages (physical ranks, `real_stage_num`).
- `MB` — number of micro-batches (`micro_batch_num`).
- `L(s)` — number of layers assigned to stage `s`.
- `T` — number of transposed (preprocess) layers, taken from stage 0's first
  chunk, `T < (#layers in stage 0's first chunk)`.
- `NT = min(PP, MB)` — number of transposed micro-batches.

Instead of stage 0 running the preprocess forward serially for all micro-batches,
the preprocess parameters are **broadcast** to every rank. For the first `NT`
micro-batches, rank `i` computes the preprocess forward of micro-batch `i` **in
parallel** during what would otherwise be its warmup idle time, then ships the
result back to stage 0. The rest of the schedule is ordinary 1F1B.

### Training-step flow

1. Stage 0 broadcasts the parameters of the `T` preprocess layers to every rank.
2. Every rank reads its own micro-batch from the dataset (dataloader replicated
   per rank, each reading different data).
3. Micro-batch `i` (for `i < NT`) has its preprocess forward computed on rank `i`.
4. Each transposed rank `i > 0` sends to stage 0:
   - **(a)** the preprocess **output** — so stage 0 can run its body forward
     (layers `T..L(0)`), and
   - **(b)** the preprocess **input** — so the preprocess **backward** can be
     recomputed centrally on stage 0.
5. Warmup → 6. Steady → 7. Cooldown proceed as ordinary interleaved 1F1B on the
   *body* model.

---

## 2. Key design decisions

| ID | Decision |
|----|----------|
| D1 | **Backward = recompute on stage 0.** A live PyTorch autograd graph cannot cross ranks, so "backward on stage 0" requires stage 0 to recompute `preprocess(input_i)` from the shipped input, then backprop the gradient that the body backward deposited on the detached input buffer. `MPIPE_GRAPH_SEND/RECV` therefore ships the **input** (not a literal graph). Pairs naturally with a recompute-wrapped preprocess block. |
| D2 | **Preprocess is a separate replicated module**, scheduled via dedicated `MPIPE_*` steps. Stage 0's `PipelineStage` wraps only `body0` (chunk 0 minus the first `T` layers); the inherited interleaved-1F1B logic runs unchanged on the body model. |
| D3 | **Option A** — the caller passes `preprocess_module` on **every** rank (standalone, identical architecture). On rank 0 it holds the trained parameters; on other ranks it is a structural copy overwritten each step by the broadcast. Required because ranks `1..PP-1` do not otherwise hold stage 0's layers. |
| D4 | **Per-rank micro read** — every rank reads its own micro-batch `i`; this is why the preprocess input must be shipped to stage 0 (`MPIPE_GRAPH_SEND`). |
| D5 | **Torch first**, MindSpore parity afterward. |

### Micro-batch categories

Every preprocess forward is **detached**: its output reaches body0 as a plain
buffer, and the preprocess gradient comes from an explicit backward rather than
from body0's graph. Which rank owns which micro-batch depends on
`pp_mpipe_transpose_overflow`:

- `"full"` (default) — round-robin: micro `m` is owned by rank `m % NT`, so the
  `MB > NT` overflow is spread evenly and every micro is transposed somewhere.
- `"min"` — only `i ∈ [0, NT)` are transposed; rank 0 absorbs the overflow
  `i ∈ [NT, MB)`, running their preprocess inline (still detached). This trades
  a longer rank-0 phase for less inter-rank transport.

The preprocess backward is either the centralized `MPIPE_TRANSPOSE_BWD` on
stage 0 (recomputing from the shipped input), or, under
`pp_mpipe_owner_backward`, run by the owning rank after stage 0 ships
`dL/dfeatures` back — see `_build_owner_backward_suffix`.

---

## 3. New `MetaStepType` members

| Type | Where | Action |
|------|-------|--------|
| `MPIPE_PARAM_BROADCAST` | all ranks | broadcast preprocess params 0→all |
| `MPIPE_TRANSPOSE_FWD` | the owning rank; rank 0 inline for `"min"` overflow | run preprocess forward, detached, retaining the input for the backward |
| `MPIPE_FWD_SEND` / `MPIPE_FWD_RECV` | `i→0` | ship preprocess **output** into stage 0's body input slot |
| `MPIPE_GRAPH_SEND` / `MPIPE_GRAPH_RECV` | `i→0` | ship preprocess **input** for recompute backward (deferrable) |
| `MPIPE_TRANSPOSE_BWD` | rank 0, `micro<NT` | recompute preprocess forward from input, then `autograd.backward` |

**Why a dedicated `MPIPE_FWD_SEND` rather than reusing `FWD_SEND`?** The output
transport `i→0` cannot reuse the existing pair: (1) the receiver is stage 0, the
*first* stage, whose `forward_one_chunk` reads `args` (not `args_recv_info`), and
the source rank **varies per micro-batch** while `src_stage` is fixed; (2) it is
**forward-only** — the transposed backward is recomputed on stage 0, so no
gradient is sent back to rank `i`, whereas `exec_fwd_send_ops` wires up a
backward grad-recv path. Keeping it separate from `MPIPE_GRAPH_SEND` is also what
makes the `T=0` case expressible (see below).

### `T = 0` — transpose only the dataload

`T = 0` is valid and degenerate: there is no preprocess block, so each rank
loads its micro-batch and ships the **raw input** to stage 0, moving data loading
off stage 0's critical path. The schedule keeps `MPIPE_TRANSPOSE_FWD` (= the
dataload) and `MPIPE_FWD_SEND` / `MPIPE_FWD_RECV` (ship the input), and **omits**
`MPIPE_PARAM_BROADCAST`, `MPIPE_GRAPH_SEND` / `MPIPE_GRAPH_RECV`, and
`MPIPE_TRANSPOSE_BWD` (nothing to broadcast, no params to backprop). Non-transposed
micro-batches are read locally by stage 0 as usual.

---

## 4. Schedule construction

`ScheduleMPipeTranspose` **subclasses `ScheduleInterleaved1F1B`** and reuses its
per-stage warmup/steady/cooldown construction and `add_send_recv` for the body.

`construct_exec_order()`:

1. Build the **body order** via `super().construct_exec_order()` (body stages,
   with P2P comms inserted).
2. **Patch stage 0's (rank 0) body order**:
   - before each `FWD(stage 0, micro ≥ NT)`: insert inline `MPIPE_TRANSPOSE_FWD(micro)`;
   - after each `BWD(stage 0, micro < NT)`: insert `MPIPE_TRANSPOSE_BWD(micro)`.
3. **Prepend a transpose-phase prefix** to every rank:
   - `MPIPE_PARAM_BROADCAST`;
   - if `rank < NT`: `MPIPE_TRANSPOSE_FWD(rank)`, and for `rank > 0`
     `MPIPE_FWD_SEND(rank→0)` + `MPIPE_GRAPH_SEND(rank→0)`;
   - if `rank == 0`: `MPIPE_FWD_RECV(i)` for `i∈[1,NT)`, then `MPIPE_GRAPH_RECV(i)`.

Invariant: stripping all `MPIPE_*` steps from any rank's order recovers the plain
interleaved-1F1B body exactly — MPipe only *layers* steps around an unchanged
body schedule.

### Worked examples

- **PP=4, n_local=1, MB=4** (`NT=4`): every rank computes one preprocess
  forward; ranks 1–3 send output+input to stage 0; stage 0 receives 3 outputs
  and 3 inputs and appends one `MPIPE_TRANSPOSE_BWD` after each `BWD(0, i)`.
- **PP=4, n_local=2, MB=8** (`NT=4`): additionally inserts inline
  `MPIPE_TRANSPOSE_FWD(4..7)` before `FWD(0, 4..7)` with automatic backward.
- **PP=4, n_local=1, MB=2** (`NT=2`): only ranks 0,1 transposed; ranks 2,3 only
  join the broadcast.

---

## 5. Files

1. `core/pipeline_parallel/scheduler.py` — `MetaStepType` members and the
   in-schedule dataload steps; `core/pipeline_parallel/mpipe/schedule.py` —
   `ScheduleMPipeTranspose` (ordering + executor registration), platform-aware
   `MPIPE_TRANSPOSE_BWD` emission, and the overflow / owner-backward layouts.
2. `core/pipeline_parallel/mpipe/executor_base.py` — `MPipeTransposeExecutorBase`:
   backend-agnostic broadcast / P2P transport / step orchestration, with
   abstract autograd hooks.
3. `platform/torch/pipeline_parallel/mpipe_transpose.py` and
   `platform/mindspore/pipeline_parallel/mpipe_transpose.py` — thin subclasses
   implementing the autograd hooks (detached forward, mark-requires-grad,
   stage-0 backward, and on torch the owner-backward hooks).
4. `core/pipeline_parallel/__init__.py` + `hyper_parallel/__init__.py` — export
   `ScheduleMPipeTranspose`.
5. Per-rank dataload convention: every rank passes the batch; rank `i` uses its
   micro-batch `i` for the transposed forward.
6. Tests: `tests/ut/core/pipeline_parallel/test_mpipe_transpose.py` (ordering),
   `test_mpipe_transpose_exec.py` (single-process recompute equivalence).
   Distributed correctness: `tests/torch/pipeline_parallel/_test_mpipe_transpose.py`
   (worker; the backend adapts to the device — gloo on CPU, hccl on Ascend NPU)
   launched by `test_mpipe_transpose_dist.py` via `parallel_run`/`TorchCase`.

### Platform-aware backward

Torch's `autograd.backward` traverses the connected graph to all leaves, so a
non-transposed micro-batch's body backward flows into the preprocess params
automatically — no `MPIPE_TRANSPOSE_BWD` needed for it. MindSpore's captured
`grad_fn` is scoped to the body submodule's own weights, so it only deposits the
*input* grad on the preprocess output; there, **every** micro-batch needs an
explicit recompute backward. The schedule therefore emits `MPIPE_TRANSPOSE_BWD`
for non-transposed micro-batches only when the backend requires it
(`platform_type == MINDSPORE`). Whether shipping the recompute input
(`MPIPE_GRAPH_*`) beats recomputation is a perf question to settle later and may
change this choice.

---

## 6. Performance

Measured on Ascend: with `T = 0` (dataload-only) MPipe step time ≈ 1F1B, but
`T > 0` is **much slower**. The gap is the *inherent* per-step cost, not the
transport plumbing:

- **Dominant cost: the per-step parameter broadcast.** `MPIPE_PARAM_BROADCAST`
  re-broadcasts the preprocess weights to every PP rank each step (they change
  after each optimizer step). For an LLM the preprocess includes `embed_tokens`,
  which is **param-heavy / compute-light** — the worst thing to transpose — and
  the embedding broadcast is unavoidable if any layer is transposed (rank `i`
  must embed before running the layers).
- **Secondary: recompute** — transposed micro-batches compute the preprocess
  forward twice (rank `i` for the shipped output, stage 0 for the recompute
  backward).
- **Transport is negligible** — confirmed by `T = 0 ≈ 1F1B` (it still ships
  `input_ids` and runs the same transport).

MPipe Transpose only wins when `bubble_saved > broadcast_cost + recompute_cost`,
i.e. a large warmup bubble (high PP, low MB) **and** a param-light, compute-heavy
preprocess. A trainable LLM embedding/decoder block is the opposite (param-heavy,
broadcast every step), so MPipe is comm-bound and loses to 1F1B for text.

**The frozen VL visual tower is the case where MPipe wins:** it is compute-heavy
(the image encode) but **frozen**, so there is **no per-step broadcast and no
recompute** — only the image features are shipped. This is exactly the
param-light-cost / compute-heavy regime MPipe is designed for, and why the VL
integration transposes the visual tower rather than text layers.
