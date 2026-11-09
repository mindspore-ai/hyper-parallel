# KDA Context Parallelism Guide

**English** | [简体中文](kda_context_parallel.zh.md)

Kimi Delta Attention (KDA) connects adjacent sequence segments through recurrent states. This guide covers KDA P2P, AllGather (AG),
Ulysses, and their combinations for Kimi K3 KDA layers and the library's `KimiDeltaAttention` module.

KDA AG gathers state summaries and transition matrices, not entire K/V sequences. Before model invocation, partition the input into contiguous sequence slices across
CP ranks, keeping CP mesh rank order consistent with global token order.
Hybrid partitioning and AG also require this logical order to match the communication group's actual rank order; noncontiguous ascending rank groups are supported.

## Parallel Methods

Let total CP size be `P`, Ulysses size `U`, state CP size `R = P / U`, and within-group AG size `g`.

| Method | Execution | Considerations |
|------|----------|----------|
| P2P | Each rank computes local summaries and passes boundary states in sequence order; backward passes boundary gradients in reverse order. | Small communication workspace; dependency chains grow with state CP size. |
| AG | Each rank gathers state summaries and transition matrices, then fused operators merge the required prefixes or backward suffixes. | Removes rank-by-rank communication dependencies; temporary gather buffers grow with state CP size. |
| RD | Compose summaries in order at distances 1, 2, 4, and so on; Triton backward reuses forward transitions, while Ascend C backward scans native G/N. | Long contiguous documents have logarithmic dependency stages; TND uses actual resets to remove communication, production, and caching without consumers. |
| Ulysses | AllToAll converts sequence shards to head shards, then restores sequence shards after computation. | Reduces heads per rank; head counts must be divisible by U. |
| Ulysses + P2P / AG | Exchange sequence and head dimensions within Ulysses groups, then run P2P or AG within state CP groups. | Head partitioning reduces state communication and state CP group size. |
| AG + P2P | Run AG within consecutive groups of g state ranks. The last rank of each group forms a P2P chain, then broadcasts the input boundary within its group. | Bounds temporary AG buffers by group width g while retaining serial inter-group dependencies. |
| Ulysses + AG + P2P | Apply Ulysses first, then within-group AG and inter-group P2P over R state ranks. | Tune head partitioning, gather buffers, and inter-group chain length together. |

For within-group AG + inter-group P2P, forward first gathers S/M. After receiving the previous group's boundary, the group owner applies local summaries in order and sends the result to the next group.
Each rank reconstructs its input state from the broadcast group-entry state. Triton backward gathers G/M; Ascend C backward gathers native G/N, traversing groups and ranks in reverse order.
The inter-group chain has `R/g` owners, but each owner still applies its group's summaries sequentially; fewer communication hops do not directly imply less computation.

## Configuration Entry Point

Select the unified wrapper for KDA layers in training `plan_overrides`:

```yaml
plan_overrides:
  - match: "*.self_attn.forward"
    when: cp
    replace:
      kind: method
      implementation:
        _target_: hyper_parallel.models.kimi_k3.adapter.distributed.context_parallel.kimi_delta_attention_cp_wrapper
        backend: triton
        state_cp_method: grouped_allgather_p2p
        ulysses_degree: 2
        allgather_group_size: 2
```

`replace.kind: method` replaces a method; `match` contains the module's fully qualified name (FQN) plus the literal method suffix `.forward`.
For example, `"*.self_attn.forward"` matches `model.layers.0.self_attn.forward`; for a root-level module, use
`"self_attn.forward"`. The module portion uses a case-sensitive glob, where `*` can cross dots in the path; it does not identify module types.
If KDA and other attention modules both use the name `self_attn`, match only the actual KDA layers.
`when: cp` enables the rule only when CP exceeds 1. Forward-method replacement currently defaults to `region_dispatch: false`;
wrappers containing communication and custom operators execute on local tensors, so this default is omitted from the example.
Place the wrapper and its arguments under `replace.implementation`.
This example uses the current Trainer method-replacement interface; see [YAML Configuration](trainer/yaml_config.md).

With CP8, this example creates U2, state CP4, within-group AG2, and a two-owner P2P chain.
Total CP size comes from the training parallel configuration. `state_cp_method` selects only the state CP method; the full strategy also depends on
`ulysses_degree` and `allgather_group_size`. For example, `state_cp_method: p2p` with `ulysses_degree: 2`
means Ulysses + P2P. The unified entry point fixes the chunk size at 64 tokens and does not expose `chunk_size`;
all backends use this size.

| Parameter | Default | Meaning and constraints |
|------|--------|------------|
| `backend` | `triton` | Local KDA backend. P2P/Ulysses supports `eager`; P2P, AG, grouped AG, RD, and their Ulysses combinations support `triton` and `ascendc`. |
| `state_cp_method` | `p2p` | State CP method: `p2p`, `allgather`, `grouped_allgather_p2p`, or `recursive_doubling`. |
| `ulysses_degree` | `1` | U, a positive integer dividing CP size; both query/key and value head counts must be divisible by U. |
| `allgather_group_size` | `1` | g, the number of state ranks per within-group AllGather, which must divide R. Used only for `grouped_allgather_p2p`; pure `allgather` gathers the full R group automatically and leaves this at 1. |

Common CP8 combinations:

| Method | `state_cp_method` | `ulysses_degree` | `allgather_group_size` |
|------|---------------------|------------------|--------------|
| P2P | `p2p` | 1 | 1 |
| AG | `allgather` | 1 | 1 |
| Ulysses + P2P | `p2p` | 2 | 1 |
| Ulysses + AG | `allgather` | 2 | 1 |
| RD | `recursive_doubling` | 1 | 1 |
| Ulysses + RD | `recursive_doubling` | 2 | 1 |
| AG + P2P | `grouped_allgather_p2p` | 1 | 2 |
| Ulysses + AG + P2P | `grouped_allgather_p2p` | 2 | 2 |
| Pure Ulysses | `p2p` | 8 | 1 |

With `U=1`, Ulysses is skipped. With `U=P`, state CP size is 1 and only local KDA runs.
The grouped protocol reduces to P2P at `g=1` and to AG at `g=R`.
Existing `kimi_delta_attention_p2p_cp_wrapper` and `kimi_delta_attention_ulysses_cp_wrapper` remain available with unchanged default parallel methods.
Their `chunk_size` parameters, and those of the underlying eager operators, remain available for direct invocation and reference validation.

## Input and Mesh Constraints

- The current fused AG path targets BF16 training on Ascend, with key/value head dimensions of 128 and a lower-bounded gate.
- Each CP rank has the same local sequence length. Fused operators use 64-token chunks; the final chunk of a fixed-length sequence or TND document fragment may be partial.
- Use a 1-D CP mesh with ascending ranks matching sequence order. A CP submesh of a root mesh, such as DP×CP, is supported;
  CP need not equal WORLD, and subgroup global ranks need not be contiguous.
- Hybrid methods must retain the association between the CP submesh and root mesh to create subgroups consistently. Arbitrary rank permutations are unsupported.
  The final root mesh used for splitting must cover WORLD; retaining only a local root is insufficient for safely creating all sibling groups.
- ShortConv exchanges halos between neighbors in the original CP sequence before Ulysses, keeping convolution boundaries independent of head rearrangement.
- For packed/TND inputs with contiguous equal-length shards, see [TND Context Parallelism](linear_attention_tnd_cp.md). Inference caches and simultaneous TP and CP are unsupported. CP does not reduce parameter gradients;
  training still requires the framework's corresponding data parallelism and gradient management configuration.

## Memory and Selection Guidance

S is the local output state from a zero initial state, M is the local segment's state transition matrix, and G is the backward gradient summary from a zero boundary.
Triton AG and grouped AG retain only one local M per in-flight call, gathering G and that M again during backward.
Forward-gathered S/M is not cached across forward and backward. Fused state merging changes execution only, not local KDA S/M/G arithmetic.

For `B=1, H=96, K=V=128`, one FP32 state occupies 6 MiB. Without Ulysses, packed S/M occupies
12 MiB. Pure AG receive buffers scale with R; grouped AG receive buffers scale with g. Ulysses reduces each state's head count
to `1/U` of the original. These figures describe state buffers, not total layer activation memory or peak training memory.

Consider head divisibility, state communication chain length, and temporary gather buffers together. AG is an option when memory is sufficient and state CP is small.
Use grouped AG + P2P to bound gather size; when head counts allow, add Ulysses first to reduce state CP size.
Different combinations change floating-point operation order and do not guarantee bitwise equality with single-device recurrence. Choose the training scale according to model-specific numerical requirements.

## Local Operators and Numerical Scope

The Triton path reuses FLA prepare, h, dAv, dhu, and local gradient operators; custom summaries output only states needed for communication.
The currently validated FLA commit is `6b6f546f9bd10a97be1e8cf2f0120de70cfca482`. Passing the minimum-version check does not imply all later versions have been validated.
S follows Ascend FLA h's FP32 state and residual feedback, corresponding to the final h state from zero initialization. G preserves Ascend FLA dhu's
order of scaling before conversion to matrix-input dtype and its two K64 groups' addition/subtraction structure, matching dhu's output-state gradient with zero terminal gradient.

M follows the FLA CP formulation, first constructing each chunk's explicit transition matrix, then multiplying the chain in FP32 without BF16 conversion during feedback.
Its output doubles as exclusive loop-feedback storage for each program, avoiding a full-segment chunk-transition workspace. The compiler may still allocate temporary workspace,
so local M cache size is not the operator's peak runtime memory. Backward reuses the transpose of forward's local M;
generic FLA CP instead regenerates transition matrices during backward. The real-number formulas are equivalent, but floating-point multiplication order and recomputed inputs may differ.

S/G parity is measured against Ascend's local h/dhu; rounding in generic FLA CP summaries is not identical. Zero-state parity also does not imply
bitwise equality between full CP and native nonzero-state recurrence: native dhu includes BF16 rounding dependent on the input-state gradient, while affine communication propagates gradients through G
and the transpose of M. Different CP merge orders also change rounding. Recheck parity whenever the FLA implementation changes.

For pure P2P with a zero initial state, the first state rank does not generate or save M when no downstream consumer needs it. AG and grouped AG retain
local M on every rank. Each invocation owns its state cache independently to support multiple in-flight forwards, reverse-order backwards, and checkpoint recomputation.

## Recursive Doubling and TND

`state_cp_method: recursive_doubling` propagates state summaries through ordered recursive doubling and computes backward state gradients using transitions cached during forward.
It composes with `ulysses_degree`, with `allgather_group_size` fixed at 1; it does not use intragroup AllGather.
This option reuses the same Triton local operators and supports dense inputs and the linked
[TND contiguous sharding](linear_attention_tnd_cp.md). Actual resets use zero effective transitions and do not propagate state across documents.
The TND entry therefore prunes sends/receives, summary production, and caches that have no consumers, without extra configuration or changes to existing process groups.

Planning reads only actual document boundaries on CPU. Each rank locates the rank interval covered by its final document and retains doubling rounds
and adjacent entry handoffs with consumers in that interval. Backward reverses the same plan. Without consumers, no summaries or full zero-state placeholders are allocated.
Receiving for the first document and sending for the final document are evaluated separately, so an internal reset does not remove a still-required first-document entry.

Calls across multiple layers should reuse one `PackedSequenceInfo`: its interval/Ulysses views share an immutable document snapshot validated once,
and RD reuses plans for that snapshot in a bounded CPU cache. The existing Kimi runtime adapter constructs this object once per batch.
For manual calls, reuse it as follows. Each new batch needs its own metadata; do not mutate indices already passed to backward.

```python
from hyper_parallel.components.functional.linear_attention_packed import PackedSequenceInfo

info = PackedSequenceInfo(global_offsets, cp_rank * local_tokens, (cp_rank + 1) * local_tokens)
for layer in local_layers:
    local_hidden = layer(local_hidden, packed_sequence_info=info)
```

Passing raw `cu_seq_lens` to every layer remains correct but rebuilds the snapshot and loses the benefit of reusing metadata within a batch.
Sharing CPU plans does not share communication buffers or backward-state caches, which remain isolated per call.

Communication and caches are isolated per call, supporting validated paths for reverse-order backward and non-reentrant checkpointing. Triton RD transition caches grow with the number of rounds;
AG's single-local-M retention rule cannot be used to estimate RD. Simultaneous TP, graph capture, empty ranks, and unequal-length ranks are currently unsupported.
All low-level communication uses the public `torch.distributed.batch_isend_irecv`, leaving internal implementation selection to the backend.
The framework does not switch private coalescing APIs by version or device model. Protocol qualification and full-layer training qualification must be assessed separately;
logarithmic round counts do not guarantee performance at arbitrary CP sizes.

## Ascend C Ulysses Backend

Pure Ulysses can select `backend: ascendc`, using fla-npu local attention and ShortConv together.
For dependencies and public operator interfaces, see the [functional documentation](../../hyper_parallel/components/functional/README.md#linear-attention-backends).

```yaml
plan_overrides:
  - match: "*.self_attn.forward"
    when: cp
    replace:
      kind: method
      implementation:
        _target_: hyper_parallel.models.kimi_k3.adapter.distributed.context_parallel.kimi_delta_attention_cp_wrapper
        ulysses_degree: 8
        state_cp_method: p2p
        backend: ascendc
```

When `ulysses_degree` equals the CP mesh size, the state communication group has size 1 and only Ulysses AllToAll runs.
The Ascend C backend also supports state P2P, AG, grouped AG, and RD described below.

Inputs retain raw Q/K, gate, and beta logits. Prepare handles Q/K normalization and the gate/beta forward operations.
Backward uses saved normalization intermediates. Native fused beta backward is used directly when available; otherwise, the corresponding FLA helper is used,
without applying the sigmoid chain rule twice. Forward and backward use save mode without exposing internal operator recomputation options.

TND calls pass global document offsets; after Ulysses exchange, local operators execute using complete document boundaries.
Document ends need not align to chunks; the same public native interface handles final document chunks.
Each call saves independent intermediates, supporting multiple in-flight forwards and checkpoint replay.
Initial/final states and their gradients, and double backward, are not yet exposed; missing capabilities raise explicit errors.

## Ascend C State P2P, AllGather, RD, and Hybrids

With an unmodified fla-npu package providing native CP summaries and separate backward-phase interfaces, `backend: ascendc` supports
`state_cp_method: p2p`, `allgather`, `grouped_allgather_p2p`, and `recursive_doubling`.
All compose with `ulysses_degree`; the grouped method also uses `allgather_group_size`.
Every combination in the preceding CP8 configuration table can use this backend.

Local execution, pure Ulysses, and all four state protocols share prepare, h, finalize, and separate bwd_prepare, dhu, and bwd_finalize phases.
Only state protocols add S/M and G/N summaries and communication; there is no separate local implementation with composite backward.
After receiving entry state, P2P first computes and sends `S + M @ H`, then completes local h. Backward first computes and sends `G + N @ D`,
then completes local dhu. Receives are posted early, and send buffers remain alive per call until completion; the full local recurrence is excluded from the state relay chain.

Ordinary AG gathers native packed summaries and uses native fused merge. Grouped AG gathers intragroup summaries, with owners relaying state between groups.
Scans with nonzero group-entry state currently use FP32 Torch NPU matrix multiplication and addition, without Triton summary or scan operators.
State application explicitly disables autocast so outer BF16 AMP cannot reduce state-computation precision. Backward uses native N directly, without transposing it.

RD reuses existing ordered communication groups, doubling sends/receives, and reset planning. Forward scans S/M; backward scans G/N in mirrored document order,
then maps logical peers back to the original communication group without creating a backward group. N is already a backward transition and is not replaced by a transpose of M.
Actual resets remove irrelevant rounds. Fragments without exit consumers skip summary production, while receive-only fragments still recover their entry state.
Composition currently uses FP32 matrix multiplication and addition with autocast disabled, without device pipelining.

Unlike Triton's local/per-round M caching, this path regenerates G/N during backward and does not cache M across forward and backward.
Intragroup gather buffers, sent states, and receive buffers belong only to the current call. Save mode still retains activations such as local h/v_new.
Temporary gather size is group width g for grouped AG and state-domain size R for ordinary AG; these figures do not represent peak memory for the whole layer.

The state path currently requires BF16, B=1, equal Q/K/V head counts, K=V=128, safe gate, and BT64.
It supports dense inputs and contiguous equal-length TND shards; document-fragment tails may contain fewer than 64 tokens.
Forward injects only the first document's entry state; backward injects only the final document's exit gradient. Ordinary P2P skips adjacent communication without cross-rank document continuation.
Grouped paths zero effective transitions at actual resets, preventing connections across documents. Group owners still communicate along a group-level chain.

For native dependencies and interface requirements, see the [functional backend documentation](../../hyper_parallel/components/functional/README.md).
Missing required interfaces raise explicit errors without switching to another backend.
