# Experiment Matrix and Acceptance

## Minimum Matrix

Select applicable cases without pretending an unsupported isolated topology is
valid:

| Case | Purpose |
| --- | --- |
| Native/HF module parity | Isolate each replaced or custom mathematical module |
| Optimized replacement reference/candidate A/B | Prove module and end-to-end precision plus performance at one identical topology |
| Same-topology uninterrupted vs resume | Prove complete checkpoint recovery |
| DP/FSDP reference | Establish the reference curve |
| EP-only | Isolate routing, dispatch, expert compute, and combine |
| CP-only | Isolate sequence communication, mask, and RoPE |
| TP-only or TP+EP | Isolate the model's supported TP path |
| TP+CP+EP | Validate combined strategy |
| TP with CP=1 in Validate mode | Expose attention internals hidden by CP regions |
| Production vs Validate | Compare execution modes with identical inputs/state |
| Explicit `layer_count`/`layer_indices` selections | Validate checkpointed activation and shared-state paths |
| Online full/cropped pipeline | Smoke by default; formal only with replay/hash proof |

Run each numerical case long enough to expose update divergence; ten optimizer
steps is the default unless the user selects another count. Recompute and memory
reuse checks require at least two steps because a second-step peak may differ.

## Mandatory Coverage Ledger

Build the ledger before writing the validation manifest. Inventory capabilities
from the final replaced model, adapter providers, plan overrides, recipe, and
available device budget. At minimum, classify:

- DP/FSDP degrees and uneven-shard behavior;
- every supported TP, CP, EP, and sequence-parallel degree;
- meaningful hybrid strategies rather than only isolated axes;
- `layer_count: 0`, an intermediate prefix, and a maximum-safe prefix or explicit
  `layer_indices` selection;
- same- and cross-topology continuation;
- Production and strict Validate execution;
- language, vision, projector, and modality-fusion ownership independently.

Every row is either `formal`, with resolved case names, or `excluded`, with an
observed limitation and evidence. Do not silently omit a row because the first
smoke failed. Do not call a multimodal topology "TP-covered" when only the
language subtree is TP-sharded and the vision/projector subtrees are replicated;
report both placements.

After `--generate-only`, inspect `cases/resolved_cases.json` and reconcile it with
the ledger. Report counts in disjoint categories: baseline, strategy
generalization, recomputation, resume, and Production/Validate. A matrix with one
strategy candidate plus two recomputation and two resume cases has one strategy
generalization case, not five. Stop before the expensive launch when the category
counts, topology degrees, or case names differ from the ledger.

Use a staged launch order. First prove structure/module/checkpoint handoff evidence
is complete and readable. Next run the topology most likely to activate a dormant
divisor or ownership rule, then one nonuniform EP case when applicable, followed
by a real-media default-world-size multimodal case when in scope. Start the full
matrix only after these smokes emit runtime evidence with the expected rank count.
This ordering localizes construction and topology defects before they consume a
multi-case reservation.

## Identity Requirements

Candidates must share:

- exact starting model, optimizer, main-parameter, scheduler, and RNG state;
- one common full-state distributed checkpoint at a declared initial or warm-start
  step when topology-dependent sharded initialization would otherwise consume
  different RNG streams; both the formal baseline and every candidate must restore
  it before their measured steps;
- identical global token/label/mask sequence and sample order;
- identical clipping policy and norm reporting point;
- learned router behavior and seeds;
- identical optimizer parameter grouping and LR values;
- identical custom-model construction path, replacement set, and runtime-input
  semantics; only the dimension under test may change.

For an optimized replacement A/B, the target replacement selection is the only
permitted difference. Keep topology, attention/collective algorithm outside the
target, compile mode, dtype, recomputation, and measured workload identical. A
CP1/EP1 versus CPn/EPn comparison is a scaling experiment, not replacement A/B.

Do not infer state identity from equal seeds. Meta-device materialization and
shard-local initialization can consume random numbers in topology-dependent
orders. For a cropped scratch model, save one common full-state DCP and restore the
measured baseline and every candidate from it whenever TP/EP/FSDP geometry changes.
Record its actual step. The standard matrix launcher currently materializes a
step-1 warm-start checkpoint by running one baseline-topology optimizer step; do
not label that artifact step 0.

Checkpoint tensor resharding and dataloader-cursor resharding are separate
capabilities. Before launching, compare the saved and target DP world sizes and
the selected loader's resume contract. If a stateful Online loader cannot rebuild
one global cursor for the target DP geometry, use fixed Offline replay or record
the case as `BLOCKED`/excluded with evidence. The standard shared warm-start is
such an explicitly defined experiment: it restores model/optimizer/scheduler/RNG
but intentionally requests replay from the same configured input start. Because
DP-dependent sharding or packing can still produce different samples, equal global
input hashes remain a hard prerequisite for numerical comparison. Skipping the
cursor is not valid for K-to-K+N resume.

## Metrics

Record each optimizer step's:

- loss and LR;
- global gradient norm, stating whether it is pre- or post-clipping;
- FP32 main-gradient norm and update norm;
- model/main-parameter synchronization or alias evidence;
- selected parameter fingerprints before and after update;
- routing/top-k summary for MoE;
- global batch/sample hash, peak memory, and non-finite/fatal events;
- loss dtype, warm-up/steady-state step time, token throughput, and recomputation
  wrapper count/FQNs;
- representative parameter/main-gradient/optimizer-state logical and local shapes,
  including an empty uneven shard when one exists.

`max_grad_norm=1` does not imply a logged pre-clip norm below one. Compare the
same norm definition between cases.

Validate evidence by record schema. Numeric parameter/gradient records must carry
finite status and no missing ranks. Structural optimizer-state records may carry
layout/absence metadata without a numeric `finite` field; do not count an absent
field as a non-finite tensor. Conversely, a process exit code or success marker
cannot replace required numeric records. Require the expected stages and probe
classes before comparing a case.

## Acceptance

Same-topology continuation and identical Production/Validate cases should be
exact or satisfy a separately justified dtype-level tolerance. Cross-topology
BF16-forward policies may declare both absolute (`loss_max_abs`, `norm_max_abs`)
and scale-aware relative (`loss_max_rel`, `norm_max_rel`) bounds. The default
combination is `all`; an explicitly declared `combination: any` means absolute
or relative agreement for each scalar independently. It never means that loss
may compensate for a failed norm. Record every chosen threshold and combination
in the manifest and report.

A case passes only when:

- all required steps complete;
- loss and norm both meet their tolerances;
- LR and data identity match;
- errors remain centered without systematic drift;
- no NaN, Inf, fatal error, unexplained checkpoint gap, or hidden fallback occurs;
- optimizer-state DCP metadata is rank-consistent and survives same- and
  cross-topology restore;
- the scalar objective remains FP32 when the required precision policy is used.

Loss agreement with norm disagreement is `FAIL`. OOM or missing assets is
`BLOCKED`, not a partial pass.

Performance is descriptive, not a substitute for precision acceptance. Compare
steady-state steps after separating initialization, first collective, compile, and
checkpoint I/O. Do not call a topology optimal without representative model scale,
warm-up, multiple measured steps, and communication/recomputation overlap evidence.

For a target explicitly claimed as a high-performance replacement, apply the
separate predeclared performance thresholds from `optimization_plan.yaml` after
precision passes. Report `PERFORMANCE_PASS`, `PERFORMANCE_FAIL`, or
`PERFORMANCE_INCONCLUSIVE`; do not change the core precision verdict based on
throughput, and do not make the high-performance claim without `PERFORMANCE_PASS`.

## Diagnostic Evidence Invariants

Diagnostics are part of the distributed program. Every rank must execute the
same collective sequence even when a lazily materialized optimizer-state key is
absent locally. Gather the global union of state names first, then contribute an
explicit missing value per rank. Never let one rank skip a collective based on
local state.

Optimizer internals may be topology-local. Across distinct HSDP/TP geometries,
require live model/main-gradient parity plus optimizer layout, finite status, and
DCP recovery, but compare optimizer-state numeric values only when their
mathematical ownership is topology-invariant. Same-topology continuation remains
strict.

Parameter probes contain two evidence regimes. A probe with complete `values`
can establish true elementwise `max_abs` and tensor `relative_l2`; keep their
default combination strict (`all`). If a predeclared BF16 policy uses `any`, state
it explicitly in the manifest and report; never change it merely to turn an
unexplained failure green. A summary-only probe (`sum/l2/min/max`) cannot establish
either elementwise max error or tensor relative L2. Report both as unavailable
and gate only explicitly named summary metrics, such as
`parameters.summary.l2_norm_relative` (and optional `l2_norm_max_abs`). Never label
the largest aggregate-statistic delta as `max_abs`.

Keep module-parity and same-topology parameter limits strict when cross-topology
BF16 needs a wider scale-aware policy. Put strict inherited defaults directly
under `acceptance.parameters`, and declare only the justified override under
`parameters.cross_topology` (or, when genuinely needed, `module_parity` and
`same_topology`). The selected case acceptance class must select the matching
parameter profile; a global relaxation that also weakens same-topology resume is
not acceptable.
