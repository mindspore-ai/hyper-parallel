---
name: hf-model-integration
description: >
  Integrate or upgrade an HF-native model or an HF-component-backed custom model
  under hyper_parallel/models, including lazy registration, custom architecture
  construction, module replacement, TP/CP/EP adaptation, checkpoint conversion,
  runtime inputs, recipes, examples, and focused tests. Use for implementation and
  functional bring-up; use hf-model-precision-validation for numerical acceptance.
---

# HF and Custom Model Integration

Add the smallest model-owned implementation and adapter that preserve the
authoritative model's mathematics while composing HyperParallel's generic
components. A model repository may use HF configs, tokenizers, or building blocks
without its architecture being representable by the installed HF model class.

## Load

1. Read `AGENTS.md` and `.agent/rules/code-style.md` before editing.
2. Read [references/model-analysis-and-adapter.md](references/model-analysis-and-adapter.md)
   before deciding the adapter surface.
3. Read
   [references/optimized-capability-discovery.md](references/optimized-capability-discovery.md)
   when the model needs optimized replacements, fused kernels, CP, EP, or a new
   reusable performance component.
4. Read [references/custom-model-integration.md](references/custom-model-integration.md)
   when HF cannot construct the exact training architecture, or when the model has
   cross-layer state, native multimodality, custom checkpoint names, or runtime-only
   forward inputs.
5. Read [references/parallel-and-precision.md](references/parallel-and-precision.md)
   when adding module replacement, TP, CP, EP, FSDP, optimizer integration, or
   plan boundaries for modules introduced by replacement.
6. Read [references/examples-tests-handoff.md](references/examples-tests-handoff.md)
   before adding recipes, examples, tests, or the validation handoff.
7. Read [references/bringup-field-checklist.md](references/bringup-field-checklist.md)
   at the start of any bring-up: version-pinned survey, hardware profile,
   weight acquisition with size verification, offline dependency overlay,
   HF-native-first registration, the released-layout contract test, and the
   cheapest on-device first checks.
8. Read [references/cleanroom-protocol.md](references/cleanroom-protocol.md)
   when auditing workflow reproducibility with a memory-free agent, or when
   running the asset-vs-memory ablation of a longitudinal evaluation.

## Bring-up stage machine

This table is the living source of truth for the bring-up workflow: each
run's post-run review amends it here, and any rendered diagram follows it.

A model bring-up runs these stages in order. Each stage names the skills it
uses, the artifact it must persist (the only interface to the next stage),
and its exit gate. Resume by reading the run directory: the first missing
artifact is the current stage. The run directory is designated at intake
(default: a `run/` directory beside the model directory). A resume card
found anywhere else is input evidence, not authoritative state — re-validate
it against the current assets before acting, because the assets may have
advanced past what the card records.

| Stage | Uses | Persisted artifact | Exit gate |
|---|---|---|---|
| S0 survey & planning | this skill's [bring-up field checklist](references/bringup-field-checklist.md) §1-§3; parallel-strategy-analyzer | migration spec: structure table, hardware + dependency profile (fill [templates/environment-profile.md](templates/environment-profile.md)), parallel plan, risk list | spec complete; every no-precedent semantic marked for escalation |
| S1 adapter implementation | this skill; add-unit-test; feature-doc† | recipe on the generic kit (adapter only if a provider is needed) + released-layout pin (checklist §5 script, exit 0, stating its shape coverage: full-shape vs names-only) | layout pin passes; any family tests green |
| S2 single-node correctness | checklist §6; hf-model-precision-validation | same-process bitwise parity report vs the plain HF build (on the zero-registration route this persisted report replaces `integration_handoff.yaml` as the S3 entry precondition), smoke loss curve | forward parity holds; loss finite and falling |
| S3 distributed equivalence | hf-model-precision-validation; cluster-debug† | equivalence matrix on the runtime's structured evidence (`model_integration.mode=runtime`: full-precision loss and gradient norms plus a per-step global input hash): recompute on/off exact; deterministic rerun exact; cross-layout from identical released weights and a token-uniform corpus — early steps exact, divergence onset and growth reported; judged by the generic kit's `judge_equivalence.py`, which delegates each step to the framework's `compare_scalar_curves` | every target layout matches its criterion |
| S4 profile & optimize | profiling†; memory-analysis†; perf-playbook†; autoresearch† | profile baseline with all consistency checks passing + bottleneck list | stated perf bar met, or budget spent with a written remainder |
| S5 acceptance & integration | precision-acceptance rule†; autogit; gate-doctor; code-review | acceptance run, reports, **a save/restore/resume record** (see below), PR past the CI gate | gate green + deliverables checklist |
| S6 formal verification (optional) | formal-verification† | spec + counterexample review | no open counterexample |

† Not yet in this checkout; delivered by in-flight PRs. When a marked skill is
absent, do not stall: apply the equivalent discipline recorded in this skill's
references, record the substitution in the run ledger (the first pilot run did
exactly this), and skip S6 — it is optional.

## Cross-layout equivalence: control the inputs before reading the number

Comparing two data-parallel layouts is only a statement about the framework
if everything except the layout is held fixed. Two things are easy to leave
free and both poison the result:

- **Weights.** `from_config` initialization materializes per rank under the
  device mesh, so a different mesh yields different initial weights and the
  comparison measures initialization, not parallelism. Load the released
  checkpoint on both sides (`load_pretrained=true` in the generic kit), so
  both layouts start from one set of numbers.
- **Packing.** Keep the global batch and the seed fixed so each step
  consumes the same global samples under either layout, and *prove* it per
  step rather than assuming the sampler is layout-independent. The runtime's
  `global_input_sha256` hashes every sample separately and order-free, so it
  stays comparable across data-parallel degrees, and the judge blocks any
  step where it differs. Text logs offer only `consumed_samples` and
  `consumed_tokens` agreeing step for step, which shows the counts match,
  not the content.

With both controlled, the only remaining source of difference is the order
of the reduction. A difference reported without these controls is a
statistical envelope over three confounded effects — usable as a smoke
signal, not as an equivalence claim, and it must be labelled that way in the
run record.

**Then judge the early steps, not the late ones.** Field measurement with
every control in place (identical released weights, a token-uniform corpus
where every sample is the same sequence, fixed global batch and seed):
layouts agreed to the trainer log's printed precision, six significant
digits, for the first three to four steps and then drifted, reaching 4e-2 by
step ten with one optimizer and 5e-4 with another. What that early agreement
shows is bounded by the precision it was read at: any layout-induced
difference started below about 1e-6 relative. That rules out the coarse
defects, which read near 1e-2 at step one, but it does not make the steps
bitwise identical; a difference can exist from step one below the last
printed digit and be amplified into view. A bitwise statement needs the
structured runtime evidence, where every step is compared exactly and the
inputs are checked. The later numbers are accumulation differences amplified
by training dynamics, and gating them with a fixed band only decides which
optimizer's amplification rate you happened to pick. So the criterion is:

1. the first steps must match **exactly at the precision of the evidence**
   across layouts, and the record names that precision: full precision from
   the runtime's structured metrics where available, otherwise the log's six
   printed digits (one step is a claim; several are a stronger one);
2. the **divergence onset and growth** are reported as measurements, with
   the optimizer named, not graded against a threshold;
3. a late-step band may still be recorded as a smoke signal, labelled as
   amplification rather than as equivalence.

A stack whose step one already differs at the scale of a real defect — a
layout-dependent data path, an initialization that follows the mesh, a
reduction that is not associative-safe, all of which read near 1e-2 — is
what this gate exists to catch. Read a full-precision step-one difference of
a few units in the last place differently: two layouts reduce the loss and
the gradients through different summation trees, and floating-point
addition does not associate. Record it as the measured onset, not as a
defect. No bound between the two scales has been calibrated yet on
full-precision cross-layout runs; the check therefore stays exact, and the
reported magnitude is what tells the two apart until runs that are not under
judgement settle a bound.

**How many early steps is itself a per-model measurement.** The default of
three came from five models whose divergence began at step four or later.
The sixth — a hybrid linear-attention backbone, whose chunked state
recurrence is more sensitive to reduction order — agreed at printed precision
for two steps and parted visibly at the third, so it fails a three-step gate
while showing no defect signal under clause 1: step one matched to six
digits. Record the
failure as a failure, state the onset beside it, and raise the constant as
something to decide per model; a divergence that starts at step one, and
the growth curve after onset, remain the things being judged. **Do not relax
your own gate because your own run tripped it.** Only evidence from runs
that are not the one under judgement can move a threshold, and the move is
an operator decision with that evidence attached — a gate loosened by the
run it just failed measures nothing afterwards.

**An untestable cell is recorded as untestable, never skipped.** A matrix
cell can be impossible at a given shape rather than merely unrun — a
comparison arm that needs recompute off, or a smaller shard count holding a
larger optimizer-state share, can exceed device memory when the other arm
fits. Report the cell with the reason and the measured number that made it
impossible (peak allocation against capacity), so a reader can tell "we did
not measure this" from "this passed"; silently dropping it leaves a matrix
that looks complete.

## Gate design

When a stage needs a new exit criterion, or an existing one has to be
changed, these rules decide whether it is worth anything. Each was paid for
by a field failure.

1. **Make gates independent.** A gate earns its place by the class of error
   it can refuse while every other gate passes. Four gates carry a bring-up:
   same-process bitwise parity, the distributed-equivalence matrix, the
   profiler consistency checks, and the released-layout pin. Adding a fifth
   that fails only when an existing one already fails buys nothing.
2. **Keep one gate that compares against the declaration, not the
   behaviour.** Behavioural gates assume the thing measured is the thing
   intended. When that assumption breaks they are blind in the same
   direction: a composite checkpoint built through a causal-LM facade
   yielded a model missing an entire tower, and parity was still bitwise
   exact because the shared path really was identical. The layout pin builds
   the class `config.architectures` names and requires a bijection with the
   released catalog, so it is the only gate that can refuse "you built the
   wrong model". Keep it, keep it cheap, and run it before any device is
   booked.
3. **Calibrate a check on at least two operating points.** A tolerance fitted
   to one regime encodes that regime. One fitted to a multi-second step time
   rejected a model whose steps were five times faster, because the benign
   drift was a per-step constant, not a proportion.
4. **Ship positive controls with every probe.** Validate a check against the
   condition it must catch, not only the condition it must allow. A check
   exercised solely on healthy inputs fails open silently; two independent
   occupancy signals, each validated only against idle nodes, agreed that a
   fully occupied node was free. A known-bad fixture the probe must flag,
   run as a unit test, is the cheapest insurance in this repository.
5. **Treat a gate's silence as unproven, not as proven.** Where no gate
   looks, the artifact must say so. Five bring-ups recorded stable
   thousand-step runs while two of them could not have saved a checkpoint;
   nothing contradicted them because the long-run never saved. Either the
   stage records a result or it records an explicit, justified out-of-scope
   statement.
6. **Report outliers with an attribution, not a smoothed average.** A
   long-run's acceptance states the median and a high percentile and names
   every excursion: one run sat 852 s on a single step, recovered itself,
   and finished with no error in any log — a mean would have hidden it and a
   bare maximum would not have said whether it was the model, the storage or
   the fabric. If an excursion cannot be attributed, say that, and record
   what telemetry would have attributed it.
7. **Find the gate that already exists before writing one, and state the
   precision of every verdict.** The S3 judge was first written to parse
   trainer text logs while the repository already recorded the same
   quantities at full precision with a per-step input hash
   (`model_integration.mode=runtime`, compared by
   `hyper_parallel.tools.model_integration.case_compare`, documented in
   `hf-model-precision-validation`). The duplicate was weaker on both axes
   that matter: it compared six printed digits and never checked inputs. Its
   verdicts were still worded as bitwise, and that wording travelled into a
   report before anyone asked how the comparison was made. Search the skills
   and `hyper_parallel/tools` for the quantity you are about to judge, extend
   what is there, and make each verdict carry the precision it was read at.

## Diagnosing a resource ceiling

When a run dies against a resource limit, the plateau value proves only that
the total exceeded the limit — it does not name the tensor, the phase, or
the cause, and reading it as if it did produces confident wrong answers (one
run burned three attempts on three different wrong attributions that way).
Find two *passing* configurations instead, fit the trend across them, and
extrapolate: with two points the slope separates a per-layer cost from a
constant overhead, and the extrapolation predicts the largest configuration
that fits. A single failing data point supports no attribution at all.

**An initialization ceiling is not a model-size ceiling.** Before recording
"too large for this cluster", check whether the framework can build the
model on the host instead of sharding it across devices: the peak that
refused the run may belong to the construction path alone, and the
steady-state footprint can sit far below it. Measured: a 64-layer 32B dense
model whose device-side sharded init exceeded a 64 GB die trained on two
nodes at a **37.4 GB** peak once construction moved host-side — the recorded
blocker was real, and it was also two configuration flags wide. State both
in the record: the defect still stands (the device path should not peak
that way) and the model is no longer blocked, because those are different
facts and conflating them either hides a framework bug or strands a model
that runs.

**When a blocker lifts, the next wall is the evidence that it lifted.** A
framework defect that stops a model at initialization leaves nothing to
measure; after the fix, the run reaching a *different* failure — one step
completed, then device memory — is what proves the first defect is gone,
and the recovery from that second wall (recompute, a smaller shard, a
shorter sequence) is an ordinary engineering step rather than a second
blocker. Record the progression as a row per state, with the number that
moved each time (steps completed, peak allocation, step time): "0 steps,
layout inference failed → 1 step, 56.8 GB against 64 → 30 steps, 24.5 GB,
4.03 s/step" is an argument; "it works now" is not. Each wall must be
reached and named, or a fix that changed nothing reads the same as one
that worked.

Autonomy boundary: escalate to a human only for (1) a design decision with
no in-repo precedent, (2) an acceptance threshold that cannot be derived
from an existing baseline, (3) anything touching shared resources
destructively, (4) the same exit gate failing twice. Budget (device-hours,
wall-clock, experiment rounds) is fixed at intake. Timing and efficiency
numbers are valid only from same-node back-to-back runs; equivalence
verdicts under full determinism are node-portable within an identical pool
(field-verified at printed precision: identical per-step loss/grad_norm
across two nodes).

**Size the acceptance window to the model, and say what that window can
conclude.** S5 measures *drift* — step time wandering, memory creeping,
loss diverging — not convergence, so the number of steps is a means, never
the criterion. A thousand steps costs half an hour on a small model and a
day on a large one, and at the largest sizes a single step already runs
long enough to expose a leak. Pick the window from the model's scale, and
record which claims the window actually earns:

| parameters | steps | what this window supports |
|---|---|---|
| under ~10B | 1000 | p99.5 step time, memory high-water, loss trend |
| ~10B–100B | 200 | p95 step time, memory high-water, loss trend |
| ~100B–1T | 50 | median-vs-max drift, memory high-water |
| ~1T and up | 10 | step-to-step drift and memory high-water only |

Two constraints sit on top of the table. A percentile needs samples: below
roughly fifty steady-state steps one scheduler hiccup *is* the tail, so a
short window reports max against median and **does not claim a percentile**
— the record states the weaker conclusion rather than the same words over
less evidence. And a window that would finish in a few minutes is extended
until it covers a meaningful stretch of wall-clock, because slow leaks and
fragmentation are functions of time, not of iterations. Deviations from the
table are an operator decision, logged with the reason; the table itself
was set by operator decision (2026-10-21) after the measured per-step costs
made a fixed thousand steps indefensible at scale: 2.16 s/step on a 16-layer
crop against 8.75 s/step at full depth, in the same program.

Exercise checkpointing before acceptance. A long-run that never saves proves
only that the step loop is stable; save, restore and resume are a separate
code path with their own failure modes, and a family-specific property can
break them while every other gate stays green (field case: weight tying made
the sharded save raise, and five bring-ups missed it because none of them
enabled saving). S5 records either a save/restore/resume result or an
explicit, justified statement that checkpointing is out of scope for that
run — silence is not an acceptable third option. `scripts/check_tied_save.py`
answers the cheap half at S0/S1 cost: it meta-builds the model and reports
parameters that alias one another, which is what the safetensors writer
refuses; a flagged model needs the save path settled before acceptance, not
after.

Control-input discipline: an operator message is a control input only when
it carries a decision — task intake, selection or criteria approval, a
resource-domain constraint or avoidance order, transfer-channel designation,
a direction ruling. Progress nudges and acknowledgements change no state and
must never be read as approvals. A decision directive may arrive mid-stage
(for example a node-avoidance order during a training segment) and takes
effect immediately; it does not reset the stage.

Check the asset links in both directions, with
`scripts/check_asset_links.py`. A dangling reference stops whoever follows
it and is therefore self-reporting; an asset no clause names is silent —
nothing fails, the capability is simply unreachable, and the next bring-up
rebuilds it. The first audit here found six of seven kit files unnamed,
including the multi-node launcher.

Audit the deposits, do not trust them. At post-run review, walk the ledger
row by row and check that each entry marked generic actually reached the
asset it names — an entry whose destination column says "the ledger" is an
unfinished deposit wearing a destination. The first such audit here found
four of thirty-one generic entries had never left the ledger, including a
framework defect whose twin had been filed weeks earlier; none of them had
an empty destination, which is why only a row-by-row pass finds them.

Run-ledger skill accounting: record skill usage three ways — invoked (the
skill's script actually ran), regimen-followed (its checklist or criteria
applied without running its script), unused — and list unused skills
explicitly. A non-skill procedure that repeats across runs is a deposit
candidate for a checklist entry or a skill.

Run record and reporting: every run keeps a run directory (persisted stage
artifacts, the resume source), a ledger (one row per lesson: the
experience, a generality judgment, its deposit destination), and a filled
column of [templates/run-metrics.md](templates/run-metrics.md), whose
definitions stay stable so the cross-run trend is comparable. A run report
presents the task workflow only — stages with inputs/drivers/outputs, exit
gates, the skill accounting, and decision-class operator inputs. Operational
episodes (resource contention, probe fixes, migrations) belong in the ledger
and the operations checklist, not in the report; the report cites the ledger
as its audit trail, so scoping the story never means losing the record.

## Required Precision Policy

All onboarding recipes and validation handoffs use this policy unless the user
explicitly changes it:

```yaml
model_init_dtype: float32

fsdp_config:
  mix_precision:
    param_dtype: bfloat16
    reduce_dtype: float32
    cast_forward_inputs: false

optimizer:
  fp32_main_params: true
```

`model_init_dtype` is the top-level Trainer field. Do not substitute the
model target's `torch_dtype` for it. Verify the resolved configuration and the
runtime parameter dtypes.

Leave `fsdp_config.mix_precision.output_dtype` absent unless the model contract
explicitly requires an output cast. Absence resolves to `None` and preserves an
FP32 scalar loss produced by the model.

`optimizer.fp32_main_params=true` must select
`Float16OptimizerWithFloat16Params`. Every trainable parameter must expose an
FP32 `main_param`. An FP32 model parameter may alias its own `main_param`; a
BF16/FP16 model parameter requires a distinct FP32 main parameter. FSDP gradient
reduction must remain FP32, and FSDP must not cast forward inputs automatically.
For a DTensor parameter, verify that parameter, `main_param`, `main_grad`, and
lazily created optimizer state expose one consistent logical global layout.

## Workflow

1. Establish the authoritative implementation, installed Transformers version and
   source path, model/checkpoint revision, task facade, full-pretrained versus
   cropped mode, target hardware, parallel axes, and required optimized modules.
   When the user supplies only a Hugging Face repo ID or URL, first create a
   local source checkout with Git LFS smudging disabled, pin the resolved commit,
   and use that checkout as the authoritative source path. For example:

   ```bash
   git ls-remote https://huggingface.co/ORG/REPO.git refs/heads/main
   GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none \
     https://huggingface.co/ORG/REPO.git /local/source/REPO
   GIT_LFS_SKIP_SMUDGE=1 git -C /local/source/REPO checkout --detach COMMIT
   ```

   Record the resolved commit and verify that tracked model-weight files are LFS
   pointers rather than downloaded payloads. This source-only bootstrap does not
   authorize weight download: ask for the weight cache/download directory and
   explicit permission before fetching any weight payload.
   When resources require a crop but the goal is structural coverage, preserve
   the released text and modality-tower depths and every native layer-role index.
   Classify every changed value as a parameter dimension, semantic cardinality,
   topology divisor, role index, or table/bucket size before scaling it. Preserve
   semantic cardinalities unless equivalence is independently proven, and check
   every topology divisor against the complete requested matrix. Record the crop
   invariant ledger with released/cropped values and dependencies. A shallow layer crop is a
   smoke-only exception and cannot claim all-structure coverage when it removes
   or renumbers a distinct layer role.
2. Write a schema-v1 local-only integration manifest and run:

   ```bash
   python -m hyper_parallel.tools.model_integration inspect --manifest <manifest>
   ```

   Review `inventory/reference.json`, `inventory/candidate.json`, and the classified
   `inventory/structure_diff.json` before choosing HF-native, HF-component custom,
   or fully native construction. Successful `AutoConfig` loading is not evidence
   that `AutoModel` implements the released architecture.
3. Before scaffolding or implementing optimized code, complete the mandatory
   high-performance capability discovery and reuse design. Inventory the final
   mathematical modules, search generic components and existing adapters, and
   classify each target as direct reuse, generic-component-plus-thin-wrapper,
   reusable-component extraction, or a justified new implementation. Record the
   target hardware/backend, semantic compatibility, weight/state-dict mapping,
   TP/CP/EP boundary, known fallbacks, and the paired precision/performance
   experiment that will prove the choice. Store the inventory and plan as
   `analysis/optimized_capability_inventory.json` and
   `analysis/optimization_plan.yaml`. Do not claim an implementation is optimized
   merely because it is fused, distributed, or resembles another model family.
4. If the family is new, use the public scaffold command to create only the adapter
   skeleton, registration test, and validation provider:

   ```bash
   python -m hyper_parallel.tools.model_integration scaffold \
     --manifest <manifest> --scaffold-dir hyper_parallel/models/<family>
   ```

   Complete the generated declarations; do not accept generated mathematical code.
5. Put architecture invariants and state lifetimes in the model, and mapping,
   replacements, parallel rules, and runtime-input translation in its adapter.
   Never add a model family, architecture, class, FQN, or config-field special
   case to generic framework paths. If the existing contracts cannot express a
   required capability, treat that as a framework extension gap: define the
   smallest model-independent provider/protocol/declarative contract, keep its
   default path behavior-preserving, and implement the model semantics in the
   adapter. If changing the framework contract is outside the authorized scope,
   stop with a concrete extension design and a `BLOCKED` handoff instead of
   hardcoding the model or hiding the behavior in a one-off runner. Keep
   shared precision changes out of model bring-up: before proposing any change
   to generic reduction dtype/order or cast policy, show the first divergent
   intermediate, why the adapter or recipe cannot address it, and the expected
   effects across models; obtain an explicit user decision before editing.
   The proposed four-BF16-Partial FP32 sum for SP parity was rejected here.
   Keep experiment choices in declarative recipes or manifests. Classify every
   registered buffer against the authoritative model's persistence contract. For
   deterministic non-persistent state invalidated by meta-device `to_empty()`,
   use the public rebuildable-buffer lifecycle described in the custom-model
   reference; never turn it into checkpoint state merely to survive
   materialization.
   Define the complete production architecture in the model package's
   `modeling*.py` files. The registered class must construct a canonical,
   executable model before any recipe replacement is applied, including every
   claimed text, vision, MoE, memory, and residual-stream branch. Run a small
   direct forward/backward smoke with all replacements disabled. A fused kernel,
   distributed wrapper, or replacement may optimize an already complete source
   module or add its parallel execution, but must never create missing layers or
   parameters, provide the only executable forward, or otherwise complete the
   base architecture. Do not use parameter-owning or non-executable placeholders
   as production model structure. If authoritative code is available only in a
   source repository, port the required architecture into `modeling*.py` first.
   Treat reference construction and optimized replacement as two independent
   gates. Gate A disables every recipe replacement and proves model semantics,
   initialization, gradients, and checkpoint FQNs. Gate B enables the selected
   replacements and proves parameter/state identity, numerical parity,
   TP/CP/EP/FSDP behavior, and same-topology performance. A Gate B pass never
   compensates for a Gate A failure. Reject an integration whose construction or
   direct forward/backward depends on replacement. A shared semantic base is
   acceptable only when the production model explicitly selects its portable
   reference branch and the replacement enables the device-optimized branch.
6. Register custom model classes and the minimum `ModelAdapterSpec` providers
   lazily. Do not edit the central registry when family discovery or a justified
   family alias is sufficient.
7. Implement optional optimized paths and parallel adapters according to the
   recorded reuse decision and by composing generic components. Existing family adapters are
   reference implementations, not cross-family dependencies: move shared kernels,
   collectives, or dispatch skeletons into a model-independent component instead
   of importing another family's private adapter. Preserve a selectable reference
   path for paired validation until the optimized implementation is accepted.
   Preserve forward/state-dict contracts or declare explicit weight transforms.
   Activate optional replacements and CP/EP behavior through recipe `plan_overrides`.
   Before adding a model or feature-specific replacement hook to a generic builder,
   check whether an existing `replace_module` entry can select the source module
   by FQN glob and `module_type`. Keep rule selection in the recipe; put a
   factory in the model adapter only when model semantics require it. Do not
   append adapter-owned replacement lists to the generic builder. Test with
   and without the recipe action, including match count and parameter identity.
   Inspect the final replaced model tree: a uniform glob may create boundaries
   only when it declares a concrete parameter or I/O contract. A later partial
   glob may merge behavior into those boundaries; it must not silently target an
   unplanned module.
8. Add text or multimodal Offline and Online examples appropriate to the load
   mode. Missing local assets fail explicitly; launchers must not download
   resources implicitly or contain environment-specific `PYTHONPATH` settings.
   Resolve each launcher's default world size and require its mesh plus global
   batch to be feasible before running it. For multimodal examples, compare plan
   matches with exact final-tree subtree counts and run real media at the default
   world size with `model_integration.mode=runtime`; require structured data,
   modality-gradient, probe, and observed-rank evidence rather than a success
   marker alone.
9. Point `model.builder` at the final production construction path, then run the
   complete model-integration structure check. Resolve every structured ERROR;
   never bypass an unresolved finalization,
   boundary, materialization invariant, checkpoint mapping, gradient domain, or
   required parameter probe.

   ```bash
   python -m hyper_parallel.tools.model_integration check --manifest <manifest>
   ```

   Before handoff, also produce a parallel-ownership ledger for every major
   subtree. For a multimodal model, list language, vision tower, projector, and
   fusion separately with their actual TP/CP/EP/FSDP placements. Executing a
   replicated vision tower inside a TP launch validates composition with TP but
   is not vision-TP coverage. Record exact unsupported DTensor callable names;
   do not infer coverage from a related fused operator or another convolution
   dimensionality.

10. Declare authoritative final-module cases in `ModelValidationSpec` and run parity
   on FP32 first, then production dtype. Candidate builders must select the module
   after the real replacement/materialization path. Reference and candidate
   implementations must be independent.

   ```bash
   python -m hyper_parallel.tools.model_integration parity \
     --manifest <manifest> --device cpu --dtype float32
   ```

   A passing run writes `integration_handoff.yaml`. The parity CLI requires a
   registered adapter, so it does not apply on the zero-registration route
   (checklist §4): there, run the same-process bitwise parity of checklist
   §6.2 and persist its report — that report is the S2→S3 handoff artifact
   (see the stage machine). Resolve and parse every
   referenced structure/parity/checkpoint artifact before handing this file,
   rather than an ad-hoc log summary, to `hf-model-precision-validation`. The
   handoff directory must also retain the capability inventory and optimization
   plan so the precision stage can run the declared replacement, CP, and EP A/B
   experiments.

## Completion Gate

The integration is complete only when model construction and registration remain
lazy, the production architecture passes direct forward/backward without mandatory
recipe replacements, optional replacement and checkpoint contracts are proven, changed modules have an
authoritative parity oracle, the final model has complete boundary coverage,
focused tests pass, meta materialization restores every declared derived buffer
without mutating checkpoint-owned state, examples fail clearly on missing inputs,
launcher defaults pass topology/batch feasibility and runtime-evidence checks,
the capability inventory accounts for every required optimized module and CP/EP
path, each reuse/new-implementation decision has a paired validation plan, and the
handoff is ready for `hf-model-precision-validation`. Audit the final diff:
generic framework behavior must not depend on new model-family identifiers,
concrete family imports, or family-specific FQN branches. Any new framework API
must be model-independent, optional when no adapter provides it, exercised by a
generic contract test, and consumed through the model adapter. Also reject a
model/feature-specific mechanism when a declared YAML glob, existing provider,
or generic replacement executor already expresses the same behavior.
