# Model Integration Validation Guide

**English** | [简体中文](model_integration_validation.zh.md)

## Start Integration Validation with an Agent

At minimum, tell the agent the model source and available devices. The source can be a local path, a Hugging Face repository ID,
or an `https://huggingface.co/...` URL. If no data is available, ask the agent to create it automatically. Add a sentence for any explicit requirements
such as parallel strategies or sequence length.

```text
Use hf-model-integration and hf-model-precision-validation to integrate and validate
[model name]. The authoritative implementation/model source is [local path, HF repository ID, or HF URL],
and [device count] [NPUs/GPUs] are available.

If no data is available, automatically create a small, deterministic local validation dataset and reuse the same inputs for every case.
Before implementing, scan existing generic high-performance modules, replacements, CP/EP communication, and model adapters in the repository. Produce
a capability-matching table and reuse decisions; do not reimplement existing kernels or depend directly on another model family's private adapter.
For every replacement, CP, or EP path claimed to be high performance, use the same weights, inputs, dtype, topology, and
Trainer path for numerical comparisons before/after replacement, module microbenchmarks on the target device, and end-to-end A/B on the same topology. Report
CP/EP scalability separately. Without fair A/B evidence, mark PERFORMANCE_INCONCLUSIVE and do not claim a speedup.
First construct a cropped model covering all critical layer roles for fast validation. If full weights and sufficient resources are available, finally load
the full-size weights to validate short continued training and checkpoint resumption. Fix integration issues until PASS; if progress is impossible, report
BLOCKED. Do not hard-code models in the framework or download resources without permission. Finally provide the configurations actually run,
commands, results, evidence directory, and uncovered items.
```

For example, append: `Validate TP1/TP2, CP1/CP2, and EP1/EP8 at sequence length 4K.` Without explicit settings, the agent
creates a minimal but discriminating validation matrix based on the model structure and available devices.

If full-size weights have already been downloaded, simply append the local path:

```text
After cropped validation passes, load the local full-size model from [checkpoint directory] and validate short continued training with
[target device count/parallel strategy]. Do not download the weights again. Save the complete training state at step K, resume for N steps, and compare against
an uninterrupted K+N-step run. Report BLOCKED if resources cannot accommodate the full model.
```

The agent checks the directory's complete config, weight shards and index, tokenizer/processor, and checkpoint
coverage. The full-weight path is used only in stage two; stage one still constructs a cropped model from the same release config.

If weights have not been downloaded and you want the agent to download and validate them, append:

```text
After cropped validation passes, you may download the full weights from [HF repository ID or URL] to [local directory]. Use the complete
release config and weights to validate short continued training with [target device count/parallel strategy]. Save the full training state at step K,
resume for N steps, and compare against an uninterrupted K+N-step run. If resources cannot accommodate the full model, report BLOCKED; do not
crop it again and claim that the full-size model passed.
```

Providing a Hugging Face URL alone does not authorize downloading large weights. The agent should first obtain a local source-only
copy of the original repository with `GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none <URL> <DIR>`, pin the resolved
commit, and verify that weight files remain Git LFS pointers rather than payloads. The training launcher reads config, tokenizer, and source code from this local
directory without ad hoc network access during multi-device startup. Weight payloads may be downloaded separately only after the user explicitly authorizes it
and supplies a destination directory.

### Default Two-stage Validation

Stage one quickly validates a cropped model. Starting from the full config, the agent preserves structural distinctions such as dense/MoE, different attention
roles, shared state, and vision paths. It initializes a small model with `from_config` and validates final-module
parity, numerical self-consistency across parallelism/recomputation, and checkpoint resumption for the cropped model. Passing this stage establishes only the
`cropped_due_to_resource` scope.

Stage two continues training from full-size weights. The corresponding Trainer recipe must switch to the complete config and
`from_pretrained`; it must not continue using the cropped builder. For example, an HF-native model is loaded as follows:

```yaml
model:
  _target_: hyper_parallel.models.HyperAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: /local/path/to/full-hf-checkpoint
```

For custom architectures, the custom-model registry resolves the production model class, and checkpoint mapping loads the same complete local
checkpoint. The model adapter separately exposes parallelism, FSDP, recomputation, and validation capabilities. Full-size acceptance checks at least weight
coverage, native/candidate alignment on the first step, and short-run loss/norm.

Hugging Face model repositories usually provide only model weights. Starting training from them means continuing training from pretrained weights,
not restoring the complete training state. Resume validation must also run an uninterrupted K+N-step baseline from those weights, alongside a run that
saves HyperParallel DCP after K steps, then restores the model, FP32 main parameters, optimizer, scheduler,
global step, RNG, and dataloader cursor for another N steps. Both paths must use the same inputs and produce aligned results.

### How the Agent Creates Validation Data by Default

When the user supplies no data, the agent should first create a small reproducible dataset, without requiring the user to understand Dataset, collator,
or label-shift details.

For text models, the default is local JSONL. Previous DeepSeek V4.1 validation used 128 long texts with unique IDs,
repeating fixed sentences so the Online tokenizer and packing could reliably form 4K sequences:

```bash
python -m hyper_parallel.models.deepseek_v41.tools.prepare_deepseek_v41_online_data \
  --output output/training_demo/deepseek_v41/online_4k.jsonl \
  --num-samples 128 \
  --sequence-length 4096
```

Each line has only one generic field, such as `{"text": "..."}`. The model's transform handles tokenization,
packing, and labels. With fixed files, random seed, and sampler, all parallel cases must produce the same
`global_input_sha256`.

For multimodal models, the default is a small set of images and JSONL in OpenAI messages format. Previous DeepSeek V4.1 VLM
validation selected fixed rows from ChartQA `val` at a pinned revision, saved the images, and recorded the dataset
revision, row index, and image SHA256 in each record:

```json
{"id":"chartqa_val_1656","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"images/train/chartqa_val_1656.jpg"}},{"type":"text","text":"What is the answer in the chart?"}]},{"role":"assistant","content":"China"}],"source":{"dataset":"HuggingFaceM4/ChartQA","revision":"pinned revision","row_index":1656,"image_sha256":"..."}}
```

Without network permission, the agent can generate local test images with fixed dimensions and colors and corresponding question-answer pairs. This is sufficient to validate image
loading, the processor, vision tower, fusion, and backward path, but cannot establish real-data quality. The agent downloads a small public-data subset at a fixed revision only when
the user explicitly permits network access.

For both text and multimodal data, Online data qualifies as formal numerical-accuracy evidence only if all cases have the same global input hash;
otherwise it counts only as a smoke test. When strict token-by-token alignment with an external baseline is required, the agent saves already
tokenized batches as Offline replay.

### Limits of Cropped and Full-model Conclusions

With limited devices, large models usually require cropped validation first. Cropping narrows the model instance covered by the acceptance conclusion without lowering
the acceptance criteria: per-module forward/backward alignment, multistep numerical self-consistency, recomputation, parallelism, and checkpoint
validation are still required. Results for the cropped instance must not be extrapolated into a full-model pass.

Tool status is always `PASS`, `FAIL`, or `BLOCKED`. The agent automatically marks reports as
`cropped_due_to_resource` or `full_pretrained` according to whether the release config was modified; users need not include that field in their prompts.

| `validation_scope` and status | What it establishes | What it does not establish |
|---|---|---|
| `cropped_due_to_resource` + `PASS` | Mathematics, gradients, data, parallelism, and resume paths of covered modules and layer roles passed under the executed configuration | Full checkpoint coverage, layer roles absent from the cropped graph, full-model memory/performance, full-scale communication behavior, and training convergence |
| `full_pretrained` + `PASS` | The complete released structure and checkpoint passed the required gates under the executed matrix | Topologies, sequence lengths, hardware, or precision configurations that were not run |
| `BLOCKED` | Existing evidence is insufficient for a pass or fail conclusion | It cannot be rewritten as PASS by relaxing thresholds, shrinking the model, or skipping cases |

Even if the cropped model retains every role, it must still report `validation_scope=cropped_due_to_resource`. Module parity may use
the same module dimensions as the authoritative implementation; the end-to-end crop validates the real
Trainer lifecycle of those modules in combination on available devices. Once sufficient resources are available, reuse the same manifest and evidence gates with the full config/checkpoint
to complete `validation_scope=full_pretrained` validation.

## Tutorial Objectives

Model integration validation answers four distinct questions: whether the final constructed model structure is safe; whether replaced critical modules match the authoritative implementation; whether optimized paths meet both numerical-accuracy and performance targets against a fair baseline; and whether the same training state remains numerically self-consistent across parallel strategies. It neither replaces ordinary training configuration nor guesses model-family semantics at runtime.

The production architecture implements the core model semantics. Model-specific validation contracts and safe recomputation boundaries are declared through
`ModelAdapterSpec.validation` and `ModelAdapterSpec.recompute`, respectively. Generic tools only execute, collect evidence, compare, and
report. Adding a model requires no model-name, class-name, or FQN special cases in the Trainer, CLI, or diagnostic modules.

This guide distinguishes three easily confused terms:

| Term | Meaning |
|---|---|
| Production reference path | The executable, unoptimized fallback included in the production architecture; it can also serve as a performance A/B baseline |
| Correctness oracle | A released-repository implementation or formula independent of production code, used to establish correctness of outputs, gradients, and intermediate values |
| Optimized candidate | A replacement, CP/EP wrapper, or fused kernel explicitly selected by the recipe |

The reference path and optimized candidate may share a semantic base class to prevent state-machine drift, but shared code cannot serve as an independent oracle.
Their common mathematics must still be compared against the released implementation or independent formulas. The performance baseline and candidate must also use the same device, dtype,
inputs, and topology; speedup cannot be calculated by comparing a CPU correctness oracle with an NPU candidate.

## 1. Validation Boundaries

Validation has four evidence gates:

1. `DISCOVERED`: record the manifest, source environment, and candidate/reference structures.
2. `STRUCTURE_VALIDATED`: check the directly executable production architecture tree, the final tree after optional replacements, materialization, FSDP parameter ownership, checkpoint coverage, and probe coverage.
3. `MODULE_PARITY_PASSED`: compare forward outputs, intermediate values, input gradients, and parameter gradients between authoritative modules and the final production-path modules.
4. `MATRIX_PASSED`: compare inputs, loss, norm, parameter lifecycles, and checkpoint layouts across TP/CP/EP/FSDP, recomputation depths, and resume cases.

States advance only in order; `FAILED` and `BLOCKED` are terminal. Do not reuse terminal-state directories; fix the issue and use a new output directory.

High-performance capability discovery and A/B testing are design/experimental gates spanning these states, without adding state-machine enum values. Core model numerical-accuracy
status and optimization performance status are reported separately. Only `PRECISION_PASS + PERFORMANCE_PASS` qualifies a target
as an accepted high-performance implementation.

Training exposes only three mode values:

| `model_integration.mode` | Behavior | Use case |
|---|---|---|
| `off` | Use a no-op session without collecting evidence | Normal training; default |
| `build` | Check structure, layout, data, and checkpoint contracts after construction | Low-cost integration checks |
| `runtime` | Include `build`, then collect inputs, gradients, parameters, optimizer, shared state, performance, and checkpoint layouts step by step | Formal numerical-accuracy matrix |

`debug.check_fsdp_runtime` independently controls FSDP execution-order diagnostics and is not implicitly enabled by `model_integration.mode`.

## 2. Preparation

Install from the current checkout and verify the actual import path:

```bash
pip install -e .
python -c "import hyper_parallel; print(hyper_parallel.__file__)"
```

Users can provide local paths for models, weights, tokenizers, processors, and reference repositories, or provide a Hugging Face
source and authorize the agent to download to a specified directory. Data can be supplied by the user or created by the agent
using the defaults at the beginning of this guide. Validation does not download resources without permission or configure device, communication-library, or Python
environment variables on the user's behalf.

First prepare a recipe that starts successfully with the standard Trainer. For formal numerical validation, the following settings are recommended:

```yaml
model_init_dtype: float32

fsdp_config:
  mix_precision:
    param_dtype: bfloat16
    reduce_dtype: float32
    cast_forward_inputs: false

optimizer:
  fp32_main_params: true

model_integration:
  mode: off  # The validate command overrides this to runtime for matrix subprocesses.
```

Unless explicitly required by the model contract, leave `fsdp_config.mix_precision.output_dtype` unset to preserve the model's FP32 scalar loss.

### 2.1 High-performance Capability Discovery and Reuse Design

Before creating an adapter or writing optimized code, the agent must extract attention, normalization, positional encoding,
shared/recurrent state, router, expert activation, multimodal paths, and target dtype/backend from the authoritative implementation, then scan:

```bash
rg --files hyper_parallel/components/modules
rg --files hyper_parallel/distributed/context_parallel
rg --files hyper_parallel/distributed/expert_parallel
rg --files hyper_parallel/models | \
  rg 'adapter/(conversion|distributed|policies)|registration.py'
rg -n 'replacements=|context_parallel=|expert_parallel=' \
  hyper_parallel/models/*/adapter/registration.py
```

Search results must be saved to `analysis/optimized_capability_inventory.json`, not merely stated in the agent conversation.
For each final FQN, CP path, and EP path to optimize, record at least:

- Authoritative source code and mathematical/state contracts;
- Reusable generic components, collectives, and routing/packing/dispatch primitives;
- Existing model adapters that can inform the design;
- Forward/backward, state-dict, materialization, checkpoint, and TP/CP/EP/FSDP compatibility;
- Target hardware, kernel/backend dependencies, and fallbacks;
- A `direct_reuse`, `thin_wrapper`, `extract_generic`, `new_generic`, `model_owned`, or
  `unsupported` decision, with specific reasons for rejected candidates;
- Whether the expected improvement concerns computation, communication, overlap, memory, or scalability;
- Corresponding numerical-accuracy and performance experiment IDs.

Reuse priorities are: reuse generic components directly; compose generic primitives with a thin model wrapper; extract common behavior from an existing model's private implementation
into a generic component; add a model-independent component; and only then retain irreducible model-private logic. Existing model
adapters are references only and must not become direct dependencies of another model family. Similar names or tensor shapes do not establish reusability:
masks, parameter packing, state lifecycles, activations, and checkpoint identity must each match.

Before implementation, also write `analysis/optimization_plan.yaml`, declaring the correctness reference, performance
baseline, candidate implementation, Trainer A/B on the same topology, module-microbenchmark shapes, warm-up/measurement policy, numerical tolerances,
performance thresholds, fallback/communication-overlap evidence, and conclusion scope. Performance thresholds come from user or project goals; do not assume a universal
speedup ratio. If no target exists, first run a baseline labeled pilot to estimate variability across repeated runs, then freeze
the formal plan. Candidate improvements must exceed that range without unexplained p90 or memory regressions; thresholds must not be chosen after seeing
candidate results.

### 2.2 Boundaries Between the Core Model and High-performance Implementations

The registered production architecture must directly construct an executable core model and pass a small forward/backward run without applying recipe
replacements. Mathematical semantics such as Attention, MoE/Router, Engram/Memory, residual connections, and cross-layer
state must not exist only in high-performance replacement functions. Production and optimized modules may share a semantic base class without device-specific optimizations,
but construction must not default to treating an optimized replacement as the core model. A device fused kernel may be a disableable
branch inside a module or a recipe replacement; either way, disabling optimizations must preserve the model's
basic semantics.

Acceptance requires two stages: first, apply no recipe replacements and validate the production architecture's
direct forward/backward, parameter initialization, and checkpoint FQNs. Second, apply optimized replacements and validate before/after
state-dict identity, numerical accuracy, TP/CP/EP/FSDP capabilities, and same-topology performance. Passing the second stage cannot replace the first,
or a Placeholder or mandatory wrapper could still be mistaken for the production model.

Temporary parameter Placeholders are allowed only when a planner or loader must first inspect the official parameter tree and direct construction cannot preserve FQNs, parameter aliases, initialization,
meta materialization, or checkpoint contracts. In that case, document the
lifecycle constraint, prove before/after state-dict/FQN identity, and complete
finalization before the final production builder returns. The structure gate must fail on leftover Placeholders. Using Placeholders to assemble a model
merely for development convenience is not an acceptable reason.

Applying a generic wrapper directly must also reproduce the source class's type-specific initialization. For example, once a wrapper means a module is no longer an upstream
Attention/HyperConnection type, upstream `post_init()` may no longer zero attention sinks or the residual
mixing base, or set scale to 1. In addition to type assertions, check these special parameter initial values and run
end-to-end backward to verify finite loss and gradients.

## 3. Declare Model-side Validation Contracts

### 3.1 Registration Entry

Each model family performs three independent registrations in `hyper_parallel/models/<family>/adapter/registration.py`. Registration modules
are discovered lazily by directory convention, with no central-registry changes:

| Registration mechanism | Registered relationship | Outside its responsibility |
|---|---|---|
| `register_custom_model()` | `config.architectures[0]` to a production Python class | Does not construct models, apply recipes, or choose parallel strategies |
| Checkpoint-mapping registration | Released checkpoint keys to production parameter keys | Does not create parameters, determine placements, or download weights |
| `register_model_adapter()` | Model type/architecture to a model-capability provider | Only makes the provider discoverable; does not imply it has been called or an optimization enabled |

A complete entry has the following structure. The checkpoint-registration API depends on the upstream weight system and is therefore wrapped in a family-specific function:

```python
from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.registry import register_custom_model, register_model_adapter

from .conversion.checkpoint_mapping import register_example_checkpoint_mapping


def _load_validation():
    from .validation.model_validation_spec import get_validation_spec

    return get_validation_spec()


MODEL_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="ExampleForCausalLM",
    model_type="example",
    validation=_load_validation,
)
register_custom_model(
    "ExampleForCausalLM",
    "hyper_parallel.models.example.modeling_example",
    "ExampleForCausalLM",
)
register_example_checkpoint_mapping()
register_model_adapter(MODEL_ADAPTER_SPEC)
```

Importing this registration module only registers entries; it does not immediately resolve model classes, call providers, install replacements, or start
parallelism. `validation` must load lazily so ordinary training does not import authoritative reference repositories, test operators, or validation-only dependencies.
In the current recipe-driven path, `replacements/context_parallel/expert_parallel` form a capability catalog; actual activation is still
explicitly declared through `plan_overrides.replace.implementation`: module replacements use `kind: module` with module paths,
while CP/EP forward replacements use `kind: method` with `.forward` paths. Sharding,
FSDP, recompute, and validation providers are consumed directly by their corresponding planner/Trainer/tools.

### 3.2 Minimal Validation Declaration

Every `ModelValidationSpec` field is optional; declare fields according to the model's risks:

```python
from hyper_parallel.models.validation_spec import (
    DataValidationSpec,
    ModelValidationSpec,
    ParameterProbeSpec,
    StateInvariantSpec,
)


def _canonical_tree(model, _context):
    invalid = [
        name for name, module in model.named_modules()
        if type(module).__name__.endswith("Placeholder")
    ]
    return None if not invalid else {"non_executable_modules": invalid}


def get_validation_spec():
    return ModelValidationSpec(
        parameter_probes=(
            ParameterProbeSpec("model.layers.*.experts.*.weight"),
        ),
        state_invariants=(
            StateInvariantSpec(
                name="example.canonical_executable_tree",
                checker=_canonical_tree,
                phase="structure",
            ),
        ),
        data=DataValidationSpec(
            required_forward_fields=("input_ids", "labels"),
            labels_are_shifted=True,
        ),
    )
```

Common declarations include:

| Declaration | Purpose |
|---|---|
| `ModuleParityCase` | Define comparisons between authoritative modules and final production modules |
| `ObservationSpec` | Add intermediate-value comparisons for routing results, indices, shared state, and similar observations |
| `ParameterProbeSpec` | Select parameters and lifecycle phases by final FQN |
| `StateInvariantSpec` | Check model semantics during structure/materialization/checkpoint/runtime phases |
| `SharedStateValidationSpec` | Verify that cross-layer state is published once and consumers read the correct source |
| `DataValidationSpec` | Declare forward fields, runtime fields, modality fields, and label shift |
| `CheckpointValidationSpec` | Declare legitimate missing training-only targets and inference-only sources |
| `TopologyConstraint` | Reject parallel combinations unsupported by the model before launch |

### 3.3 Module Numerical Alignment

Candidates for in-process parity must come from the production construction/materialization path. If an optional
replacement is under test, also pass the candidate through that real replacement path. The reference must be an independent implementation. The following example compares
a final linear module:

```python
def build_candidate(context):
    return build_final_production_model(context.manifest.model)


def build_reference(_context):
    return AuthoritativeLinear(16, 32)


def build_inputs(_context):
    return (torch.randn(2, 16),)


def copy_weights(reference, candidate, _context):
    with torch.no_grad():
        reference.weight.copy_(candidate.weight)
    return {"weight": "weight"}


case = ModuleParityCase(
    name="linear",
    candidate_selector="model.layers.0.proj",
    candidate_builder=build_candidate,
    reference_builder=build_reference,
    input_builder=build_inputs,
    weight_adapter=copy_weights,
)
```

When the reference repository conflicts with production module names or can only be imported in a separate interpreter, use `execution="isolated_process"` and `isolated_runner`. DeepSeek-V4.1 uses this approach for Engram, MoE, and CSA2.

### 3.4 Safe Recomputation Regions

Producers/consumers sharing cross-layer state may participate in recomputation, but two kinds of actions must first be handled on that path: **side effects that must execute only once** (writing cross-layer state) and **discrete, nonreproducible selections** (Top-K/candidate pools). A single model-adapter declaration specifies both which modules become regions and which descendants must remain outside recomputation. `on_applied` installs runtime behavior that depends on the final region structure, such as pinning selections for layers that will be recomputed:

```python
from hyper_parallel.models.adapter_spec import RecomputePolicy


def build_recompute_policy():
    """Use one region per layer; derive attention recomputation scope from the same role table."""
    recomputed = ("full", "reindex", "reuse")
    return RecomputePolicy(
        region_patterns=("model.layers.*",),
        exclude_predicate=...,  # Keep attention roles outside recomputed out of the region.
        on_applied=...,  # Pin selections for index sources that will be recomputed.
    )
```

DeepSeek-V4.1 expresses this as a tier switch in `models/deepseek_v41/adapter/policies/activation_checkpointing.py`: `ATTENTION_RECOMPUTE_TIER` is `reuse` (default), `reuse_reindex`, `all`, or `none`. The same role table derives `exclude_predicate` for roles outside the recomputation scope, and the same tuple passes recomputed index sources to `configure_shared_state` to pin their selections. With default `reuse`, layers that only read cross-layer state are recomputed with their layer, while producers such as Full/Reindex remain outside the region and retain activations. `configure_shared_state` pins the indexer's discrete selection as an excluded region, reusing the forward-selected set during recomputation and avoiding violations of checkpoint tensor-count invariants caused by skipping differentiable code. Cross-layer state itself needs no scheduling: checkpointing holds it as an input to the recomputation region, and published entries live with it until backward ends. `publish_*` executes only during forward, as determined by `is_recomputing()`, so recomputation does not write it again. Pinning covers only discrete selection; compressor and indexer projections outside that selection are recomputed with the layer or kept outside the region according to the tier.

`region_patterns` uses the **outermost match** as each region root. A whole-layer pattern saves one copy of the layer input per region; a leaf-module pattern such as `model.layers.*.mlp` creates a separate region for each module, usually with empty `exclude_patterns`. `exclude_patterns` must match descendants of a region root or raise an error; the same module cannot be both a region root and excluded. `on_applied` is a model-owned hook invoked after regions are installed to enable runtime behavior depending on their final structure, such as DeepSeek-V4.1 pinning selections for recomputed layers.

The Trainer recipe selects this policy through generic configuration:

```yaml
activation_checkpoint:
  mode: selective
  selection:
    source: model_adapter_safe_regions
    layer_count: 16
```

`layer_count` selects the specified number of consecutive layers starting at index `0`, in the flattened order of all discovered layer containers. For example, `layer_count: 2` selects layers `0, 1`, and `0` selects none. For nonconsecutive layers, use the mutually exclusive `layer_indices`:

```yaml
activation_checkpoint:
  mode: selective
  selection:
    source: model_adapter_safe_regions
    layer_indices: [0, 4, 8, 12]
```

`layer_indices` is zero-based and must be nonempty, nonnegative, and duplicate-free. `source: default` accepts neither field.

## 4. Write a Validation Manifest

The validation manifest describes only model identity, standard Trainer launch configuration, the experiment matrix, and acceptance thresholds. Model structure, datasets, dtype, tokenizer, and local assets belong in the Trainer recipe. Environment variables, source revisions, and hashes are automatically recorded as evidence and should not be duplicated in YAML.

```yaml
schema_version: 2

model:
  adapter: example
  # inspect/check require the final production-model builder; kwargs are passed unchanged.
  builder:
    _target_: your_package.build_final_model
    config_path: /local/model/config.json

reference:
  # Local authoritative repository for parity; no implicit downloads.
  source_path: /local/reference/repository

launcher:
  module: examples.training_demo.train_text
  config: /local/recipes/train_example.yaml
  timeout_seconds: 7200

matrix:
  devices: 16
  steps: 10
  resume_split_step: 5
  performance_warmup_steps: 2
  # First run one baseline initialization step and save a step-1 warm-start DCP. Then restore the same
  # model, optimizer, scheduler, and RNG for the formal baseline and all candidates, requesting replay from the same configured data start.
  shared_initial_checkpoint: true
  baseline:
    tp: 1
    cp: 1
    ep: 1
    fsdp: 16
    sequence_parallel: false
    recompute: {layer_count: 0}
  axes:
    ep: [1, 16]
  combined:
    - {tp: 2, cp: 1, ep: 16, fsdp: 8, sequence_parallel: true}
    - {tp: 1, cp: 2, ep: 16, fsdp: 16, sequence_parallel: false}
  recompute_selections:
    - {layer_count: 0}
    - {layer_count: 2}
    - {layer_indices: [0, 1, 2, 3]}
  recompute_topology: {tp: 1, cp: 1, ep: 16, fsdp: 16}
  same_topology_resume: true
  cross_topology_resume: {tp: 2, cp: 1, ep: 16, fsdp: 8}

acceptance:
  same_topology:
    loss_max_abs: 5.0e-4
    norm_max_abs: 2.0e-3
  cross_topology:
    loss_max_abs: 5.0e-3
    loss_max_rel: 5.0e-3
    norm_max_abs: 1.25e-1
    norm_max_rel: 5.0e-3
    # Apply absolute-or-relative acceptance separately to loss and norm; both must still pass.
    combination: any
  parameters:
    # Strict base tier for module parity and the same topology; cross-topology BF16 tolerances must not relax either.
    max_abs: 1.0e-6
    relative_l2: 1.0e-6
    combination: all
    summary:
      l2_norm_relative: 1.0e-6
    cross_topology:
      max_abs: 1.0e-3
      relative_l2: 2.0e-2
      # Applies only to tensors whose complete values were recorded.
      combination: any
      summary:
        # With only aggregate summaries for large tensors, only relative drift of the L2 norm itself can be evaluated.
        l2_norm_relative: 2.0e-2
```

### 4.1 Matrix Fields

| Field | Description |
|---|---|
| `devices` | `--nproc_per_node` for each `torchrun` launch |
| `steps` | Training steps for ordinary cases and the final step after resume |
| `resume_split_step` | Step at which resume preparation saves state; must satisfy `0 < split < steps` |
| `shared_initial_checkpoint` | Generate a shared step-1 warm-start DCP; baseline, candidates, and resume preparation restore the same model/optimizer/scheduler/RNG, but request replay from the configured data start instead of restoring the warm-start dataloader cursor. The global-input-hash gate still determines whether logical inputs match |
| `baseline` | Reference topology for all cases |
| `axes` | Override one field at a time; values duplicating the baseline are deduplicated |
| `combined` | Hybrid strategies explicitly overriding multiple baseline fields |
| `recompute_selections` | Exact layer counts or indices for safe-region recomputation; each entry contains only `layer_count` or `layer_indices` |
| `recompute_topology` | Fixed topology override for the recomputation-selection group |
| `production_validate_pair` | Set `model.validate_placement=false/true` separately on the same topology |
| `same_topology_resume` | Save and restore on the baseline topology |
| `cross_topology_resume` | Save on the baseline topology and restore on the target topology |

The standard launcher accepts only `tp`, `cp`, `ep`, `fsdp`, `sequence_parallel`, and `recompute`. Fields that cannot map to Trainer commands raise errors instead of producing duplicate-command cases that appear to pass without testing anything new.

Before writing the matrix, create a coverage ledger listing the parallel axes the model actually supports, critical hybrid topologies, recomputation layer selections, same-/cross-topology resume, Production/Validate, and language/vision/projector/fusion subtrees for multimodal models. Each item must be either a formal case or an exclusion supported by evidence. After `--generate-only`, reconcile the ledger item by item with `cases/resolved_cases.json` and count baseline, strategy generalization, recomputation, resume, and Production/Validate separately. Recomputation and resume do not count as strategy-generalization cases. If only one strategy candidate ran, report only one strategy-generalization comparison even if two recomputation and two resume cases also ran.

Multimodal models must also record actual placements for each subtree. If ViT completes forward/backward in a process group with TP but its parameters are `Replicate()` along TP, this establishes only that the multimodal path composes with language TP; it does not establish ViT TP coverage. Audit operator support using the exact callable key intercepted by the dispatcher: `npu_rms_norm` does not substitute for `rms_norm`, and `conv3d` does not substitute for `conv1d`/`conv2d`.

The acceptance tier is selected automatically according to whether the launched TP/CP/EP/FSDP/sequence-parallel configuration matches the baseline. Recomputation layer selection alone does not change parallel topology. If `recompute_topology` changes EP or another field, that group uses `cross_topology` thresholds. `cross_topology_resume` must actually change at least one parallel field; there is currently no case that resumes after changing only the recomputation strategy.

Scalars support absolute and relative thresholds. The default is `combination: all`. With explicit `any`, the comparator accepts either absolute or relative error separately for loss and norm, while still requiring loss, pre-clip norm, post-clip norm, input identity, LR, and finiteness all to pass. A passing loss-relative threshold cannot compensate for a norm failure, and `any` cannot bypass the data-identity gate.

Parameter probes must distinguish topology tiers and evidence strength. Root `parameters` fields define the strict base tier for module parity and same-topology comparisons. `parameters.module_parity`, `parameters.same_topology`, and `parameters.cross_topology` may override it separately; undeclared fields inherit the base tier. Thus BF16 cross-topology scale tolerances do not accidentally relax module parity or same-topology resume. Only small tensors with complete `values` recorded support true elementwise `max_abs` and tensor `relative_l2`; their default combination is `all`. BF16 cross-topology comparisons may use `parameters.cross_topology.combination: any` when declared in advance with numerical justification. Large tensors recording only `sum/l2/min/max` lack elementwise evidence; the comparator reports `max_abs` and `relative_l2` as `n/a` and evaluates L2-norm summary drift using the current tier's `summary.l2_norm_relative` and optional `l2_norm_max_abs`. Differences between aggregate statistics must not be labeled as elementwise maximum errors. Both modes always check finiteness and global shape separately.

### 4.2 Reuse a Parity Handoff

Run all gates sequentially in one output directory, or reference a previously generated handoff in the numerical-validation manifest:

```yaml
integration_handoff: /local/evidence/integration_handoff.yaml
```

The handoff must use schema-v2, have `status: PASS`, and have a `family` matching `model.adapter`. It proves only that structure and module parity passed; it does not replace the current matrix's Trainer recipe and input evidence.

### 4.3 Paired Experiments for High-performance Replacements

Every replacement, CP, or EP path described as high performance requires four layers of evidence:

| Experiment | Fixed conditions | Question to answer |
|---|---|---|
| Module numerical parity | Same weights, inputs, state, and dtype | Do outputs, intermediate values, input gradients, and parameter gradients match an independent authoritative implementation? |
| Target-device module microbenchmark | Same shape, dtype, autograd, compilation, and synchronization boundaries | Do forward/backward p50/p90, throughput, peak memory, and kernel fallback meet predeclared targets? |
| Same-topology Trainer A/B | Same complete initial/warm-up-step DCP, data hash, topology, precision, optimizer, recomputation, and step count | Does the optimized path preserve numerical accuracy and improve end-to-end performance in the real training lifecycle? |
| CP/EP scalability | Same global workload and model semantics | How do communication, load balancing, memory, and scaling efficiency change as CP/EP degree increases? |

The correctness reference and performance baseline need not be the same. An authoritative eager formula can establish numerical accuracy, but a CPU
reference cannot be compared with an NPU candidate to compute speedup. Performance baseline and candidate must use the same device, topology, shape,
dtype, compilation, and synchronization boundaries. An ordinary validation matrix switches only parallel topology and
recomputation, not arbitrary replacement providers. End-to-end A/B therefore uses two recipes or `plan_overrides` differing only in the target implementation
selection, writes to separate evidence directories, and shares the same
complete DCP with its actual step number recorded. The standard matrix launcher currently generates a step-1 warm-start DCP; do not label it step 0.

Another limitation must be preflighted after `--generate-only`: DCP's ability to reshard model/optimizer state does not imply that a stateful
dataloader cursor can be resharded across DP world sizes. The standard launcher distinguishes two scenarios. Shared warm-start restores
model/optimizer/scheduler/RNG but explicitly does not restore the cursor after the initialization step consumed data, allowing measured cases to request replay from the configured data
start. Actual K→K+N resume still restores the cursor after step K. Restarting from the beginning does not automatically mean identical inputs: if
Online-loader rank sharding, randomization, or packing depends on DP world size, changing TP/CP and thus DP degree may still change the global input hash.
Such a case establishes only topology executability and cannot support numerical comparisons. Formal cross-topology numerical validation should use fixed, DP-invariant
Offline replay. The current Online dataloader also fails explicitly when actual resume changes DP world size. A loader capable of reconstructing the cursor from
global sample positions is required; otherwise mark the capability `BLOCKED` or exclude it with evidence. Do not drop the cursor and still claim
input continuity, or classify mismatched inputs or pre-training failures as TP/CP model numerical failures.

Implementation A/B for CP must compare synchronous/reference and optimized communication at the same CP/TP degree, global sequence, mask, and
attention kernel. CP1 versus CPn is only a scalability experiment. EP similarly requires comparing dispatch/compute/combine with the same EP degree, learned router,
expert layout, capacity, and grouped-kernel settings. EP1 versus EPn
cannot replace implementation A/B.

Write results to `analysis/optimization_results.json`, recording per-target numerical errors, p50/p90, throughput, peak
memory, collective bytes/time, asynchronous launch/wait and overlap, actual kernel/backend, fallbacks,
sample count/dispersion, and scope. Report separately:

- `PRECISION_PASS` or `PRECISION_FAIL`;
- `PERFORMANCE_PASS`, `PERFORMANCE_FAIL`, or `PERFORMANCE_INCONCLUSIVE`.

Only `PRECISION_PASS + PERFORMANCE_PASS` qualifies a target as an accepted high-performance implementation. Cropped-model
end-to-end results apply only to the crop. Even full-shape module microbenchmarks cannot be extrapolated to full-model throughput or communication
scalability.

The generic `report` command currently neither executes nor reads arbitrary replacement A/B experiments. Keep its generated `summary.md` and
`summary.zh-CN.md`, then have the agent use the inventory, plan, and results to generate an adjacent English
`optimization-summary.md` and Chinese `optimization-summary.zh-CN.md`. Do not manually rewrite the matrix's first-line
status to imply performance acceptance.

## 5. Execution Workflow

Use the same absolute `--output-dir` for the following commands. Relative paths resolve against the manifest directory, not the current shell
directory, so absolute paths are recommended for all formal runs:

```bash
RUN_DIR=output/model_integration/example
MANIFEST=/local/manifests/example_validation.yaml

python -m hyper_parallel.tools.model_integration inspect \
  --manifest "$MANIFEST" --output-dir "$RUN_DIR"

python -m hyper_parallel.tools.model_integration check \
  --manifest "$MANIFEST" --output-dir "$RUN_DIR"

python -m hyper_parallel.tools.model_integration parity \
  --manifest "$MANIFEST" --output-dir "$RUN_DIR" \
  --device cpu --dtype float32

# Follow analysis/optimization_plan.yaml for production-dtype module parity, target-device
# microbenchmarks, and reference/optimized same-topology Trainer A/B in separate directories.
# These experiments depend on the model's builder, recipe, and plan_overrides; there is no generic
# implicit replacement switch. Write results to analysis/optimization_results.json.

python -m hyper_parallel.tools.model_integration validate \
  --manifest "$MANIFEST" --output-dir "$RUN_DIR" --generate-only

python -m hyper_parallel.tools.model_integration validate \
  --manifest "$MANIFEST" --output-dir "$RUN_DIR"

python -m hyper_parallel.tools.model_integration report \
  --output-dir "$RUN_DIR"
```

Check category counts and the complete topology list in `cases/resolved_cases.json` against the coverage ledger before using multiple devices. Remove `--generate-only` for actual runs.

Command semantics:

| Command | Input | Main artifacts |
|---|---|---|
| `inspect` | Local path or explicit builder | Reference/candidate inventories, structure differences, resource fingerprints |
| `scaffold` | Model identity | Minimal adapter registration, validation provider, manifest, and registration-test skeleton |
| `check` | Final production-model builder | Findings, parameter ownership, FSDP units, checkpoint coverage |
| `parity` | Adapter validation provider and reference | Module comparisons and `integration_handoff.yaml` |
| `validate` | Trainer recipe and matrix | Per-case training evidence, cross-case comparisons, performance summaries |
| `report` | Existing evidence directory | English `summary.md` and Chinese `summary.zh-CN.md`, each with PASS/FAIL/BLOCKED consistently on the first line |

Exit codes are `0=PASS`, `1=FAIL or execution error`, and `2=BLOCKED or manifest error`.

## 6. Enable Checks Only Within the Trainer

Without running a matrix, enable checks directly in an ordinary Trainer recipe:

```yaml
model_integration:
  mode: build
```

Or:

```yaml
model_integration:
  mode: runtime
```

If `HYPER_PARALLEL_MODEL_INTEGRATION_OUTPUT_DIR` is unset, evidence defaults to `<checkpoint_dir>/model_integration`. The matrix launcher automatically assigns a separate directory to each case, so ordinary users need no output-path field in YAML.

`runtime` checks gradients and layouts before the optimizer update, avoiding parameter corruption when it detects an error. It adds diagnostic overhead and should be used only for validation runs.

## 7. Evidence Directory

A typical directory is:

```text
<run>/
├── manifest.resolved.yaml
├── environment.json
├── integration_state.json
├── source_fingerprints.json
├── inventory/
│   ├── reference.json
│   ├── candidate.json
│   └── structure_diff.json
├── check/
│   ├── findings.json
│   ├── parameter_ownership.csv
│   ├── fsdp_units.json
│   └── optimizer_layout_initial.json
├── checkpoint/
│   └── coverage.json
├── module_parity/
│   ├── comparison.json
│   └── <case>/comparison.json
├── integration_handoff.yaml
├── analysis/
│   ├── operator_support.json
│   ├── parallel_ownership.json
│   ├── issues.json
│   ├── optimized_capability_inventory.json
│   ├── optimization_plan.yaml
│   └── optimization_results.json
├── cases/
│   ├── resolved_cases.json
│   └── <case>/
│       ├── preflight.json
│       ├── stdout.log
│       ├── stderr.log
│       └── cases/runtime/
│           ├── metrics.jsonl
│           └── performance.jsonl
├── comparison.json
├── summary.md
├── summary.zh-CN.md
├── optimization-summary.md
└── optimization-summary.zh-CN.md
```

Parameter probes and checkpoint layouts are written per rank. The comparator aggregates by logical global shape, mesh/placement, and logical shard identity instead of gathering complete large parameters onto one device.

Performance data in an ordinary parallel matrix is descriptive. Failures in loss, norm, input identity, parameter probes, layouts, or
checkpoint continuation still determine core numerical-accuracy status; throughput does not alter it. For replacements, CP, or EP explicitly claimed as high performance,
use separately predeclared same-topology paired experiments as the performance gate. Their results determine only that
optimization target's `PERFORMANCE_PASS/FAIL/INCONCLUSIVE` and cannot conceal numerical failures.

## 8. DeepSeek-V4.1 Example

Start with the [DeepSeek-V4.1 validation manifest](../../../hyper_parallel/models/deepseek_v41/tools/deepseek_v41_validation.yaml). Model structure, Online data, 4K sequence length, and dtype are still defined by Trainer recipes in `examples/training_demo/deepseek_v41/`, referenced through relative paths in the manifest. Users provide the shell environment and local assets.

The DeepSeek adapter's validation declarations include:

- Parameter-lifecycle probes for Engram `q_weight`, `k_weight`, and `wkv.weight`;
- Probes for CSA2 sinks and MoE gate bias;
- Direct construction of executable Attention, Engram, mHC, and Router modules in the production model, without mandatory replacements;
- Correct special initialization and finite gradients for attention sinks, mHC base/scale, and similar parameters covered by direct wrappers;
- Official initialization of Engram q/k in scratch crops;
- Cross-layer publish/consume checks for `compressed_kv`, `index_key`, `topk_indices`, and `candidate_blocks`;
- Checks for packed runtime fields, vision input fields, and vision/aligner gradients;
- Native Engram/MoE/CSA2 forward/backward comparisons in isolated processes;
- Whole-layer recomputation as one region, including CSA2 attention: state writes occur only during forward, as determined by `is_recomputing()`; discrete selections are pinned as excluded regions, and autograd holds cross-layer entries until backward ends.

DeepSeek-V4.1 core model modules and optimized paths correspond one to one, but do not replace each other's responsibilities:

| Model semantics | Production reference module | Optional recipe candidate | Selection |
|---|---|---|---|
| pipelined mHC | `DeepseekV41PipelinedHyperConnection` | `PipelinedMhcModule` | `replace.kind: module` + `implementation` |
| Engram | `DeepseekV41Engram` | `EngramModule` | `replace.kind: module` + `implementation` |
| CSA2 Attention | `DeepseekV41Attention` | `SharedCompressedDSAAttention` | `replace.kind: module` + `implementation` |

Both Attention implementations share `SharedCompressedDSAAttentionBase`. The production module fixes
`use_optimized_sparse_attention=False`, while the candidate fixes it to `True`. Providers only register factories; the current demo's recipe
explicitly selects three replacements. Removing those entries still allows forward/backward and falls back to the unoptimized reference path.

The example uses EP16 for `recompute_topology` and EP1 for the baseline, so the recomputation group deliberately uses cross-topology thresholds. To isolate the effect of recomputation, make the parallel fields in `recompute_topology` match the baseline.

## 9. Common Failures and Remedies

| Symptom | Meaning | Remedy |
|---|---|---|
| Replacement errors | Optional replacement did not match, or the final tree still contains a nonexecutable Placeholder/violated forward-state contract | First move core model semantics into the production architecture; only for genuine lifecycle constraints, fix finalization, adapter patterns, or explicit mappings |
| FSDP ownership errors | An FSDP unit mixes dense/expert domains, experts fall into the root, or alias ownership conflicts | Declare homogeneous child units and correct boundaries in the adapter |
| Materialization errors | Meta tensors, derived buffers, or model initialization remain incomplete | Fix materialization hooks/model initialization; do not mask derived-state issues with checkpoints |
| Checkpoint coverage errors | Checkpoint keys are uncovered or transformations are not reversible | Add mappings/transforms/reverse transforms or explicitly legitimate exclusions |
| Forward data-contract errors | Conflicting forward-field ownership, missing fields, or incorrect label shift/modality gradients | Fix Omni dataset transforms, get-batch, or runtime input-adapter declarations |
| Unmatched parameter probes | A required probe matches no final parameters | Update adapter declarations using final FQNs after replacement |
| Shared-state contract errors | Cross-layer state lacks a producer/consumer, producers publish twice, or traces exceed limits | Restrict state writes to forward using `is_recomputing()`, pin discrete selections as excluded regions, or narrow recomputation boundaries |
| Online dataloader rejects a cursor across DP world sizes, or input hashes still differ after skipping the cursor | DCP model/optimizer resharding does not imply that the loader can reconstruct global sample positions; starting at the same configured point does not guarantee DP-invariant packing | Warm-start may request replay with `restore_dataloader_state=false`, but must check the global input hash. Use fixed Offline replay for formal cross-topology numerical validation and a loader supporting global-cursor reconstruction for actual continued training; otherwise mark `BLOCKED` |
| `BLOCKED` | Evidence, devices, assets, or an independent oracle are unavailable | Satisfy prerequisites; do not relax thresholds or silently shrink the experiment |

If two cases have different `global_input_sha256` values, fix data replay before evaluating loss/norm. If aggregate loss/norm passes but parameter probes fail, the parameter probes take precedence: they detect local errors in small parameters, expert-local parameters, and optimizer states.

## 10. Completion Criteria for a New Model Integration

An integration is complete only when all of the following hold:

- No model-name, model-class, model-FQN, or model-config-field special cases were added to the central framework;
- The high-performance capability inventory covers all required optimized modules, CP paths, and EP paths, recording search scope, reuse decisions, rejected candidates, and fallbacks;
- Every high-performance claim has a predeclared optimization plan, independent numerical oracle, target-device module microbenchmark, and same-topology end-to-end A/B;
- Optimization targets are accepted only with `PRECISION_PASS + PERFORMANCE_PASS`; missing fair baselines or stable evidence are explicitly marked `PERFORMANCE_INCONCLUSIVE`;
- Final production modules participate in structure checks and parity, rather than test substitutes;
- The production architecture directly completes forward/backward with finite results without recipe replacements, and high-performance implementations do not carry the core model semantics;
- Every checkpoint source/target key has an explicit classification;
- Dense/expert parameter ownership and DTensor global layouts are demonstrable;
- The coverage ledger fully reconciles with `resolved_cases.json`, and report counts use mutually exclusive categories;
- Actual TP/CP/EP/FSDP placements for each multimodal subtree are recorded separately; launch topology is not used as evidence of actual sharding;
- Required DTensor operators have their implementations, signatures, and registrations verified by exact callable keys;
- Input identity, loss, pre-/post-clip norm, and critical parameter/gradient/optimizer states are all compared;
- The adapter explicitly declares shared cross-layer state and recomputation boundaries;
- Declared vision parameters receive finite gradients when multimodal inputs reach the model;
- Same-topology, cross-topology, EP1-control, recomputation, and resume coverage matches the model's actual use cases;
- English and Chinese final reports begin with `PASS`, and the written conclusion explicitly states whether validation covered a cropped or full model.
