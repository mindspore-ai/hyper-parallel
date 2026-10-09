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
   experiment choices in declarative recipes or manifests. Classify every
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

   A passing run writes `integration_handoff.yaml`. Resolve and parse every
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
generic contract test, and consumed through the model adapter.
