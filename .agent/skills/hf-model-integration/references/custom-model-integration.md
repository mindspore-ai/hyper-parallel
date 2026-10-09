# Custom Model Construction and Ownership

## Choose the Construction Path

Classify the model before writing an adapter:

- **HF-native:** the installed HF class matches the released training architecture
  and checkpoint. Let `AutoModel` construct it and add only necessary adapters.
- **HF-component-backed custom:** HF provides config, tokenizer, or common blocks,
  but misses released layers, shared state, multimodal routing, or training
  semantics. Register a lazy custom model class and reuse correct HF components.
- **Fully native/non-HF:** no suitable HF config or model implementation exists.
  Register the native class and an explicit config/checkpoint boundary.

Compare source code, config, parameter names, and forward behavior. A matching
model ID, `model_type`, or loadable `AutoConfig` is insufficient evidence.

Use `register_custom_model()` for a custom architecture and
`register_model_adapter()` for its adapter providers. Preserve the common
AutoModel/Trainer build pipeline; a custom model is not permission to create a
second launcher-only construction path.

## Model, Adapter, Framework, and Recipe Boundaries

- **Model:** layer graph, algorithmic crop rules, modality fusion, state lifetime,
  training forward, and architecture invariants.
- **Adapter:** checkpoint mapping, high-performance replacement factories,
  TP/CP/EP and FSDP hints, runtime-input translation, and source-version guards.
- **Generic components:** reusable kernels and communication implementations with
  no model-family naming assumptions.
- **Framework APIs:** model-independent extension contracts used by integrations.
  If the requested semantics cannot be expressed, add or design a generic,
  optional contract and implement its model-specific provider in the adapter. A
  framework branch on the motivating model is not an extension mechanism.
- **Trainer recipe:** sequence length, dtype policy, data/model asset paths,
  checkpoint mode, activation policy, and model construction arguments.
- **Validation manifest:** adapter selection, Trainer recipe entry, topology and
  recomputation/resume matrix, step count, and acceptance thresholds.
- **Experiment harness:** evidence collection and generated case enumeration only.
  Do not hide model semantics or permanent configuration in a one-off Python file.

The ownership boundary is judged by dependency direction, not merely by file
location. Generic framework code may depend on an adapter protocol or registry;
it must not import a concrete family, recognize family-specific names, or carry a
DeepSeek/Qwen/Kimi-shaped default. Conversely, adapter code may import public
framework contracts and generic components to implement those contracts.

For a cropped validation model, scale architecture-owned tables or buckets in the
model/config adapter. Do not truncate a loaded tensor merely to make an example fit.
The recipe selects the crop; it does not redefine the algorithm.

## Complete Base Architecture Before Replacement

The registered production architecture must be completely implemented in the
model package's `modeling*.py` files and must directly construct an executable
canonical model. This includes every claimed text, vision, MoE, shared-state, and
residual-stream branch. Model mathematics belongs to that architecture;
high-performance modules are implementations or optimizations of existing
semantics, not a hidden assembly step required to make the architecture runnable.
Prove a small direct forward/backward before applying recipe replacements.

Run two distinct acceptance gates. The first disables all recipe replacements
and validates architecture semantics, initialization, gradients, and checkpoint
FQNs. The second enables optimized replacements and validates state identity,
numerical parity, distributed behavior, and performance. Do not use an optimized
replacement pass to waive a failure in the reference architecture. A model-owned
reference class and an optimized replacement may share a semantic base, provided
the reference class explicitly disables device-specific optimized execution.

Construct each production module directly while preserving the official parameter
FQNs, aliases, initialization, meta-device materialization, planner inspection,
FSDP wrapping, and checkpoint loading contracts. Device-specific kernels may be
selected inside that module while a portable eager/reference path preserves
executability. That path is not automatically an independent correctness oracle:
if it shares semantic code with the optimized path, validate the shared mathematics
against authoritative source code or an independently derived formula.

Do not register a parameter-owning or non-executable placeholder as production
model structure. Replacement is allowed only after both its source and target are
complete modules with a declared semantic and state-transfer contract. It may:

- replace an executable attention, MoE, vision, or other module with a numerically
  equivalent high-performance implementation;
- install TP/CP/EP-aware computation, collectives, or sharded parameter behavior;
- select a device-optimized kernel while retaining a valid reference path.

It must not create missing layers or required parameters, invent the model's
forward semantics, repair an otherwise invalid architecture, or be required merely
to instantiate and execute the model. For example, replacing an executable
`DeepseekV41Attention` with `SharedCompressedDSAAttention` is valid; constructing a
`PlaceholderAttention` that owns weights but cannot run until the same replacement
is applied is invalid. If a source-only HF repository is the only architecture
reference, port its required implementation into `modeling*.py` before adding
replacement rules.

Gate A must inspect the replacement-disabled model for placeholder/debug classes
and run direct forward/backward over every claimed structural branch. Gate B then
checks source/target FQN and state-dict identity, numerical parity, distributed
behavior, and performance. Any dependency on replacement for construction or
basic execution is a failed integration, even when Gate B passes.

Direct wrappers can hide an upstream class from type-based initialization. When a
canonical wrapper is installed before `post_init()`, explicitly preserve every
special initialization rule, not only generic Linear/Embedding initialization.
For example, zero attention sinks and residual-mixing bases, initialize projection
weights, and set scale vectors exactly as the authoritative class does. Test both
the initialized values and finite end-to-end gradients; structural type assertions
alone do not catch uninitialized parameters.

## Shared State and Recomputation

Models with cross-layer KV, index, candidates, recurrent coefficients, or another
producer/consumer relationship need an explicit per-forward state object. Key
entries by source layer or another stable identity and pass the object through the
model forward. Do not store a mutable "latest value" globally: concurrent forwards
and backward replay can read or overwrite the wrong state.

The state belongs to the model contract; communication and packing metadata used
to populate it belong to the adapter. Tensors required by downstream consumers
must remain in the autograd graph. If whole-layer checkpoint replay would publish
or overwrite shared values, define a model-owned safe recomputation boundary such
as norms and MLP/MoE submodules, and validate multiple checkpoint depths.

## Runtime Inputs

Implement `RuntimeInputAdapter` for model-owned inputs derived from a generic
device batch and runtime context. Typical outputs include packed-sequence offsets,
CP descriptors, modality routing, cache descriptors, or shared-attention metadata.
The generic batch path owns movement and collision checks; the model adapter owns
meaning and validation. New integrations must not introduce an attention-specific
batch API when the requirement is not inherently limited to attention.

With the Omni data lifecycle, put source decoding and sample/batch encoding in a
model-owned `OmniDataTransform` (`encode_sample` and, when needed,
`encode_batch`). Attach only forward-time metadata through
`dataloader.get_batch.runtime_input_adapter`. Do not restore the removed
`DataBatchAdapter`/`batch_adapter` lifecycle to add padding, collation, modality,
or packed-sequence behavior. Both `TextParallelBatch` and `OmniParallelBatch`
must reject a runtime field that replaces an already encoded/framework-owned
field.

## Checkpoint Mapping

Classify every mismatch:

1. **Direct rename:** source and target tensors have the same semantics and shape.
2. **Transform:** fused/split projections, permutations, transposes, packing, or
   another reversible layout conversion. Implement and test a round trip.
3. **Architectural mismatch:** semantics or shapes differ. Do not relabel the
   weight; change construction/replacement or report it unsupported.
4. **Training-only initialization:** the released checkpoint intentionally omits a
   training parameter. Declare and test the initialization rule.

Report loaded, renamed, transformed, missing, unexpected, and newly initialized
tensors. Cache/decode-only buffers may be excluded from training only after source
inspection proves that forward/backward does not consume them.

## Meta Materialization and Derived Buffers

Large-model construction may instantiate parameters on `meta`, establish
TP/EP/FSDP layouts, and then call `model.to_empty(device)`. `to_empty()` allocates
new storage for every registered buffer as well as every parameter; it does not
preserve a CPU buffer merely because `init_empty_weights(include_buffers=False)`
was used. Treat materialization as a state-lifecycle boundary.

First inspect the authoritative model and classify each buffer:

- persistent model state uses ordinary `register_buffer(..., persistent=True)`
  and is restored from the checkpoint;
- deterministic derived state that the authoritative model marks
  `persistent=False` remains non-persistent and is rebuilt after materialization;
- caches, workspaces, and step-local statistics use their model-defined clear or
  lazy-allocation rule and must not be mistaken for construction constants.

Do not promote a non-persistent authoritative buffer to a parameter or persistent
buffer just to make DCP save it. Doing so changes the checkpoint contract and can
make a checkpoint appear compatible while its tokenizer/config-derived semantics
are different.

For a constant or deterministic tensor, use the composition-style registration
API. It works with a custom module and with an unmodified native HF class:

```python
class HashState(nn.Module):
    def __init__(self, token_map, primes):
        super().__init__()
        register_rebuildable_buffer(
            self,
            "token_map",
            value=token_map,
            persistent=False,
        )
        register_rebuildable_buffer(
            self,
            "offsets",
            factory=self._build_offsets,
            persistent=False,
        )

    def _build_offsets(self, context: MaterializationContext):
        del context
        return build_offsets_from_config()
```

Do not add a framework mixin to an authoritative model's inheritance tree. When
the HF module already registered the correct non-persistent buffer, the family
adapter can capture its construction value without replacing it:

```python
def register_materialization(model):
    register_rebuildable_buffer(model.hash_state, "token_map")
    register_rebuildable_buffer(model.hash_state, "offsets")

MODEL_ADAPTER_SPEC = ModelAdapterSpec(
    ...,
    materialization=register_materialization,
)
```

The `materialization` provider runs after configured module replacements and
before TP/EP/FSDP sharding plus `to_empty()`. It mutates registration metadata and
returns `None`; it must not execute the rebuild itself. This is the preferred path
for native HF or third-party model classes that HyperParallel does not own.

`value` and `factory` are mutually exclusive. A value is retained as a detached
CPU source outside `Module._apply()`. A factory is called once with a CPU
registration context to establish shape/dtype and again with the final
`MaterializationContext`; it must not depend on mutable training state or produce
side effects. The framework copies the result into the existing registered buffer,
slicing a global source to the buffer's final DTensor layout when needed. It does
not replace the buffer object or layout.

For several interdependent derived states or model-specific validation, implement
the explicit module-local hook when the model class is owned by the integration:

```python
@torch.no_grad()
def rebuild_materialized_state_(self, context: MaterializationContext) -> None:
    ...
```

For an unmodified HF class, register the equivalent adapter-owned callback with
`register_materialized_state_hook(module, hook)`; the callback receives
`(module, context)`. Do not monkey-patch the HF class or replace its `forward()`
solely to gain a materialization callback.

The hook is invoked after either model-native random initialization or pretrained
checkpoint finalization and before compile/optimizer construction. It must be
deterministic and idempotent, update derived storage with `copy_()`, and avoid
collectives by default. It must not recurse, replace or mutate parameters,
persistent buffers, or established DTensor layouts, and it must not overwrite
loaded checkpoint values. The framework validates these protected invariants in
strict mode and reports the owning module FQN, materialization reason, and device
when rebuild fails.

Keep the authoritative recovery boundary:

- if the source model derives a non-persistent buffer from config/tokenizer
  assets on every construction, retain those assets or an equivalent deterministic
  recipe and rebuild it;
- if the source model checkpoints a buffer, load it normally and do not register
  it as rebuildable;
- if the value changes during training, it is training state and needs an explicit
  checkpoint contract rather than a construction-time rebuild recipe.

Tests must cover scratch and pretrained materialization paths as applicable. At a
minimum, prove that `meta -> to_empty -> rebuild` reproduces the authoritative
values, the derived names remain absent from `state_dict`, a repeated rebuild is
identical, and parameters/persistent buffers keep their identity, value, and
layout. DCP resume normally occurs after initial model materialization; verify that
resume does not introduce a second, model-specific `to_empty()` path.

## Native Multimodal Data

Keep modality decoding and transformation in the data layer, but keep the model's
token/patch/grid contract in a model-specific transform or adapter. Validate:

- processor/tokenizer revision and image normalization;
- placeholder expansion, modality masks, token types, patch/grid offsets, and
  label masking;
- ordering and alignment of multiple images and text spans;
- packing boundaries required by compressed or shared attention;
- FSDP units and execution order for vision towers, projectors, and conditional
  branches.

An image token placeholder alone is not multimodal training. The test must show
that visual parameters receive finite gradients and that text-only samples follow
the intended bypass path.
