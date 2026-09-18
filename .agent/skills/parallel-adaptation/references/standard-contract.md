# Standard Parallel Adaptation Contract

## Layering

Use the narrowest layer that owns the behavior:

| Layer | Owns | Does not own |
| --- | --- | --- |
| `components/functional` | Reusable kernels/functions | Model names, topology, benchmark rules |
| `components/modules` | Parameter-owning reusable high-performance modules | Cross-rank orchestration |
| `distributed` | Mesh, placements, collectives, generic plan application | Model-family naming and forward quirks |
| `models/<family>/adapter` | Runtime class mapping, replacements, attention/loss/CP/EP glue | Duplicate generic kernels or Trainer lifecycle |
| `trainer` | Component construction, lifecycle, loss/update boundaries | Benchmark task semantics |

`ModelAdapterSpec` is the family registration point. Provider imports remain lazy so registry discovery works without all
optional runtimes. Generic builders must not grow `if model_type == ...` branches.

## Capability Declaration

Record one row per model/backend/dtype combination:

| Axis or combination | Status | Constraints | Evidence |
| --- | --- | --- | --- |
| FSDP/HSDP | supported/unsupported/unverified | | |
| TP | | degree, sequence/loss parallel | |
| CP | | algorithm, layout, packed/media support | |
| EP | | experts, top-k, grouped kernel/fallback | |
| TP+CP | | | |
| TP+EP | | | |
| CP+EP | | | |
| FSDP/HSDP + model parallel | | | |
| Flagship hybrid | | | |

`supported` requires a real nontrivial topology and parity evidence. `unsupported` requires a fail-fast check and message.
Everything else is `unverified`.

## Ownership Table

Before implementation, fill this table for every important tensor/state:

| Object | Global meaning | Local owner/layout | Replicated over | Reduction/assembly |
| --- | --- | --- | --- | --- |
| Samples/batch IDs | | | | |
| Hidden states | | | | |
| Attention Q/K/V | | | | |
| Logits/vocabulary | | | | |
| Labels/loss mask | | | | |
| Loss sum/token count | | | | |
| Router decisions | | | | |
| Expert parameters/grads | | | | |
| KV cache | | | | |
| Prediction/metric record | | | | |

An implementation is not standardized until ownership can be stated without relying on `world_rank` as a proxy for every
axis.

## Plan Requirements

- Every boundary declares parameter, input, internal, natural-output, and downstream placements where applicable.
- Explicitly declare non-dispatchable injected regions and their local/global tensor contract.
- Divide cached head/expert metadata only when its runtime meaning becomes local; keep global config immutable.
- Preserve tied parameters and deferred row-parallel bias semantics.
- Resolve partial values before consumers read them; wait for asynchronous handles and streams per `distributed.md`.
- Validate mesh-axis divisibility and model constraints before parameter materialization or collectives.
- Unsupported attention, packing, media, cache, or checkpoint combinations fail before the first training step.

## Fast Adaptation Sequence

1. Register the family and make the non-parallel path pass.
2. Build a tiny deterministic checkpoint and stable batch fixture.
3. Add TP plan and local metadata; pass forward/backward/update parity.
4. Add CP field ownership and attention wrapper; pass tail/padding/position parity.
5. Add EP ownership and dispatch/combine; pass zero-token and uneven-routing parity.
6. Add selected hybrids and verify every reduction happens once in the correct group.
7. Validate checkpoint save/resume and, when supported, generation/cache behavior.

This sequence is an evidence order, not a requirement that every model support every axis.
