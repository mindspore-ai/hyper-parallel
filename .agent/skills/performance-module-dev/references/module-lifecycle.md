# High-Performance Module Lifecycle

## Placement Decision

```text
Stateless reusable operation               -> components/functional
Parameter/buffer/state-owning drop-in block -> components/modules
Model-family symbol or weight mapping       -> models/<family>/adapter
Cross-rank ownership/collectives            -> distributed
Construction/lifecycle/config               -> trainer
```

Do not put model-family conditionals in generic components or embed collective topology in a reusable local kernel.

## Development Stages

1. **Watch**: record upstream project/paper, revision, license, supported hardware/dtype, claimed benefit, and maintainer.
2. **Reference**: establish a readable mathematical implementation and edge-case corpus.
3. **Kernel/function**: implement the reusable operation with explicit capability checks and optional dependency loading.
4. **Module**: add only when a parameter-owning or drop-in contract is needed; preserve state-dict meaning.
5. **Adapter**: map exact runtime classes/symbols and define weight transforms or parameter reuse.
6. **Standalone validation**: forward, backward, parameter gradient, dtype, shape, and error-path parity.
7. **Parallel composition**: run module off/on crossed with the target FSDP/TP/CP/EP topology.
8. **Impact**: measure representative end-to-end training, not only isolated kernel time.
9. **Promotion**: choose opt-in, auto-selected, or default-on according to acceptance evidence.
10. **Maintenance**: record last validated revisions, deprecation/fallback, and upstream API drift.

## Inventory Record

Use a table or machine-readable registry containing:

- public name and category;
- reference implementation and upstream source/revision/license;
- owner and status: watch/reference/experimental/supported/default/deprecated;
- model/backend/device/dtype/shape/topology matrix;
- optional dependencies and fallback;
- numerical tier and last passing accuracy artifacts;
- workload, speed/memory/MFU result, and last performance date;
- known limitations and next action.

## Integration Order

Apply source-class patching or replacement before model construction when the upstream library requires it. Otherwise build
the reference model, replace modules before optimizer creation and parallelization, materialize/load weights in the declared
order, then create optimizer groups. A replacement after optimizer construction is invalid unless the optimizer is rebuilt
and state migration is proven.

Parameter names/layout changes require explicit transforms and checkpoint round-trip tests. When names and layouts are
identical, reuse the source parameters rather than copying values and breaking tied storage.
