# KDA AllGather and hybrid context parallelism

This integration composes three CP operations: Ulysses token/head exchange,
group-local summary AllGather, and P2P between group owners. The local S/M/G
producers and FLA token kernels remain those of the existing KDA backend.

## Configuration

Select the unified model adapter in a training recipe:

```yaml
plan_overrides:
  - match: "*.self_attn"
    when: cp
    region_dispatch: false
    inner_target: self
    inner_wrapper:
      _target_: hyper_parallel.models.kimi_k3.adapter.distributed.context_parallel.kimi_delta_attention_cp_wrapper
      backend: triton
      chunk_size: 64
      boundary_protocol: grouped_allgather_p2p
      ulysses_degree: 2
      group_size: 2
```

For CP8 the following configurations are supported:

| Operation | `boundary_protocol` | `ulysses_degree` | `group_size` |
| --- | --- | ---: | ---: |
| Existing P2P | `p2p` | 1 | 1 |
| Summary AG | `allgather` | 1 | 1 |
| Ulysses + P2P | `p2p` | 2 | 1 |
| Ulysses + AG | `allgather` | 2 | 1 |
| AG + P2P | `grouped_allgather_p2p` | 1 | 2 |
| Ulysses + AG + P2P | `grouped_allgather_p2p` | 2 | 2 |

After Ulysses degree U, the state CP size is R=P/U. Group width g must divide R;
there are R/g owner groups. The three-way CP8/U2/g2 example has two head lanes,
two ranks in each gather, and two owners in each state chain. Group width one
reduces to P2P; group width R reduces to fused AG. U=P reduces to local KDA after
Ulysses exchange. Existing dedicated P2P and Ulysses wrappers remain available.

U must divide both query/key and value head counts. Group size is only accepted
for the grouped protocol. The initial implementation supports dense BF16 Ascend
training, K=V=128, BT64, and the existing lower-bounded gate contract. Gather
protocols require the Triton backend; unsupported eager configurations fail
before tensor communication. Packed sequences, recurrent inference cache and
simultaneous TP/CP retain the existing restrictions.

## Cache and communication

Every rank retains only its own forward M for backward, including group owners.
The protocol object stores communication metadata and no invocation tensors.

- Forward AG collects S+M; the pure AG path uses a fused ordered prefix merge.
- Backward AG collects newly computed G together with cached local M. Pure AG
  uses a fused ordered transposed suffix merge.
- Group owners apply the gathered summaries to the received state in the same
  chronological order as P2P, then forward the boundary to the next owner.
  A group broadcast distributes the incoming boundary to its local ranks.
  Backward uses the reverse ordered scan over G/M. Group matrices are not
  precomposed: reassociating products can change BF16 rounding in downstream
  kernels even when the FP32 boundary difference is small. No M_all or M_group
  survives forward; M_group is not constructed.
- ShortConv halos use the original CP sequence before Ulysses exchange.
  Learned head parameters follow the same Ulysses head partition.

For B1/H96/K=V128, local M is 6 MiB, or 6/U MiB after Ulysses. This is transfer
storage only: local activations and incoming states are also saved or recomputed.
AG still has O(R) temporary gather space; grouped AG has O(g) temporary space.
AllGather's fused merge remains an ordered scan, and a fixed-width owner chain
still has O(P/(Ug)) communication dependency depth. Ordered group application
performs up to g state updates on each owner; it does not reduce the total
serial state arithmetic to O(P/(Ug)).

## Mesh requirements

Pass the chronological CP submesh from the root DeviceMesh, preserving DP and
other sibling axes. Splits use the existing root-aware mesh machinery during
executor construction; forward and checkpoint replay create no process groups.
Naturally ordered noncontiguous CP groups are supported. Arbitrary permutations
are rejected rather than silently interpreting collective slots in the wrong order.
A detached non-WORLD mesh cannot be split without its root relationship.

## Numerical contract and validation

The new protocols preserve master's affine summaries and local mixed-precision
arithmetic. They do not promise bitwise equivalence to a single sequential FLA
call: matrix association and native mixed-precision state feedback can differ.
Compatibility checks do not establish native-FLA equivalence or large-scale training qualification.

The tests include independent FP64 boundary recurrences, CPU FP32 full-layer
layout/halo/parameter gradients, real subgroup communication, and local-M storage
ownership. NPU layer tests use B1/H96/D128, local length 4096 by default, the same
weights/input/upstream gradient for every mode, and FP32 parameter-gradient
aggregation outside the measured layer interval.

```bash
pytest -q tests/ut/distributed/context_parallel/test_kimi_delta_attention_mesh.py
pytest -q tests/torch/context_parallel/test_kda_ag_hybrid.py -k boundary_cpu
pytest -q tests/torch/context_parallel/test_kda_ag_hybrid.py -k boundary_cp4
pytest -q tests/torch/context_parallel/test_kda_ag_hybrid.py -k hybrid_layout_cpu
KDA_TEST_LENGTH=8192 pytest -q tests/torch/context_parallel/test_kda_ag_hybrid.py -k layer_cp4
pytest -q tests/torch/context_parallel/test_kda_ag_hybrid.py -k layer_cp8
```

Run multi-card cases in an exclusive device window. The layer worker checks two
live invocations with reverse backward for AG, and non-reentrant checkpoint
replay for grouped AG. Boundary tests exercise interleaved DP/CP subgroups and
Ulysses-derived state groups. The eight-card layer case additionally exercises
all three nontrivial axes. These are correctness tests, not benchmarks.

## Implementation layout

The existing `functional/kimi_delta_attention.py` keeps the local KDA forward
and backward, with an optional boundary executor. S/M/G arithmetic is unchanged.
`distributed/context_parallel/kimi_delta_attention_mesh.py` owns group creation,
AG/owner-chain communication and merge dispatch. The fused merge kernel lives
beside the existing KDA kernels in `functional/_triton/kimi_delta_attention/`.
There is no second functional KDA implementation.
