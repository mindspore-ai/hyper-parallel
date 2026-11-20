# Runtime evidence contract

A normal parameter checkpoint does not contain routing or gradient evidence. Missing captures yield
`UNVERIFIED`; a mismatch in observed captures yields `FAIL`. All required checks must pass for overall `PASS`.
`--strict` converts missing required evidence and warnings to failure. Unverified CLI runs exit nonzero.

Token conservation requires independent values on **each** log record:

```text
rank=0 step=1 Layer 3 input_tokens=2 top_k=2 tokens_per_expert=[2,2]
```

The counts must sum to `input_tokens * top_k`. A previous line's metadata is not reused.
Routing captures can be JSON (CPU-only) or `.pt`/`.pth` (loaded with Torch on CPU using `weights_only=True`).
The top level must be a mapping. Example:

```json
{
  "routing": [{
    "layer_id": 3, "step": 1, "input_tokens": 2, "top_k": 2,
    "input_expert_indices": [1,0,1,0],
    "expert_assignments": [0,0,1,1],
    "tokens_per_expert": [2,2],
    "token_permutation": [1,3,0,2]
  }],
  "state_dict": {"layers.3.experts.0.weight": [[1.0,2.0]]},
  "gradients": {"layers.3.experts.0.weight": [[0.1,0.2]]}
}
```

`token_permutation` indexes flattened **assignment slots** before dispatch, not original token IDs.
It must be a bijection, and indexing `input_expert_indices` by it must produce the expert-major assignments.
Captures can be nested or use sibling keys such as `layers.3.token_permutation`. Every record and shard is checked.

Gradients must match full parameter keys and shapes and be finite with nonzero norm. Weights alone never pass.
Supported expert parameter names contain `expert` and end in `.weight`, `.bias` or `_weight`; unsupported naming
requires an exporter mapping to this contract. `experts_checked` is a legacy parameter-count field;
`parameters_checked` names that count accurately. PASS covers captured shards, not unsupplied ranks or steps.
A missing gradient in partially captured evidence fails. No captured gradients yields UNVERIFIED.
Nonzero gradients establish observed flow only, not numerical equivalence.
