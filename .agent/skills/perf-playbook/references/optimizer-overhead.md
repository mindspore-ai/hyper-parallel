# Optimizer and non-model compute overhead

> **Evidence provenance:** Mixed. General engineering practice and the published
> Newton–Schulz algorithm, with a utilization audit from a MoE pretraining run in this
> repository. See [evidence provenance](../SKILL.md#evidence-provenance).

The MFU numerator counts model computation only; see [mfu-targets.md](mfu-targets.md),
section 1. Optimizer iterations, routing corrections, and selective recomputation are
**executed work outside that numerator**. Omitting them can make a busy device appear
underutilized and direct optimization at the wrong bottleneck. It can also obscure
opportunities to reduce optimizer overhead.

This reference defines the accounting convention, scriptable estimates, and criteria
for deciding whether and how to optimize this work.

## 1. Measure the phase share before its absolute cost

First obtain the optimizer's fraction of step time from the `profiling` skill's
compute, exposed-communication, device-idle, and optimizer breakdown.

| Optimizer share of step time | Interpretation |
|---|---|
| < 5% | Not a primary bottleneck. Record the cost without prioritizing changes. |
| 5%–15% | Include it in the HFU numerator. Its optimization potential is bounded; prioritize larger compute or communication costs. |
| > 15% | A primary bottleneck. Apply section 4 and include its work in HFU to interpret MFU correctly. |

## 2. Derive Newton–Schulz FLOPs

Muon orthogonalizes gradient matrices using Newton–Schulz iterations. For a matrix
`X` of shape m×n, transposed first to ensure m ≤ n, the reference implementation
performs three matrix multiplications per iteration.

| Operation | Shapes | FLOPs |
|---|---|---|
| `A = X Xᵀ` | (m,n)×(n,m) | 2·m²·n |
| `A A` | (m,m)×(m,m) | 2·m³ |
| `B X` | (m,m)×(m,n) | 2·m²·n |

The cost **per matrix per optimizer step** is
`ns_flops = steps × (4·m²·n + 2·m³)`, where m and n are the smaller and larger
matrix dimensions. Elementwise scaling and norms are O(mn) and comparatively small.
Include any additional matrix multiplications performed by the actual implementation.

**Use the optimizer's parameter groups to determine which matrices count.** Muon acts
on matrix parameters. One-dimensional normalization and bias parameters, and usually
embeddings and output heads, use an AdamW path and do not belong in this term.
Including embeddings can multiply the estimate because their long dimension is the
vocabulary size. In the example in section 3, this mistake changes the estimate from
0.1416 to 0.4798 TFLOPs/step, an overstatement of approximately 3.4×.

## 3. Aggregate `hfu_extra_tflops_per_step`

```bash
python .agent/skills/perf-playbook/scripts/extra_flops.py \
    --spec extra_flops_spec.json --markdown --out extra_flops.json
```

Every item in the `perf-playbook/extra-flops/v1` specification must include `source`.
The script rejects entries without provenance because an extra-work term without a
derivation is not evidence. Four item types are supported.

| kind | Purpose | Required fields |
|---|---|---|
| `newton_schulz` | Muon Newton–Schulz iterations | `steps` and either `shapes` or `shapes_file` |
| `matmul` | Explicit matrix multiplications, such as routing corrections | `dims` = [m, k, n], with optional `count` |
| `forward_fraction` | Selective recomputation as a fraction of forward work | `fraction`, `model_tflops_per_step_per_device` |
| `measured` | Executed work measured directly by profiling | `tflops_per_step` |

Each item can specify `shard_devices`, the number of devices that share its work
uniformly. Use the optimizer-shard count for unsharded shapes, or 1 if the shapes already
represent one device's shard. The default is 1. Output includes an itemized breakdown,
the total, a ready-to-use `hfu_extra_tflops_per_step:` configuration line, and, with
`--markdown`, a derivation block for `templates/tuning-summary.md`.

Export shapes **under the target parallel configuration**. Model configuration alone
cannot determine the shapes resident on each device after sharding. When the optimizer
is available, read its Muon parameter group directly. Use name filtering only as a
fallback and check its count against the intended group.

```python
import json

EXCLUDE = ("embed_tokens", "lm_head", "wte", "wpe")  # Embeddings and output heads use AdamW.

shapes = [list(p.shape) for name, p in model.named_parameters()
          if p.dim() == 2 and not any(key in name for key in EXCLUDE)]
with open("shapes.json", "w", encoding="utf-8") as handle:
    json.dump({"shapes": shapes}, handle)
```

Name filtering is only a fallback. Model families can use other names, including tied
weights, `tok_embeddings`, or `output`. A single incorrectly included embedding can
multiply the extra-work term. Check the exported count and largest shapes first.

**Measured example.** A four-layer Llama structure with hidden size 512, FFN size 1376,
vocabulary 32000, five Newton–Schulz iterations, and one device has 28 eligible
matrices and `0.1416` TFLOPs/step. Incorrectly including `embed_tokens` and `lm_head`
produces 30 matrices and `0.4798` TFLOPs/step. The same model appears to perform 3.4×
as much optimizer work because of one accounting error. This is why every item must
provide `source`.

## 4. Optimization techniques in priority order

1. **Rule out fallback.** If the optimizer phase contains AI CPU execution or integer
   sorting hotspots, apply [fusion-methodology.md](fusion-methodology.md), section 5,
   first; the possible slowdown is much larger.
2. **Newton–Schulz iteration count.** FLOPs scale linearly with the iteration count.
   Reducing it is an algorithm change and requires evidence about orthogonalization
   quality and convergence. Evaluate loss curves as well as throughput.
3. **Optimizer sharding or offload.** Distribute computation across more devices or
   move states out of HBM. Include Host round-trip costs described in the device-idle
   and optimizer section of `feature-scenarios.md`.
4. **Dtype and layout.** Address inefficient dtypes or noncontiguous inputs to the
   Newton–Schulz matrix multiplications before changing the algorithm.
5. **Interpret recomputation jointly.** Full recomputation and an expensive optimizer
   can make HFU substantially higher than MFU. Include both terms in HFU before
   interpreting low MFU as an execution defect.

## 5. A measured accounting example

In one sparse-MoE pretraining utilization audit with a matrix-heavy optimizer, the
optimizer dominated overhead and a healthy run reported PaLM MFU near 12%. This is an
accounting observation for that configuration. Importing a 40%+ target from a dense
model would imply a several-fold gap that the measured breakdown did not support.
Set targets from a baseline multiplier or measured roofline deductions, as described
in [mfu-targets.md](mfu-targets.md), section 3, instead of transferring absolute targets
across configurations.

## 6. Related references

- Utilization conventions, target-setting methods, and waterfall decomposition:
  [mfu-targets.md](mfu-targets.md).
- Operator localization and timing boundaries: [fusion-methodology.md](fusion-methodology.md).
- Step breakdown and optimizer phase share: the `profiling` skill.
- Techniques by scenario: [feature-scenarios.md](feature-scenarios.md).
