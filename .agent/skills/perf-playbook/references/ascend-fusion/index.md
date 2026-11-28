# Ascend fused-operator knowledge index

> **Evidence provenance:** Official documentation and integration constraints.
> Entries do not imply a verified speedup in the current training environment.

Eight official documentation pages were retrieved on 2026-11-04 from the `Pytorch/700`
branch for `PyTorch2.5.1`. Their contents can change and do not establish compatibility
with torch_npu 2.9 or CANN 9.0. The repository provides [searchable entries](catalog.json).
On-demand retrieval records the requested URL, resolved URL, retrieval time, and content hash.

| Interface | Official documentation | Training integration requirements |
|---|---|---|
| `torch_npu.npu_fusion_attention` | [Reference](https://www.hiascend.com/document/detail/zh/Pytorch/700/apiref/apilist/ptaoplist_000156.html) | Documented for training only. Verify masks, scale, dropout, layout, variable lengths, and backward. Do not substitute inference FlashAttention. |
| `torch_npu.npu_rms_norm` | [Reference](https://www.hiascend.com/document/detail/zh/Pytorch/700/apiref/apilist/ptaoplist_000139.html) | Preserve epsilon, normalization dimensions, and gamma dtype. The documented outputs include backward intermediates; validate actual autograd behavior independently. |
| `torch_npu.npu_swiglu` | [Reference](https://www.hiascend.com/document/detail/zh/Pytorch/700/apiref/apilist/ptaoplist_000143.html) | The selected dimension must be divisible by 2, and the input must be contiguous and nonempty. Verify gate/up ordering and backward. |
| `torch_npu.npu_rotary_mul` | [Reference](https://www.hiascend.com/document/detail/zh/Pytorch/700/apiref/apilist/ptaoplist_000141.html) | Verify half/interleaved layout and cos/sin gradients. The documentation notes that training may retain inputs and increase device memory. |
| `torch_npu.npu_grouped_matmul` | [Reference](https://www.hiascend.com/document/detail/zh/Pytorch/700/apiref/apilist/ptaoplist_000160.html) | This documentation version explicitly restricts the interface to inference. Training integration requires a matching training/backward implementation and independent validation. |
| `torch_npu.npu_moe_init_routing` | [Reference](https://www.hiascend.com/document/detail/zh/Pytorch/700/apiref/apilist/ptaoplist_000169.html) | This version documents inference only and does not establish autograd support. Validate permutation, token conservation, and gradients. |
| `torch_npu.npu_moe_finalize_routing` | [Reference](https://www.hiascend.com/document/detail/zh/Pytorch/700/apiref/apilist/ptaoplist_000167.html) | This version documents inference only. Training requires a separately implemented and validated backward that preserves scale weights and routing-index semantics. |
| `torch_npu.optim.NpuFusedAdamW` | [Reference](https://www.hiascend.com/document/detail/zh/Pytorch/700/apiref/apilist/ptaoplist_000298.html) | Applies to AdamW parameters only and cannot replace Muon. Verify optimizer states, bias correction, decoupled weight decay, epsilon, and per-step parameter updates. |

**An inference-only declaration is a compatibility restriction.** BF16 support or a
training-series device name does not establish training support for grouped matmul or
the two MoE routing interfaces. If a newer release supplies a training path or the
framework provides custom backward, record that implementation's provenance and validate
all input and parameter gradients separately.

## Retrieval and updates

```bash
python .agent/skills/perf-playbook/scripts/fetch_ascend_docs.py --query "rmsnorm rsqrt"
python .agent/skills/perf-playbook/scripts/fetch_ascend_docs.py --out /path/to/doc-cache
```

Keyword search returns candidate entries. Read the corresponding documentation and
match the actual shapes, dtypes, devices, and versions. Refresh only the catalog's
explicit official links. For a new version, locate its content URL in the official
index before registering it. WAF, login, and error pages remain invalid even with HTTP
200. A failed refresh must not overwrite the previous manifest.

The downloader preserves readable page text. Formula images and diagrams are not
converted to text; consult the official page when they are needed.

## From an API to a validated optimization

1. Identify the runtime hotspot, shapes, dtypes, strides, and source call site.
2. Check documented version, device, and training restrictions, then inspect the
   installed interface and backward support.
3. Run forward/backward comparisons, fallback checks, and microbenchmarks across the
   actual shape distribution.
4. Record fused-path invocation counts and supported-shape coverage in the training
   process, and verify them against the trace.
5. Judge end-to-end benefit from complete training steps with profiling disabled and
   the fixed model MFU convention.

See [fusion methodology](../fusion-methodology.md). Preserve the full evidence and
applicability limits of historical cases. Add accepted cases here or in model-specific
files; retain negative results and keep unverified explanations as hypotheses.

## Accepted training cases

- [Qwen3 MoE + Muon: Q/K RMSNorm fusion and two negative results](../cases/qwen3-muon-qk-rms-20261105.md).
  Loaded 30B weights, four nodes with 32 die, native routing, and 16 recomputed layers.
  Q/K normalization fusion improved complete-step throughput by approximately 2.9%.
  Muon layout/foreach changes and a causal FlashAttention mode both engaged but each
  added less than the 2% acceptance threshold. The user approved a 5% numerical tolerance
  for this short run. These results do not establish long-term convergence or gains
  on other models or software versions.
