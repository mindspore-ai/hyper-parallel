# High-performance functions

`hyper_parallel.components.functional` provides reusable high-performance functions for Ascend NPU. CPU and GPU fallback implementations
are not included.

```python
from hyper_parallel.components.functional import rms_norm

output = rms_norm(x, weight)
```

RoPE functions keep the Transformers call contract:

```python
from hyper_parallel.components.functional import apply_rotary_pos_emb

query, key = apply_rotary_pos_emb(query, key, cos, sin)
```

Use `apply_rotary_pos_emb_interleave` for Transformers models whose source
attention calls the interleaved variant.

## MoE functions

The reusable MoE interfaces include:

- `grouped_matmul`: NPU grouped matrix multiplication with backward support.
- `moe_token_permute`: reorder routed tokens into expert-major order.
- `moe_token_unpermute`: restore expert outputs to token-major order.
- `swiglu`: fused NPU SwiGLU used by grouped experts.

```python
from hyper_parallel.components.functional import (
    moe_token_permute,
    moe_token_unpermute,
)

permuted_tokens, sorted_indices = moe_token_permute(tokens, expert_indices)
expert_outputs = experts(permuted_tokens, tokens_per_expert, routing_weights)
output = moe_token_unpermute(expert_outputs, sorted_indices, routing_probs)
```

In this example, `experts` is an instance of `hyper_parallel.components.modules.GroupedExperts`. It contains only local expert
weights and computation; routing, expert-parallel communication, and shared experts remain outside the module.
`hyper_parallel.components.modules.SharedExpert` provides the parameter-owning dense MLP used as a shared expert.

## Auxiliary-loss functions

`hyper_parallel.components.functional` also exports `aux_loss_auto_scale` and `set_aux_loss_scale` for auxiliary-loss gradient injection.

## DSA causal training operators

The public DSA interfaces use `dsa_*` names and import CANN operators directly
from `cann_ops_transformer`. V3.2 sparse execution no longer depends on Omni
custom-op names. Dense warm-up is selected by the source module's
`dsa_dense_warm_up` state in the CP wrapper; `use_dense: true/false` can force
one branch for a complete run. Install a matching Python extension and native
OPP before importing or calling these NPU-only functions.

| Public interface | Positional parameters | Keyword parameters | CANN operators |
| --- | --- | --- | --- |
| `dsa_indexer` | `index_query, index_key, merge_weight, actual_seq_qlen, actual_seq_kvlen, sparse_count` | none | `lightning_indexer` |
| `dsa_sparse_attention` | `query_nope, compressed_kv, query_rope, key_rope, topk_indices, scale, actual_seq_qlen, actual_seq_kvlen` | none | `sparse_flash_attention`, `sparse_flash_attention_grad` |
| `dsa_sparse_indexer_kl_loss` | `index_query, index_key, merge_weight, query, key, topk_indices, softmax_max, softmax_sum, query_rope, key_rope, actual_seq_qlen, actual_seq_kvlen, scale, loss_coeff` | none | `sparse_lightning_indexer_kl_loss_grad` + metadata |
| `dsa_dense_indexer_lse` | `index_query, index_key, head_weights` | `query_lengths=None, key_lengths=None, layout="TND"` | `dense_lightning_indexer_softmax_lse` + metadata |
| `dsa_dense_indexer_kl_loss` | `index_query, index_key, head_weights, teacher_probabilities, indexer_lse` | `query_lengths=None, key_lengths=None, layout="TND", loss_coefficient=1.0` | `dense_lightning_indexer_kl_loss_grad` + metadata |

For the public sparse attention/indexer interfaces, `actual_seq_qlen` and
`actual_seq_kvlen` are cumulative document ends without a leading zero. The
indexer accepts BSND inputs and returns int32 top-k plus flattened indexer Q/K/
weights. Attention accepts BSND states and returns output, softmax maximum, and
exponential sum. The loss receives flattened TND tensors and returns a scalar
token-mean auxiliary loss. Dense `query_lengths`/`key_lengths` are cumulative
ends for TND or per-batch valid lengths for BSND.

### Dependencies and initialization

Load the CANN and custom-operator environments before starting Python. The
dense DSA adapter imports its required operators directly from
`cann_ops_transformer`, matching the V4.1 fused-op pattern. Importing
`dsa_dense` therefore requires the extension to be installed. Install a wheel
that exports the required DSA APIs and deploy matching native OPP kernels for
the target SoC.

Single-card precision tests passed on Ascend 910B3 with Python 3.10.20,
Torch 2.6.0+cpu, torch_npu 2.6.0.post6, and CANN 9.1.0-beta.1 using the
standard ops-transformer package and its matching native OPP. Other
toolkit/vendor combinations require validation.

### Tensor contracts and training

The public API uses the entry points in `dsa_indexer.py`,
`dsa_sparse_attention.py`, `dsa_kl_loss.py`, and `dsa_dense.py`. Low-level CANN
metadata adaptation lives in `dsa_ops.py`; its `_package_*` helpers are private
implementation details, not user-facing APIs. Dense KL takes detached teacher
probabilities and indexer LSE.
Q/K and rotary tensors use float16 or bfloat16; the tested indexer also requires
weights in the query dtype. Sparse indices are int32. Softmax statistics are
float32, with the following shapes for the tested single-KV-head path:

| Statistics | BSND | TND |
| --- | --- | --- |
| Dense indexer LSE | `[B,1,S]` | `[1,T]` |
| Dense attention target | `[B,S,1,MaxK]` | `[T,1,MaxK]` |
| Sparse teacher maximum/sum | `[B,1,S,Nq]` | `[1,T,Nq]` |

TND sequence metadata contains cumulative document ends, without a leading zero;
BSND metadata contains per-batch valid lengths. The public CANN APIs expose these
as `query_lengths` and `key_lengths` and build package metadata internally.

The existing high-level indexer/attention interfaces flatten BSND into TND and
preserve their output contracts. For example, with model tensors already on NPU:

```python
import torch

from hyper_parallel.components.functional.dsa_indexer import dsa_indexer
from hyper_parallel.components.functional.dsa_kl_loss import dsa_sparse_indexer_kl_loss
from hyper_parallel.components.functional.dsa_sparse_attention import dsa_sparse_attention

# q_idx/k_idx: BSND indexer tensors; weights: BSN; q/kv: absorbed MLA BSND.
# q_rope/k_rope: BSND rotary states. ends_cpu: cumulative document ends.
ends_npu = ends_cpu.to(device=q.device, dtype=torch.int32)
indices, iq, ik, iw = dsa_indexer(q_idx, k_idx, weights, ends_npu, ends_npu, 2048)
output, maximum, total = dsa_sparse_attention(
    q, kv, q_rope, k_rope, indices, scale, ends_npu, ends_npu)
aux_loss = dsa_sparse_indexer_kl_loss(
    iq, ik, iw, q.flatten(0, 1), kv.flatten(0, 1), indices, maximum, total,
    q_rope.flatten(0, 1), k_rope.flatten(0, 1), ends_cpu, ends_cpu, scale, loss_coeff)
(task_loss(output) + aux_loss).backward()
```

Dense warm-up requires native OPP support for the dense LSE/KL kernels. The
public dense KL function returns token-mean loss, applies `loss_coefficient`
once, and propagates the upstream loss gradient only to indexer Q/K/weights.
Teacher probabilities and LSE do not receive KL gradients.

Attention requires shared K/V storage, shape and strides. Each candidate row
must contain `min(causal_length, TopK)` distinct valid indices, followed by `-1`
padding, as produced by the indexer. Arbitrary shorter lists and independent V
are unsupported. Attention supports first-order Q/shared-KV/rotary gradients;
higher-order gradients are unsupported. The high-level attention output retains
zero padding for the rotary width.

These boundaries reject HP DTensors. Existing CP/TP hooks must complete
communication and provide local tensors before calling them. Single-NPU
precision and DeepSeek-V3.2 CP1/CP2 module parity were validated; full-model
multi-step convergence, performance, compression, and noncausal execution were
not validated.

Run the hardware-independent tests and the single-NPU precision tests separately:

```bash
python -m pytest tests/ut/components/functional tests/ut/core/context_parallel/test_dsa_context_parallel.py -q
ASCEND_RT_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 \
  python -m pytest tests/torch/components/functional/test_dsa_ops.py -q
ASCEND_RT_VISIBLE_DEVICES=0,1 \
  python -m pytest tests/torch/context_parallel/test_deepseek_v32_dsa_cp.py -q
```

Select an available device for the second command. The precision suite covers
BSND/TND, FP16/BF16 indexer statistics, real TopK selection, sparse attention
forward/backward, sparse KL, and the public sparse training chain against independent
FP32 Torch references. The two-card test compares the same DeepSeek-V3.2 attention
weights and inputs at CP1 and CP2, including auxiliary KL and packed boundaries;
it checks output, input-gradient, and CP-averaged parameter-gradient parity. This
DSA path has separate APIs and dependencies from the DeepSeek V4.1 fused adapters
described below.

## DeepSeek V4.1 fused operators

The V4.1 adapters use the optional `cann_ops_transformer` PyTorch extension:

| Component | Required entry points |
| --- | --- |
| Indexer | `lightning_indexer` (LI V2), `lightning_indexer_metadata` when required by the device |
| Attention | `sparse_flash_mla`, `sparse_flash_mla_grad`, `sparse_flash_mla_metadata` |
| Indexer KL | `sparse_lightning_indexer_kl_loss_grad`, `sparse_lightning_indexer_kl_loss_grad_metadata` |

### Installation

Follow the official ops-transformer guides:

1. [Build and install the native operator package](https://gitcode.com/cann/ops-transformer/blob/master/docs/zh/install/compile.md).
2. [Build and install the PyTorch extension](https://gitcode.com/cann/ops-transformer/blob/master/torch_extension/README.md).

Both the native kernels and the Python extension are required. Load the CANN and
custom-operator environment as described in the guides before training.
HyperParallel imports the standard `cann_ops_transformer` package; use the
standard wheel build (`bash build.sh --torch_extension`), without `--ops` or
`--vendor_name`, which can change the Python package name at the tested revision.

### Tested versions

| Dependency | Tested configuration |
| --- | --- |
| Hardware / CANN Toolkit | Ascend 910B3 / 9.0.0 |
| Python / PyTorch / torch-npu | 3.11 / 2.10.0 / 2.10.0 |
| ops-transformer source | `0b08075141913daeaf1e856faf7480ed393c8949` |
| Python extension | `cann_ops_transformer` 1.0.0, built from the source revision above |

The source revision identifies the tested operator interfaces; the wheel version
alone does not identify them.

### Configuration

Enable the three adapters together through the V4.1 replacement:

```yaml
plan_overrides:
  - match: "model.layers.*.self_attn"
    module_type: hyper_parallel.models.deepseek_v41.modeling_deepseek_v41.DeepseekV41Attention
    replace_module:
      _target_: >-
        hyper_parallel.components.modules.shared_compressed_dsa_attention.SharedCompressedDSAAttention
      use_fused_ops: true
```

`false` (default) uses the Torch reference implementation without this extension.
`true` requires the fused entry points used by the model's execution path and
raises on missing entries or unsupported inputs, without automatic fallback.
The fused path supports ordinary causal CSA with compression ratio 1 or 2;
candidate-pool selection is not supported.
