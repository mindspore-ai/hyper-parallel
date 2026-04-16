# Expert Parallelism

HyperParallel provides a complete MoE (Mixture of Experts) implementation with
declarative expert-parallel (EP) styles that distribute expert weights and tokens
across devices. The styles follow the same `ParallelStyle.apply(module, device_mesh)`
contract as tensor-parallel styles and compose freely with TP, FSDP, and pipeline
parallelism.

## MoE Building Blocks

### `FeedForward`

SwiGLU feed-forward network used as a shared (always-active) expert inside `MoE`.

```python
FeedForward(dim: int, hidden_dim: int, bias: bool = False)
```

Forward: `w2(silu(w1(x)) * w3(x))`

---

### `GroupedExperts`

Batched expert computation over a set of independent experts sharing the same
hidden dimension. Supports three forward paths selected automatically by device
type and `use_grouped_mm`:

| Path | Backend | Trigger |
|------|---------|---------|
| `_run_experts_for_loop` | All | `use_grouped_mm=False` (default) |
| `_run_experts_grouped_mm` | CUDA | `use_grouped_mm=True` |
| `_run_experts_grouped_mm_npu` | Ascend NPU | `use_grouped_mm=True` |

```python
GroupedExperts(
    dim: int,
    hidden_dim: int,
    num_experts: int,
    use_grouped_mm: bool = False,
)
```

**Weight shapes:**

| Parameter | Shape |
|-----------|-------|
| `w1` | `[num_experts, hidden_dim, dim]` |
| `w2` | `[num_experts, dim, hidden_dim]` |
| `w3` | `[num_experts, hidden_dim, dim]` |

When weights are `DTensor` (e.g. after `ExpertParallel.apply`), `to_local()` is
called automatically before matmul.

---

### `TokenChoiceTopKRouter`

Learnable router that assigns each token to its top-k experts.

```python
TokenChoiceTopKRouter(
    dim: int,
    num_experts: int,
    top_k: int = 1,
    score_func: str = "sigmoid",       # "sigmoid" or "softmax"
    num_expert_groups: Optional[int] = None,
    num_limited_groups: Optional[int] = None,
    route_scale: float = 1.0,
)
```

Returns `(top_scores, selected_experts, num_tokens_per_expert)`.

Node-limited routing (`num_expert_groups` / `num_limited_groups`) restricts each
token to the top-`num_limited_groups` expert groups before selecting top-k.

---

### `MoE`

Top-level orchestrator that combines router, experts, and optional load-balancing.

```python
MoE(
    dim: int,
    hidden_dim: int,
    num_experts: int,
    top_k: int = 1,
    score_before_experts: bool = True,
    load_balance_coeff: Optional[float] = None,
    shared_expert: Optional[FeedForward] = None,
    router_kwargs: Optional[dict] = None,
    use_grouped_mm: bool = False,
)
```

**Forward flow:**

```text
x [bs, slen, dim]
  → flatten → [bs*slen, dim]
  → TokenChoiceTopKRouter → top_scores, selected_experts, counts
  → inline argsort (stable) → expert-major token ordering
  → GroupedExperts → expert outputs
  → scatter-add → restore token order
  → [+ shared_expert(x)]
  → view → [bs, slen, dim]
```

**Load balancing:**

- `expert_bias` buffer (`[num_experts]`) — passed to router to shift routing
  scores; updated externally via `update_expert_bias()`.
- `tokens_per_expert` buffer — accumulates expert load counts across forward
  calls; reset by `update_expert_bias()`.
- `load_balance_coeff` — when set, attaches `_load_balance_loss` to the output
  tensor for auxiliary loss computation.

```python
def update_expert_bias(moe: MoE, lr: float = 1e-3) -> None:
    """Update expert bias using sign of load deviation from mean."""
```

---

## Quick Start

```python
from hyper_parallel import init_device_mesh
from hyper_parallel.core.expert_parallel import MoE, ExpertParallel

# 1. Build MoE and EP mesh
moe = MoE(dim=4096, hidden_dim=14336, num_experts=8, top_k=2)
ep_mesh = init_device_mesh("npu", (8,), mesh_dim_names=("ep",))

# 2. Shard experts across the EP mesh — one expert per rank
ExpertParallel().apply(moe.experts, ep_mesh)

# 3. Forward pass (dispatch and combine are transparent)
output = moe(x)  # x: [batch, seq_len, dim]
```

---

## EP Parallel Styles

### `ExpertParallel`

Standard all-to-all EP: each rank holds `num_experts // ep_degree` local experts;
tokens are dispatched and gathered via differentiable all-to-all.

**Applied to:** `GroupedExperts` module.

**Mesh requirement:** 1-D mesh, e.g. `mesh_dim_names=("ep",)`.

**What it does:**

| Stage | Operation |
|-------|-----------|
| Partition | Expert weights sharded on dim 0 — `Shard(0)` via DTensor |
| Dispatch (pre-hook) | Exchange token counts (non-diff), dispatch tokens (differentiable all-to-all); rank-major → expert-major permutation |
| Combine (post-hook) | expert-major → rank-major unpermute; reverse differentiable all-to-all |

**Sharding behavior:**

| Parameter | Shape | Placement |
|-----------|-------|-----------|
| `w1` | `[num_experts, hidden_dim, dim]` | `Shard(0)` |
| `w2` | `[num_experts, dim, hidden_dim]` | `Shard(0)` |
| `w3` | `[num_experts, hidden_dim, dim]` | `Shard(0)` |

**Token dispatch protocol:**

1. Non-differentiable `all_to_all_single` — exchange `num_tokens_per_expert` counts
2. Compute `input_splits` / `output_splits` from counts
3. Differentiable `all_to_all_single` — dispatch actual tokens (gradient flows back)
4. Local `_permute` — rank-major → expert-major reordering (no extra collective)
5. `GroupedExperts` forward on local shard
6. Local `_unpermute` — expert-major → rank-major
7. Differentiable `all_to_all_single` — combine tokens back (gradient flows back)

---

### `TensorParallel`

TP-only weight sharding for expert weights — no token dispatch. Use when the EP
degree is 1 and you want to split the expert hidden dimension across ranks
(column-wise / row-wise, analogous to MLP TP).

**Applied to:** `GroupedExperts` module.

**Mesh requirement:** 1-D mesh, e.g. `mesh_dim_names=("tp",)`.

**Sharding behavior:**

| Parameter | Shard dim | Meaning |
|-----------|-----------|---------|
| `w1` | `Shard(1)` | Split `hidden_dim` (column-wise) |
| `w3` | `Shard(1)` | Split `hidden_dim` (column-wise) |
| `w2` | `Shard(2)` | Split input `hidden_dim` (row-wise) |

---

### `ExpertTensorParallel`

Combined EP + TP on a 2-D `[ep, tp]` mesh. Expert weights are doubly sharded:
dim 0 across EP ranks and the hidden dim across TP ranks. Token dispatch uses only
the EP sub-mesh.

**Applied to:** `GroupedExperts` module.

**Mesh requirement:** 2-D mesh, `mesh_dim_names=("ep", "tp")`.

**Sharding behavior:**

| Parameter | Placement |
|-----------|-----------|
| `w1` | `[Shard(0), Shard(1)]` — expert dim × hidden dim |
| `w3` | `[Shard(0), Shard(1)]` — expert dim × hidden dim |
| `w2` | `[Shard(0), Shard(2)]` — expert dim × row dim |

---

## Typical Compositions

### Standard EP (one expert per rank)

```text
Input [bs, slen, dim]
    │
    ├─► Router          ─► top_scores, selected_experts
    │
    └─► ExpertParallel
            │  dispatch (all-to-all)
            ▼
        Local experts (Shard(0))
            │  combine (all-to-all)
            ▼
        Output [bs, slen, dim]
```

### EP with multiple experts per rank

```python
# 4 ranks, 8 experts → 2 experts per rank
ep_mesh = init_device_mesh("npu", (4,), mesh_dim_names=("ep",))
ExpertParallel().apply(moe.experts, ep_mesh)  # num_experts=8, ep_degree=4
```

### EP + TP (2-D parallelism)

```text
Input [bs, slen, dim]
    │
    ├─► Router
    │
    └─► ExpertTensorParallel  (mesh: [ep=4, tp=2])
            │  dispatch on ep sub-mesh
            ▼
        Doubly-sharded experts [Shard(0), Shard(1/2)]
            │  combine on ep sub-mesh
            ▼
        Output [bs, slen, dim]
```

---

## Combining with Other Parallelism

EP styles accept the mesh dimensions they were designed for. Slice sub-meshes from
a multi-dimensional mesh for hybrid strategies:

```python
mesh = init_device_mesh(
    "npu", (dp_size, ep_size, tp_size),
    mesh_dim_names=("dp", "ep", "tp"),
)

# EP+TP on experts
ExpertTensorParallel().apply(moe.experts, mesh["ep", "tp"])

# FSDP on the DP dimension
fully_shard(model, mesh=mesh["dp"])
```

EP hooks do not interfere with FSDP unshard/reshard or pipeline micro-batch
scheduling.

---

## Platform Support

`ExpertParallel` uses `platform.differentiable_all_to_all_single` for token
dispatch and combine, keeping all collective calls behind the platform abstraction
layer.

| Platform | `differentiable_all_to_all_single` | `GroupedExperts` kernel |
|----------|------------------------------------|-------------------------|
| PyTorch (GPU) | autograd Function wrapping `dist.all_to_all_single` | `torch._grouped_mm` (when `use_grouped_mm=True`) |
| PyTorch (NPU) | same as GPU | `torch_npu.npu_grouped_matmul` (when `use_grouped_mm=True`) |
| MindSpore | `NotImplementedError` (planned) | for-loop fallback |
