# Troubleshooting

**English** | [简体中文](faq.zh.md)

## Installation Issues

### Q: pip install reports "not a supported wheel on this platform"

**Cause**: The Python version and architecture tags in the wheel filename do not match the target environment.

**Solution**: Rebuild the wheel with Python in the target environment:

```bash
cd hyper-parallel
./build.sh
# Install the exact wheel path printed by build.sh.
wheel_path=/absolute/path/printed/by/build.sh
pip install "${wheel_path}"
```

If you only need a core-only wheel without optional native components, use
`./build.sh --multicore off`. The indexed Dataset helper is a base component and has no separate disable switch.

### Q: Import fails with `GLIBC_2.xx not found`

**Cause**: The runtime glibc version is older than the version required by the wheel's ELF binaries.

**Solution**: Rebuild the wheel in an OS image whose glibc baseline does not exceed that of the target runtime.

### Q: Importing a native component fails with `undefined symbol`

**Cause**: The native libraries loaded into the process do not match the Python, framework, CANN, or CXX11 ABI combination.

**Solution**: Record the full symbol name and the actual loaded `.so` path. Check the wheel's Python/architecture tags,
framework/CANN compatibility, and consistency with the CXX11 ABI required by the framework.

### Q: Building a PyTorch extension fails

**Cause**: The PyTorch, torch_npu, and CANN versions are incompatible, or the CXX11 ABI does not match.

**Solution**: Use PyTorch/torch_npu versions compatible with the current CANN, and build the native adapter using the
`compiled_with_cxx11_abi()` value reported by the current PyTorch. This may be 0 or 1 and must match the actual framework.

```bash
python -c "import torch; print(torch.__version__); print(torch.compiled_with_cxx11_abi())"
```

---

## Runtime Issues

### Q: DTensor redistribute is slow

**Cause**: Cache misses cause the layout to be recomputed on every redistribution.

**Solution**: Ensure the `compact_str + rank_id` cache is effective. Avoid repeatedly creating new Layout objects inside the training loop.

### Q: FSDP fails when tp>1/dp=1

**Cause**: Device mesh initialization fails when FSDP constructs the mesh in a meta context with tp>1/dp=1.

**Solution**: This issue was fixed in v1.0.0. Make sure you use the latest version.

### Q: Gradients are incorrect in HSDP overlap mode

**Cause**: The gradient synchronization stream conflicts with the compute stream.

**Solution**: Manage the gradient synchronization stream correctly with `hsdp_sync_stream`.

### Q: Some pipeline parallel stages receive no data

**Cause**: The PP stage index is configured incorrectly, or micro_batch_num values do not match.

**Solution**:

1. Check the arguments to `PipelineStage(submodule, stage_index, stage_num)`.
2. Ensure every stage uses the same `stage_num`.
3. Ensure batch_size is divisible by micro_batch_num.

### Q: overlap_b_f causes a deadlock

**Cause**: Asymmetric MoE layer counts and mismatched FWD/BWD chunk layer counts cause an indefinite barrier wait.

**Solution**: Ensure FWD/BWD chunks have matching layer counts. If they differ, align hook registration to the shorter side.

### Q: Numerical accuracy degrades or behavior changes after Activation Swap

**Common causes**:

- Swap prefetch was not configured (call `SwapManager.set_forward_prefetch_layer`), or swap is used with PP, causing offload to behave unexpectedly.
- DMA overhead for small tensors outweighs the benefit, or `policy_fn` is misconfigured.
- `wait_load` / `wait_offload` was not called before compute-stream access; see the [Activation Checkpoint Guide](./guide/activation_checkpoint.md).

**Solution**: Filter out small tensors with `policy_fn` and swap only large activations. Ensure the inter-layer prefetch chain is registered.

```python
from hyper_parallel.core.activation_checkpoint import CheckpointPolicy, SwapManager

def swap_policy(tensor):
    if tensor.numel() < 1024:
        return CheckpointPolicy.MUST_SAVE   # Keep small tensors on the device.
    return CheckpointPolicy.MUST_SWAP       # Offload large activations to the host.

for i in range(len(model.layers) - 1):
    SwapManager().set_forward_prefetch_layer(model.layers[i], model.layers[i + 1])
```

## Communication Issues

### Q: HCCL communication times out

**Cause**: A process group initialization issue or an incorrect network topology configuration.

**Solution**:

```python
from hyper_parallel import init_process_group

# Initialize the process group correctly.
init_process_group(backend="hccl")
```

Check the HCCL configuration:

- `HCCL_WHITELIST_DISABLE=1`
- `HCCL_CONNECT_TIMEOUT=1800`

### Q: The Gloo backend fails with NPUs

**Cause**: The Gloo backend does not support NPU communication.

**Solution**: Use HCCL for NPU communication. Use Gloo only for CPU tests.

---

## Memory Issues

### Q: OOM (Out of Memory)

**Common causes and solutions**:

| Cause | Solution |
|------|------|
| Model is too large | Shard parameters and optimizer states with FSDP (fully_shard) |
| Sequence is too long | Partition the sequence dimension with CP (ContextParallel) |
| Too many micro-batches | Reduce micro_batch_num or use Activation Swap |
| Unbalanced PP stages | Use SAPP-PPB for automatic balancing |

### Q: Optimizer memory usage remains high with FSDP

**Cause**: Optimizer states are not sharded correctly.

**Solution**: Create the optimizer after applying `fully_shard`. FSDP automatically shards optimizer states.

```python
from hyper_parallel.core.optimizer import get_hyper_optimizer

model = fully_shard(model, mesh=mesh)  # Apply FSDP first.
optimizer = get_hyper_optimizer(...)    # Then create the optimizer.
```

---

## Debugging Tips

### 1. Check Process Group State

```python
from hyper_parallel import get_backend, get_process_group_ranks

print(get_backend())
print(get_process_group_ranks(group))
```

### 2. Inspect DTensor Layouts

```python
# Inspect distributed tensor layout information.
print(dtensor.layout)
print(dtensor.placements)
print(dtensor.to_local().shape)
```

### 3. Debug in Deterministic Mode

Enable deterministic mode to debug numerical accuracy:

```python
# Determinism helps detect SDC and numerical inconsistencies.
```

### 4. Inspect FSDP Shard State

```python
# Inspect parameter state after FSDP sharding.
for name, param in model.named_parameters():
    if isinstance(param, DTensor):
        print(f"{name}: mesh={param.layout.mesh}, placements={param.placements}")
```
