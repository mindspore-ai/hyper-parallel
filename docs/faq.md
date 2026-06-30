# 故障排查

## 安装问题

### Q: pip install 报错 "not a supported wheel on this platform"

**原因**：whl 文件名中的 python 版本和架构 tag 与目标环境不匹配。

**解决**：在目标环境的 Python 下重新构建 whl：

```bash
cd hyper-parallel
python setup.py bdist_wheel
pip install dist/hyper_parallel-*.whl
```

### Q: 导入时报错 "undefined Symbol"

**原因**：编译机和运行机的 glibc 版本不一致。

**解决**：在目标环境所在的 OS 发布镜像内重新构建 whl。例如部署在 OpenEuler 22.03 环境时，需要在该镜像内编译。

### Q: MindSpore custom_ops 编译失败

**原因**：`ASCEND_HOME_PATH` 未正确设置，或 MindSpore 版本 < 2.8。

**解决**：

```bash
# 设置 CANN 路径
export ASCEND_HOME_PATH=/usr/local/Ascend/cann
source /usr/local/Ascend/cann/set_env.sh

# 确认 MindSpore 版本
python -c "import mindspore; print(mindspore.__version__)"
```

### Q: PyTorch 扩展构建失败

**原因**：PyTorch 版本 < 2.7 或 CXX11 ABI 不匹配。

**解决**：确保 PyTorch ≥ 2.7 且 `_GLIBCXX_USE_CXX11_ABI=1`（官方 wheel 默认满足）。

```bash
python -c "import torch; print(torch.__version__); print(torch._GLIBCXX_USE_CXX11_ABI)"
```

---

## 运行时问题

### Q: DTensor redistribute 性能差

**原因**：redistribute 缓存未命中，每次 redistribute 都重新计算 layout。

**解决**：确保 `compact_str + rank_id` 缓存生效。避免在训练循环内频繁创建新的 Layout 对象。

### Q: FSDP 在 tp>1/dp=1 场景下报错

**原因**：FSDP 在 meta 上下文中构造 device mesh 时，tp>1/dp=1 场景下 mesh 初始化异常。

**解决**：此问题已在 v1.0.0 修复。确保使用最新版本。

### Q: HSDP overlap 模式下梯度不正确

**原因**：梯度同步流与计算流冲突。

**解决**：使用 `hsdp_sync_stream` 正确管理梯度同步流。

### Q: Pipeline Parallel 某些 stage 没收到数据

**原因**：PP stage index 配置错误，或 micro_batch_num 不匹配。

**解决**：

1. 检查 `PipelineStage(submodule, stage_index, stage_num)` 参数是否正确
2. 确保所有 stage 的 `stage_num` 一致
3. MindSpore 后端确保 batch_size 整除 micro_batch_num

### Q: overlap_b_f 导致死锁

**原因**：MoE 层数不对称，FWD/BWD chunk 层数不一致导致 barrier 死等。

**解决**：确保 FWD/BWD chunk 层数一致。不一致时需要在装钩时按短边对齐。

### Q: Activation Swap 后精度下降或行为异常

**常见原因**：

- 未设置 swap 预取（须调用 `SwapManager.set_forward_prefetch_layer`），或在 PP 场景使用 swap，导致 offload 未按预期执行。
- 小 tensor 的 DMA 开销大于收益，或 `policy_fn` 误配。
- 未在 compute stream 访问前调用 `wait_load` / `wait_offload`（见 [Activation Checkpoint 指南](./guide/activation_checkpoint.md)）。

**解决**：使用 `policy_fn` 过滤小 tensor，只 swap 大激活；并确保层间 prefetch 链已注册。

```python
from hyper_parallel.core.activation_checkpoint import CheckpointPolicy, SwapManager

def swap_policy(tensor):
    if tensor.numel() < 1024:
        return CheckpointPolicy.MUST_SAVE   # 小 tensor 不 swap，保留在设备上
    return CheckpointPolicy.MUST_SWAP       # 大激活 offload 到 host

for i in range(len(model.layers) - 1):
    SwapManager().set_forward_prefetch_layer(model.layers[i], model.layers[i + 1])
```

### Q: Context Parallel 在 MindSpore 上不工作

**原因**：DSA 系列 async CP 目前仅 PyTorch 后端支持。

**解决**：MindSpore 后端使用基础 `ContextParallel`，DSA 系列为 noop 占位。

---

## 通信问题

### Q: HCCL 通信超时

**原因**：进程组初始化问题或网络拓扑配置错误。

**解决**：

```python
from hyper_parallel import init_process_group

# 确保正确初始化进程组
init_process_group(backend="hccl")
```

检查 HCCL 配置：

- `HCCL_WHITELIST_DISABLE=1`
- `HCCL_CONNECT_TIMEOUT=1800`

### Q: Gloo backend 在 NPU 场景下报错

**原因**：Gloo backend 不支持 NPU 通信。

**解决**：NPU 场景使用 HCCL backend。Gloo 仅用于 CPU 级别测试。

---

## 内存问题

### Q: OOM（Out of Memory）

**常见原因与解决**：

| 原因 | 解决 |
|------|------|
| 模型太大 | 使用 FSDP（fully_shard）切分参数/优化器状态 |
| 序列太长 | 使用 CP（ContextParallel）切分序列维度 |
| micro_batch_num 太多 | 减少 micro_batch_num 或使用 Activation Swap |
| PP stage 不平衡 | 使用 SAPP-PPB 自动平衡 |

### Q: FSDP 下 optimizer 内存仍然很高

**原因**：optimizer 状态未正确分片。

**解决**：确保使用 `fully_shard` 后再创建 optimizer。FSDP 会自动将 optimizer 状态分片。

```python
from hyper_parallel.core.optimizer import get_hyper_optimizer

model = fully_shard(model, mesh=mesh)  # 先 FSDP
optimizer = get_hyper_optimizer(...)    # 后创建 optimizer
```

---

## 调试技巧

### 1. 检查进程组状态

```python
from hyper_parallel import get_backend, get_process_group_ranks

print(get_backend())
print(get_process_group_ranks(group))
```

### 2. 检查 DTensor layout

```python
# 查看分布式张量的 layout 信息
print(dtensor.layout)
print(dtensor.placements)
print(dtensor.to_local().shape)
```

### 3. 确定性模式调试

启用确定性模式进行精度调试：

```python
# 启用确定性可帮助发现 SDC 和精度不一致问题
```

### 4. 检查 FSDP shard 状态

```python
# 查看 FSDP 分片后的参数状态
for name, param in model.named_parameters():
    if isinstance(param, DTensor):
        print(f"{name}: mesh={param.layout.mesh}, placements={param.placements}")
```