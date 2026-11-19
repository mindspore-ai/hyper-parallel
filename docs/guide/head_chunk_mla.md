# Head-chunk MLA（实验性、显式启用）

`HeadChunkMLA` 包装已经构造并加载权重的 `MLAAttention`，复用原参数和 state-dict 名称。
它不修改原生 MLA，也不默认替换模型。DeepSeek-V3 可通过显式 recipe 接入现有 TextTrainer，
模型特有的权重转换、CP 接线和 packed 元数据由 adapter 处理，不修改通用 Trainer。

## 适用范围

- Ascend NPU 原生 `npu_fusion_attention` / `npu_fusion_attention_grad`，与单卡 MLA 使用相同 FA；只声明已验证的 BF16 计算路径。
- `[batch, local_sequence, hidden]`，各 CP rank 等长、按全局 token 顺序连续切分。
- 标准 MLA：Q-LoRA、compressed KV、单头 K-RoPE、half RoPE、causal、无 bias/dropout/gate。
- CP1 或一维 Ulysses CP；每个 stage 的全局 head 数必须能整除 CP size，允许尾 stage。
- Attention 整体或它的外层模块由 FSDP 管理；不能单独包裹内部 Linear/norm。
- 只支持一阶训练反向；启用 offload 时，每次 forward 的保存态只能消费一次。
- packed 支持 batch=1、全局累计文档末端和文档内重置的位置；原生 FA 前反向使用 TND。

不支持任意 padding/custom mask、KV cache、TP、Ring/混合 CP、gate、MOME、sink、
滑窗、interleaved RoPE、量化投影和高阶导数。不要在未验证的输入上依赖隐式回退。

## 使用方式

```python
from hyper_parallel.components.modules import HeadChunkMLA
from hyper_parallel.core.activation_memory import checkpoint_exclude_wrapper

# native_mla 已经完成原生 MLA 构造和权重加载，尚未被 CP/TP/子 FSDP 包装。
attention = HeadChunkMLA(
    module=native_mla,
    head_chunk_size=32,       # 全局 head 数，不是每 rank 的 head 数
    cp_mesh=cp_mesh,
    offload_saved_stats=True,
    prefetch_slots=1,        # 可选 1/2；不是序列长度调优规则
    max_host_bytes=2 * 1024**3,
    overlap_backward=False,
)
# 若外层使用非 reentrant checkpoint，避免再次重放整个 Attention。
attention = checkpoint_exclude_wrapper(attention)
```

随后在 Attention 或其外层应用 FSDP，再创建优化器。已有参数引用和参数名称不改变。
若需要从 Hugging Face 原始布局转换权重，先完成原生 `MLAAttention` 的转换，再包装本模块；
本模块不是新的 HF checkpoint 转换器。

CP>1 时，参数梯度的外层归约必须覆盖 CP 组。本模块的 KV-B 梯度是各 head owner 的稀疏贡献；
激活梯度 ReduceScatter 不能替代参数归约。随附示例使用同轴 CP/FSDP mesh 和 FP32 参数/归约。
没有外层参数归约的“各 rank 独立更新”不受支持；不要额外手动叠加一次参数 AllReduce。

`overlap_backward=True` 只改变反向提交顺序：dO 通信覆盖 dW_o，逆 dQ 通信覆盖 owner KV VJP。
它默认关闭，不保证在所有尺寸或设备上加速。offload 与通信可能竞争带宽。

## DeepSeek-V3 与 TextTrainer

将 [head_chunk_mla.yaml](../../hyper_parallel/models/deepseek_v3/recipes/head_chunk_mla.yaml)
作为片段合并到已有的完整 DeepSeek-V3 配置，保留用户自己的模型、数据、优化器和训练参数。
该片段本身不是可独立运行的训练配置，也不会下载权重或数据。

其中 `replace.kind: module` 选择 `DeepseekV3HeadChunkMLA`：复用 Q-B、KV-B、O 与 norm 参数，
声明 HF `q_a_proj`/`kv_a_proj_with_mqa` 到 `linear_qkv` 的可逆转换。
直接包装已加载 HF 模块时也会复制这两份权重，不会留下随机初始化的融合权重。
`replace.kind: method` 只给组件绑定 CP mesh，不能再叠加原生 `mla_cp_wrapper`。

使用以下入口注册显式的 HF kernel 名称，然后执行原有 TextTrainer：

```bash
torchrun --nproc-per-node=4 examples/torch/head_chunk_mla/train.py /path/to/complete_config.yaml
```

模型配置使用 `attn_implementation: head_chunk_mla_native`，由原生 FA 负责 causal/packed mask，
避免 HF 预先生成 CP 本地 dense mask。已有自定义启动脚本也可在构造模型前调用
`register_head_chunk_mla_attention()`；普通配置不导入、不启用这一注册。

源模型须符合上述 half-RoPE 等约束；不要为了通过检查直接改变已训练模型的 RoPE 语义。
保留 BF16 计算、FP32 main parameter/梯度规约，FSDP 不设置输出强制 dtype。
外层重计算须排除 Attention，可独立重计算 MLP；子投影单独 FSDP 不受支持。

### Packed 数据

在完整配置中添加以下数据接线，并使用 batch=1、`labels_are_shifted: true` 的 Indexed 数据。
`eod_token_id` 必须与实际数据一致；下面的 `1` 只是配置示例。

```yaml
dataloader:
  collate_fn:
    _target_: hyper_parallel.models.deepseek_v3.adapter.data.packed_mla.DeepseekV3PackedIndexedCollator
    eod_token_id: 1
  get_batch:
    _target_: hyper_parallel.data.batching.TextParallelBatch
    source_type: indexed
    attention_mode: compressed
    reset_position_ids: true
    runtime_input_adapter:
      _target_: hyper_parallel.models.deepseek_v3.adapter.data.packed_mla.DeepseekV3PackedMLARuntime
```

数据流为：完整样本边界 → 按序列连续切分 CP → 本地位置 ID → 全局累计文档末端 → 每个 stage 的 TND FA。
例如全局长度 16、CP4、文档边界 `[0, 3, 11, 16]`，每个 rank 都向 Attention 传入
`actual_seq_len=(3, 11, 16)`，不能传本地截断后的边界。文档可以跨 CP rank。
模块不接收额外 dense mask；同一调用的正反向保持相同的全局文档边界。

Indexed collator 把输入为 EOD 处的下一 token target 写为 `-100`，阻止学习跨文档预测；
标签已经 shift，不再 shift 第二次。Trainer 有效 token 计数与 loss 因而读取同一份标签。
如果上游已按文档准备好标签，可沿用已有 collator，但必须保留文档隔离与标签约定。
Online、任意 padding、自定义 mask 及完整预训练收敛不在本版验证范围内。

## 前向

1. 从本地 hidden 生成 Q latent、KV latent 和单头 K-RoPE，执行 norm/RoPE。
2. 一次打包 AllGather 发布 normalized compressed KV 与单头 K-RoPE。
3. 每个 head stage 生成本地 Q，并通过 Ulysses 将 sequence 分片转换为 head owner 布局。
4. owner 只展开当前 stage 的 K-NoPE/V，拼接共享 K-RoPE，调用一次原生 FA。
5. 保存 FA 的 O/max/sum；可按 stage 异步 D2H。O 逆 Ulysses 后执行分块输出投影并累加。

输出累加保持输入 dtype（BF16 输入时为 BF16），不宣称 FP32 输出累加。

## 反向

1. 重放共享低秩投影并进行一次 compressed-KV publication；不重算 FA forward。
2. 逐 stage 重建 Q、owner K/V，预取/消费保存的 O/max/sum。
3. 计算输出投影 VJP 与 dO，调用原生 FA-grad。
4. 逆 Ulysses 返回本地 dQ；分别计算 Q-B 和 owner KV-B 的 VJP。
5. FP32 累加低秩激活梯度；最后对 KV latent、单头 K-RoPE 各做一次 SUM ReduceScatter。
6. 执行共享 norm/投影 VJP，向外层返回本地 hidden 梯度和完整形状参数梯度贡献。

`GradientEdge` 只保留 stage-local autograd 图的梯度入口，不额外持有最终投影输出引用。
autograd 必需的保存值仍由图拥有；不强行释放别名 storage。

## Offload 生命周期

只 offload O/max/sum，不缓存所有展开 Q/K/V。pool 属于模块，host lease、copy stream、slot 属于
一次调用，多个未反向的 forward 不共用 active slot。host pool 有容量上限，超限报错，不静默降级。

- copy stream 等待生产者事件后 D2H；host lease 不能在 H2D 完成前被复用。
- compute stream 在消费前等待 H2D；copy stream 在覆盖同槽下一代前等待上一代消费者事件。
- 一槽在当前 stage 结束后加载下一 stage；两槽可提前一个 stage。
- 反向/异常结束关闭调用级存储；缓存池保留可复用容量不等于活跃 lease 泄漏。
- D2H source retirement 仅针对已知在当前计算 stream 产生的私有原生 FA 输出。
  不能把这个 helper 用作任意跨 stream Tensor 的通用 offload 接口。

本版不包含跨层预取、首块提前 H2D、FA 后立即覆写槽、Q/K/V host cache，或按长度硬编码的调度。

## 验证入口与边界

CPU UT 检查独立公式的前反向、参数身份、tail stage、HF 转换、packed ABI/元数据和不支持输入。
NPU 示例见 [validate.py](../../examples/torch/head_chunk_mla/validate.py)：

```bash
HCCL_DETERMINISTIC=true torchrun --nproc-per-node=4 \
  examples/torch/head_chunk_mla/validate.py \
  --output /path/to/new-result-directory --shape deepseek \
  --sequence 32768 --chunk 32 --steps 10 --slots 1 \
  --layers 2 --micro-batches 2 --offload --overlap --checkpoint
```

先激活匹配的 CANN/torch-npu 环境。`sequence` 是全局长度；上述 CP4 为 local8K。
示例随机初始化，显式记录实际参数/输入哈希、output/dHidden/全部参数梯度、每步 loss/norm、
最终参数以及资源计数。该入口不是加载预训练权重的完整模型，也不是 Trainer/MLP/MoE 集成测试。
BF16 分 stage 和改变 KV 计算位置会改变舍入顺序；不能承诺与整头 MLA 逐元素零差。

Trainer 接线可用不等于完整 DeepSeek-V3 训练精度已验收。Dense 层和 packed 文档隔离已分别验证；
学习路由的 MoE 仍可能放大 BF16 分组差异，当前部分 router/expert 梯度未满足原精度预算。
该替换保持显式启用，不据此声明完整 MoE 预训练收敛。通信合桶、跨层预取与额外输出累加策略不包含在本版中。
