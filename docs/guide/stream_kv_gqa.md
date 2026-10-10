# GQA 的流式 KV 上下文并行

`StreamKVGQAAttention` 在 Ulysses 与 KV-owner 两个子组之间执行同步 GQA attention。它把全量 KV AllGather 和全量 dKV 中间量替换为有界 panel，供需要控制长序列层内显存的调用方显式选择。

## 支持范围与接入

当前后端支持 Ascend NPU、BF16 输入、B=1、head dimension=256、等长连续分片和单一完整 causal 序列。支持一阶反向、非 reentrant activation checkpoint 和 GQA 所需的最小 KV 复制。不支持 TND、padding、自定义 mask、滑窗、dropout、KV cache 或高阶导；TP/FSDP 组合尚未验证。

下面以独立 CP 作业为例。调用方已有 `GQAAttention` 或 `GatedGQAAttention` 组件，并负责创建 mesh、切分输入和计算原始全局位置的 RoPE：

```python
from hyper_parallel import init_device_mesh
from hyper_parallel.distributed.context_parallel import (
    StreamKVConfig,
    StreamKVGQAAttention,
)

mesh = init_device_mesh(
    "npu", (kv_degree, ulysses_degree), mesh_dim_names=("kv", "u")
)
gated_gqa.attention_interface = StreamKVGQAAttention(
    mesh["u"], mesh["kv"],
    StreamKVConfig(
        owner_panel_tokens=4096,
        key_block_tokens=65536,
        query_head_chunk=4,
        causal_load_balance=False,
    ),
)
```

两个子 mesh 必须来自同一个 `(KV owner, Ulysses lane)` 布局，输入按该父 mesh 的顺序连续分片。各 rank 的配置、局部长序列长度和头数必须一致。接口复用已有 process group，不创建新的组。Pu 必须整除 Hq；Hq 必须是 Hkv 的整数倍；Hkv/Pu 或 Pu/Hkv 必须为整数。

投影、Q/K norm、RoPE、gate、O projection 和参数梯度归约仍由模型与训练框架负责。attention 返回原来的 token 顺序，因此不要求其他层使用镜像布局。调用 `.attention(q, k, v)` 时输入/输出为 BNSD；组件回调返回 BSND 和 `None` attention weights。

## 通信与计算粒度

设 Ulysses 前每卡长度为 s，两个子组大小为 Pu/Pg，全局长度为 S。Ulysses 后本地长度 `L=Pu·s=S/Pg`，本地 Q/KV 头数记为 hq/hk。

| 参数 | 含义 |
|---|---|
| `owner_panel_tokens=t` | 每轮每个 KV owner 提供的 token 上限；总 panel 长度 `T≤Pg·t` |
| `key_block_tokens=W` | 单次 FA 的 key 长度上限；一个通信 panel 可拆成多次 FA |
| `query_head_chunk=h` | 单次 FA 在一个 compact KV-head group 内处理的 Q 头数；`None` 表示处理整个 group |
| `causal_load_balance` | 默认 `False`；设为 `True` 时镜像 Q/K/V，并配对两个半片的条带 |

h 控制 post-U attention 的临时量，完整 Q 和最终 A 仍保存。它不是 Ulysses 设备数，也不是投影级 head chunk。Pu≤Hkv 时切分 KV heads；Pu>Hkv 时只复制到满足 Ulysses 分头所需的数量。

前向采用 KV panel 外层循环：

1. 按需最小复制 K/V，执行 Q/K/V 的 Ulysses seq-to-head 交换。
2. 从每个 owner 收取一个 KV 条带，形成共享 panel。
3. 遍历 Q-head stages，再将可见 KV 区间按 W 切分。
4. 用 FP32 online softmax merge 得到全局归一化的 A/max/sum。
5. 恢复布局并执行 head-to-sequence 交换，返回原位置的输出。

反向重新扫描相同的 KV panels。每次 native FA VJP 使用最终全局 A/max/sum，不能使用该小块自己的归一化；dQ 与当前 panel 的 dK/dV 均以 FP32 累加。完成所有 head stages 后，以 FP32 SUM ReduceScatter 把 dKV 归还 owner，随后逆布局、逆 Ulysses、FP32 KV 副本 SUM，最后才转换为输入 dtype。

KV-owner SUM、KV 副本 SUM 和参数 SUM 的来源不同。本接口只负责前两者；框架必须正确安排参数归约，不能额外对本地模型 dX 做 CP SUM。checkpoint 会重跑前向的 KV 扫描，需将重计算通信计入成本。

## 因果负载均衡

开启均衡后，将全局序列划成 2Pg 个等长半片，每片 `m=L/2`；owner g 持有半片 g 和 `2Pg−1−g`。例如 Pg4 时为 `[0,7]、[1,6]、[2,5]、[3,4]`。当前实现通过已有 KV mesh 的 AllToAllV 镜像 Q/K/V，并在退出 attention 前恢复输出；反向应用对应排列和逆排列。

要求 post-U L 为偶数、t≥2。每轮从每个半片取相同的相对区间 `[a,a+w)`，其中 `w=min(floor(t/2),m-a)`。t 是两半合计预算；奇数 t 少用一个 token。每个 owner 发送 2w 个 token，每次 AG 收到所有半片对应条带，`T=2Pg·w`。小写 w 与 FA 上限 W 不同。

FA 输入按所有 early 条带、所有 late 条带排列；可见性由全局半片位置决定：

| 本地 Q | FULL KV | CAUSAL KV | 跳过 |
|---|---|---|---|
| early 半片 g | early owner <g | 自己的 early 条带 | 其余 |
| late 半片 2Pg−1−g | 全部 early，以及 late owner >g | 自己的 late 条带 | late owner <g |

FULL 使用 `sparse_mode=0` 且无 mask；own-CAUSAL 使用 `sparse_mode=4`、对应的 query suffix 和 `next_tockens`。mask 固定为 2048×2048 压缩格式，不分配 S² mask。不连续 panel 不能直接套普通三角 mask。

每个 owner、每个 Q head 在一轮完整 panel 上的有效 Q–K 对数为：

```text
own(a,w) = w(m-a) - w(w-1)/2
early(g) = g·m·w + own(a,w)
late(g)  = (2Pg-1-g)·m·w + own(a,w)
total(g) = (2Pg+1)·m·w - 2a·w - w(w-1)
```

总数不含 g，所以每轮完成两半 Q 后面积均衡；head stage 有 h 个头时统一乘 h。仅计算一半 Q，或只收取某一对 KV 半片，没有这个保证。FA 形状、调用次数、packing 和启动成本仍可能不同，因此面积相等不代表运行时间或峰值相等。

两个 Q 半片可能同时贡献给同一 KV 条带。实现将这些贡献累加到同一个 FP32 owner-major dKV panel 后统一 RS；不能让第二次输出覆盖第一次。所有 rank 保持相同 collective 次序，即使本地跳过某些未来条带的 FA。

## 显存收益与代价

每 lane 有 hk 个 KV heads、维度 D 时，一份 BF16 KV 占 `4·S·hk·D` bytes，一份 FP32 dKV 占 `8·S·hk·D` bytes。流式方案把这里的 S 替换为 T。例如 hk1/D256 下，S1Mi/10Mi 的两项合计为 3/30GiB，T256Ki 则为 768MiB。

这只是两项中间量，不是整层峰值。还需计入 Q/A、owned KV/dKV、统计量、FA workspace、打包/解包和模型投影。固定 t 时 T 随 Pg 增大；大 CP 必须按总 panel 预算反算 t。增大 T 可以减少 AG/RS 启动，同时增加 BF16 接收量与 FP32 dKV 缓冲。

相对于前向保留全量 KV 供反向复用的基线，本实现反向多扫描一次 KV，即从一次 AG＋一次 FP32 RS 变为两次 AG＋一次 FP32 RS。小 W/h 还会增加 FA 调用、dQ 重复写出、FP32 累加和 merge 成本。镜像模式额外支付 attention 内部的布局交换。

开发阶段在 910B3×8、Pu2/Pg4、Hq32/Hkv2/D256、t2048/W8192/h4 上比较过连续与镜像接口。一次预热、三次测量，取 max-rank 完整 F+B 耗时的中位数：

| Ulysses 前每卡长度 | 连续耗时/峰值 | 镜像耗时/峰值 |
|---|---|---|
| 2Ki | 66.46 ms / 516.03 MiB | 80.71 ms / 504.01 MiB |
| 8Ki | 849.97 ms / 1748.01 MiB | 718.39 ms / 2004.01 MiB |

计时包含布局交换、Ulysses、AG/RS 和 attention 前后向，不包含模型投影、参数归约与优化器。峰值为 allocated peak，包含输入和上游梯度。数据说明均衡不是所有形状都更快或更省显存；它不代表完整模型或目标超节点大 CP 的性能保证。

## 验证与后续工作

```bash
python -m pytest -q tests/ut/distributed/context_parallel/test_stream_kv.py \
    tests/ut/distributed/context_parallel/test_stream_kv_layout.py
python -m pytest -q tests/torch/context_parallel/test_stream_kv.py
```

CPU 测试使用独立全局 FP64 autograd 检查前向、全局归一化 partial VJP、owner 归属、排列/逆排列和副本 SUM，并检查逻辑 Pg1280 的调度计数。它们不替代实际 collective 验证。

分布式测试需空闲 Ascend 设备，覆盖 CP8/Pu1,2,4,8、CP6/Pu2/Pg3、连续/镜像、KV 复制、尾 panel 和非整除 head chunk。组件测试使用实际 GatedGQA，检查 RoPE、AMP、三步 SGD、Torch/Hyper checkpoint 的输出/梯度和分配增长。数值同时要求 max-abs≤0.12、relative-L2≤0.035；测试直接断言，不依赖历史结果文件。开发验证环境为 Torch 2.7.1、torch_npu 2.7.1.post4、CANN 9.0。

后续先固定 post-U 形状、W/h 和精度，扫描 T 的完整耗时与峰值；再根据实际成本推进以下工作：

- 分离通信与计算 panel，按需解包，并独立选择 RS 粒度。
- 保持 KV 连续 owner，只镜像 Q，或将布局交换与 Ulysses 合并。
- 根据可用显存和等待时间组合有界异步、stage offload 与投影重计算。
- 减少 native FA 内部重复 dQ 写出、packing 和 workspace。

这些优化需分别验证精度、buffer 生命周期和端到端收益。当前公开接口保持同步实现。
