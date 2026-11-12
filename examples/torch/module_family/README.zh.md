# 通过公开模块族构建与训练模型

[English](README.md) | **简体中文**

本示例包含完整的 causal Attention、模型拥有的路由和 SwiGLU 专家计算。导入公开模型包时，包内会注册模块族和模型 spec。用户只需通过公开接口构建模型并训练，无需调用注册或模型准备函数。

示例使用可移植的 Torch 计算和 FP32。Mega、低精度成员需要分别实现并完成数值验收。

使用与本仓库兼容的 PyTorch 环境，并通过 `pip install -e .` 将当前 checkout 安装为 editable 模式。

## 单进程训练

在仓库根目录执行以下 Python 代码。直接构造普通 `nn.Module`，内部公开族按默认条件选择单卡成员。

```python
import torch
from examples.torch.module_family import DecoderConfig, FamilyDecoder

config = DecoderConfig()
model = FamilyDecoder(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
x = torch.randn(2, 4, config.hidden_size, device=next(model.parameters()).device)
loss = model(x).square().mean()
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
print(tuple(model(x).shape))  # (2, 4, 8)
```

## 并行训练

模型和公开族可直接接收作业已有的完整 `mesh`，无须额外的装配调用。假设作业已经初始化进程组并选定当前设备：

```python
import torch

from examples.torch.module_family import DecoderConfig, FamilyDecoder
from hyper_parallel.distributed.mesh import MeshContext

# 四 rank：dense DP2×CP2，专家视图 EP2。
job_mesh = MeshContext(dp_size=2, cp_size=2, ep_size=2,
                       dp_shard_size=4, edp_shard_size=2)
job_mesh.build_meshs(device_type="npu", world_size=4)
with torch.device("npu"):
    model = FamilyDecoder(DecoderConfig(), mesh=job_mesh)
```

模型内部使用 `CausalAttention(config=config, mesh=mesh)` 和 `GroupedExperts(config=config, mesh=mesh)`。族按实际 CP/EP 域选出具体类，在返回前处理组件计算切分并借用已有通信资源；`mesh` 不进入静态请求或 `state_dict`。meta 参数保留 meta，离线逻辑 mesh 只记录布局，框架在运行时接续。

对应并行规模大于 1 时，族要求 mesh 中存在名称严格为 `cp`、`ep` 的轴。CP 从 `device_mesh` 获取；EP 优先使用 `fsdp_moe_mesh`，未提供专家视图时使用 `device_mesh`。成员构造前校验轴名与规模，离线逻辑 mesh 也遵循同一规则。

`HyperAutoModelForCausalLM.from_config(config, model_factory=FamilyDecoder, distributed_setup=setup)` 是可选托管入口。完整可运行的 [train.py](train.py) 初始化作业并使用该入口，继续接入主仓 FSDP2、物化和训练：

```bash
python -m examples.torch.module_family.train
torchrun --standalone --nproc-per-node=4 -m examples.torch.module_family.train --cp-size 2
torchrun --standalone --nproc-per-node=4 -m examples.torch.module_family.train --ep-size 2
torchrun --standalone --nproc-per-node=4 -m examples.torch.module_family.train --cp-size 2 --ep-size 2
```

四卡训练使用主仓 HyperParallel FSDP2。优化器 step 沿用现有 Trainer 的 `SkipDTensorDispatch(no_skip={torch.zeros_like})` 约定，使 Adam 状态保留 DTensor 元数据；接入已有 Trainer 时由 Trainer 执行这一步。CP/EP 成员不额外归约参数梯度。

CP2+EP2 复用四个 rank：dense 拓扑为 DP2 x CP2，EP2 是同一域中的专家视图。

构建顺序如下：

```text
导入公开模型包
  → 注册模块族声明和模型 spec
  → 直接构造模型，或由 facade 将已有 mesh 传入构造器
  → 族选定成员，应用组件计算布局并借用 CP/EP 资源
  → 框架校验共享关系，复用族已完成的切分和绑定
  → 处理模型其余参数布局
  → 应用主仓 FSDP2
  → 物化或初始化权重
  → 创建优化器并训练
```

族声明由模块拥有者放在定义文件中，经 Python import 生命周期执行一次。用户脚本不调用 `register_*()`。

## 成员选型

族根据已声明的 target 入参和作业已有的 CP/EP 配置，直接选出一个具体的 `nn.Module`。直接构造与模块替换复用同一个选型函数。`kernel`、`precision`、`backend` 等参数由需要它们的族声明，每个支持的参数组合直接对应具体成员。

本示例中每个族只有一种 Torch 算法。作业使用默认配置时，`CausalAttention` 返回 `CausalAttentionModule`，`GroupedExperts` 返回 `GroupedExpertsModule`；CP2 选择 `CPCausalAttention`，EP2 选择 `EPGroupedExperts`。因此这两个示例入口无需 `backend` 参数。直接传 mesh 时在构造阶段完成成员选择。框架仍负责全局共享关系、FSDP2、优化器和 DCP；族不执行额外的参数梯度归约。

## 归属与目录结构

```text
module_family/
├── __init__.py                      公开导出；加载模块族与模型声明
├── config.py                        算法尺寸与结构约束
├── model.py                         模型组装、router、模型 spec
├── attention/
│   ├── family.py                    统一入口、成员选择与布局声明
│   ├── attention.py                 完整默认算法与 identity 阶段
│   └── cp_attention.py              继承默认算法，交换 K/V 与序列坐标
├── grouped_experts/
│   ├── family.py                    统一入口、专家布局声明
│   ├── grouped_experts.py           dispatch/group_experts/combine 与 SwiGLU 计算
│   └── ep_grouped_experts.py        继承默认算法，将运输交给 EPContext
└── train.py                         作业入口、mesh、主仓 FSDP2、训练循环
```

模型只调用公开的族接口。模块拥有者维护族声明，模型拥有者维护模型 spec；导入公开包时加载二者。

## 适用范围与验证

CP 示例使用等长、连续序列分区及 causal mask，不包含 RoPE、KV cache 或 packed 文档 Attention。专家采用完整 SwiGLU 数学，per-slot bmm 用于可移植示范，不声称融合算子性能。

[分布式验收用例](../../../tests/torch/distributed/test_module_family_public_example.py) 分别直接构造和通过公开 facade 构建 CP2、EP2 和 CP2+EP2 作业，与全局稠密参考模型比较两步输出、全局权重和 Adam 矩；另有四进程 CPU 测试直接构造时的输出与输入梯度。

上述直接构造的单进程示例使用 CPU；托管入口选择本机可用设备。分布式训练需要主仓 FSDP2 支持的 GPU/NPU。
