# 通过公开模块族构建与训练模型

下面是可在仓库根目录直接运行的单进程用户代码。示例包拥有完整的 causal Attention、模型外置路由和 SwiGLU 专家计算；导入公开模型入口时完成族与模型 spec 的注册。使用方只构建模型并训练。该例子展示公开接入方式，使用可移植 Torch 数学；Mega/低精成员需分别实现并验收。

```python
import torch
from examples.torch.module_family import DecoderConfig, FamilyDecoder
from hyper_parallel.models import HyperAutoModelForCausalLM

config = DecoderConfig()
model = HyperAutoModelForCausalLM.from_config(
    config, model_factory=FamilyDecoder, torch_dtype=torch.float32,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
x = torch.randn(2, 4, config.hidden_size, device=next(model.parameters()).device)
loss = model(x).square().mean()
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
print(tuple(model(x).shape))  # (2, 4, 8)
```

模型包导入时安装族声明和模型 spec；使用方无需调用注册函数。单进程运行命令：

```bash
python -m examples.torch.module_family.train
```

模型作者维护 `config.py`、`model.py` 以及 `attention/`、`grouped_experts/` 中的算法和统一入口。模型只使用统一族接口表达结构。示例使用完整 causal Attention 和 SwiGLU 数学；融合算子和低精成员需要单独实现与验收。
