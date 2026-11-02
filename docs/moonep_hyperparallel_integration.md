# MoonEP 与 HyperParallel 集成

本变更为 Ascend 910C 上的 HyperParallel Expert Parallel 增加可选 MoonEP transport。默认路径保持不变；设置 `HP_MOONEP_EP=1` 后，`build_ep_compute` 通过 `moonep_routed_forward` 完成路由 token 的 dispatch、专家计算和 combine。

## 运行时依赖

MoonEP 必须使用与当前 `torch`、`torch_npu` 和 CANN 匹配的安装包。按照 MoonEP 的 910C 文档构建并安装 `moonep` wheel 后，HyperParallel 直接导入：

```python
from moonep import Buffer, MoonEPCommPlan
from npu_patch import ShmemRuntime
```

HyperParallel 不修改 `sys.path`，也不读取源码目录环境变量。缺少安装包时，只有实际进入 MoonEP 路径才报错；普通 HyperParallel import 不需要 MoonEP。

910C 的推荐环境变量：

```bash
export MOONEP_BACKEND=npu
export ASCEND_MOONEP_HEAP=shmem
export HP_MOONEP_EP=1
export HP_MOONEP_PLANNER=host
export HP_MOONEP_STRICT_HOST_PLANNER=1
```

910C 使用主机 planner；SIMT device planner、UDMA/URMA 不属于本适配范围。

## 代码边界

- `moonep_ep.py`：MoonEP Buffer 生命周期、host plan、autograd dispatch/combine 和 route-weight 传递。
- `moonep_plan.py`：向量化 Python planner，优先调用已安装包的 C++ host planner。
- `moonep_forward.py`：路由适配、冗余专家权重 gather、局部专家计算和 token-major combine；不等长权重表的 packed gather 由 `HP_MOONEP_PACKED_GATHER=1` 显式开启，默认关闭。
- `moe.py`：抽出 `_routed_experts` seam，使 transport 可以替换 token movement。
- `recipes.py`：在 MoonEP 分支中保留专家模块的 forward hook/FSDP 边界。
- `experts.py`：处理 DTensor 本地化及 `[in, out]` / `[out, in]` 两种专家权重布局。
- `hsdp_param.py`：支持 MoonEP 路径未执行普通专家 forward 时的参数回退，并保持单 rank storage 刷新和梯度清理语义。

## 验证

新增 UT 覆盖 planner 不变量、缓存共识、dispatch/combine 索引、路由权重梯度、不等长权重表和 scratch 隔离。真实 Ascend EP8 检查覆盖 host planner、强偏斜冗余专家、dispatch/combine round-trip、完整权重行梯度以及小型 SwiGLU + SGD 更新。

MoonEP 路径是显式开关，不改变默认 all-to-all。性能候选 packed gather 在 mock/LIMO、4k/16k、micro batch 2、global batch 16 的 40 次配对中均未取得收益，因此保持默认关闭。
