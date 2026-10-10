# DeepSeek 与 TorchTitan-NPU Ascend 高性能算子

HyperParallel 为 DeepSeek 开源的 Ascend 高性能算子及 TorchTitan-NPU 的公开生产算子提供可选适配层。适配层不会
复制上游内核，也不会在导入 `hyper_parallel` 时加载原生扩展；只有调用对应接口时才加载已安装的后端。

## 支持范围

| 上游项目 | HyperParallel 接口 | 典型用途 |
|---|---|---|
| DeepGEMM-Ascend | `deepseek_gemm` 及显式语义接口 | BF16、FP8/FP4、Grouped GEMM、Einsum、MQA logits |
| DeepSelect | `deepseek_select_topk`、`deepseek_select_stride_requirement` | DSA/路由场景 Top-K |
| FlashMLA | `deepseek_flash_mla_sparse_prefill`、`deepseek_flash_mla_sparse_decode` | DeepSeek V4.1 稀疏 MLA |
| TileKernels | `deepseek_tile_kernel` 及显式语义接口 | quant、MoE、mHC、RoPE、Engram 与 autograd modeling 等 TileLang 内核 |
| TorchTitan-NPU | `torchtitan_*` 显式语义接口 | AscendC、TileLang、Triton 训练算子 |

FlashMLA 另提供 `DeepseekV41SparsePrefillAttention` 和
`DeepseekV41SparseDecodeAttention` 两个 `torch.nn.Module` 门面。后者会复用调度元数据；输入形状变化前必须调用
`reset_scheduler()`。除张量形状外，变长 Top-K 的长度值发生变化时也要重置。修改
`enable_batch_invariant` 或执行 `.to()` 等设备/类型转换时，Module 会自动清理调度元数据。

TileLang 是 TileKernels 的编译和运行时依赖，不作为 HyperParallel 算子接口重复封装。DeepEP-Ascend 属于专家并行通信层，
且其仓库当前未提供明确许可证文件，因此本适配不会复制或分发其源码。

FlashMLA 的 dense varlen 接口和 `fused_norm_rope_attn_rope_cast` 当前只实现于 CUDA SM100/SM103，不属于本 Ascend
适配范围。Ascend 侧支持的是 V4.1 sparse prefill/decode。

### 显式接口

常用生产路径无需依赖字符串名称：

- DeepGEMM：支持四种转置布局的 `deepseek_bf16_gemm`、`deepseek_fp8_gemm`、`deepseek_fp8_fp4_gemm`，
  三类 `deepseek_grouped_*_gemm`，以及 `deepseek_einsum`、`deepseek_mqa_logits`、Mega-MoE、mHC pre-norm
  GEMM、paged MQA scheduler metadata 和 scaling-factor/weight layout transform；
- TileKernels quant：`deepseek_per_token_cast`、`deepseek_per_block_cast`、`deepseek_cast_back` 和
  `deepseek_swiglu_forward/backward`；
- TileKernels MoE/transform/Engram：只返回稳定 Top-K 索引的 `deepseek_topk_gate`，完整融合路由的
  `deepseek_moe_topk_gate` / `deepseek_moe_topk_gate_backward`，以及
  `deepseek_normalize_routing_weights`、`deepseek_apply_rotary`、`deepseek_engram_hash` 和带 autograd 的
  `deepseek_engram_gate`。

其余上游公开生产算子仍可通过 `deepseek_gemm` 或 `deepseek_tile_kernel` 调用。测试、benchmark、PyTorch reference、
TileKernels runtime 配置等开发接口不作为 HyperParallel 公共 API。

`deepseek_mega_moe` 仅转发 DeepGEMM-Ascend 的 fused Mega-MoE 算子。调用方仍负责按上游契约创建并持有
`deep_gemm.SymmBuffer`、管理 process group、保证各 rank 的分配/调用顺序一致并处理其生命周期；该接口不表示
HyperParallel 已接管 Mega-MoE 或 DeepEP 的通信资源。

### TorchTitan-NPU 接口

TorchTitan-NPU 适配覆盖其 `torchtitan_npu.ops` 中可直接调用且 HyperParallel 尚未覆盖的 17 个公开生产接口：

- AscendC：partial RoPE、MoE re-routing、token permute 和 token unpermute；
- TileLang：mHC head-compute-mix、mHC pre/post、SwiGLU 和 Top-K gate；
- Triton：gated delta rule，以及五个 mHC BMM/Sinkhorn 算子。

`torchtitan_npu_op` 按文档中的限定名称调用这些接口；`torchtitan_*` 显式接口提供常用的语义名称。两种形式均原样转发
上游参数和返回值。TorchTitan-NPU 中仅负责注册 decomposition/autograd 的模块、内部反向函数、SDC 诊断算子和测试接口
不作为 HyperParallel 公共 API。

## 安装与兼容性

根据各上游仓库 2026-10-08 的主分支说明，各项目的环境约束并不完全相同：

| 上游项目 | 上游声明的主要环境要求 |
|---|---|
| DeepGEMM-Ascend | Ascend 950、CANN 9.20、Python 3.10 或更高版本 |
| DeepSelect | Ascend 后端支持 BF16 输入和 int32 索引；版本组合以其构建说明为准 |
| FlashMLA | Ascend 950、CANN 9.2.0 或更高版本、PyTorch 2.0 或更高版本及 torch-npu |
| TileKernels | Ascend 950、CANN 9.2.0 或更高版本、Python 3.12 或更高版本、PyTorch 2.13 或更高版本 |
| TorchTitan-NPU | Python 3.12 或更高版本；PyTorch、torch-npu、CANN 组合以其安装教程和 requirements 为准 |

请以目标 revision 的 README 和构建脚本为最终依据：

- [DeepGEMM-Ascend](https://github.com/deepseek-ai/DeepGEMM-Ascend)
- [DeepSelect](https://github.com/deepseek-ai/DeepSelect)
- [FlashMLA](https://github.com/deepseek-ai/FlashMLA)
- [TileKernels](https://github.com/deepseek-ai/TileKernels)
- [TorchTitan-NPU](https://github.com/torchtitan-npu/torchtitan-npu)

这些包均为可选依赖，HyperParallel 不修改全局 PyTorch/CANN 版本约束。应在目标 A5 机器上从源码构建，并确保构建时和
运行时使用相同的 PyTorch、torch-npu 与 CANN 环境。

### 本适配评审的上游 revision

| 项目 | revision |
|---|---|
| TileLang | `47976f210de3598444003e9b05de8204dfa7ff88` |
| DeepGEMM-Ascend | `8491bbb4b8c02a094a2318965f50c70438a3e73c` |
| DeepSelect | `bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18` |
| FlashMLA | `2e5429fc5653bab6e081f09477126f731882a6a9` |
| TileKernels | `66258df6175d2f630ffecb04c5ab66bff8a2ae6a` |
| TorchTitan-NPU | `9323939241034e93f1d5e54142755023e7cd3b60` |

revision 用于明确 HyperParallel 适配时核对的 API 表面，并不替代上游的版本管理。使用其他 revision 时，应先核对对应
README、构建脚本以及接口签名。

### 按上游说明构建

以下命令保持各目标 revision README 中的构建方式。开始前应按 CANN 安装说明加载环境，并确认当前 Python 可以导入
`torch_npu`、`torch.npu.is_available()` 为 `True`。

#### TileLang

TileLang 的 Ascend 指南要求兼容的 Ascend 950 驱动、CANN、PyTorch 和 `torch_npu`，并要求 `bisheng`、CCE
`ld.lld` 和 Ascend runtime 可用。其 Ascend 源码构建命令为：

```bash
git clone --recursive https://github.com/tile-ai/tilelang.git
cd tilelang
git checkout 47976f210de3598444003e9b05de8204dfa7ff88
git submodule update --init --recursive
USE_ASCEND=ON USE_CUDA=OFF python -m pip install -v .
```

详见该 revision 的
[Ascend 950 backend guide](https://github.com/tile-ai/tilelang/blob/47976f210de3598444003e9b05de8204dfa7ff88/tilelang/ascend/README.md)。

#### DeepGEMM-Ascend

DeepGEMM-Ascend 声明需要 Ascend 950、提供 `bin/bisheng` 和 `bin/ld.lld` 的 CANN 9.20、`torch_npu`、
Python 3.10+、支持 C++20 `<format>` 的编译器和标准库，以及 `tree-sitter`、`tree-sitter-cpp`。安装命令为：

```bash
git clone --recursive https://github.com/deepseek-ai/DeepGEMM-Ascend.git
cd DeepGEMM-Ascend
git checkout 8491bbb4b8c02a094a2318965f50c70438a3e73c
git submodule update --init --recursive
```

该 revision 内嵌的 DeepJIT 需要 `elfutils/libdwfl.h`，但此主机开发依赖未由 Python 包元数据安装。使用 Conda
工具链时，构建前在当前环境安装：

```bash
conda install -c conda-forge elfutils
```

FP8 JIT 路径使用 CANN 的 `ascendc_assert`。先确认 CANN 中存在 `utils/debug/asc_assert.h`；若
`deep_gemm/include/deep_gemm/ascend.hpp` 尚未包含它，在构建前补充尖括号形式的 include：

```bash
grep -q 'utils/debug/asc_assert.h' deep_gemm/include/deep_gemm/ascend.hpp || \
  sed -i '3i#include <utils/debug/asc_assert.h>' deep_gemm/include/deep_gemm/ascend.hpp
```

DeepJIT 的 include 跟踪器不接受此处使用双引号形式。若 CANN 中不存在该头文件，应改用符合上游要求的 CANN
工具链，不能删除断言来绕过编译。

上游为源码目录内的开发和测试提供 `develop.sh`；该脚本构建扩展，并将生成的 `_C` 动态库链接回源码包：

```bash
./develop.sh
```

若只安装 wheel，则使用上游的安装命令：

```bash
python -m pip install . --no-build-isolation
```

常规 wheel 安装不会在源码目录原地生成 `deep_gemm._C`。因此，安装后的导入检查及上游测试应从源码树以外的目录
启动，避免源码包遮蔽 site-packages 中的已安装扩展；或者按上游开发流程先运行 `./develop.sh`。例如：

```bash
cd /tmp
python -c 'import deep_gemm'
```

上游正确性测试会一次预编译大量 JIT kernel，默认 worker 数为主机 CPU 数。多核机器出现编译器资源争用时，可以用
测试框架提供的 `DG_TEST_MAX_WORKERS` 限制并发，例如 `DG_TEST_MAX_WORKERS=4`；该变量只控制上游测试的预编译并发，
不是 HyperParallel 的运行时要求。

上游通过 `ASCEND_HOME_PATH` 或 `ASCEND_TOOLKIT_HOME` 获取 Ascend 安装目录。详见该 revision 的
[README](https://github.com/deepseek-ai/DeepGEMM-Ascend/blob/8491bbb4b8c02a094a2318965f50c70438a3e73c/README.md)。

#### DeepSelect

DeepSelect README 的源码安装命令为：

```bash
git clone https://github.com/deepseek-ai/DeepSelect.git
cd DeepSelect
git checkout bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18
git submodule update --init --recursive
python -m pip install -v . --no-build-isolation
```

DeepSelect 的构建脚本在生成 wheel metadata 时会通过 `tests.kernelkit` 导入 `torch`。关闭 build isolation 可使构建过程
使用当前环境中已安装且与 `torch_npu` 匹配的 PyTorch；不要在 pip 创建的临时构建环境中另行安装一份 PyTorch。

Ascend 实现只支持 BF16 输入和 `torch.int32` 索引；输入最后一维必须连续，行 stride 必须满足
`deep_select.get_stride_requirement()`。详见该 revision 的
[README](https://github.com/deepseek-ai/DeepSelect/blob/bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18/README.md)。

#### FlashMLA

FlashMLA 的 Huawei 平台要求为 Ascend 950、CANN 9.2.0+、`torch_npu` 和 PyTorch 2.0+。安装命令为：

```bash
git clone https://github.com/deepseek-ai/FlashMLA.git flash-mla
cd flash-mla
git checkout 2e5429fc5653bab6e081f09477126f731882a6a9
git submodule update --init --recursive
python -m pip install -v . --no-build-isolation
```

`--no-build-isolation` 是上游明确要求；构建脚本会自动检测 Ascend，也可设置
`FLASH_MLA_BUILD_TARGET_PLATFORM=ASCEND`。`ASCEND_HOME_PATH` 必须指向 CANN 安装根目录。详见该 revision 的
[README](https://github.com/deepseek-ai/FlashMLA/blob/2e5429fc5653bab6e081f09477126f731882a6a9/README.md)。

#### TileKernels

TileKernels 声明需要 Python 3.12+、PyTorch 2.13+、TileLang 0.1.15+；Ascend 后端还需要 Ascend 950 和
CANN 9.2.0+。本地开发安装命令为：

```bash
git clone https://github.com/deepseek-ai/TileKernels.git
cd TileKernels
git checkout 66258df6175d2f630ffecb04c5ab66bff8a2ae6a
git submodule update --init --recursive
python -m pip install -e ".[dev]"
```

详见该 revision 的
[README](https://github.com/deepseek-ai/TileKernels/blob/66258df6175d2f630ffecb04c5ab66bff8a2ae6a/README.md)。

#### TorchTitan-NPU

按照 TorchTitan-NPU 上游 README 从源码安装：

```bash
git clone https://github.com/torchtitan-npu/torchtitan-npu.git
cd torchtitan-npu
git checkout 9323939241034e93f1d5e54142755023e7cd3b60
python -m pip install -r requirements.txt
python -m pip install -e .
```

具体软件与硬件准备以其
[安装教程](https://github.com/torchtitan-npu/torchtitan-npu/blob/master/docs/user-guides/installation.md) 为准。

完成各原生包的上游安装和测试后，再从 HyperParallel 仓库根目录安装当前代码：

```bash
python -m pip install -e .
```

## Functional 示例

```python
import torch

from hyper_parallel.components.functional import (
    deepseek_bf16_gemm,
    deepseek_per_token_cast,
    deepseek_select_topk,
    deepseek_topk_gate,
)

output = deepseek_bf16_gemm("nt", x, weight)
values, indices = deepseek_select_topk(scores, 2048, indices_type=torch.int32)
expert_indices = deepseek_topk_gate(router_scores, 8)
quantized = deepseek_per_token_cast(hidden_states, "e4m3", 32)
```

TorchTitan-NPU 的显式接口同样保持上游参数和返回值：

```python
from hyper_parallel.components.functional import torchtitan_moe_token_permute, torchtitan_topk_gate

indices = torchtitan_topk_gate(router_scores, 8)
permuted, restore_indices = torchtitan_moe_token_permute(tokens, expert_indices)
```

接口保持上游原生参数和返回值。通用 DeepGEMM 和 TileKernels 接口使用算子名分发，以便上游新增内核无需在
HyperParallel 中逐个复制易变化的签名。不支持的包、命名空间、变体或算子会产生包含构建提示的异常。

## Module 示例

```python
from hyper_parallel.components.modules import DeepseekV41SparseDecodeAttention

attention = DeepseekV41SparseDecodeAttention(
    value_head_dim=512,
    enable_batch_invariant=True,
)
output, logsumexp = attention(query, key_cache, sparse_indices)
```

## 验证建议

普通开发机可以执行惰性导入、参数转发及模块状态测试：

```bash
pytest -q tests/ut/components/functional/test_deepseek_ascend.py \
  tests/ut/components/functional/test_torchtitan_npu.py \
  tests/ut/components/modules/test_deepseek_sparse_attention.py
```

目标 A5 环境还应分别运行各上游项目的 Ascend 单测/基准，并用相同输入与 PyTorch 参考实现比较：输出形状、dtype、
最大绝对/相对误差、NaN/Inf、正反向结果（适用时）。FlashMLA decode 需覆盖调度元数据复用、`reset_scheduler()`、
变长 Top-K、额外 KV cache 和 batch-invariant 模式；DeepSelect 需使用 BF16 输入和 int32 索引。
