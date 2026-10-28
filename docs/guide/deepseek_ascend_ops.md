# DeepSeek Ascend 算子安装

HyperParallel 提供以下上游项目的可选适配：

- [TileLang](https://github.com/tile-ai/tilelang)
- [DeepGEMM-Ascend](https://github.com/deepseek-ai/DeepGEMM-Ascend)
- [DeepEP-Ascend](https://github.com/deepseek-ai/DeepEP-Ascend)
- [TileKernels](https://github.com/deepseek-ai/TileKernels)
- [FlashMLA](https://github.com/deepseek-ai/FlashMLA)
- [DeepSelect](https://github.com/deepseek-ai/DeepSelect)

这些依赖按需加载，安装 HyperParallel 时不会自动安装上游算子包。

## 环境要求

以下安装说明适用于 Ascend 950（A5）：

| 依赖 | 要求 |
|---|---|
| Python | 3.12 |
| PyTorch | 2.13.x |
| torch-npu | 与 PyTorch 2.13.x 配套的版本 |
| NumPy | 1.26.4 |
| CANN | 9.2.0 或更高版本 |
| TileLang | 0.1.15 或更高版本 |
| C++ 工具链 | 支持 C++20 |
| Transformers | 使用 DeepSeek V4.1 模型时需要 5.13.0 |

PyTorch、torch-npu、CANN 和驱动必须使用兼容的组合。构建和运行 Ascend 原生扩展时使用同一套 CANN 环境。
DeepEP-Ascend 还需要 UBMEM、URMA、HCOMM 和多卡 HCCL 通信环境。

## 安装

以下命令安装上游算子包及 HyperParallel。构建 Ascend 原生扩展前，加载所用 CANN 的环境脚本；
环境已配置时无需重复加载。单独安装 HyperParallel 等 Python 包不需要加载 CANN。

### TileLang

```bash
git clone --recursive https://github.com/tile-ai/tilelang.git
USE_ASCEND=ON USE_CUDA=OFF python -m pip install -v ./tilelang
```

TileLang 的 Ascend 后端需要从源码构建。

### TileKernels

```bash
git clone https://github.com/deepseek-ai/TileKernels.git
python -m pip install ./TileKernels
```

### DeepGEMM-Ascend

```bash
git clone --recursive https://github.com/deepseek-ai/DeepGEMM-Ascend.git
python -m pip install --no-build-isolation ./DeepGEMM-Ascend
```

### DeepEP-Ascend

```bash
git clone https://github.com/deepseek-ai/DeepEP-Ascend.git
git -C DeepEP-Ascend submodule update --init --recursive third-party/deep_jit
python -m pip install --no-build-isolation ./DeepEP-Ascend
```

### DeepSelect

```bash
git clone --recursive https://github.com/deepseek-ai/DeepSelect.git
python -m pip install --no-build-isolation ./DeepSelect
```

### FlashMLA

```bash
git clone --recursive https://github.com/deepseek-ai/FlashMLA.git
FLASH_MLA_BUILD_TARGET_PLATFORM=ASCEND python -m pip install --no-build-isolation ./FlashMLA
```

### HyperParallel

```bash
python -m pip install --no-deps --no-build-isolation -e /path/to/hyper-parallel
```

DeepSelect Attention 使用 PyTorch 完成注意力和 Indexer KL 计算，仅替换 Indexer Top-K。
可选的 CANN 融合训练路径通过 `SharedCompressedDSAAttention(use_fused_ops=True)` 单独启用，
安装要求见[functional 依赖说明](../../hyper_parallel/components/functional/README.md#deepseek-v41-fused-operators)。

## 支持范围

| 上游项目 | HyperParallel 接入 |
|---|---|
| TileLang | 上游算子的编译和运行依赖 |
| DeepGEMM-Ascend | GEMM、Einsum、MQA logits、Mega-MoE functional |
| DeepEP-Ascend | EP dispatch/combine 及通信资源管理 functional |
| TileKernels | 量化、SwiGLU、MoE、RoPE、Engram、流形约束超连接（Manifold-Constrained Hyper-Connections，mHC）functional 和 Module |
| FlashMLA | 稀疏 prefill/decode functional 和推理 Attention Module |
| DeepSelect | Top-K functional 和 DSA Indexer Module |

FlashMLA 的 Ascend 稀疏注意力接口不提供训练反向。DeepGEMM-Ascend 的 functional 不自动替换通用 `nn.Linear`。

## 常见问题

### 导入 PyTorch 时出现 `Mismatch in kernel C++ signatures`

处理步骤：

1. 按 torch-npu 的版本配套说明，重新安装 PyTorch 2.13.x 和对应的 torch-npu；两者一起更换。
2. 先完成 PyTorch 和 torch-npu 的导入，再安装其他原生扩展。
3. 使用当前环境重新构建 DeepGEMM、DeepEP、DeepSelect、FlashMLA 等依赖 PyTorch 的扩展。

重新导入时不应再出现 C++ kernel signature mismatch。若仍报错，检查是否加载了其他 Python 环境中的包。

### 构建时找不到 `<format>` 或 `std::format`

处理步骤：

1. 更换缺少该接口的 C++ 标准库。使用 GCC/libstdc++ 时，选择 CANN 支持的 GCC 13 或更高版本及其配套标准库；
   [GCC 13 开始提供 `<format>` 和 `std::format`](https://gcc.gnu.org/gcc-13/changes.html)。
2. 同步使用该工具链的头文件和运行时库，清除构建配置中指向旧 GCC 的路径。
3. 重新构建报错的 DeepGEMM、DeepEP 或 DeepSelect 包。

编译命令应使用 C++20，并加载包含 `<format>` 的标准库头文件。仅添加 `-std=c++20` 无法补齐缺失的库接口。

### 找不到 Bisheng、`ld.lld`，或链接器不支持 AI Core 目标

处理步骤：

1. 使用包含编译器的完整 CANN Toolkit 安装，而不是仅安装运行时。
2. 在执行安装命令的同一终端中加载该 Toolkit 的 `set_env.sh`。
3. 移除构建配置中指向系统 LLVM 链接器的设置，重新安装报错的算子包。

后续编译应调用 CANN 的 Bisheng 和 CCE 链接器；系统 LLVM 的同名 `ld.lld` 不能替代它。

### 编译时出现 CANN 接口未声明

例如 FlashMLA 编译时报 `asc_float2bfloat16_rn_v2_impl` 未声明。处理步骤：

1. 按对应上游项目的安装说明选择 CANN 版本，在新终端中仅加载这一套 CANN 环境。
2. 重新执行该包的源码安装命令，排除混用其他 CANN 头文件的情况。
3. 若接口仍未声明，使用上游明确提供该问题修复的版本或 wheel；上游尚未提供修复时，提交包含软件版本和首个编译错误的 issue。

此类上游 API 兼容问题没有通用的 pip 参数修复方式。不能把未经上游确认的函数替换作为安装步骤。

### 更换 PyTorch 或 CANN 后，已安装算子出现导入或链接错误

处理步骤：

1. 卸载报错的原生扩展包，保留已经配套安装的 PyTorch 和 torch-npu。
2. 使用新的源码构建目录，加载目标 CANN 环境，并按本页对应包的命令重新安装。
3. 在源码目录之外导入已安装的包，避免源码包遮蔽新安装的扩展。

不要复制旧环境中的扩展 `.so` 或 wheel 来替代重新构建。
