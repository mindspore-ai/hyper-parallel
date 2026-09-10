# Multicore 构建与使用

## 编译

准备配套的 Torch / torch_npu、CANN toolkit/ops >= 9.1.0，以及 Python 开发头文件、
pybind11、CMake >= 3.18、GCC/G++、GNU Make、Git。先激活所选 CANN 环境。

从仓库根目录编译并打包：

```bash
source /path/to/cann/set_env.sh
bash build.sh --multicore on --strict on --jobs 24
```

Multicore 只有 `--multicore on|off` 开关，默认 on。SHMEM 是内部组件，随 Multicore
一起编译和打包，没有独立开关。off 不构建或打包 Multicore/SHMEM native 制品。
`--strict on` 使组件编译失败时立即终止；默认 off 保留主入口原有的可选组件行为。

主入口调用本目录的 `build.sh` 并统一组包。业务入口负责私有 SHMEM、Torch adapter
和正反向 vendor；依赖版本固定在 `_build/dependencies.lock.json`。
默认目标是 `ascend910b,ascend910_93`；可以通过 `--soc-list` 选择其中的目标。
不同 Python、CPU、Torch/PTA 或 CANN 构建环境的制品不能默认互换。

组件会在生成私有 SHMEM、复用缓存、生成单 SoC vendor、合并多 SoC vendor 和安装 Torch adapter 后校验
最终制品。校验覆盖 ACLNN API、kernel object/metadata、ELF 架构、私有 SONAME、相对 RUNPATH 和运行时
依赖；失败会由本组件入口直接返回非零，不依赖额外 validation 脚本。

只编译 Multicore、不生成 wheel：

```bash
bash hyper_parallel/core/multicore/build.sh --soc-list ascend910b --jobs 24
```

`--clean` 重建组件工作缓存，保留下载依赖缓存。生成物放在 `build/native`，
不写入业务源码。可使用的本地制品位于
`build/native/payload/hyper_parallel/core/multicore/`，完整 wheel 位于 `dist/`。

## 交付与导入

Multicore 保持显式预编译，不在 Python 导入时自动编译 CANN vendor。
交付物包含 CANN custom OPP 激活脚本，必须先激活 CANN 和对应 OPP 制品，再启动 Python。
源码 payload 与 wheel 安装态的具体激活路径由 HyperParallel
[安装指南](../../../../docs/installation.md)统一说明。

```python
from hyper_parallel.core.multicore import MegaMoeExperts
```

HyperParallel 根包不导出 Multicore 的业务接口。
