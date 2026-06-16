# 安装指南

## 系统要求

### 通用要求（构建 + 运行）

- **Python**：3.10、3.11 或 3.12（构建出的 whl 仅可装到对应小版本的解释器上）
- **架构**：linux_aarch64 或 linux_x86_64（whl 内含预编译 .so，文件名 tag 与编译机一致；跨架构无法安装）
- **GCC**：主机 GCC 版本须落在 [7.3.0, 11.3.0] 区间（与 MindSpore 编译策略一致）
- **glibc**：主机 glibc 需不低于编译机的 glibc 版本；如部署到旧系统，请在更老的发布镜像内编译。例如在 OpenEuler 22.03（glibc 2.34）编出的 whl 无法在 glibc < 2.34 的环境运行

### 额外构建工具（仅编译 whl 时需要）

- **CMake** ≥ 3.18
- **CANN 工具链**（需要可用的 `ASCEND_HOME_PATH`；脚本会尝试 source `/usr/local/Ascend/cann/set_env.sh`）
- **bisheng 编译器**（来自 CANN，用于 symmetric_memory CCE 内核编译）
- **MindSpore** ≥ 2.8（构建期由 `CustomOpBuilder` 调用以编 custom_ops 和 multicore .so；同时也是大多数用户的运行时依赖）
- 可选：启用 `BUILD_TORCH_EXTENSION=true` 时还需 **PyTorch** ≥ 2.7（CXX11 ABI=1）+ torch_npu；如设 `USE_NINJA=1` 还需 ninja

### 运行时依赖

- 安装深度学习框架
- 推荐安装的 MindSpore 版本 ≥ 2.8，最好使用最新的 MindSpore 版本，参考 [此处](https://atomgit.com/mindspore/mindspore#%E5%AE%89%E8%A3%85)
- 若使用 PyTorch 路径，需要 PyTorch ≥ 2.7（编译时启用 `_GLIBCXX_USE_CXX11_ABI=1` 的版本，官方 wheel 即满足）

---

## 从源码构建安装

当前仅支持从源码安装：

```bash
git clone https://gitcode.com/mindspore/hyper-parallel.git
cd hyper-parallel
python setup.py bdist_wheel
# whl 文件名按当前 Python 和架构生成，例如 cp310-cp310-linux_aarch64
pip install dist/hyper_parallel-*.whl
```

### 构建选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `BUILD_TORCH_EXTENSION` | `false` | 设为 `true` 同时构建 PyTorch 后端扩展 |
| `USE_NINJA` | `false` | 设为 `true` 使用 ninja 加速编译 |

### 仅构建 MindSpore 后端

```bash
python setup.py bdist_wheel
pip install dist/hyper_parallel-*.whl
```

### 同时构建 PyTorch + MindSpore 后端

```bash
BUILD_TORCH_EXTENSION=true python setup.py bdist_wheel
pip install dist/hyper_parallel-*.whl
```

---

## 验证安装

```python
import hyper_parallel as hp

# 检查平台类型
print(hp.get_platform())

# 验证核心接口可导入
from hyper_parallel import (
    fully_shard, HSDPModule, hsdp_sync_stream,
    DTensor, Layout, DeviceMesh, init_device_mesh, get_current_mesh,
    shard_module, custom_shard, DFunction,
    PipelineStage, Schedule1F1B, ScheduleGPipe, ScheduleInterleaved1F1B,
    ContextParallel, AsyncContextParallel,
    ColwiseParallel, RowwiseParallel, SequenceParallel, parallelize_module,
    init_process_group, destroy_process_group,
    MetaStep, MetaStepType, BatchDimSpec,
    manual_seed,
)
```

---

## 环境变量

| 变量 | 说明 |
|------|------|
| `ASCEND_HOME_PATH` | CANN 工具链路径 |
| `BUILD_TORCH_EXTENSION` | 设为 `true` 构建 PyTorch 后端 |
| `USE_NINJA` | 设为 `true` 使用 ninja 编译 |

---

## 常见安装问题

### Q: pip install 报错 "not a supported wheel on this platform"

whl 文件名中的 python 版本和架构 tag 必须与目标环境一致。请在目标环境的 Python 下重新构建 whl。

### Q: 导入时报错 "undefined symbol"

可能原因是编译机和运行机的 glibc 版本不一致。请在目标环境所在的 OS 发布镜像内重新构建 whl。

### Q: MindSpore custom_ops 编译失败

确保 `ASCEND_HOME_PATH` 已正确设置，并且 MindSpore ≥ 2.8 已安装。构建脚本会尝试 source `/usr/local/Ascend/cann/set_env.sh`。

### Q: PyTorch 扩展构建失败

确保 PyTorch ≥ 2.7 已安装且使用 CXX11 ABI=1（官方 wheel 默认满足）。若使用 torch_npu，确保 torch_npu 版本与 PyTorch 版本匹配。