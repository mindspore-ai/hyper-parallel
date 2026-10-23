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

SHMEM 构建保持锁定的上游 Git 源码不变，只在隔离副本应用经过 SHA256 校验的
`cann_runtime_symbols.patch`。该适配使 SDK 从 `libruntime.so` 加载必需的 `rt*` 接口，
兼容 `libascendcl.so` 不依赖 Runtime 库的 CANN 布局；缺失必需符号仍会导致初始化失败。

主入口调用本目录的 `build.sh` 并统一组包。业务入口负责私有 SHMEM、Torch adapter
和正反向 vendor；依赖版本固定在 `_build/dependencies.lock.json`。
默认目标是 `ascend910b,ascend910_93`；`--soc-list ascend950` 选择 950 家族，要求所选 CANN
包含对应的工具链、ops 和 `libhcomm.so`。SoC 入参使用 CANN 家族名，不填写 PR/DT 子型号。
950 使用 CANN 的 `Ascend950PR_9599` 代表编译目标；实际设备核数由运行时查询，
不从代表型号推断。一次组件构建选择 A2/A3 或 A5 的 SHMEM 后端，不能混合这两代目标。
不同 Python、CPU、Torch/PTA 或 CANN 构建环境的制品不能默认互换。

MegaMHC 的融合 worker 使用 `MhcPreSinkhornTilingData` 任务布局。
它复用匹配的 task tiler 并查询实际平台资源，不使用上游独立算子的 arch35 tiling 布局；
独立算子的架构分派不能直接替换融合入口。

组件会在生成私有 SHMEM、复用缓存、生成单 SoC vendor、合并多 SoC vendor 和安装 Torch adapter 后校验
最终制品。校验覆盖 ACLNN API、kernel object/metadata、ELF 架构、私有 SONAME、运行时依赖及共享库安全
属性。随包共享库不携带 RPATH/RUNPATH，并启用 RELRO、BIND_NOW、不可执行栈、栈保护和
FORTIFY_SOURCE；Release 共享库会删除静态符号表。失败由本组件入口直接返回非零。

只编译 Multicore、不生成 wheel：

```bash
bash hyper_parallel/core/multicore/build.sh --soc-list ascend910b --jobs 24
```

`--clean` 重建组件工作缓存，保留下载依赖缓存。生成物放在 `build/native`，
不写入业务源码。可使用的本地制品位于
`build/native/payload/hyper_parallel/core/multicore/`，完整 wheel 位于 `dist/`。

## 本地 Lizard 预检

使用门禁固定的 Lizard 1.17.19，同时检查 Python、C++ 和 Java；圈复杂度上限为 19，
函数有效代码行数上限为 100。推送前从仓库根目录运行：

```bash
python -m pip install 'lizard==1.17.19'
python -m lizard --version
set -o pipefail
git diff --name-only --diff-filter=ACMR -z upstream/master -- \
  '*.py' '*.c' '*.cc' '*.cpp' '*.cxx' '*.cu' '*.h' '*.hpp' '*.java' \
  ':!tests/**' ':!third_party/**' \
  | xargs -0 -r python -m lizard -l cpp -l python -l java -C 19 -L 100 \
      -x '*/tests/*' -x '*/test/*' -x '*/third_party/*' -w
```

`upstream/master` 应指向 PR 目标分支的最新本地引用。命令检查整个 PR 涉及的完整源文件，
包含已跟踪文件的未提交修改；新增文件需先加入 Git 索引。修改函数、合并目标分支或 rebase 后
重新运行，确认检查通过再推送。测试目录和第三方源码使用与门禁一致的排除规则。

## 交付与导入

Multicore 保持显式预编译，不在 Python 导入时自动编译 CANN vendor。
交付物包含 CANN custom OPP 激活脚本，必须先激活 CANN 和对应 OPP 制品，再启动 Python。
源码 payload 与 wheel 安装态的具体激活路径由 HyperParallel
[安装指南](../../../../docs/installation.md)统一说明。

```python
from hyper_parallel.core.multicore import MegaMoeExperts
```

HyperParallel 根包不导出 Multicore 的业务接口。

## 测试

从已安装当前源码的开发环境执行无卡 UT：

```bash
python -m pytest tests/ut/core/multicore
```

ST 使用仓库公共 launcher 启动 worker。准备对应交付的 CANN/OPP 环境，并按用例需要
提供空闲 NPU 后，使用标准 pytest 收集与运行：

```bash
python -m pytest --collect-only -q tests/torch/multicore -m 'level0 or level1'
python -m pytest -v tests/torch/multicore -m 'level0 or level1'
```

可以按文件、node ID 或 level 选择子集；完整目录包含八卡 SHMEM 和四卡 MegaMoe 用例。
wheel 验证应在源码目录之外运行，并避免将业务源码加入 `PYTHONPATH`，确保测试的是安装包。
