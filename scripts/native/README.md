# Optional native build

`./build.sh` is the unified user entry. It checks and prepares locked source dependencies, incrementally compiles the
selected optional components, assembles `build/native/payload/hyper_parallel`, and always creates a wheel in `dist/`.
The payload can also be used directly through `PYTHONPATH`; wheel packaging uses the same payload.

The host must provide Python 3.10--3.12 with development headers, setuptools, wheel and pybind11; GCC/G++ with C++17;
CMake, GNU Make, Git, binutils, coreutils, tar, sed and awk; and a complete CANN >= 9.1.0 toolkit/ops development
environment. The CANN environment must expose bisheng, asc_opc, headers, libopapi.so and ops_base. The indexed Dataset
C++ helper is a required wheel artifact and is compiled on every `build.sh` invocation. Supported build hosts are
aarch64 and x86_64; each wheel targets one host architecture and one CPython ABI. When `ASCEND_HOME_PATH` is unset,
the unified entry sources `/usr/local/Ascend/cann/set_env.sh` if that default installation exists. A custom CANN
installation must be sourced by the caller. MindSpore targets additionally require MindSpore >= 2.10 and Ninja. Torch
targets use the compatible Torch/torch_npu pair already selected by the
repository extras or external build environment and require Torch's `_GLIBCXX_USE_CXX11_ABI=1`.

```bash
# Source this explicitly for a custom installation. build.sh automatically
# sources /usr/local/Ascend/cann/set_env.sh when ASCEND_HOME_PATH is unset.
source /custom/path/to/cann/set_env.sh

# Optional failures are warnings by default, so a core-only wheel can still be produced.
./build.sh --multicore mindspore --shmem mindspore --custom-ops on

# Rebuild selected work/install outputs while retaining downloaded dependencies.
./build.sh --clean --jobs 32

# PYTHONPATH development uses the payload produced by the same command.
export PYTHONPATH=/path/to/hyper-parallel:${PYTHONPATH:-}
source build/native/payload/hyper_parallel/core/multicore/lib/set_env.bash
```

The automatic source only affects the `build.sh` process and its children. Before running an application or ST in the
calling shell, source CANN and the packaged multicore environment explicitly.

Dependency preparation uses the versions and hashes in `config/dependencies.lock.json`. It reuses a matching ignored
`build/native/deps` cache and automatically downloads or refreshes an absent/inconsistent entry. SHMEM compilation
uses an isolated `git archive` export of the locked commit.
Framework adapter work directories are separated by the active CPython ABI and framework installation identity.
Each adapter is rebuilt from a clean identity directory.

Normal builds keep `build/native/work` and reuse the heavy SHMEM and per-SoC vendor caches. `--clean` removes the
selected component's work/install outputs, but not downloaded dependency sources. Every invocation freshly assembles
the payload from component install roots. `--strict off` is the
default: an optional component failure emits a stable reason code and warning, then wheel packaging continues;
`--strict on` stops immediately.

Multicore has a one-way dependency on the SHMEM base capability. Selecting multicore automatically enables the
matching symmetric-memory framework target. `--soc-list` controls multicore kernel fan-out and SHMEM hardware gating;
supported targets are `ascend910b` (910B) and `ascend910_93` (910C). Selecting `ascend950` reports
`SOC_SOURCE_NOT_SUPPORTED` as an optional component failure. The assembled HyperParallel vendor kernel package is
built once per selected SoC and the kernel/config trees are combined into one package. The merge requires consistent
vendor build inputs and host ABI, and the package carries one common host payload.
Each SoC payload contains compiled objects under `tbe/kernel/<soc>`, runtime binary indexes under
`tbe/kernel/config/<soc>`, and the CANN operator registry under `tbe/config/<soc>`.

Using multicore always requires sourcing its packaged custom-OPP environment before starting the application or
framework Python process. For an installed wheel, source the shell locator in the active environment's `bin` directory:

```bash
source "$(command -v hyper_parallel_multicore_set_env.bash)"
```

For a PYTHONPATH build, source the payload path shown by `build.sh`. The loader validates
`ASCEND_CUSTOM_OPP_PATH` and `LD_LIBRARY_PATH`. Missing activation raises `HP-NATIVE-OPP-NOT-ACTIVATED`; if MindSpore or torch/torch_npu is already loaded, it
raises `HP-NATIVE-OPP-ACTIVATION-TOO-LATE`, prints the exact script path, and requires a new Python process.

The component scripts are stable independent boundaries and accept their own explicit command-line arguments:

```bash
bash scripts/build_symmetric_memory.sh --framework all --soc-list ascend910b,ascend910_93 --jobs 32
bash scripts/build_multicore.sh --framework all --soc-list ascend910b,ascend910_93 --jobs 32
bash scripts/build_custom_ops.sh --framework mindspore --jobs 32
```

Each supports `--clean`; the first two accept `mindspore|torch|all`, while custom ops accepts the same values but
supports MindSpore as its native target; a `torch` selection emits a warning. A successful component command refreshes
that component's slice under `build/native/payload/hyper_parallel`, so a local incremental build is immediately visible
to the PYTHONPATH runtime. Use `./build.sh` when a wheel is required.
The Python helpers prepare dependencies, assemble the isolated multicore source tree, and merge
verified per-SoC vendor inputs. Vendor log, artifact, symbol, ELF, and runtime-link checks execute inside
`build_multicore.sh`. Build status is carried by logs under `build/native/logs`.

Build and packaging behavior is verified by the unified/component build logs and
the wheel/ST validation flow. Runtime payload lookup and loader behavior is
covered by the standard repository tests under `tests/ut/native`.

Native source dependency revisions, hashes, and supported environment versions are declared in
`scripts/native/config/dependencies.lock.json`. The assembler extracts the required SwiGLU/Grad and GroupedMatmul
implementations and applies `swi_glu_fusion.patch` and `grouped_matmul_fusion.patch` in an isolated build tree. A
dependency update must update the lock metadata, adapter hashes, and source-assembly checks together.
