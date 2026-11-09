# Multicore Build and Usage

**English** | [简体中文](build.zh.md)

## Build

Prepare compatible Torch / torch_npu, CANN toolkit/ops >= 9.1.0, Python development headers,
pybind11, CMake >= 3.18, GCC/G++, GNU Make, and Git. Activate the selected CANN environment first.

Build and package from the repository root:

```bash
source /path/to/cann/set_env.sh
bash build.sh --multicore on --strict on --jobs 24
```

Multicore has a single `--multicore on|off` switch, defaulting to on. SHMEM is internal and is built
and packaged with Multicore, without a separate switch. Off skips both Multicore and SHMEM native artifacts.
`--strict on` aborts immediately if a component build fails; the default off preserves optional-component behavior.

The SHMEM build uses pinned upstream Git sources and applies only SHA256-verified
`cann_runtime_symbols.patch` and `cann_shmem_log_api.patch` to an isolated copy. `cann_runtime_symbols.patch` makes the SDK load
required `rt*` interfaces from `libruntime.so`, supporting CANN layouts where `libascendcl.so` does not depend on the Runtime library;
`cann_shmem_log_api.patch` provides the private `hpshmem_log` interface so HP-SHMEM can reuse CANN SHMEM's built-in log sink.
The sole C ABI declaration of `hpshmem_log` is in `include/host/utils/hpshmem_log.h`; CANN's include installation rules install it into
the private SDK. Missing required symbols cause build or initialization failure.

The main entry point calls this directory's `build.sh` and packages the result. The component entry point owns
private SHMEM, the Torch adapter, and forward/backward vendors. Dependencies are pinned in `_build/dependencies.lock.json`.
Default targets are `ascend910b,ascend910_93`; use `--soc-list` to select among them.
Artifacts built with different Python, CPU, Torch/PTA, or CANN environments are not interchangeable by default.

The component validates final artifacts after generating private SHMEM, reusing caches, generating single-SoC vendors,
merging multi-SoC vendors, and installing the Torch adapter. Validation covers ACLNN APIs, kernel objects/metadata, ELF architecture, private SONAMEs, runtime dependencies, and shared-library security
properties. Packaged shared libraries have no RPATH/RUNPATH and enable RELRO, BIND_NOW, a non-executable stack, stack protection, and
FORTIFY_SOURCE; Release shared libraries have their static symbol tables stripped. Failures return a nonzero status directly from this component's entry point.

Build only Multicore, without a wheel:

```bash
bash hyper_parallel/core/multicore/build.sh --soc-list ascend910b --jobs 24
```

`--clean` rebuilds the component work cache while retaining downloaded dependency caches. Generated files reside in
`build/native`, outside business source directories. Usable local artifacts are under
`build/native/payload/hyper_parallel/core/multicore/`; complete wheels are in `dist/`.

## Local Lizard Precheck

Use the gate's pinned Lizard 1.17.19 to check Python, C++, and Java together. The cyclomatic complexity limit is 19
and the function effective-line limit is 100. Before pushing, run from the repository root:

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

`upstream/master` should point to the latest local reference of the PR target branch. The command checks complete source files affected by the PR,
including uncommitted modifications to tracked files; new files must first be added to the Git index. Rerun after modifying functions, merging the target branch, or rebasing,
and ensure the check passes before pushing. Test directories and third-party sources use the same exclusions as the gate.

## Delivery and Imports

Multicore uses explicit precompilation; importing Python does not compile CANN vendors automatically.
Artifacts include a CANN custom OPP activation script. Activate CANN and the matching OPP payload before starting Python.
The HyperParallel [Installation Guide](../../../../docs/installation.md) documents activation paths for source payloads
and installed wheels.

```python
from hyper_parallel.core.multicore import MegaMoeExperts
```

The HyperParallel root package does not export Multicore's public component APIs.
