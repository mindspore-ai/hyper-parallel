#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Huawei Technologies Co., Ltd.
set -euo pipefail

docker_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
registry=quay.io/hyper-parallel/hyper-parallel-npu
source_repo=https://github.com/mindspore-ai/hyper-parallel
mirror_repo=https://gitcode.com/mindspore/hyper-parallel.git

die() { printf 'Error: %s\n' "$*" >&2; exit 1; }
source_ref() {
    local branch=$1 revision=$2
    if [[ $branch ]]; then
        [[ -z $revision ]] || die 'Choose either a HyperParallel branch or a HyperParallel commit'
        [[ $branch != -* ]] && git check-ref-format "refs/heads/$branch" >/dev/null || die "Invalid HyperParallel branch: $branch"
        printf 'refs/heads/%s\n' "$branch"
    else
        [[ $revision =~ ^[0-9a-fA-F]{40}$ ]] || die 'HyperParallel commit must be a full 40-character Git SHA'
        printf '%s\n' "${revision,,}"
    fi
}

fetch_source() {
    local directory=$1 ref=$2 url
    git -C "$directory" init -q
    for url in "$source_repo.git" "$mirror_repo"; do
        if GIT_TERMINAL_PROMPT=0 git -C "$directory" -c http.version=HTTP/1.1 \
            -c http.lowSpeedLimit=1 -c http.lowSpeedTime=30 fetch --quiet --no-tags --depth=1 "$url" "$ref"; then return; fi
    done
    die "Could not fetch $ref from either official HyperParallel repository"
}

# Resolve a moving branch once for both OS images. Read package metadata without running source on the host.
resolve_source() (
    local ref=$1 version=$2 directory revision
    directory=$(mktemp -d)
    trap 'rm -rf -- "$directory"' EXIT
    fetch_source "$directory" "$ref"
    revision=$(git -C "$directory" rev-parse --verify 'FETCH_HEAD^{commit}')
    [[ $ref == refs/heads/* || $revision == "$ref" ]] || die "Fetched $revision, expected $ref"
    if [[ -z $version ]]; then
        version=$(git -C "$directory" show "$revision:setup.py" | sed -nE "s/^[[:space:]]*version[[:space:]]*=[[:space:]]*['\"]([^'\"]+)['\"].*/\\1/p") || :
        [[ $version ]] || die 'Set --hp-version to the wheel version for this source revision'
    fi
    printf '%s %s\n' "$revision" "$version"
)

checkout_source() {
    local ref revision
    if [[ ${HYPER_PARALLEL_BRANCH:-} ]]; then ref=$(source_ref "$HYPER_PARALLEL_BRANCH" "${HYPER_PARALLEL_COMMIT:-}")
    else ref=$(source_ref '' "${HYPER_PARALLEL_COMMIT:-${DEFAULT_HYPER_PARALLEL_COMMIT:?}}"); fi
    fetch_source "$PWD" "$ref"
    revision=$(git rev-parse --verify 'FETCH_HEAD^{commit}')
    [[ $ref == refs/heads/* || $revision == "$ref" ]] || die "Fetched $revision, expected $ref"
    git checkout --quiet --detach "$revision"
    printf '%s\n' "$revision" > /tmp/hyper-parallel-revision
}

build_wheel() {
    local torch_api
    # CANN loads the tiling library during CPU compilation, before a host driver is available.
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/devlib"
    if [[ -d hyper_parallel/core/multicore/torch ]] && torch_api=$(python3 -I -c \
        'from importlib.metadata import version; major, minor, *_ = version("torch").split("."); print(f"V{major}R{minor}")' 2>/dev/null); then
        # The op-plugin headers select their C++ API with CURRENT_VERSION.
        export CXXFLAGS="${CXXFLAGS:+$CXXFLAGS }-UCURRENT_VERSION -DCURRENT_VERSION=$torch_api"
    fi
    bash build.sh
}

install_runtime() {
    local wheel_dir=/tmp/hp-wheel expected_pta=${TORCH_NPU_VERSION:-} installed_pta torch
    local expected_torch=${TORCH_VERSION:-${expected_pta%.post*}}
    if [[ $expected_pta && $expected_torch != "${expected_pta%.post*}" ]]; then
        die "PyTorch $expected_torch does not match TorchNPU $expected_pta"
    fi
    local -a packages=("$wheel_dir"/hyper_parallel-*.whl)
    local -a pip=(python3 -I -m pip --disable-pip-version-check)
    installed_pta=$(python3 -I -c 'from importlib.metadata import version; print(version("torch-npu"))' 2>/dev/null) || installed_pta=''
    if [[ $installed_pta ]]; then
        [[ -z $expected_pta || $expected_pta == "$installed_pta" ]] || die "Base provides torch_npu $installed_pta, requested $expected_pta"
        torch=$(python3 -I -c 'from importlib.metadata import version; print(version("torch").split("+")[0])')
        [[ -z $expected_torch || $expected_torch == "$torch" ]] || die "Base provides PyTorch $torch, requested $expected_torch"
        python3 -I -c 'from importlib.metadata import version; print(*(f"{p}=={version(p)}" for p in ("torch", "torch-npu")), sep="\n")' > "$wheel_dir/constraints.txt"
    else
        [[ $expected_pta ]] || die 'A CANN base requires TORCH_NPU_VERSION'
        torch=$expected_torch
        "${pip[@]}" download --no-cache-dir --no-deps --only-binary=:all: --index-url https://download.pytorch.org/whl/cpu \
            --dest "$wheel_dir" "torch==$torch+cpu"
        printf 'torch==%s+cpu\ntorch-npu==%s\n' "$torch" "$expected_pta" > "$wheel_dir/constraints.txt"
        packages+=("$wheel_dir"/torch-*.whl "torch-npu==$expected_pta")
    fi
    # Let pip match the base's SciPy to the NumPy range required by the selected source.
    if "${pip[@]}" show scipy >/dev/null 2>&1; then
        packages+=(scipy)
    fi
    "${pip[@]}" install --no-cache-dir --index-url https://pypi.org/simple -c "$wheel_dir/constraints.txt" \
        "${packages[@]}" "hyper-parallel==${HP_VERSION:?Missing HP_VERSION build argument}"
    python3 -I -c 'from importlib.metadata import version; import sys; sys.exit(version("hyper-parallel") != sys.argv[1])' "$HP_VERSION"
    "${pip[@]}" check
}

case ${1:-} in
    _checkout-source) checkout_source; exit ;;
    _build-wheel) build_wheel; exit ;;
    _install-runtime) install_runtime; exit ;;
esac

usage() {
    cat <<'HELP'
Usage: bash docker/build.sh --npu-type PRODUCT [options]
  -t, --npu-type, -c, --chip  a2 (910b), a3 or a5 (950); required for builds
  -o, --os OS                 ubuntu22.04 or openeuler24.03 (default: both)
  -a, --arch ARCH             arm64/aarch64 or amd64/x86_64 (default: host)
      --python-version X.Y    3.10, 3.11 or 3.12 (default: preferred PTA base)
      --hyper-parallel-branch NAME  Build from an official HyperParallel branch
      --hyper-parallel-commit SHA  Build from a full 40-character commit instead of a branch
      --hp-version V          Expected HyperParallel wheel version (custom source: read setup.py by default)
      --cann-version V        Select a CANN release, e.g. 9.2.0 or 9.0.0
      --torch-version V       Select a PyTorch release in that CANN matrix
      --torch-npu-version V   Exact TorchNPU package version (alias: --pta-version)
      --network MODE          Docker build network, e.g. host (default: Docker's build network)
      --print-command         Print the resolved native docker build command for each target
      --no-cache              Build without Docker's layer cache
      --tags                  Print matching image names
      --push                  Build both OS images before pushing the pair to Quay
      --docs | --check        Update or verify the two overviews in docker/
      --export DIR            Export the recipe and bilingual pages to a new or empty directory
  -h, --help                  Show this help
HELP
}

# One record per Linux architecture. Product aliases stay independent of AscendHub's tag spelling.
add_target() {
    local chip=$1 os=$2 python=$3 cann=$4 pta=$5 base=$6 product arch tag
    case $chip in
        910b) product=a2 ;;
        a3) product=a3 ;;
        950) product=a5 ;;
        *) die "Unknown Ascend chip $chip" ;;
    esac
    tag="v${hp_version//+/_}-cann$cann-torch_npu$pta-$chip-$os-py$python"
    for arch in arm64 amd64; do
        images+=("$product $chip $os $python $arch $cann $pta $base $tag-$arch-g${hp_revision:0:12}")
    done
}

read_recipe() {
    local directive assignment key value base name pta cann chip os python default_base
    local -A args=() versions=()
    local -a bases=() cann_bases=() compatible_pta=()
    while read -r directive assignment _; do
        [[ $directive != FROM ]] || break
        [[ $directive == ARG && $assignment == *=* ]] || continue
        key=${assignment%%=*}
        value=${assignment#*=}
        args[$key]=${value%$'\r'}
        if [[ $key == BASE_* && $key != BASE_IMAGE ]]; then bases+=("$key"); fi
    done < "$docker_dir/Dockerfile"
    hp_version=${version_override:-${args[HP_VERSION]:?missing HP_VERSION}}
    hp_revision=${args[DEFAULT_HYPER_PARALLEL_COMMIT]:?missing DEFAULT_HYPER_PARALLEL_COMMIT}
    if [[ $hp_branch || $hp_commit ]]; then
        local ref resolved
        ref=$(source_ref "$hp_branch" "$hp_commit")
        if [[ $ref != "$hp_revision" ]]; then
            if [[ $ref == refs/heads/* || -z $version_override ]]; then
                resolved=$(resolve_source "$ref" "$version_override")
                hp_revision=${resolved%% *}
                hp_version=${resolved#* }
            else
                hp_revision=$ref
            fi
        fi
    fi
    [[ $hp_revision =~ ^[0-9a-f]{40}$ ]] || die 'Default HP revision must be a full Git commit'
    [[ $hp_version =~ ^[0-9][0-9a-zA-Z.+_-]*$ ]] || die "Invalid HP package version: $hp_version"
    default_base=${args[BASE_IMAGE]:?missing BASE_IMAGE}
    local suffix='-(910b|a3|950)-(ubuntu[0-9.]+|openeuler[0-9.]+)-py(3\.[0-9]+)@sha256:[0-9a-f]{64}$'
    local pta_pattern="^.+/torch-npu:([0-9.]+(\.post[0-9]+)?)(-cann([0-9.]+))?$suffix"
    local cann_pattern="^.+/cann:([0-9.]+)$suffix"
    for name in "${bases[@]}"; do
        base=${args[$name]}
        if [[ $base =~ $cann_pattern ]]; then cann_bases+=("$base"); continue; fi
        [[ $base =~ $pta_pattern ]] || die "Invalid AscendHub manifest in $name"
        pta=${BASH_REMATCH[1]} cann=${BASH_REMATCH[4]:-${args[LEGACY_CANN]}}
        chip=${BASH_REMATCH[5]} os=${BASH_REMATCH[6]} python=${BASH_REMATCH[7]}
        versions["$cann/$chip/$os"]+=" $pta"
        preferred_pta[$cann]=${preferred_pta[$cann]:-$pta}
        base_python[$pta]=$python
        add_target "$chip" "$os" "$python" "$cann" "$pta" "$base"
        if [[ $default_base == "$base" || $default_base == "\${$name}" ]]; then
            default_python=$python default_pta=$pta default_cann=$cann
        fi
    done
    [[ $default_python ]] || die 'BASE_IMAGE must select a declared PTA base'

    preferred_pta[$default_cann]=$default_pta
    for base in "${cann_bases[@]}"; do
        [[ $base =~ $cann_pattern ]] || die "Invalid CANN base: $base"
        cann=${BASH_REMATCH[1]} chip=${BASH_REMATCH[2]} os=${BASH_REMATCH[3]} python=${BASH_REMATCH[4]}
        read -ra compatible_pta <<< "${versions[$cann/$chip/$os]:?No matching PTA release for this CANN base}"
        for pta in "${compatible_pta[@]}"; do add_target "$chip" "$os" "$python" "$cann" "$pta" "$base"; done
    done
}

render_overview() {
    local template=$1 link_prefix=$2 zh=0
    shift 2
    [[ $template != *.zh.md ]] || zh=1
    awk -v zh="$zh" \
        -v default_py="$default_python" -v default_pta="$default_pta" -v repo="$source_repo" \
        -v registry="$registry" -v url="$link_prefix" -v ver="$hp_version" -v rev="$hp_revision" '
        function add(values, field, key, item) {
            if (!indexed[key, item, field]++) values[key] = values[key] (values[key] ? ", " : "") item
        }
        NR == FNR {
            key = $6 SUBSEP $7
            if (!(key in stacks)) {
                order[++n] = key
                torch = $7; sub(/\.post[0-9]+$/, "", torch)
                stacks[key] = sprintf("| %s | %s | %s | ", $6, torch, $7)
            }
            py[key, $4] = 1
            minor=$4; sub(/^3\./, "", minor)
            if (minor+0 > max_minor) max_minor=minor+0
            add(systems, "os", key, $3)
            add(chips, "chip", key, $2)
            tags[++count] = sprintf("| %s | %s | `%s` |", $2, $5, $9)
            if (FNR == 1) {
                chip=$2; os=$3; python=$4; arch=$5; cann=$6; pta=$7; base=$8; tag=$9
            }
            next
        }
        { sub(/\r$/, "") }
        /^<!-- BEGIN generated / {
            if (section || seen_section[$0]++) exit 1
            section = $0; sub(/^<!-- BEGIN generated /, "", section); sub(/ -->$/, "", section)
            print; print ""
            if (section == "support matrix") {
                print zh ? "| CANN | PyTorch | torch_npu (PTA) | Python | 操作系统 | 芯片 |" : "| CANN | PyTorch | torch_npu (PTA) | Python | OS | Chips |"
                print "| --- | --- | --- | --- | --- | --- |"
                for (i=1; i<=n; i++) {
                    key=order[i]; versions=""
                    for (minor=0; minor<=max_minor; minor++) {
                        v="3." minor
                        if (py[key, v]) versions=versions (versions ? ", " : "") v
                    }
                    print stacks[key] versions " | " systems[key] " | " chips[key] " |"
                }
                print ""
                printf (zh ? "脚本默认使用 Python %s 和 PTA %s。" : "The script defaults to Python %s and PTA %s."), default_py, default_pta
                printf " [%s](%s/blob/%s/requirements.txt)\n", (zh ? "HyperParallel 完整 Python 依赖" : "HyperParallel Python requirements"), repo, rev
            } else if (section == "image tags") {
                print zh ? "| 芯片 | CPU | 完整版本标签 |" : "| Chip | CPU | Full image tag |"
                print "| --- | --- | --- |"; for (i=1; i<=count; i++) print tags[i]
            } else if (section == "build example") {
                print "```bash"
                printf "bash docker/build.sh -t %s -o %s -a %s --cann-version %s --python-version %s --pta-version %s\n", chip, os, arch, cann, python, pta
                print "IMAGE=" registry ":" tag
                print "```"
            } else if (section == "docker build") {
                print "```bash"
                print "docker build --platform linux/" arch " \\"
                print "  --build-arg BASE_IMAGE=" base " \\"
                torch=pta; sub(/\.post[0-9]+$/, "", torch)
                print "  --build-arg CANN_VERSION=" cann " --build-arg TORCH_VERSION=" torch " \\"
                print "  --build-arg PYTHON_VERSION=" python " --build-arg TORCH_NPU_VERSION=" pta " \\"
                print "  --build-arg HYPER_PARALLEL_COMMIT=" rev " \\"
                print "  --build-arg HP_VERSION=" ver " \\"
                print "  -f docker/Dockerfile -t " registry ":" tag " docker/"
                print "```"
            } else exit 1
            next
        }
        /^<!-- END generated / {
            if ($0 != "<!-- END generated " section " -->") exit 1
            print ""; print; section=""; next
        }
        !section { if (url) gsub(/\]\(\.\//, "](" url "/"); print }
        END { if (section || length(seen_section) != 4) exit 1 }
    ' <(printf '%s\n' "$@") "$template"
}

generate_pages() (
    local action=$1 directory=${2:-} url=.. name image tag tmp parent
    if [[ $action == export ]]; then
        directory=$(realpath -m -- "$directory")
        parent=$(dirname -- "$directory")
        mkdir -p -- "$parent"
        tmp=$(mktemp -d -- "$parent/.hyperparallel-export.XXXXXX")
    else
        tmp=$(mktemp -d)
    fi
    trap 'rm -rf -- "$tmp"' EXIT

    if [[ $action == export ]]; then
        # Published per-tag pages can link back to an exact, clean checkout.
        if [[ $(git -C "$docker_dir" rev-parse --show-prefix 2>/dev/null) == docker/ ]] &&
            git -C "$docker_dir" ls-files --error-unmatch -- Dockerfile build.sh OVERVIEW.md OVERVIEW.zh.md >/dev/null 2>&1 &&
            git -C "$docker_dir" diff --quiet HEAD -- Dockerfile build.sh OVERVIEW.md OVERVIEW.zh.md; then
            url="$source_repo/blob/$(git -C "$docker_dir" rev-parse HEAD)/docker"
        fi
        cp -- "$docker_dir/Dockerfile" "$docker_dir/build.sh" "$tmp/"
    fi
    for name in OVERVIEW.md OVERVIEW.zh.md; do
        render_overview "$docker_dir/$name" '' "${images[@]}" > "$tmp/$name" || die "Invalid generated sections in $name"
    done
    for name in OVERVIEW.md OVERVIEW.zh.md; do
        case $action in
            check) diff -u --strip-trailing-cr "$docker_dir/$name" "$tmp/$name" || die "Refresh $name with --docs" ;;
            docs) cmp -s "$tmp/$name" "$docker_dir/$name" || cp -- "$tmp/$name" "$docker_dir/$name" ;;
            export)
                for image in "${images[@]}"; do
                    tag=${image##* }
                    mkdir -p -- "$tmp/$tag"
                    render_overview "$docker_dir/$name" "$url" "$image" > "$tmp/$tag/$name" || die "Could not render $tag"
                done ;;
        esac
    done
    if [[ $action == export ]] && ! mv -T -- "$tmp" "$directory" 2>/dev/null; then
        [[ -d $directory ]] && diff -qr -- "$tmp" "$directory" >/dev/null && exit 0
        die 'Export to a new or empty directory to preserve its existing files'
    fi
)

mode=build
npu=''
target_os=''
target_arch=$(uname -m)
python_version=''
pta_version=''
torch_version=''
cann_version=''
hp_branch=''
hp_commit=''
version_override=''
build_flags=()
export_dir=''
while (($#)); do
    option=$1
    shift
    if [[ $option == --*=* ]]; then
        set -- "${option#*=}" "$@"
        option=${option%%=*}
    fi
    case $option in
        -h|--help) usage; exit 0 ;;
        -t|--npu-type|-c|--chip) npu=${1:?"$option needs a product"}; shift ;;
        -o|--os) target_os=${1:?"$option needs an OS"}; shift ;;
        -a|--arch) target_arch=${1:?"$option needs an architecture"}; shift ;;
        --python-version) python_version=${1:?"$option needs a Python version"}; shift ;;
        --hyper-parallel-branch|--hp-branch) hp_branch=${1:?"$option needs a branch"}; shift ;;
        --hyper-parallel-commit|--hp-commit) hp_commit=${1:?"$option needs a full commit SHA"}; shift ;;
        --hp-version) version_override=${1:?"$option needs a package version"}; shift ;;
        --cann-version) cann_version=${1:?"$option needs a CANN version"}; shift ;;
        --network) build_flags+=(--network "${1:?"$option needs a Docker build network"}"); shift ;;
        --torch-version) torch_version=${1:?"$option needs a PyTorch version"}; shift ;;
        --torch-npu-version|--pta-version) pta_version=${1:?"$option needs a PTA version"}; shift ;;
        --no-cache) build_flags+=(--no-cache) ;;
        --tags|--print-command|--push|--docs|--check|--export)
            [[ $mode == build ]] || die 'Choose one operation'
            mode=${option#--}
            if [[ $mode == export ]]; then export_dir=${1:?'--export needs a directory'}; shift; fi ;;
        *) die "Unknown option: $option (see --help)" ;;
    esac
done

images=()
default_python=''
default_pta=''
default_cann=''
declare -A base_python=() preferred_pta=()
case $mode in
    docs|check|export) [[ -z $hp_branch$hp_commit$version_override ]] || die 'Release documentation uses the pinned default HP source' ;;
esac
read_recipe
case $mode in docs|check|export) generate_pages "$mode" "$export_dir"; exit ;; esac
[[ $mode == tags || $npu ]] || die 'Select an Ascend product with --npu-type'
[[ $mode != push || -z $target_os ]] || die '--push publishes both operating systems; omit --os'
case $target_arch in
    aarch64|arm64) target_arch=arm64 ;;
    x86_64|amd64) target_arch=amd64 ;;
    *) die "Unsupported CPU architecture: $target_arch" ;;
esac
npu=${npu,,}
if [[ -z $cann_version && -z $pta_version ]]; then cann_version=$default_cann; fi
if [[ $cann_version && ! ${preferred_pta[$cann_version]:-} ]]; then die "Unknown CANN release: $cann_version"; fi
if [[ -z $pta_version && -z $torch_version ]]; then pta_version=${preferred_pta[$cann_version]}; fi
if [[ -z $python_version ]]; then
    preferred=${pta_version:-${preferred_pta[$cann_version]}}
    python_version=${base_python[$preferred]:-$default_python}
fi
selected=()
declare -A systems=()
for image in "${images[@]}"; do
    read -r product chip os python arch cann pta _ <<< "$image"
    [[ -z $npu || $npu == "$product" || $npu == "$chip" ]] || continue
    [[ $arch == "$target_arch" && ( -z $target_os || $os == "$target_os" ) ]] || continue
    [[ $python == "$python_version" && ( -z $cann_version || $cann == "$cann_version" ) ]] || continue
    [[ -z $pta_version || $pta == "$pta_version" ]] || continue
    [[ -z $torch_version || ${pta%.post*} == "$torch_version" ]] || continue
    selected+=("$image")
    systems[${os%%[0-9]*}]=1
done
(( ${#selected[@]} )) || die "No build target for product=${npu:-any}, CANN=${cann_version:-any}, Python=${python_version:-any}, PyTorch=${torch_version:-default}, PTA=${pta_version:-default}, OS=${target_os:-both}, CPU=$target_arch"
if [[ $mode == push ]]; then
    [[ ${systems[ubuntu]:-} && ${systems[openeuler]:-} ]] || die 'Publication requires Ubuntu and openEuler'
    bash "$docker_dir/build.sh" --check
fi
platform_flags=(--platform "linux/$target_arch")
if [[ $mode != tags ]] && command -v docker >/dev/null; then
    # Old engines can build natively but gate --platform behind experimental features.
    engine=$(docker version --format '{{.Server.APIVersion}} {{.Server.Arch}}' 2>/dev/null) || engine=''
    if [[ $engine =~ ^1\.([0-9]+)[[:space:]]+(arm64|amd64)$ && ${BASH_REMATCH[1]} -lt 40 ]]; then
        if [[ ${BASH_REMATCH[2]} == "$target_arch" ]]; then
            platform_flags=()
        elif [[ $mode != print-command ]]; then
            die "This Docker engine only supports native ${BASH_REMATCH[2]} builds"
        fi
    fi
fi
for image in "${selected[@]}"; do
    read -r _ _ _ python arch cann pta base tag <<< "$image"
    if [[ $mode == tags ]]; then
        printf '%s:%s\n' "$registry" "$tag"
    else
        command=(docker build ${platform_flags[@]+"${platform_flags[@]}"} ${build_flags[@]+"${build_flags[@]}"}
            --build-arg "BASE_IMAGE=$base" --build-arg "CANN_VERSION=$cann" --build-arg "PYTHON_VERSION=$python"
            --build-arg "TORCH_VERSION=${pta%.post*}" --build-arg "TORCH_NPU_VERSION=$pta"
            --build-arg "HYPER_PARALLEL_COMMIT=$hp_revision" --build-arg "HP_VERSION=$hp_version"
            -f "$docker_dir/Dockerfile" -t "$registry:$tag" "$docker_dir")
        if [[ $mode == print-command ]]; then
            printf '%q' "${command[0]}"; printf ' %q' "${command[@]:1}"; printf '\n'
        else
            "${command[@]}"
        fi
    fi
done
if [[ $mode == push ]]; then
    for image in "${selected[@]}"; do docker push "$registry:${image##* }"; done
fi
