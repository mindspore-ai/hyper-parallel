#!/usr/bin/env bash
# Deprecated compatibility entrypoint. Use run_qwen3_4b_agentic_docker.sh.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
printf 'run_qwen35_agentic_docker.sh is deprecated; running the Qwen3-4B launcher\n' >&2
exec "${script_dir}/run_qwen3_4b_agentic_docker.sh" "$@"
