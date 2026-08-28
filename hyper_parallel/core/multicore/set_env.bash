# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Required runtime environment for the packaged HyperParallel multicore custom OPP vendor.

_hp_multicore_lib=$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
_hp_multicore_vendor="${_hp_multicore_lib}/vendors/hyper_parallel_multicore_nn"
_hp_multicore_op_api="${_hp_multicore_vendor}/op_api/lib"

if [[ ! -d "${_hp_multicore_vendor}" || ! -f "${_hp_multicore_op_api}/libcust_opapi.so" ]]; then
    echo "[HP-NATIVE-PAYLOAD-MISSING] multicore vendor is incomplete under ${_hp_multicore_vendor}." >&2
    unset _hp_multicore_lib _hp_multicore_vendor _hp_multicore_op_api
    return 1 2>/dev/null || exit 1
fi

case ":${ASCEND_CUSTOM_OPP_PATH:-}:" in
    *":${_hp_multicore_vendor}:"*) ;;
    *) export ASCEND_CUSTOM_OPP_PATH="${_hp_multicore_vendor}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}" ;;
esac
case ":${LD_LIBRARY_PATH:-}:" in
    *":${_hp_multicore_op_api}:"*) ;;
    *) export LD_LIBRARY_PATH="${_hp_multicore_op_api}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
esac

unset _hp_multicore_lib _hp_multicore_vendor _hp_multicore_op_api
