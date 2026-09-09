#!/usr/bin/env bash
# SHMEM + MegaMoe 全量回归(cpp UT + python UT + ST): bash run_shmem_mega_moe_regression.sh
# 卡数要求见每条注释, 按环境裁剪对应行即可(如注释掉 8 卡用例)
set -uo pipefail

CASE_TIMEOUT_SECONDS="${CASE_TIMEOUT_SECONDS:-900}"

# megamoe 自定义算子 OPP 环境, 必须在 pytest 启动前 source (对 shmem ST 同样生效, 无害)
source build/native/payload/hyper_parallel/core/multicore/lib/set_env.bash

LOG="shmem_mega_moe_regression_$(date +%Y%m%d_%H%M%S).log"

CASES=(
    # ---- SHMEM UT (host, 无需 NPU) ----
    "tests/ut/core/multicore/shmem/test_cpp.py"           # cpp UT, session fixture 自动 cmake 增量构建
    "tests/ut/core/multicore/shmem/test_runtime.py"       # python UT, mock native 边界
    "tests/ut/core/multicore/shmem/test_api.py"           # python UT, Allocation debug 边界
    "tests/ut/core/multicore/test_multicore_boundary.py"  # 边界: 惰性 native 加载

    # ---- SHMEM ST: binding (真 NPU, torchrun) ----
    "tests/torch/multicore/shmem/test_binding.py::test_binding_allocation_and_release"                  # 1卡
    "tests/torch/multicore/shmem/test_binding.py::test_binding_single_process_without_distributed"      # 1卡
    "tests/torch/multicore/shmem/test_binding.py::test_binding_inactive_runtime_hint"                   # 1卡
    "tests/torch/multicore/shmem/test_binding.py::test_all_gather_output_input_overlap_rejected"        # 1卡
    "tests/torch/multicore/shmem/test_binding.py::test_all_gather_output_size_mismatch_rejected"        # 1卡
    "tests/torch/multicore/shmem/test_binding.py::test_binding_storage_assoc_double_free"               # 1卡
    "tests/torch/multicore/shmem/test_binding.py::test_binding_orphan_storage_warning"                  # 1卡
    "tests/torch/multicore/shmem/test_binding.py::test_binding_stale_tensor_rejected_by_one_sided_ops"  # 1卡
    "tests/torch/multicore/shmem/test_binding.py::test_binding_error_projection"                        # 1卡
    "tests/torch/multicore/shmem/test_binding.py::test_binding_init_timeout_injection"                  # 1卡
    "tests/torch/multicore/shmem/test_binding.py::test_binding_reinit_with_different_heap_sizes"        # 2卡
    "tests/torch/multicore/shmem/test_binding.py::test_binding_final_release_with_active_allocation_is_retryable"  # 2卡
    "tests/torch/multicore/shmem/test_binding.py::test_binding_put_get"                                 # 2卡
    "tests/torch/multicore/shmem/test_binding.py::test_binding_barrier_blocking_and_config"             # 2卡
    "tests/torch/multicore/shmem/test_binding.py::test_binding_signal_and_pull"                         # 2卡

    # ---- SHMEM ST: all_gather (真 NPU, torchrun) ----
    "tests/torch/multicore/shmem/test_all_gather.py::test_all_gather_boundary_matrix_1_rank"            # 1卡
    "tests/torch/multicore/shmem/test_all_gather.py::test_all_gather_boundary_matrix_2_ranks"           # 2卡
    "tests/torch/multicore/shmem/test_all_gather.py::test_all_gather_boundary_matrix_4_ranks"           # 4卡
    "tests/torch/multicore/shmem/test_all_gather.py::test_all_gather_boundary_matrix_8_ranks"           # 8卡
    "tests/torch/multicore/shmem/test_all_gather.py::test_all_gather_correctness_large"                 # 4卡
    "tests/torch/multicore/shmem/test_all_gather.py::test_all_gather_stream_order_and_visibility"       # 4卡
    "tests/torch/multicore/shmem/test_all_gather.py::test_runtime_device_guard_on_real_npu"             # 1进程，需2卡

    # ---- MegaMoe UT ----
    "tests/ut/core/multicore/modules/mega_moe/test_module.py"
    "tests/ut/core/multicore/modules/mega_moe/test_workspace.py"
    "tests/ut/core/multicore/modules/mega_moe/test_route.py"

    # ---- MegaMoe ST (launcher 内部 torchrun) ----
    "tests/torch/multicore/test_mega_moe.py::test_mega_moe_level0_precision"             # 2卡
    "tests/torch/multicore/test_mega_moe.py::test_mega_moe_shared_resources"             # 2卡
    "tests/torch/multicore/test_mega_moe.py::test_mega_moe_default_interface"            # 4卡
    "tests/torch/multicore/test_mega_moe.py::test_mega_moe_representative_performance"   # 4卡
)

pass=0
fail=0
idx=0

{
    echo "=== SHMEM + MegaMoe regression started at $(date) ==="
    for case in "${CASES[@]}"; do
        # 每条用例独立 HCCL NPU socket 端口段, 避免串行用例间端口互踩(EI0020 Bind_IP_Port)
        export HCCL_NPU_SOCKET_PORT_RANGE="$((41000 + idx * 100))-$((41099 + idx * 100))"
        idx=$((idx + 1))
        echo
        echo "================ RUN: ${case} (hccl_port_range=${HCCL_NPU_SOCKET_PORT_RANGE}) ================"
        timeout --signal=TERM --kill-after=30s "${CASE_TIMEOUT_SECONDS}s" pytest -s "${case}"
        rc=$?
        echo "================ EXIT: ${case} rc=${rc} ================"
        if [ ${rc} -eq 0 ]; then
            pass=$((pass + 1))
        else
            fail=$((fail + 1))
            if [ ${rc} -eq 124 ]; then
                echo "TIMEOUT (${CASE_TIMEOUT_SECONDS}s): ${case}"
            else
                echo "FAILED: ${case}"
            fi
        fi
    done
    echo
    echo "=== Summary: pass=${pass} fail=${fail} ==="
    echo "=== SHMEM + MegaMoe regression finished at $(date) ==="
    if [ ${fail} -ne 0 ]; then
        exit 1
    fi
} 2>&1 | tee "${LOG}"
