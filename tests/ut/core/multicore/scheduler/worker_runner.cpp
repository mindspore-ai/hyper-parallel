/*
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifdef __DAV_C220_CUBE__
#define MulticoreRuntime TestCubeRuntime
#define TestWorker TestCubeWorker
#else
#define MulticoreRuntime TestVectorRuntime
#define TestWorker TestVectorWorker
#endif
#include "hyper_parallel/core/multicore/ops/runtime/worker_kernel.h"

class TestWorker : public MulticoreRuntime::KernelWorkerBase<TestWorker> {
 public:
  static constexpr uint32_t TILING_IDX = 0;
  static constexpr uint32_t EVENT_IDX = 1;
  static constexpr uint32_t PROFILE_IDX = 2;

  void ExecuteComputeKernel(MulticoreRuntime::TaskDesc task) { AscendC::compute_callback(task.task_index); }
};

#ifdef _WIN32
#define TEST_EXPORT __declspec(dllexport)
#else
#define TEST_EXPORT
#endif

extern "C" TEST_EXPORT bool run_worker(uint32_t worker_id, uint8_t *runtime, int64_t *tiling, int32_t *events,
                                       uint8_t *profile, void (*compute)(uint32_t), void (*barrier)()) {
  AscendC::compute_callback = compute;
  AscendC::barrier_callback = barrier;
  GM_ADDR inputs[] = {reinterpret_cast<GM_ADDR>(tiling), reinterpret_cast<GM_ADDR>(events), profile};
  try {
    TestWorker worker;
    worker.Init(worker_id, runtime, inputs);
    worker.Process();
    return true;
  } catch (const std::runtime_error &) {
    return false;
  }
}
