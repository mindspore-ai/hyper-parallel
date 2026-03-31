/**
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

// =============================================================================
// PYBOOST MODE IMPLEMENTATION — moe_ffn_fwd
//
// Uses AclnnOpRunner + LAUNCH_ACLNN_FUNC so that GetWorkspaceSize, executor
// caching, and workspace allocation are handled transparently by the MindSpore
// framework.  The function name "aclnnMegaKernelGmm" is resolved at runtime
// from the loaded libcust_opapi.so — no compile-time declaration is needed.
//
// This op is INPLACE with 5 declared returns:
//   dispatch_target, up_proj_y, swiglu_out, down_proj_y, combine_target
//
// Following the pyboost_aclnn_batch_norm.cpp pattern:
//   - Inplace output tensors appear in BOTH the inputs and outputs vectors
//   - The function returns std::vector<ms::Tensor> containing the 5 outputs
//   - No separate pyboost wrapper; npu_moe_ffn_fwd is bound directly
//
// Input tensor positions (for the runner->Run inputs vector):
//   [0]  dispatch_target      [inplace write, declared return]
//   [1]  dispatch_target_off
//   [2]  dispatch_src
//   [3]  dispatch_src_off
//   [4]  dispatch_size
//   [5]  up_proj_weight
//   [6]  up_proj_glist
//   [7]  up_proj_y            [inplace write, declared return]
//   [8]  swiglu_out           [inplace write, declared return]
//   [9]  down_proj_weight
//   [10] down_proj_glist
//   [11] down_proj_y          [inplace write, declared return]
//   [12] combine_target       [inplace write, declared return]
//   [13] combine_target_off
//   [14] combine_src_off
//   [15] combine_size
//   [16] gmm_workspace
//   [17] up_proj_tiling
//   [18] swiglu_tiling
//   [19] down_proj_tiling
//   [20] runtime_config
//   [21] all_event_counters
// =============================================================================

#include <memory>
#include <vector>

#include "framework/module.h"
#include "include/kernel/ascend/custom/pyboost_impl/aclnn_op_runner.h"

namespace ms_multicore {

// npu_moe_ffn_fwd — returns the 5 inplace output tensors
std::vector<ms::Tensor> npu_moe_ffn_fwd(
    const ms::Tensor &dispatch_target,      // pos  0 — inplace write, declared return
    const ms::Tensor &dispatch_target_off,  // pos  1
    const ms::Tensor &dispatch_src,         // pos  2
    const ms::Tensor &dispatch_src_off,     // pos  3
    const ms::Tensor &dispatch_size,        // pos  4
    const ms::Tensor &up_proj_weight,       // pos  5
    const ms::Tensor &up_proj_glist,        // pos  6
    const ms::Tensor &up_proj_y,            // pos  7 — inplace write, declared return
    const ms::Tensor &swiglu_out,           // pos  8 — inplace write, declared return
    const ms::Tensor &down_proj_weight,     // pos  9
    const ms::Tensor &down_proj_glist,      // pos 10
    const ms::Tensor &down_proj_y,          // pos 11 — inplace write, declared return
    const ms::Tensor &combine_target,       // pos 12 — inplace write, declared return
    const ms::Tensor &combine_target_off,   // pos 13
    const ms::Tensor &combine_src_off,      // pos 14
    const ms::Tensor &combine_size,         // pos 15
    const ms::Tensor &gmm_workspace,        // pos 16
    const ms::Tensor &up_proj_tiling,       // pos 17
    const ms::Tensor &swiglu_tiling,        // pos 18
    const ms::Tensor &down_proj_tiling,     // pos 19
    const ms::Tensor &runtime_config,       // pos 20
    const ms::Tensor &all_event_counters,   // pos 21
    int64_t rank_id, int64_t ep, int64_t expert_num,
    int64_t hidden_size, int64_t seq_size) {
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("MoeFfnFwd");
  MS_EXCEPTION_IF_NULL(runner);
  // LAUNCH_ACLNN_FUNC captures all args via Arg() helpers (ms::Tensor →
  // TensorPtr, scalars pass through), then LAUNCH_ACLNN internally:
  //   1. calls aclnnMegaKernelGmmGetWorkspaceSize (via dynamic lookup + cache)
  //   2. allocates workspace through MindSpore's device memory manager
  //   3. dispatches aclnnMegaKernelGmm asynchronously on the current stream
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnMegaKernelGmm,
      dispatch_target, dispatch_target_off,
      dispatch_src, dispatch_src_off, dispatch_size,
      up_proj_weight, up_proj_glist,
      up_proj_y, swiglu_out,
      down_proj_weight, down_proj_glist, down_proj_y,
      combine_target, combine_target_off, combine_src_off, combine_size,
      gmm_workspace, up_proj_tiling, swiglu_tiling, down_proj_tiling,
      runtime_config, all_event_counters,
      rank_id, ep, expert_num, hidden_size, seq_size));

  // All 22 tensors passed as inputs so the framework prepares their device
  // addresses before the async launch.
  // The 5 inplace output tensors also appear in outputs so the framework
  // tracks them as written — following pyboost_aclnn_batch_norm.cpp pattern.
  std::vector<ms::Tensor> inputs = {
      dispatch_target, dispatch_target_off,
      dispatch_src, dispatch_src_off, dispatch_size,
      up_proj_weight, up_proj_glist,
      up_proj_y, swiglu_out,
      down_proj_weight, down_proj_glist, down_proj_y,
      combine_target, combine_target_off, combine_src_off, combine_size,
      gmm_workspace, up_proj_tiling, swiglu_tiling, down_proj_tiling,
      runtime_config, all_event_counters,
  };
  // outputs = the 5 inplace tensors declared as returns in moe_ffn_fwd_op.yaml
  runner->Run(inputs, {});

  return {dispatch_target, up_proj_y, swiglu_out, down_proj_y, combine_target};
}
}  // namespace ms_multicore

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("moe_ffn_fwd", PYBOOST_CALLER(5, ms_multicore::npu_moe_ffn_fwd));
}
