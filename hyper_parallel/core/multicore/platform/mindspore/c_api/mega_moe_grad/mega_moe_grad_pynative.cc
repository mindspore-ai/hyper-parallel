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
// PYBOOST MODE IMPLEMENTATION — mega_moe_grad
//
// Uses AclnnOpRunner + LAUNCH_ACLNN_FUNC so that GetWorkspaceSize, executor
// caching, and workspace allocation are handled transparently by the MindSpore
// framework.  The function name "aclnnHyperMegaMoeGrad" is resolved at
// runtime from the loaded libcust_opapi.so — no compile-time declaration needed.
//
// This op is INPLACE with 7 declared returns:
//   dispatch_target, hidden_dw, act_grad_y, grad_gate, gate_dx, grad_x, gate_dw
// (permute_out is written in-place but NOT a declared return in the YAML)
//
// Input tensor positions (for the runner->Run inputs vector):
//   [0]  dispatch_target      [inplace write, declared return]
//   [1]  dispatch_target_off
//   [2]  dy
//   [3]  dispatch_src_off
//   [4]  dispatch_size
//   [5]  hidden
//   [6]  hidden_dw            [inplace write, declared return]
//   [7]  w2
//   [8]  act_grad_y           [inplace write, declared return]
//   [9]  gate
//   [10] grad_gate            [inplace write, declared return]
//   [11] w1
//   [12] gate_dx              [inplace write, declared return]
//   [13] grad_x               [inplace write, declared return]
//   [14] combine_target_off
//   [15] combine_src_off
//   [16] combine_size
//   [17] permute_out          [inplace write, NOT a declared return]
//   [18] gate_dw              [inplace write, declared return]
//   [19] group_list
//   [20] act_grad_tiling
//   [21] gate_grad_tiling
//   [22] w1_grad_tiling
//   [23] w2_grad_tiling
//   [24] swiglu_grad_tiling
//   [25] gmm_workspace
//   [26] swiglu_grad_workspace
//   [27] runtime_config
//   [28] all_event_counters
// =============================================================================

#include <memory>
#include <vector>

#include "framework/module.h"
#include "include/kernel/ascend/custom/pyboost_impl/aclnn_op_runner.h"

namespace ms_multicore {

// npu_mega_moe_grad — returns the 7 inplace output tensors declared in YAML
std::vector<ms::Tensor> npu_mega_moe_grad(
    const ms::Tensor &dispatch_target,      // pos  0 — inplace write, declared return
    const ms::Tensor &dispatch_target_off,  // pos  1
    const ms::Tensor &dy,                   // pos  2
    const ms::Tensor &dispatch_src_off,     // pos  3
    const ms::Tensor &dispatch_size,        // pos  4
    const ms::Tensor &hidden,               // pos  5
    const ms::Tensor &hidden_dw,            // pos  6 — inplace write, declared return
    const ms::Tensor &w2,                   // pos  7
    const ms::Tensor &act_grad_y,           // pos  8 — inplace write, declared return
    const ms::Tensor &gate,                 // pos  9
    const ms::Tensor &grad_gate,            // pos 10 — inplace write, declared return
    const ms::Tensor &w1,                   // pos 11
    const ms::Tensor &gate_dx,              // pos 12 — inplace write, declared return
    const ms::Tensor &grad_x,              // pos 13 — inplace write, declared return
    const ms::Tensor &combine_target_off,  // pos 14
    const ms::Tensor &combine_src_off,     // pos 15
    const ms::Tensor &combine_size,        // pos 16
    const ms::Tensor &permute_out,         // pos 17 — inplace write (not a declared return)
    const ms::Tensor &gate_dw,             // pos 18 — inplace write, declared return
    const ms::Tensor &group_list,          // pos 19
    const ms::Tensor &act_grad_tiling,     // pos 20
    const ms::Tensor &gate_grad_tiling,    // pos 21
    const ms::Tensor &w1_grad_tiling,      // pos 22
    const ms::Tensor &w2_grad_tiling,      // pos 23
    const ms::Tensor &swiglu_grad_tiling,  // pos 24
    const ms::Tensor &gmm_workspace,       // pos 25
    const ms::Tensor &swiglu_grad_workspace,  // pos 26
    const ms::Tensor &runtime_config,      // pos 27
    const ms::Tensor &all_event_counters,  // pos 28
    int64_t rank_id, int64_t ep, int64_t expert_num,
    int64_t hidden_size, int64_t seq_size) {
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("MoeBwd");
  MS_EXCEPTION_IF_NULL(runner);
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnHyperMegaMoeGrad,
      dispatch_target, dispatch_target_off,
      dy, dispatch_src_off, dispatch_size,
      hidden, hidden_dw,
      w2, act_grad_y,
      gate, grad_gate,
      w1, gate_dx, grad_x,
      combine_target_off, combine_src_off, combine_size,
      permute_out, gate_dw, group_list,
      act_grad_tiling, gate_grad_tiling, w1_grad_tiling, w2_grad_tiling,
      swiglu_grad_tiling, gmm_workspace, swiglu_grad_workspace,
      runtime_config, all_event_counters,
      rank_id, ep, expert_num, hidden_size, seq_size));

  std::vector<ms::Tensor> inputs = {
      dispatch_target, dispatch_target_off,
      dy, dispatch_src_off, dispatch_size,
      hidden, hidden_dw,
      w2, act_grad_y,
      gate, grad_gate,
      w1, gate_dx, grad_x,
      combine_target_off, combine_src_off, combine_size,
      permute_out, gate_dw, group_list,
      act_grad_tiling, gate_grad_tiling, w1_grad_tiling, w2_grad_tiling,
      swiglu_grad_tiling, gmm_workspace, swiglu_grad_workspace,
      runtime_config, all_event_counters,
  };
  // outputs = the 7 inplace tensors declared as returns in mega_moe_grad_op.yaml
  runner->Run(inputs, {});

  return {dispatch_target, hidden_dw, act_grad_y, grad_gate, gate_dx, grad_x, gate_dw};
}
}  // namespace ms_multicore

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("mega_moe_grad", PYBOOST_CALLER(7, ms_multicore::npu_mega_moe_grad));
}
