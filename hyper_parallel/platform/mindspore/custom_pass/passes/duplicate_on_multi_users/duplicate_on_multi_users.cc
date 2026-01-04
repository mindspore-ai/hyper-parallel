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

#include "duplicate_on_multi_users/duplicate_on_multi_users.h"

#include <memory>
#include <vector>

#include "abstract/abstract_function.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/framework_ops.h"
#include "primitive/other_ops.h"
#include "utils/log_adapter.h"

namespace mindspore::opt {

namespace {
constexpr auto kAttrDuplicateOnMultipleUsers = "duplicate_on_multiple_users";
constexpr auto kDistCommAllGatherIntoTensorBufferIndex = 1;
constexpr auto kDistCommAllGatherIntoTensorUmonadIndex = 5;
constexpr auto kExpectedDistCommAllGatherIntoTensorInputSize = 6;

constexpr auto kLoadTensorIndex = 1;
constexpr auto kLoadStateIndex = 2;

// Helper to create a primitive value node with proper abstract
AnfNodePtr CreatePrimitiveValueNode(const PrimitivePtr &primitive, const FuncGraphPtr &func_graph) {
  auto value_node = NewValueNode(primitive);
  value_node->set_abstract(std::make_shared<abstract::PrimitiveAbstractClosure>(primitive));
  return value_node;
}

// Duplicate communication chain for a new load site (different UMonad context)
AnfNodePtr DuplicateCommunicationChainForLoadSite(const FuncGraphManagerPtr &manager,
                                                  const CNodePtr &original_buffer_alloc_cnode,
                                                  const CNodePtr &original_allgather_cnode,
                                                  const AnfNodePtr &original_load_node) {
  MS_EXCEPTION_IF_NULL(manager);
  MS_EXCEPTION_IF_NULL(original_buffer_alloc_cnode);
  MS_EXCEPTION_IF_NULL(original_allgather_cnode);
  MS_EXCEPTION_IF_NULL(original_load_node);

  auto func_graph = original_allgather_cnode->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);

  auto load_cnode = original_load_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(load_cnode);
  // Input layout: Load(prim, tensor, umonad)
  auto umonad = load_cnode->input(kLoadStateIndex);

  // Step 1: Create new buffer allocation (e.g., PrimFuncZeros) with same inputs
  auto new_buffer = func_graph->NewCNode(original_buffer_alloc_cnode->inputs());
  auto original_buffer_alloc_cnode_abs = original_buffer_alloc_cnode->abstract();
  MS_EXCEPTION_IF_NULL(original_buffer_alloc_cnode_abs);
  new_buffer->set_abstract(original_buffer_alloc_cnode_abs->Clone());

  // Step 2: Create new AllGather node with new buffer and current UMonad state
  auto new_allgather = func_graph->NewCNode(original_allgather_cnode->inputs());
  // Input layout: DistCommAllGatherIntoTensor(prim, buffer, ..., ..., umonad)
  manager->SetEdge(new_allgather, kDistCommAllGatherIntoTensorBufferIndex, new_buffer);
  manager->SetEdge(new_allgather, kDistCommAllGatherIntoTensorUmonadIndex, umonad);
  auto original_allgather_cnode_abs = original_allgather_cnode->abstract();
  MS_EXCEPTION_IF_NULL(original_allgather_cnode_abs);
  new_allgather->set_abstract(original_allgather_cnode_abs->Clone());

  // Step 3: Update state with new communication handle
  auto update_state_prim = CreatePrimitiveValueNode(prim::kPrimUpdateState, func_graph);
  auto state_after_comm = func_graph->NewCNode({update_state_prim, umonad, new_allgather});
  auto umonad_abs = umonad->abstract();
  MS_EXCEPTION_IF_NULL(umonad_abs);
  state_after_comm->set_abstract(umonad_abs->Broaden());

  // Step 4: Load the result under the updated state
  auto load_prim = CreatePrimitiveValueNode(prim::kPrimLoad, func_graph);
  auto new_load = func_graph->NewCNode({load_prim, new_allgather, state_after_comm});
  auto original_load_node_abs = original_load_node->abstract();
  MS_EXCEPTION_IF_NULL(original_load_node_abs);
  new_load->set_abstract(original_load_node_abs->Clone());

  return new_load;
}

// Duplicate communication chain per consumer of a loaded value
void DuplicatePerLoadConsumer(const FuncGraphManagerPtr &manager, const CNodePtr &original_buffer_alloc_cnode,
                              const std::vector<AnfNodePtr> &load_nodes_to_process,
                              const AnfNodePtr &original_allgather_node) {
  MS_EXCEPTION_IF_NULL(manager);
  MS_EXCEPTION_IF_NULL(original_allgather_node);

  auto func_graph = original_allgather_node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);

  // Get all users of this load node
  auto node_users_map = manager->node_users();

  for (const auto &load_node : load_nodes_to_process) {
    auto load_cnode = load_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(load_cnode);

    auto users_iter = node_users_map.find(load_node);
    if (users_iter == node_users_map.end() || users_iter->second.empty()) {
      continue;
    }
    const auto &users = users_iter->second;

    // Skip if only one consumer (no duplication needed)
    if (users.size() <= 1) {
      continue;
    }

    // Identify MakeTuple node that collects results (required for graph integrity)
    AnfNodePtr result_tuple = nullptr;
    for (auto user_iter = users.begin(); user_iter != users.end(); ++user_iter) {
      const auto &user_node = user_iter->first;
      if (IsPrimitiveCNode(user_node, prim::kPrimMakeTuple)) {
        result_tuple = user_node;
        break;
      }
    }

    if (result_tuple == nullptr) {
      MS_LOG(DEBUG) << "No MakeTuple collector found for load node: " << load_node->DebugString()
                    << ", skipping duplication for this load site";
      continue;
    }

    // Process consumers beyond the first one (first consumer keeps original load)
    bool is_first_consumer = true;
    for (const auto &[consumer_node, input_index] : users) {
      // Skip non-computation nodes (e.g., MakeTuple itself)
      if (IsPrimitiveCNode(consumer_node, prim::kPrimMakeTuple)) {
        continue;
      }

      if (is_first_consumer) {
        MS_LOG(DEBUG) << "Preserving first consumer of load node: " << load_node->DebugString() << " -> "
                      << consumer_node->DebugString();
        is_first_consumer = false;
        continue;
      }

      MS_LOG(DEBUG) << "Duplicating communication chain for consumer: " << consumer_node->DebugString()
                    << " of load node: " << load_node->DebugString();

      // Step 1: Advance state after previous consumption
      auto umonad = load_cnode->input(kLoadStateIndex);
      auto update_state_prim = CreatePrimitiveValueNode(prim::kPrimUpdateState, func_graph);
      auto state_after_load = func_graph->NewCNode({update_state_prim, umonad, load_node});
      auto umonad_abs = umonad->abstract();
      MS_EXCEPTION_IF_NULL(umonad_abs);
      state_after_load->set_abstract(umonad_abs->Broaden());

      // Step 2: Create new buffer allocation
      auto new_buffer = func_graph->NewCNode(original_buffer_alloc_cnode->inputs());
      auto original_buffer_alloc_cnode_abs = original_buffer_alloc_cnode->abstract();
      MS_EXCEPTION_IF_NULL(original_buffer_alloc_cnode_abs);
      new_buffer->set_abstract(original_buffer_alloc_cnode_abs->Clone());

      // Step 3: Create new AllGather with updated state
      auto new_allgather = func_graph->NewCNode(original_allgather_node->cast<CNodePtr>()->inputs());
      manager->SetEdge(new_allgather, kDistCommAllGatherIntoTensorBufferIndex, new_buffer);
      manager->SetEdge(new_allgather, kDistCommAllGatherIntoTensorUmonadIndex, state_after_load);
      auto original_allgather_node_abs = original_allgather_node->abstract();
      MS_EXCEPTION_IF_NULL(original_allgather_node_abs);
      new_allgather->set_abstract(original_allgather_node_abs->Clone());

      // Step 4: Update state after new communication
      auto state_after_new_comm = func_graph->NewCNode({update_state_prim, state_after_load, new_allgather});
      state_after_new_comm->set_abstract(state_after_load->abstract()->Broaden());

      // Step 5: Load the new result
      auto load_prim = CreatePrimitiveValueNode(prim::kPrimLoad, func_graph);
      auto new_load = func_graph->NewCNode({load_prim, new_allgather, state_after_new_comm});
      auto load_node_abs = load_node->abstract();
      MS_EXCEPTION_IF_NULL(load_node_abs);
      new_load->set_abstract(load_node_abs->Clone());

      // Step 6: Redirect consumer to use new load result
      manager->SetEdge(consumer_node, input_index, new_load);

      // Step 7: Add new load to result tuple for graph integrity
      manager->AddEdge(result_tuple, new_load);
    }
  }
}

/*
This pass implements a communication-for-memory tradeoff by duplicating distributed
collective communications (e.g., all-gather) instead of materializing and reusing their
results. When the same communication output is consumed at multiple points in the program:

  • Stage 1: Duplicate the entire communication chain per distinct load site
            (i.e., when loaded under different UMonad state contexts)
  • Stage 2: Further duplicate per consumer of each loaded value to break state dependencies

Benefits:
  ✓ Eliminates need to store large intermediate communication results in memory
  ✓ Breaks false dependencies in UMonad state chains
  ✓ Enables finer-grained scheduling and potential pipelining

Cost:
  ✗ Increases communication volume (N× all-gathers instead of 1)

Applies when memory pressure outweighs communication overhead, especially for large
distributed tensors where storing intermediate results is prohibitive.

--- Original pattern: single communication reused across state boundaries ---

%0 = PrimFuncZeros(...)                     // allocate zero-initialized buffer for all-gather input
...
%1 = Some UMonad                            // initial state
%2 = DistCommAllGatherIntoTensor(%0, ..., %1)  // collective communication (output not stored)
%3 = UpdateState(%1, %2)                    // advance state with communication handle
%4 = Load(%2, %3)                           // materialize result under state %3
%5 = Op1(..., %4, ...)                      // consumer 1 of loaded value
%6 = Op2(..., %4, ...)                      // consumer 2 of loaded value (shares %4)
...
%7 = Some UMonad                            // later state (after other stateful ops)
%8 = Load(%2, %7)                           // attempt to reuse same communication under new state
%9 = Op3(..., %8, ...)                      // consumer 3
%10 = Op4(..., %8, ...)                     // consumer 4


--- Stage 1: Duplicate per load site (different UMonad contexts) ---

%0 = PrimFuncZeros(...)                     // original buffer
...
%1 = Some UMonad
%2 = DistCommAllGatherIntoTensor(%0, ..., %1)
%3 = UpdateState(%1, %2)
%4 = Load(%2, %3)
%5 = Op1(..., %4, ...)
%6 = Op2(..., %4, ...)
...
%7 = Some UMonad                            // different state context → duplicate communication
%11 = PrimFuncZeros(...)                    // new buffer for duplicated communication
%12 = DistCommAllGatherIntoTensor(%11, ..., %7)  // independent all-gather
%13 = UpdateState(%7, %12)                  // independent state update
%14 = Load(%12, %13)                        // materialize under new state
%9 = Op3(..., %14, ...)                     // consumer uses fresh load result
%10 = Op4(..., %14, ...)                    // shares the fresh load

--- Stage 2: Duplicate per load consumer (break intra-site dependencies) ---

%0 = PrimFuncZeros(...)                     // original buffer
...
%1 = Some UMonad
%2 = DistCommAllGatherIntoTensor(%0, ..., %1)
%3 = UpdateState(%1, %2)
%4 = Load(%2, %3)
%5 = Op1(..., %4, ...)                      // first consumer uses original load
%15 = UpdateState(%3, %4)                   // advance state after first consumption
%16 = PrimFuncZeros(...)                    // new buffer for second consumer
%17 = DistCommAllGatherIntoTensor(%16, ..., %15)  // independent all-gather for Op2
%18 = UpdateState(%15, %17)
%19 = Load(%17, %18)                        // fresh load for second consumer
%6 = Op2(..., %19, ...)                     // second consumer uses independent result
...
%7 = Some UMonad
%11 = PrimFuncZeros(...)                    // buffer for third consumer group
%12 = DistCommAllGatherIntoTensor(%11, ..., %7)
%13 = UpdateState(%7, %12)
%14 = Load(%12, %13)
%9 = Op3(..., %14, ...)                     // third consumer uses first load of this site
%20 = UpdateState(%13, %14)                 // advance state after Op3
%21 = PrimFuncZeros(...)                    // new buffer for fourth consumer
%22 = DistCommAllGatherIntoTensor(%21, ..., %20)  // independent all-gather for Op4
%23 = UpdateState(%20, %22)
%24 = Load(%22, %23)                        // fresh load for fourth consumer
%10 = Op4(..., %24, ...)                    // fourth consumer uses independent result
*/
void DuplicateDistCommAllGatherIntoTensor(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  // Step 1: Collect all AllGather nodes marked for duplication
  std::vector<AnfNodePtr> target_allgather_nodes;
  for (const auto &node : manager->all_nodes()) {
    if (!IsPrimitiveCNode(node, prim::kPrimDistCommAllGatherIntoTensor)) {
      continue;
    }

    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto prim = GetCNodePrimitive(cnode);
    if (prim == nullptr || !prim->HasAttr(kAttrDuplicateOnMultipleUsers)) {
      continue;
    }
    // Verify expected input structure (prim, buffer, ..., ..., ..., umonad)
    if (cnode->size() != kExpectedDistCommAllGatherIntoTensorInputSize) {
      MS_LOG(WARNING) << "Skipping AllGather with unexpected input count: " << cnode->size()
                      << ", node: " << cnode->DebugString();
      continue;
    }

    target_allgather_nodes.push_back(node);
  }

  // Step 2: Process each AllGather node
  for (const auto &allgather_node : target_allgather_nodes) {
    auto allgather_cnode = allgather_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(allgather_cnode);

    // Verify buffer allocation node has required marker attribute
    auto buffer_alloc = allgather_cnode->input(kDistCommAllGatherIntoTensorBufferIndex);
    auto buffer_prim = GetCNodePrimitive(buffer_alloc);
    if (buffer_prim == nullptr || !buffer_prim->HasAttr(kAttrDuplicateOnMultipleUsers)) {
      continue;
    }
    auto buffer_alloc_cnode = buffer_alloc->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(buffer_alloc_cnode);

    // Step 2.1: Collect all Load nodes consuming this AllGather result
    std::vector<AnfNodePtr> load_consumers;
    for (const auto &[user_node, input_idx] : manager->node_users()[allgather_node]) {
      if (IsPrimitiveCNode(user_node, prim::kPrimLoad) && input_idx == kLoadTensorIndex) {
        load_consumers.push_back(user_node);
      }
    }

    if (load_consumers.empty()) {
      continue;
    }

    // Step 2.2: Stage 1 - Duplicate per load site (different UMonad contexts)
    std::vector<AnfNodePtr> duplicated_loads{load_consumers[0]};  // First load remains original
    for (size_t idx = 1; idx < load_consumers.size(); ++idx) {
      auto new_load =
        DuplicateCommunicationChainForLoadSite(manager, buffer_alloc_cnode, allgather_cnode, load_consumers[idx]);
      manager->Replace(load_consumers[idx], new_load);
      duplicated_loads.push_back(new_load);
    }

    // Step 2.3: Stage 2 - Duplicate per consumer within each load site
    DuplicatePerLoadConsumer(manager, buffer_alloc_cnode, duplicated_loads, allgather_node);
  }
}

constexpr auto kDependStateIndex = 2;
constexpr auto kAllGatherDataInputIndex = 1;
constexpr auto kUpdateStateValueIndex = 2;
constexpr auto kUpdateStatePriorStateIndex = 1;
constexpr auto kDependDataIndex = 1;
/*
Eliminate redundant Depend nodes that directly follow UpdateState nodes in a specific pattern.

Transformation pattern:
  Before (redundant synchronization):
    %u0 = UMonad()                          // initial state
    %x0 = Load(%weight, %u0)                // load weight under state %u0
    %x1 = AllGather(%x0) [duplicate_on_multi_users]  // communication
    %u1 = UpdateState(%u0, %x1)             // advance state with communication handle
    %x2 = Depend(%x1, %u1)                  // ⚠️ REDUNDANT: UpdateState already enforces ordering
    %x3 = Op(%x2)                           // consumer

  After (optimized):
    %u0 = UMonad()
    %x0 = Load(%weight, %u0)
    %x1 = AllGather(%x0) [duplicate_on_multi_users]
    %x3 = Op(%x1)                           // direct consumption

Eligibility conditions (ALL must be satisfied):
  1. Depend node's second input is an UpdateState node
  2. UpdateState node has EXACTLY ONE user (this Depend node)
  3. Depend node's first input is an AllGather node marked with kAttrDuplicateOnMultipleUsers
  4. AllGather's data input (index 1) is a Load node
  5. UpdateState's state-modifying input (index 2) is the SAME Load node
  6. Load's state input (index 2) and UpdateState's prior state (index 1) are the SAME UMonad
*/
void EliminateRedundantDependAfterUpdateState(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  const auto &node_consumers = manager->node_users();
  std::vector<AnfNodePtr> depend_nodes_to_eliminate;

  // Step 1: Identify eligible Depend nodes for elimination
  for (const auto &node : manager->all_nodes()) {
    // Condition 1: Must be a Depend node
    if (!IsPrimitiveCNode(node, prim::kPrimDepend)) {
      continue;
    }
    auto depend_cnode = node->cast<CNodePtr>();

    // Condition 2: Second input must be an UpdateState node
    auto updatestate_candidate = depend_cnode->input(kDependStateIndex);
    if (!IsPrimitiveCNode(updatestate_candidate, prim::kPrimUpdateState)) {
      continue;
    }
    auto updatestate_cnode = updatestate_candidate->cast<CNodePtr>();

    // Condition 3: UpdateState must have exactly one consumer (this Depend node)
    auto updatestate_consumers = node_consumers.find(updatestate_cnode);
    if (updatestate_consumers == node_consumers.end() || updatestate_consumers->second.size() != 1) {
      continue;
    }

    // Condition 4: First input must be an AllGather node with duplication marker
    auto allgather_candidate = depend_cnode->input(kDependDataIndex);
    auto allgather_cnode = allgather_candidate->cast<CNodePtr>();
    if (allgather_cnode == nullptr) {
      continue;
    }
    auto allgather_prim = GetCNodePrimitive(allgather_cnode);
    if (allgather_prim == nullptr || !allgather_prim->HasAttr(kAttrDuplicateOnMultipleUsers)) {
      continue;
    }

    // Condition 5: AllGather's data input must be a Load node
    auto allgather_load = allgather_cnode->input(kAllGatherDataInputIndex);
    if (!IsPrimitiveCNode(allgather_load, prim::kPrimLoad)) {
      continue;
    }
    auto load_cnode_for_allgather = allgather_load->cast<CNodePtr>();

    // Condition 6: UpdateState's modified value must be the SAME Load node

    auto updatestate_modified_value = updatestate_cnode->input(kUpdateStateValueIndex);
    if (!IsPrimitiveCNode(updatestate_modified_value, prim::kPrimLoad)) {
      continue;
    }
    auto load_cnode_for_updatestate = updatestate_modified_value->cast<CNodePtr>();

    if (load_cnode_for_allgather != load_cnode_for_updatestate) {
      continue;
    }

    // Condition 7: Load's state input and UpdateState's prior state must be the SAME UMonad
    auto load_state_input = load_cnode_for_allgather->input(kLoadStateIndex);
    auto updatestate_prior_state = updatestate_cnode->input(kUpdateStatePriorStateIndex);

    if (!IsValueNode<UMonad>(load_state_input) || !IsValueNode<UMonad>(updatestate_prior_state)) {
      continue;
    }
    if (load_state_input != updatestate_prior_state) {
      continue;
    }

    // All conditions satisfied - mark for elimination
    depend_nodes_to_eliminate.push_back(node);
    MS_LOG(DEBUG) << "Marked redundant Depend node for elimination: " << depend_cnode->DebugString();
  }

  // Step 2: Perform elimination (replace Depend with its data input)
  for (const auto &depend_node : depend_nodes_to_eliminate) {
    auto depend_cnode = depend_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(depend_cnode);
    auto data_input = depend_cnode->input(kDependDataIndex);
    manager->Replace(depend_cnode, data_input);
    MS_LOG(DEBUG) << "Eliminated redundant Depend node, replaced with: " << data_input->DebugString();
  }
}

/*
Duplicate AllGather nodes with multiple consumers via shallow copying.

Two-phase transformation pipeline:
  Phase 1: Eliminate redundant Depend nodes after UpdateState (EliminateRedundantDependAfterUpdateState)
           - Removes unnecessary synchronization when UpdateState already enforces ordering
           - Prepares graph for safe duplication by simplifying state dependencies

  Phase 2: Duplicate communication primitives for multiple consumers
           a) DistCommAllGatherIntoTensor: Full state-aware duplication (buffer + comm + load)
           b) AllGather: Shallow duplication (input copy only, no state handling)

Critical invariant: Phase 1 MUST precede Phase 2 because Depend elimination exposes
the true consumer structure required for correct duplication.

Transformation:
  Before:
    %x1 = AllGather(%input) [duplicate_on_multi_users]
    %x2 = Op1(%x1)
    %x3 = Op2(%x1)

  After:
    %x1 = AllGather(%input)   // original (kept but may become dead)
    %x4 = AllGather(%input)   // duplicate for Op2
    %x2 = Op1(%x1)
    %x3 = Op2(%x4)
*/
void DuplicateAllGatherForMultipleConsumers(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  // Preprocessing: Eliminate redundant Depend nodes to expose true consumer structure
  EliminateRedundantDependAfterUpdateState(func_graph);

  const auto &all_graph_nodes = manager->all_nodes();
  auto &node_consumers_map = manager->node_users();

  // Step 1: Collect AllGather nodes marked for duplication with their consumers
  std::map<CNodePtr, AnfNodeIndexSet> allgather_nodes_to_duplicate;
  for (const auto &node : all_graph_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimAllGather)) {
      continue;
    }

    auto primitive = GetCNodePrimitive(node);
    if (primitive == nullptr || !primitive->HasAttr(kAttrDuplicateOnMultipleUsers)) {
      continue;
    }

    auto allgather_cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(allgather_cnode);

    auto consumers_iter = node_consumers_map.find(node);
    if (consumers_iter != node_consumers_map.end() && !consumers_iter->second.empty()) {
      allgather_nodes_to_duplicate[allgather_cnode] = consumers_iter->second;
      MS_LOG(DEBUG) << "Collected AllGather for duplication with " << consumers_iter->second.size()
                    << " consumers: " << allgather_cnode->DebugString();
    }
  }

  // Step 2: Duplicate each AllGather node for ALL its consumers
  for (const auto &[allgather_cnode, consumers] : allgather_nodes_to_duplicate) {
    if (consumers.size() <= 1) {
      MS_LOG(DEBUG) << "Skipping AllGather duplication (single consumer): " << allgather_cnode->DebugString();
      continue;
    }

    auto fg = allgather_cnode->func_graph();
    MS_EXCEPTION_IF_NULL(fg);

    MS_LOG(DEBUG) << "Duplicating AllGather node for " << consumers.size()
                  << " consumers: " << allgather_cnode->DebugString();

    // Create independent copy for each consumer (including first)
    for (const auto &[consumer_node, input_index] : consumers) {
      // Shallow copy: duplicate inputs but NOT state edges
      auto duplicated_allgather = fg->NewCNode(allgather_cnode->inputs());
      duplicated_allgather->set_abstract(allgather_cnode->abstract()->Clone());

      // Redirect consumer to use duplicated node
      manager->SetEdge(consumer_node, input_index, duplicated_allgather);

      MS_LOG(DEBUG) << "Redirected consumer " << consumer_node->DebugString()
                    << " to new AllGather node: " << duplicated_allgather->DebugString();
    }
  }
}
}  // namespace

bool DuplicatePrimOnMultiUsersPass::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);

  // Phase 1: Duplication for distributed communication primitives (best effort).
  DuplicateDistCommAllGatherIntoTensor(func_graph);

  // Phase 2: Duplication for standard AllGather primitives (best effort).
  DuplicateAllGatherForMultipleConsumers(func_graph);

  return true;
}
}  // namespace mindspore::opt
