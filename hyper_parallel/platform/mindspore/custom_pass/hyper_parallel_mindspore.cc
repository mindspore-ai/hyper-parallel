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
#include <string>
#include <memory>
#include <vector>
#include <unordered_map>

#include "mindspore/core/include/utils/log_adapter.h"
#include "mindspore/ccsrc/include/utils/custom_pass/custom_pass_plugin.h"
#include "duplicate_on_multi_users/duplicate_on_multi_users.h"

namespace mindspore {
namespace opt {
// Helper function to create pass with logging
template <typename PassType>
std::shared_ptr<Pass> CreatePassWithLogging(const std::string &pass_name) {
  auto pass = std::make_shared<PassType>();
  MS_LOG(INFO) << "Created mindspore custom pass '" << pass_name << "' successfully.";
  return pass;
}

class HyperParallelMindsporePlugin : public CustomPassPlugin {
 public:
  std::string GetPluginName() const override { return "hyper_parallel_mindspore"; }

  std::vector<std::string> GetAvailablePassNames() const override {
    const std::vector<std::string> kPassNames = {"DuplicatePrimOnMultiUsersPass"};
    return kPassNames;
  }

  std::shared_ptr<Pass> CreatePass(const std::string &pass_name) const override {
    const auto &kPassCreators = *[]() {
      auto *map = new std::unordered_map<std::string, std::function<std::shared_ptr<Pass>()>>{
        {"DuplicatePrimOnMultiUsersPass",
         []() { return CreatePassWithLogging<DuplicatePrimOnMultiUsersPass>("DuplicatePrimOnMultiUsersPass"); }}};
      return map;
    }();

    auto it = kPassCreators.find(pass_name);
    if (it != kPassCreators.end()) {
      return it->second();
    }

    std::string available_list;
    const auto &available_names = GetAvailablePassNames();
    for (size_t i = 0; i < available_names.size(); ++i) {
      if (i > 0) available_list += ", ";
      available_list += available_names[i];
    }

    MS_LOG(WARNING) << "Mindspore custom pass '" << pass_name << "' not found. Available passes: " << available_list;
    return nullptr;
  }
};

}  // namespace opt
}  // namespace mindspore

EXPORT_CUSTOM_PASS_PLUGIN(mindspore::opt::HyperParallelMindsporePlugin)
