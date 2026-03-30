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
#ifndef SYMMETRIC_MEMORY_PLATFORM_MINDSPORE_C_API_ALLOCATOR_SYMMETRIC_MEMORY_ALLOCATOR_H_
#define SYMMETRIC_MEMORY_PLATFORM_MINDSPORE_C_API_ALLOCATOR_SYMMETRIC_MEMORY_ALLOCATOR_H_

#include <memory>
#include <cstdint>

#define OPEN_API __attribute__((visibility("default")))

namespace ms_custom_ops {

class SymmetricMemoryAllocator {
 public:
  explicit SymmetricMemoryAllocator() {}
  virtual ~SymmetricMemoryAllocator() = default;
  SymmetricMemoryAllocator(const SymmetricMemoryAllocator &) = delete;
  SymmetricMemoryAllocator &operator=(const SymmetricMemoryAllocator &) = delete;
  static std::shared_ptr<SymmetricMemoryAllocator> &GetInstance();

  void *Alloc(size_t size, uint32_t stream_id);  // override;
  bool Free(void *address_ptr);                  // override;
  bool IsPinned();                               // override;

 private:
  static std::shared_ptr<SymmetricMemoryAllocator> instance;
};

}  // namespace ms_custom_ops
#endif  // SYMMETRIC_MEMORY_PLATFORM_MINDSPORE_C_API_ALLOCATOR_SYMMETRIC_MEMORY_ALLOCATOR_H_
