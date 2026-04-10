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
#include "symmetric_memory_allocator.h"
#include <memory>
#include <string>
#include <cstdlib>

#include "acl/acl.h"

#include "shmem.h"
#include "mindspore/include/custom_op_api.h"

namespace ms_custom_ops {

constexpr int64_t kShmemSize = 1024 * 1024 * 1024;
constexpr char kShmemIpPort[] = "tcp://127.0.0.1:8769";

int32_t test_set_attr(int32_t my_pe, int32_t n_pes, uint64_t local_mem_size, const char *ip_port,
                      aclshmemx_uniqueid_t default_flag_uid, aclshmemx_init_attr_t *attributes) {
  size_t ip_len = 0;
  if (ip_port != nullptr) {
    ip_len = std::min(strlen(ip_port), (size_t)(ACLSHMEM_MAX_IP_PORT_LEN - 1));

    std::copy_n(ip_port, ip_len, attributes->ip_port);
    if (attributes->ip_port[0] == '\0') {
      return ACLSHMEM_INVALID_VALUE;
    }
  }

  int attr_version = (1 << 16) + sizeof(aclshmemx_init_attr_t);
  attributes->my_pe = my_pe;
  attributes->n_pes = n_pes;
  attributes->ip_port[ip_len] = '\0';
  attributes->local_mem_size = local_mem_size;
  attributes->option_attr = {attr_version, ACLSHMEM_DATA_OP_MTE, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
  attributes->comm_args = reinterpret_cast<void *>(&default_flag_uid);
  // aclshmemx_uniqueid_t *uid_args = (aclshmemx_uniqueid_t *)(attributes->comm_args);

  return ACLSHMEM_SUCCESS;
}

const std::string GetShmemIpPort() {
  // get MS_SCHED_HOST and MS_SCHED_PORT of scheduler from environment variables
  auto sched_host_env = std::getenv("MS_SCHED_HOST");
  auto sched_port_env = std::getenv("MS_SCHED_PORT");
  std::string ip_port = kShmemIpPort;
  if (sched_host_env != nullptr && sched_port_env != nullptr) {
    try {
      std::string sched_host(sched_host_env);
      int sched_port = std::stoi(sched_port_env);
      int new_port = sched_port + 13;
      ip_port = "tcp://" + sched_host + ":" + std::to_string(new_port);
      MS_LOG(INFO) << "Using environment variables for shmem IP port: " << ip_port;
    } catch (const std::invalid_argument &e) {
      MS_LOG(WARNING) << "Invalid MS_SCHED_PORT value: " << sched_port_env << ", using default: " << kShmemIpPort;
    } catch (const std::out_of_range &e) {
      MS_LOG(WARNING) << "MS_SCHED_PORT value " << sched_port_env
                      << " is out of range, using default: " << kShmemIpPort;
    }
  } else {
    MS_LOG(INFO) << "Using default shmem IP port: " << kShmemIpPort;
  }

  return ip_port;
}

void initialize_npushmem() {
  static bool is_initialized = false;
  if (is_initialized) {
    return;
  }
  // get shmem para
  auto rank_id_env = std::getenv("RANK_ID");
  auto rank_size_env = std::getenv("RANK_SIZE");

  if (rank_id_env == nullptr || rank_size_env == nullptr) {
    MS_LOG(EXCEPTION) << "RANK_ID or RANK_SIZE is empty";
  }
  int32_t my_pe = std::stoi(rank_id_env);
  int32_t n_ranks = std::stoi(rank_size_env);

  uint64_t local_mem_size = kShmemSize;
  auto shmem_size = std::getenv("SYMMETRIC_MEMORY_HEAP_SIZE");
  if (shmem_size != nullptr) {
    try {
      // str to uint64_t
      uint64_t temp_shmem_size = std::stoull(shmem_size);
      local_mem_size = temp_shmem_size;
      MS_LOG(INFO) << "Manually set SYMMETRIC_MEMORY_HEAP_SIZE to " << local_mem_size;
    } catch (const std::invalid_argument &e) {
      // handle invalid argument
      MS_LOG(WARNING) << "Invalid SYMMETRIC_MEMORY_HEAP_SIZE value: " << shmem_size << ", using default value "
                      << local_mem_size;
    } catch (const std::out_of_range &e) {
      // handle out of range
      MS_LOG(WARNING) << "SYMMETRIC_MEMORY_HEAP_SIZE value " << shmem_size << " is out of range, using default value "
                      << local_mem_size;
    }
  }
  const std::string ip_port_str = GetShmemIpPort();

  // init status
  int32_t status = aclshmemx_set_conf_store_tls(false, nullptr, 0);
  if (status != 0) {
    MS_LOG(EXCEPTION) << "Aclshmem set conf store tls failed, error code:" << status;
  }
  aclshmemx_uniqueid_t default_flag_uid;
  aclshmemx_init_attr_t attributes;
  int32_t set_attr_status = test_set_attr(my_pe, n_ranks, local_mem_size,
    ip_port_str.c_str(), default_flag_uid, &attributes);
  if (set_attr_status != 0) {
    MS_LOG(EXCEPTION) << "Aclshmem set attr failed, error code:" << set_attr_status;
  }
  int32_t init_status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
  if (init_status != 0) {
    MS_LOG(EXCEPTION) << "Aclshmem init failed, error code:" << init_status;
  }
  is_initialized = true;
  MS_LOG(INFO) << "Finish initialize_npushmem: " << init_status;
}

std::shared_ptr<SymmetricMemoryAllocator> &SymmetricMemoryAllocator::GetInstance() {
  static std::shared_ptr<SymmetricMemoryAllocator> instance = std::make_shared<SymmetricMemoryAllocator>();
  return instance;
}

void *SymmetricMemoryAllocator::Alloc(size_t size, uint32_t stream_id) {
  initialize_npushmem();
  MS_LOG(INFO) << "Allocate symmetric memory, size: " << size;
  void *device_shmem_ptr = aclshmem_malloc(size);
  if (device_shmem_ptr == nullptr) {
    MS_LOG(ERROR) << "Allocate symmetric memory failed, size: " << size;
  }
  MS_LOG(DEBUG) << "Allocate symmetric memory success, ptr: " << device_shmem_ptr;
  return device_shmem_ptr;
}

bool SymmetricMemoryAllocator::Free(void *address_ptr) {
  aclshmem_free(address_ptr);
  MS_LOG(DEBUG) << "Free symmetric memory, ptr:" << address_ptr;
  return true;
}

bool SymmetricMemoryAllocator::IsPinned() { return false; }

}  // namespace ms_custom_ops

extern "C" {
void *Alloc(size_t size, int device, aclrtStream stream) {
  auto g_allocator = ms_custom_ops::SymmetricMemoryAllocator::GetInstance();
  if (g_allocator) {
    return g_allocator->Alloc(size, 0);
  }
  return nullptr;
}

void Free(void *ptr, size_t size, aclrtStream stream) {
  auto g_allocator = ms_custom_ops::SymmetricMemoryAllocator::GetInstance();
  if (g_allocator && ptr) {
    g_allocator->Free(ptr);
  }
}
}
