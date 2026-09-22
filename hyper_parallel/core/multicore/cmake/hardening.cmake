# Copyright 2026 Huawei Technologies Co., Ltd.
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

function(hp_enable_compile_hardening target_name)
    target_compile_options(${target_name} PRIVATE
        -D_FORTIFY_SOURCE=2
        -fstack-protector-strong
    )
endfunction()

function(hp_enable_link_hardening target_name)
    target_link_options(${target_name} PRIVATE
        -Wl,-z,relro
        -Wl,-z,now
        -Wl,-z,noexecstack
        $<$<CONFIG:Release>:-s>
    )
    set_target_properties(${target_name} PROPERTIES
        SKIP_BUILD_RPATH TRUE
        SKIP_INSTALL_RPATH TRUE
    )
endfunction()

function(hp_enable_elf_hardening target_name)
    hp_enable_compile_hardening(${target_name})
    hp_enable_link_hardening(${target_name})
endfunction()
