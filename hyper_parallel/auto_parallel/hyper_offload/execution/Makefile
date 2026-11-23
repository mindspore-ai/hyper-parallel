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

PYTHON ?= python3
CXXFLAGS += -O2 -Wall -Wextra -shared -std=c++17 -fPIC -fstack-protector-strong
LDFLAGS += -Wl,-z,relro,-z,now
TORCH_ROOT = $(shell $(PYTHON) -c "import importlib.util; print(next(iter(importlib.util.find_spec('torch').submodule_search_locations)))")
CPPFLAGS += $(shell $(PYTHON) -c "import sysconfig; print('-I' + sysconfig.get_path('include'))")
CPPFLAGS += -I$(TORCH_ROOT)/include -I$(TORCH_ROOT)/include/torch/csrc/api/include
CPPFLAGS += -DTORCH_EXTENSION_NAME=_native_saved_tensor
CPPFLAGS += -D_GLIBCXX_USE_CXX11_ABI=$(shell $(PYTHON) -c "import torch; print(int(torch.compiled_with_cxx11_abi()))")
LDFLAGS += -L$(TORCH_ROOT)/lib -ltorch_python -ltorch_cpu -ltorch -lc10
LIBRARY_EXTENSION = $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
OUTPUT = _native_saved_tensor$(LIBRARY_EXTENSION)

.PHONY: all clean
all: $(OUTPUT)

$(OUTPUT): native_saved_tensor.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@ $(LDFLAGS)

clean:
	$(RM) _native_saved_tensor*.so
