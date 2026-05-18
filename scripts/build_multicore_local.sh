#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# 
set -e
cd hyper_parallel/core/multicore/ops
rm -rf build
rm -rf ops-nn
rm -rf ops-transformer
mkdir build
mkdir build/mega_moe
cp --parents -r mega_moe mega_moe_grad build/mega_moe
cp -r runtime build/mega_moe/mega_moe/op_kernel/
cp -r runtime build/mega_moe/mega_moe_grad/op_kernel/
mkdir build/mega_moe/{mega_moe,mega_moe_grad}/op_kernel/{swi_glu,grouped_matmul}
mkdir build/mega_moe/mega_moe_grad/op_kernel/swi_glu_grad
git clone --no-checkout --depth 1 https://gitcode.com/cann/ops-nn.git -b v9.0.0-beta.1
cd ops-nn
git sparse-checkout init --cone
git sparse-checkout set cmake common control scripts conv/common/op_kernel matmul/common/cmct activation/swi_glu/op_kernel activation/swi_glu_grad/op_kernel build.sh CMakeLists.txt install_deps.sh
git checkout
git apply ../ops-nn.patch
cp --parents -r cmake common control scripts conv/common/op_kernel matmul/common/cmct build.sh CMakeLists.txt install_deps.sh ../build
cp -r activation/swi_glu/op_kernel/. ../build/mega_moe/mega_moe/op_kernel/swi_glu/
cp -r activation/swi_glu/op_kernel/. ../build/mega_moe/mega_moe_grad/op_kernel/swi_glu/
cp -r activation/swi_glu_grad/op_kernel/. ../build/mega_moe/mega_moe_grad/op_kernel/swi_glu_grad/
cd ..
rm -rf ops-nn
git clone --no-checkout --depth 1 https://gitcode.com/cann/ops-transformer.git -b v8.5.0
cd ops-transformer
git sparse-checkout init --cone
git sparse-checkout set gmm/grouped_matmul/op_kernel
git checkout
git apply ../ops-transformer.patch
cp -r gmm/grouped_matmul/op_kernel/. ../build/mega_moe/mega_moe_grad/op_kernel/grouped_matmul/
cp -r gmm/grouped_matmul/op_kernel/. ../build/mega_moe/mega_moe/op_kernel/grouped_matmul/
cd ..
rm -rf ops-transformer
# 之后在hyper-parallel/hyper_parallel/core/multicore/ops/build目录下进行编译，SOC_VALUE改为对应硬件版本
# cd build
# bash build.sh --pkg --soc=$SOC_VALUE --ops=mega_moe --vendor_name=mega_moe
# bash build.sh --pkg --soc=$SOC_VALUE --ops=mega_moe_grad --vendor_name=mega_moe_grad
