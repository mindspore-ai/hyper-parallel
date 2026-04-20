#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
import os
import sys
import numpy as np
import torch

THRESH = 2**-6


def get_eb(golden: torch.Tensor, actual: torch.Tensor):
    golden = golden.to(torch.float16)
    golden_nmax = torch.clamp(torch.abs(golden), min=1)
    actual_error = actual.to(torch.float16) - golden
    eb = torch.mean(actual_error / golden_nmax)
    result = eb <= 2 ** (-7)
    return result


def ref_compare(golden: torch.Tensor, actual: torch.Tensor, thresh: float):
    golden = golden.to(torch.float16)
    golden_nmax = torch.clamp(torch.abs(golden), min=1)
    abs_error = torch.abs(actual.to(torch.float16) - golden)
    result = (abs_error <= thresh * golden_nmax).all()
    return result

def verify_result(real_result, golden):
    dtype = np.float16
    real_result = np.fromfile(real_result, dtype=dtype) # 从bin文件读取实际运算结果
    golden = np.fromfile(golden, dtype=dtype) # 从bin文件读取预期运算结果
    print("real_result.shape:", real_result.shape)
    print("real_result:", real_result)
    print("golden.shape:", golden.shape)
    print("golden:", golden)

    real_result = torch.tensor(real_result)
    golden = torch.tensor(golden)
    eb = get_eb(golden, real_result)
    cmp = ref_compare(golden, real_result, THRESH)
    result = eb and cmp
    if not result:
        print("[ERROR] result error")
        return False
    print("test pass")
    return True


def verify_result_matmul_nm(real_result, golden):
    dtype = np.float16
    real_result = np.fromfile(real_result, dtype=dtype) # 从bin文件读取实际运算结果
    golden = np.fromfile(golden, dtype=dtype) # 从bin文件读取预期运算结果
    print("real_result.shape:", real_result.shape)
    # print("real_result:", real_result)
    print("golden.shape:", golden.shape)
    # print("golden:", golden)

    matrix_ones = np.ones((4096, 7168), dtype=dtype)

    value_per = 1024 * 256
    for i in range(112):
        data = real_result[i*value_per: (i+1) * value_per]
        data = data.reshape((1024,256))

        hengzhou = (int)(i % 4)
        shuzhou = (int)(i // 4)
        matrix_ones[hengzhou*1024:(hengzhou+1)*1024, shuzhou*256:(shuzhou+1)*256] = data

    matrix_ones = matrix_ones.reshape((4096*7168, ))

    print("real_result:", matrix_ones[111110:111150])
    print("golden:", golden[111110:111150])

    real_result = torch.tensor(matrix_ones)
    golden = torch.tensor(golden)
    eb = get_eb(golden, real_result)
    cmp = ref_compare(golden, real_result, THRESH)
    result = eb and cmp
    if not result:
        print("[ERROR] result error")
        return False
    print("test pass")
    return True

def verify_result_matmul_mn(real_result, golden):
    dtype = np.float16
    real_result = np.fromfile(real_result, dtype=dtype) # 从bin文件读取实际运算结果
    golden = np.fromfile(golden, dtype=dtype) # 从bin文件读取预期运算结果
    print("real_result.shape:", real_result.shape)
    # print("real_result:", real_result)
    print("golden.shape:", golden.shape)
    # print("golden:", golden)

    matrix_ones = np.ones((4096, 7168), dtype=dtype)

    value_per = 256 * 1024
    for i in range(112):
        data = real_result[i*value_per: (i+1) * value_per]
        data = data.reshape((256,1024))

        hengzhou = (int)(i // 7)
        shuzhou = (int)(i % 7)

        matrix_ones[hengzhou*256:(hengzhou+1)*256, shuzhou*1024:(shuzhou+1)*1024] = data

    matrix_ones = matrix_ones.reshape((4096*7168, ))

    print("real_result:", matrix_ones[111110:111150])
    print("golden:", golden[111110:111150])

    real_result = torch.tensor(matrix_ones)
    golden = torch.tensor(golden)
    eb = get_eb(golden, real_result)
    cmp = ref_compare(golden, real_result, THRESH)
    result = eb and cmp
    if not result:
        print("[ERROR] result error")
        return False
    print("test pass")
    return True

def verify_result_matmul_nm_reshape_trans(real_result, golden):
    dtype = np.float16
    real_result = np.fromfile(real_result, dtype=dtype) # 从bin文件读取实际运算结果
    golden = np.fromfile(golden, dtype=dtype) # 从bin文件读取预期运算结果
    print("real_result.shape:", real_result.shape)
    print("golden.shape:", golden.shape)

    real_result = real_result.reshape((4, 28, 1024, 256)).transpose(0,2,1,3).reshape((4096, 7168))
    matrix_ones = real_result.reshape((4096*7168, ))

    # matrix_ones = np.ones((4096 * 28, 256), dtype=dtype)
    # value_per = 1024 * 256
    # for i in range(112):
    #     data = real_result[i*value_per: (i+1) * value_per]
    #     data = data.reshape((1024,256))

    #     hengzhou = (int)(i % 4)
    #     shuzhou = (int)(i // 4)
    #     matrix_ones[hengzhou*1024:(hengzhou+1)*1024, shuzhou*256:(shuzhou+1)*256] = data
    # matrix_ones = matrix_ones.reshape((4096*7168, ))

    real_result = torch.tensor(matrix_ones)
    golden = torch.tensor(golden)
    eb = get_eb(golden, real_result)
    cmp = ref_compare(golden, real_result, THRESH)
    result = eb and cmp
    if not result:
        print("[ERROR] result error")
        return False
    print("test pass")
    return True

if __name__ == '__main__':
    # verify_result(sys.argv[1],sys.argv[2])
    # verify_result_matmul_nm("./output/matmul_result.bin", "./output/matmul_golden.bin")
    # verify_result_matmul_nm_reshape_trans("./output/matmul_result.bin", "./output/matmul_golden.bin")
    # verify_result("./output/add_golden.bin", "./output/add_golden.bin")

    data = "/home/z00797459/workspace/cann-ops-adv/ops-nn/my_mega_kernel/input_data/input_grad_32_24_task_reverse_alltoall_dynamic/gmmTilingG4.bin"
    data = np.fromfile(data, dtype=np.int32) # 从bin文件读取实际运算结果
    res = []
    for i in range(24):
        temp = []
        for j in range(504):
            index = i * 504 + j
            temp.append(data[index])
        res.append(temp)
        print(temp)
    # print(res)