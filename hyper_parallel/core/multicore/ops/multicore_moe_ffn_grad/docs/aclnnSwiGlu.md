# aclnnSwiGlu

## 支持的产品型号
- Atlas A2 训练系列产品。
- Atlas 推理系列产品。
- Atlas A3 训练系列产品/Atlas A3 推理系列产品。

## 接口原型
每个算子分为两段式接口，必须先调用“aclnnSwiGluGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSwiGlu”接口执行计算。

- `aclnnStatus aclnnSwiGluGetWorkspaceSize(const aclTensor *x, int64_t dimOptional, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnSwiGlu(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述
- 算子功能：Swish门控线性单元激活函数，实现x的SwiGlu计算。
- 计算公式：
  <p style="text-align: center">
  out<sub>i</sub> = SwiGlu(x<sub>i</sub>)=Swish(A<sub>i</sub>)*B<sub>i</sub>
  </p>
  其中，A<sub>i</sub>表示x<sub>i</sub>按指定dim维度一分为二的前半部分张量，B<sub>i</sub>表示x<sub>i</sub>按指定dim维度一分为二的后半部分张量。

## aclnnSwiGluGetWorkspaceSize
- **参数说明**：

  - x（aclTensor*，计算输入）：表示待计算的数据，公式中的x<sub>i</sub>，Device侧的aclTensor，shape维度必须大于0维，小于8维，且shape必须在入参dimOptional对应维度上可以整除2。不支持非连续的Tensor，不支持空Tensor。数据格式支持ND。
    - Atlas 推理系列产品：数据类型支持FLOAT16、FLOAT32。shape不支持非64字节对齐。
    - Atlas A2 训练系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品：数据类型支持BFLOAT16、FLOAT16、FLOAT32。
  - dimOptional（int64_t，入参）：需要进行切分的维度序号，对x相应轴进行对半切，Host侧的整型，数据类型支持INT64。取值范围为[-x.dim(), x.dim()-1]。
  - out（aclTensor*，计算输出）：表示计算结果，公式中的out<sub>i</sub>，Device侧的aclTensor，数据类型与计算输入x的类型一致，不支持非连续的Tensor。数据格式支持ND。
  - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。

- **返回值**：
  aclnnStatus：返回状态码。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：传入的x或out是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. x或out的数据类型不在支持的范围之内。
                                        2. dimOptional不在取值范围内。
  ```

## aclnnSwiGlu
- **参数说明**：
  - workspace（void*，入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnSwiGluGetWorkspaceSize获取。
  - executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream，入参）：指定执行任务的AscendCL Stream流。

- **返回值**：
  aclnnStatus：返回状态码。

## 约束与限制
无。