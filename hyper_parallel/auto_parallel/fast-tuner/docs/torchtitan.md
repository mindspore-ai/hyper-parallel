# 基于Torchtitan的自动搜索

## 1 快速上手

在安装了fast-tuner工具之后，使用如下命令：

```bash
fast-tuner-parallel --config ./config/args_for_parallel_tool_titan.json
```

其中 json 文件示例如下：

```json
{
    // 基础配置
    "rank_num" : 8,
    "gbs": 32,
    "npus_per_node": 8,
    "nnodes": 1,
    "output_path": "./output/nd_output/",

    // Torchtitan相关配置
    "toml_path" : "./config/example/torchtitan/debug_model.toml",
    "torchtitan_path" : "./torchtitan/run_train.sh"
}
```

参考文件：[配置文件 args_for_parallel_tool_titan.json](../config/setup_config/args_for_parallel_tool_titan.json)

输出推荐并行配置:

```js
dp, tp, pp, ep, step_time(μs)
 2,  1,  4,  1,  14383
 1,  1,  8,  1,  15811
 2,  2,  2,  1,  22738
 4,  1,  2,  1,  N/A
```

## 2 参数说明

**基础参数**

| 参数            | 含义                                         | 默认值                                                                    |
|---------------|--------------------------------------------|------------------------------------------------------------------------|
| rank_num      | 工具可用于做profile的卡数                           | 8                                                                      |
| npus_per_node | 用户用于训练的每个节点的卡数                             | 8                                                                      |
| nnodes        | 用户用于训练的节点数                                 | 1                                                                      |
| strategy      | 工具搜索的并行策略，可选策略有 [DP, TP, PP, CP, EP, FSDP] | {"DP":true, "TP":true, "EP":true, "FSDP":true, "CP":false, "PP":false} |
| gbs           | global batch size                          | 32                                                                     |
| output_path   | 日志输出路径                                     | ./output/nd_output/                                                    |

**加速库参数**

<table>
<tr>
 <th>加速库</th>
 <th>参数</th>
 <th>含义</th>
 <th>示例</th>
</tr>
<tr>
 <td rowspan="3"> torchtitan</td>
 <td>torchtitan_path</td>
 <td>训练脚本路径，本工具需要此文件拉起profile</td>
 <td>./torchtitan/run_train.sh</td>
</tr>
<tr>
 <td>toml_path</td>
 <td>模型参数配置文件</td>
 <td>./config/example/torchtitan/debug_model.toml</td>
</tr>
</table>

通常用户只需要配置好基础参数，也就是模型规模，支持的并行范式等信息，以及所用的大模型训练套件信息，即可开始配置搜索。对于熟悉本工具的用户，也提供一些更高自由度的选项。

**高级参数**

| 参数                  | 含义                                         | 默认值                                    |
|---------------------|--------------------------------------------|----------------------------------------|
| select_recompute    | 是否搜索自定义选重                                  | True                                   |
| profile_data_dir    | 已有profile数据路径                              | None                                   |
| parallel_num        | dryrun并行数                                  | 2                                      |
| max_expert_parallel | 搜索范围的最大EP数                                 | 64                                     |
| parser_result       | profile解析结果csv文件，不需要解析profile时填写此参数即可      | None                                   |
| alg_phase           | 选择使用搜索算法阶段；0--全流程搜索, 1--ND搜索, 2--流水线负载均衡搜索 | 1                                      |

## 3 参数配置

对于想要修改的参数，直接在json配置文件中修改即可。比如说在支持DP，TP，EP和PP，不支持CP，FSDP的场景下，可以按照如下方式修改：

```json
{
    // 其他配置
    "xxx": x,
    ...

    // 自定义需要调优的并行配置
    "strategy": {
        "DP": true,
        "TP": true,
        "PP": true,
        "EP": true,
        "CP": false,
        "FSDP": false,
    }
}
```

[返回原文档](../README.md)
