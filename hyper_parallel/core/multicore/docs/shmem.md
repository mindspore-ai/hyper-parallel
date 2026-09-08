# Multicore 私有 SHMEM

SHMEM 仅在 Multicore 内部提供对称内存和单边通信。
实现、Torch binding、算子、构建脚本和 native 制品均位于本组件的 `shmem/`。

## 生命周期

托管 MegaMoE 通过 `shmem/lifecycle.py` 的 `acquire_symmetric_memory()` 获取独立 owner。
同一进程的多个 owner 共用一个 native manager、whole-world communicator 和对称 heap。
最后一个 owner 关闭时才 finalize；关闭前必须完成所有相关 backward 和设备操作。

- 仅支持覆盖整个 distributed world 且 rank 顺序一致的 group。
- 每个 owner 记录自己的分配；只能释放本 owner 的张量。
- `empty()` 和 `aligned_empty()` 分配对称内存；释放后 storage 被置为零，不能再访问数据。
- `close()` 幂等；所有 rank 必须以一致顺序完成关闭，再销毁 HCCL 进程组。
- `SYMMETRIC_MEMORY_HEAP_SIZE` 单位为字节；正整数，默认 1 GiB。初始化后不能扩大现有 heap。
- `SHMEM_IP_PORT` 在一个生命周期内各 rank 相同；重建生命周期前所有 rank 完成关闭并使用新的端口。

## 内部通信能力

`shmem/_bindings.py` 定位并加载随组件交付的 Torch native manager/ops。
基础操作包括 put、get、signal、wait 和 put-with-signal。调用方负责 stream 依赖、
通信完成和源/目标内存生命周期，不可把异步提交当作完成。

MegaMoE 通过自己的计算图与 kernel 使用私有 SHMEM 能力，用户无需直接操作 binding。

## 构建与制品

SHMEM 没有独立对外开关。启用 `--multicore on` 会构建 Torch Multicore 和必需的 SHMEM；
`--multicore off` 不交付二者的 native payload。内部 `shmem/build.sh` 由组件入口调用。

私有库位于 `core/multicore/shmem/lib`，使用专属 SONAME 和相对 RUNPATH，
避免与框架自带的通用 SHMEM 库冲突。详见 [构建与使用](build.md)。
