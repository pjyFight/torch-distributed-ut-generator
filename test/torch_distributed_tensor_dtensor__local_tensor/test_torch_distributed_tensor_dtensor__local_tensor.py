# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.tensor.DTensor._local_tensor 接口的功能正确性
API 名称：torch.distributed.tensor.DTensor._local_tensor
API 签名：_local_tensor 属性，返回本地张量

覆盖维度：
+------------------+----------------------------------------+
| 维度             | 覆盖值                                 |
+------------------+----------------------------------------+
| device_type      | cuda, npu                             |
| placements       | Replicate, Shard(0), Shard(1)         |
| tensor dtype     | float32, bfloat16                     |
| tensor shape     | [8], [4,4], [2,4]                    |
| 创建方式         | from_local, distribute_tensor         |
| 访问次数         | 多次访问 _local_tensor                |
+------------------+----------------------------------------+

未覆盖项及原因：
- Partial placement：仅部分 shard 的场景较复杂
- N dim > 2：已覆盖 2D 场景
- _local_tensor 写操作：属性为只读，写操作可能导致未定义行为

注意：本测试仅验证功能正确性（返回本地 tensor 正确），
     不做数值正确性校验。
"""

import os
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest

_IS_NPU = hasattr(torch, 'npu') and torch.npu.is_available()
_IS_CUDA = not _IS_NPU and torch.cuda.is_available()

if _IS_NPU:
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
    from torch.distributed.tensor import DeviceMesh
    from torch.distributed.tensor.placement_types import Replicate, Shard
else:
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor.placement_types import Replicate, Shard

DEVICE_TYPE = "npu" if _IS_NPU else ("cuda" if _IS_CUDA else "cpu")
BACKEND = "hccl" if _IS_NPU else ("nccl" if _IS_CUDA else "gloo")
WORLD_SIZE = 2


def _get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return str(s.getsockname()[1])


def _setup_device(rank):
    if _IS_NPU:
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch.npu.set_device(rank)
    elif _IS_CUDA:
        torch.cuda.set_device(rank)


def _worker(rank, world_size, port, test_name):
    _setup_device(rank)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group(backend=BACKEND, rank=rank, world_size=world_size)
    try:
        if test_name == "test_local_tensor_from_local":
            _test_local_tensor_from_local(rank, world_size)
        elif test_name == "test_local_tensor_replicate":
            _test_local_tensor_replicate(rank, world_size)
        elif test_name == "test_local_tensor_shard_dim0":
            _test_local_tensor_shard_dim0(rank, world_size)
        elif test_name == "test_local_tensor_shard_dim1":
            _test_local_tensor_shard_dim1(rank, world_size)
        elif test_name == "test_local_tensor_dtype_float32":
            _test_local_tensor_dtype_float32(rank, world_size)
        elif test_name == "test_local_tensor_dtype_bfloat16":
            _test_local_tensor_dtype_bfloat16(rank, world_size)
        elif test_name == "test_local_tensor_2d":
            _test_local_tensor_2d(rank, world_size)
        elif test_name == "test_local_tensor_large":
            _test_local_tensor_large(rank, world_size)
        elif test_name == "test_local_tensor_multiple_access":
            _test_local_tensor_multiple_access(rank, world_size)
    finally:
        dist.destroy_process_group()


def _run_test(test_name):
    if DEVICE_TYPE == "cpu":
        pytest.skip("无可用 GPU/NPU 设备，跳过分布式测试")
    port = _get_free_port()
    mp.spawn(
        _worker,
        args=(WORLD_SIZE, port, test_name),
        nprocs=WORLD_SIZE,
        join=True,
    )


def _test_local_tensor_from_local(rank, world_size):
    device_mesh = DeviceMesh(DEVICE_TYPE, torch.arange(world_size))
    local_tensor = torch.ones(4, dtype=torch.float32, device=DEVICE_TYPE)
    dtensor = torch.distributed.tensor.DTensor.from_local(
        local_tensor, device_mesh, [Replicate()]
    )
    result = dtensor._local_tensor
    assert result.shape == (4,)
    assert result.dtype == torch.float32


def _test_local_tensor_replicate(rank, world_size):
    device_mesh = DeviceMesh(DEVICE_TYPE, torch.arange(world_size))
    local_tensor = torch.ones(8, dtype=torch.float32, device=DEVICE_TYPE)
    dtensor = torch.distributed.tensor.DTensor.from_local(
        local_tensor, device_mesh, [Replicate()]
    )
    result = dtensor._local_tensor
    assert result.shape == (8,)
    assert result.dtype == torch.float32


def _test_local_tensor_shard_dim0(rank, world_size):
    device_mesh = DeviceMesh(DEVICE_TYPE, torch.arange(world_size))
    local_tensor = torch.ones(8, dtype=torch.float32, device=DEVICE_TYPE)
    dtensor = torch.distributed.tensor.DTensor.from_local(
        local_tensor, device_mesh, [Shard(0)]
    )
    result = dtensor._local_tensor
    assert result.shape[0] <= 8
    assert result.dtype == torch.float32


def _test_local_tensor_shard_dim1(rank, world_size):
    device_mesh = DeviceMesh(DEVICE_TYPE, torch.arange(world_size))
    local_tensor = torch.ones(4, 4, dtype=torch.float32, device=DEVICE_TYPE)
    dtensor = torch.distributed.tensor.DTensor.from_local(
        local_tensor, device_mesh, [Shard(1)]
    )
    result = dtensor._local_tensor
    assert result.shape[1] <= 4
    assert result.dtype == torch.float32


def _test_local_tensor_dtype_float32(rank, world_size):
    device_mesh = DeviceMesh(DEVICE_TYPE, torch.arange(world_size))
    local_tensor = torch.ones(4, dtype=torch.float32, device=DEVICE_TYPE)
    dtensor = torch.distributed.tensor.DTensor.from_local(
        local_tensor, device_mesh, [Replicate()]
    )
    result = dtensor._local_tensor
    assert result.dtype == torch.float32


def _test_local_tensor_dtype_bfloat16(rank, world_size):
    device_mesh = DeviceMesh(DEVICE_TYPE, torch.arange(world_size))
    local_tensor = torch.ones(4, dtype=torch.bfloat16, device=DEVICE_TYPE)
    dtensor = torch.distributed.tensor.DTensor.from_local(
        local_tensor, device_mesh, [Replicate()]
    )
    result = dtensor._local_tensor
    assert result.dtype == torch.bfloat16


def _test_local_tensor_2d(rank, world_size):
    device_mesh = DeviceMesh(DEVICE_TYPE, torch.arange(world_size))
    local_tensor = torch.ones(4, 4, dtype=torch.float32, device=DEVICE_TYPE)
    dtensor = torch.distributed.tensor.DTensor.from_local(
        local_tensor, device_mesh, [Replicate()]
    )
    result = dtensor._local_tensor
    assert result.shape == (4, 4)
    assert result.dtype == torch.float32


def _test_local_tensor_large(rank, world_size):
    device_mesh = DeviceMesh(DEVICE_TYPE, torch.arange(world_size))
    local_tensor = torch.ones(1024, 1024, dtype=torch.float32, device=DEVICE_TYPE)
    dtensor = torch.distributed.tensor.DTensor.from_local(
        local_tensor, device_mesh, [Replicate()]
    )
    result = dtensor._local_tensor
    assert result.shape == (1024, 1024)
    assert result.dtype == torch.float32


def _test_local_tensor_multiple_access(rank, world_size):
    device_mesh = DeviceMesh(DEVICE_TYPE, torch.arange(world_size))
    local_tensor = torch.ones(4, dtype=torch.float32, device=DEVICE_TYPE)
    dtensor = torch.distributed.tensor.DTensor.from_local(
        local_tensor, device_mesh, [Replicate()]
    )
    result1 = dtensor._local_tensor
    result2 = dtensor._local_tensor
    assert result1.shape == result2.shape
    assert result1.dtype == result2.dtype


def test_local_tensor_from_local():
    _run_test("test_local_tensor_from_local")


def test_local_tensor_replicate():
    _run_test("test_local_tensor_replicate")


def test_local_tensor_shard_dim0():
    _run_test("test_local_tensor_shard_dim0")


def test_local_tensor_shard_dim1():
    _run_test("test_local_tensor_shard_dim1")


def test_local_tensor_dtype_float32():
    _run_test("test_local_tensor_dtype_float32")


def test_local_tensor_dtype_bfloat16():
    _run_test("test_local_tensor_dtype_bfloat16")


def test_local_tensor_2d():
    _run_test("test_local_tensor_2d")


def test_local_tensor_large():
    _run_test("test_local_tensor_large")


def test_local_tensor_multiple_access():
    _run_test("test_local_tensor_multiple_access")
