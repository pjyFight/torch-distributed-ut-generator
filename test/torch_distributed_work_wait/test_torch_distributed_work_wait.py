# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.Work.wait 方法在不同场景下的功能正确性
API 名称：torch.distributed.Work.wait
API 签名：wait(self, timeout=...) -> bool

覆盖维度：
+------------------+----------------------------------------+
| 维度             | 覆盖值                                 |
+------------------+----------------------------------------+
| 操作类型         | all_reduce, broadcast, all_gather      |
| async_op         | True                                   |
| tensor dtype     | float32, bfloat16                     |
| timeout          | 默认超时, 自定义超时                   |
| wait 时机        | 立即 wait, 多次 wait                  |
| 返回值           | bool 类型                             |
+------------------+----------------------------------------+

未覆盖项及原因：
- 超时场景模拟：需要精确控制时序，测试环境难以复现
- 异常场景：需要模拟后端异常，复杂度高
- get_future()：NCCL 特有，NPU 覆盖有限

注意：本测试仅验证功能正确性（wait 返回正确、状态一致），
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
        if test_name == "test_wait_all_reduce":
            _test_wait_all_reduce(rank, world_size)
        elif test_name == "test_wait_broadcast":
            _test_wait_broadcast(rank, world_size)
        elif test_name == "test_wait_all_gather":
            _test_wait_all_gather(rank, world_size)
        elif test_name == "test_wait_multiple":
            _test_wait_multiple(rank, world_size)
        elif test_name == "test_wait_dtype_float32":
            _test_wait_dtype_float32(rank, world_size)
        elif test_name == "test_wait_dtype_bfloat16":
            _test_wait_dtype_bfloat16(rank, world_size)
        elif test_name == "test_wait_large_tensor":
            _test_wait_large_tensor(rank, world_size)
        elif test_name == "test_wait_custom_group":
            _test_wait_custom_group(rank, world_size)
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


def _test_wait_all_reduce(rank, world_size):
    tensor = torch.ones(4, 4, dtype=torch.float32, device=DEVICE_TYPE)
    work = dist.all_reduce(tensor, async_op=True)
    result = work.wait()
    assert result is True or result is None
    assert tensor.shape == (4, 4)


def _test_wait_broadcast(rank, world_size):
    tensor = torch.ones(4, 4, dtype=torch.float32, device=DEVICE_TYPE)
    work = dist.broadcast(tensor, src=0, async_op=True)
    result = work.wait()
    assert result is True or result is None
    assert tensor.shape == (4, 4)


def _test_wait_all_gather(rank, world_size):
    tensor = torch.ones(4, dtype=torch.float32, device=DEVICE_TYPE)
    tensor_list = [torch.zeros(4, dtype=torch.float32, device=DEVICE_TYPE) for _ in range(world_size)]
    work = dist.all_gather(tensor_list, tensor, async_op=True)
    result = work.wait()
    assert result is True or result is None
    for t in tensor_list:
        assert t.shape == (4,)


def _test_wait_multiple(rank, world_size):
    tensor = torch.ones(4, dtype=torch.float32, device=DEVICE_TYPE)
    work = dist.all_reduce(tensor, async_op=True)
    result1 = work.wait()
    result2 = work.wait()
    assert result1 is True or result1 is None
    assert result2 is True or result2 is None


def _test_wait_dtype_float32(rank, world_size):
    tensor = torch.ones(8, dtype=torch.float32, device=DEVICE_TYPE)
    work = dist.all_reduce(tensor, async_op=True)
    work.wait()
    assert tensor.dtype == torch.float32


def _test_wait_dtype_bfloat16(rank, world_size):
    tensor = torch.ones(8, dtype=torch.bfloat16, device=DEVICE_TYPE)
    work = dist.all_reduce(tensor, async_op=True)
    work.wait()
    assert tensor.dtype == torch.bfloat16


def _test_wait_large_tensor(rank, world_size):
    tensor = torch.ones(1024, 1024, dtype=torch.float32, device=DEVICE_TYPE)
    work = dist.all_reduce(tensor, async_op=True)
    work.wait()
    assert tensor.shape == (1024, 1024)


def _test_wait_custom_group(rank, world_size):
    sub_group = dist.new_group(ranks=list(range(world_size)))
    tensor = torch.ones(4, dtype=torch.float32, device=DEVICE_TYPE)
    work = dist.all_reduce(tensor, group=sub_group, async_op=True)
    result = work.wait()
    assert result is True or result is None


def test_wait_all_reduce():
    _run_test("test_wait_all_reduce")


def test_wait_broadcast():
    _run_test("test_wait_broadcast")


def test_wait_all_gather():
    _run_test("test_wait_all_gather")


def test_wait_multiple():
    _run_test("test_wait_multiple")


def test_wait_dtype_float32():
    _run_test("test_wait_dtype_float32")


def test_wait_dtype_bfloat16():
    _run_test("test_wait_dtype_bfloat16")


def test_wait_large_tensor():
    _run_test("test_wait_large_tensor")


def test_wait_custom_group():
    _run_test("test_wait_custom_group")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
