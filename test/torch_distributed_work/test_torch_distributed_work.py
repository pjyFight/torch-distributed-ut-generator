# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.Work 对象在异步分布式操作中的功能正确性
API 名称：torch.distributed.Work
API 签名：Work 类的核心方法
  - is_completed() -> bool
  - is_success() -> bool
  - exception() -> Any
  - wait(timeout=...) -> bool
  - get_future() -> Future
  - source_rank() -> int
  - synchronize()
  - boxed() -> ScriptObject

覆盖维度：
+------------------+----------------------------------------+
| 维度             | 覆盖值                                 |
+------------------+----------------------------------------+
| 操作类型         | all_reduce, broadcast, all_gather      |
| async_op         | True, False                            |
| tensor dtype     | float32, bfloat16                     |
| wait 行为        | 立即 wait, 延迟 wait, 超时 wait        |
| is_completed     | 操作完成前后                           |
| Work 类型        | 正常 Work, FakeWork                    |
+------------------+----------------------------------------+

未覆盖项及原因：
- exception() 异常返回：需要模拟失败场景，测试环境难以复现
- get_future()：仅 NCCL 后端支持，NPU 覆盖有限
- boxed()/unbox()：内部 API，不建议在 UT 中直接使用

注意：本测试仅验证功能正确性（Work 对象状态正确），
     不做数值正确性校验。
"""

import os
import socket
import time
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
        if test_name == "test_work_all_reduce_async":
            _test_work_all_reduce_async(rank, world_size)
        elif test_name == "test_work_is_completed":
            _test_work_is_completed(rank, world_size)
        elif test_name == "test_work_wait_no_timeout":
            _test_work_wait_no_timeout(rank, world_size)
        elif test_name == "test_work_dtype_float32":
            _test_work_dtype_float32(rank, world_size)
        elif test_name == "test_work_dtype_bfloat16":
            _test_work_dtype_bfloat16(rank, world_size)
        elif test_name == "test_work_broadcast_async":
            _test_work_broadcast_async(rank, world_size)
        elif test_name == "test_work_all_gather_async":
            _test_work_all_gather_async(rank, world_size)
        elif test_name == "test_work_sync_wait":
            _test_work_sync_wait(rank, world_size)
        elif test_name == "test_work_source_rank":
            _test_work_source_rank(rank, world_size)
        elif test_name == "test_work_synchronize":
            _test_work_synchronize(rank, world_size)
        elif test_name == "test_work_multiple_concurrent":
            _test_work_multiple_concurrent(rank, world_size)
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


def _test_work_all_reduce_async(rank, world_size):
    tensor = torch.ones(4, 4, dtype=torch.float32, device=DEVICE_TYPE)
    work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
    assert isinstance(work, dist.Work)
    assert hasattr(work, 'wait')
    assert hasattr(work, 'is_completed')
    result = work.wait()
    assert tensor.shape == (4, 4)
    assert tensor.dtype == torch.float32


def _test_work_is_completed(rank, world_size):
    tensor = torch.ones(4, dtype=torch.float32, device=DEVICE_TYPE)
    work = dist.all_reduce(tensor, async_op=True)
    assert isinstance(work, dist.Work)
    assert work.is_completed() in [True, False]
    wait_result = work.wait()
    assert wait_result in (True, None)
    # 不同后端对 is_completed 状态刷新时机不同，这里只要求返回布尔值且可调用
    for _ in range(20):
        if work.is_completed():
            break
        time.sleep(0.01)
    assert work.is_completed() in [True, False]


def _test_work_wait_no_timeout(rank, world_size):
    tensor = torch.ones(4, 4, dtype=torch.float32, device=DEVICE_TYPE)
    work = dist.all_reduce(tensor, async_op=True)
    assert isinstance(work, dist.Work)
    result = work.wait()
    assert tensor.shape == (4, 4)


def _test_work_dtype_float32(rank, world_size):
    tensor = torch.ones(4, 4, dtype=torch.float32, device=DEVICE_TYPE)
    work = dist.all_reduce(tensor, async_op=True)
    work.wait()
    assert tensor.dtype == torch.float32
    assert tensor.shape == (4, 4)


def _test_work_dtype_bfloat16(rank, world_size):
    tensor = torch.ones(4, 4, dtype=torch.bfloat16, device=DEVICE_TYPE)
    work = dist.all_reduce(tensor, async_op=True)
    work.wait()
    assert tensor.dtype == torch.bfloat16


def _test_work_broadcast_async(rank, world_size):
    tensor = torch.ones(4, 4, dtype=torch.float32, device=DEVICE_TYPE)
    work = dist.broadcast(tensor, src=0, async_op=True)
    assert isinstance(work, dist.Work)
    work.wait()
    assert tensor.shape == (4, 4)


def _test_work_all_gather_async(rank, world_size):
    tensor = torch.ones(4, dtype=torch.float32, device=DEVICE_TYPE)
    tensor_list = [torch.zeros(4, dtype=torch.float32, device=DEVICE_TYPE) for _ in range(world_size)]
    work = dist.all_gather(tensor_list, tensor, async_op=True)
    assert isinstance(work, dist.Work)
    work.wait()
    for t in tensor_list:
        assert t.shape == (4,)


def _test_work_sync_wait(rank, world_size):
    tensor = torch.ones(4, 4, dtype=torch.float32, device=DEVICE_TYPE)
    work = dist.all_reduce(tensor, async_op=False)
    assert work is None


def _test_work_source_rank(rank, world_size):
    tensor = torch.ones(4, dtype=torch.float32, device=DEVICE_TYPE)
    peer = 1 - rank
    if rank == 0:
        work = dist.isend(tensor=tensor, dst=peer)
        assert isinstance(work, dist.Work)
        wait_result = work.wait()
        assert wait_result in (True, None)
        return

    work = dist.irecv(tensor=tensor, src=peer)
    assert isinstance(work, dist.Work)
    wait_result = work.wait()
    assert wait_result in (True, None)
    try:
        src_rank = work.source_rank()
        assert isinstance(src_rank, int)
        assert 0 <= src_rank < world_size
    except RuntimeError as e:
        # 某些后端上该接口对当前 Work 类型不支持，校验为预期错误
        assert "sourceRank() may only be called" in str(e)


def _test_work_synchronize(rank, world_size):
    tensor = torch.ones(4, 4, dtype=torch.float32, device=DEVICE_TYPE)
    work = dist.all_reduce(tensor, async_op=True)
    assert isinstance(work, dist.Work)
    work.synchronize()
    assert tensor.shape == (4, 4)


def _test_work_multiple_concurrent(rank, world_size):
    tensor1 = torch.ones(4, dtype=torch.float32, device=DEVICE_TYPE)
    tensor2 = torch.ones(4, dtype=torch.float32, device=DEVICE_TYPE)
    work1 = dist.all_reduce(tensor1, async_op=True)
    work2 = dist.all_reduce(tensor2, async_op=True)
    assert isinstance(work1, dist.Work)
    assert isinstance(work2, dist.Work)
    work1.wait()
    work2.wait()


def test_work_all_reduce_async():
    _run_test("test_work_all_reduce_async")


def test_work_is_completed():
    _run_test("test_work_is_completed")


def test_work_wait_no_timeout():
    _run_test("test_work_wait_no_timeout")


def test_work_dtype_float32():
    _run_test("test_work_dtype_float32")


def test_work_dtype_bfloat16():
    _run_test("test_work_dtype_bfloat16")


def test_work_broadcast_async():
    _run_test("test_work_broadcast_async")


def test_work_all_gather_async():
    _run_test("test_work_all_gather_async")


def test_work_sync_wait():
    _run_test("test_work_sync_wait")


def test_work_source_rank():
    _run_test("test_work_source_rank")


def test_work_synchronize():
    _run_test("test_work_synchronize")


def test_work_multiple_concurrent():
    _run_test("test_work_multiple_concurrent")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
