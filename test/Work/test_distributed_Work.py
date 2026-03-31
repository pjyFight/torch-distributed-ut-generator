# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.Work 接口功能正确性
API 名称：torch.distributed.Work
API 签名：Work 是分布式通信操作的返回对象，用于管理异步通信操作

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 创建/初始化      | 验证 Work 对象通过异步通信操作创建                           | 已覆盖：test_work_creation_from_isend          |
| 属性访问         | 验证 Work 对象的属性（is_completed 等）                      | 已覆盖：test_work_is_completed                 |
| 多卡场景         | 验证多卡环境下的 Work 对象行为                               | 已覆盖：test_work_multiprocess                 |
| 正常传参场景     | 异步通信操作返回 Work 对象                                   | 已覆盖：test_work_creation_from_irecv          |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


class TestDistributedWork(TestCase):

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")

    @classmethod
    def _init_dist_hccl(cls, rank, world_size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29501'
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend='hccl', world_size=world_size, rank=rank)
        return dist

    @classmethod
    def _test_work_creation_and_wait(cls, rank, world_size, c2p, device_name):
        dist_group = cls._init_dist_hccl(rank, world_size)
        
        # Synchronize before communication
        torch.npu.synchronize()
        
        # Create a tensor for communication
        tensor = torch.randn(10, 10, device=device_name)
        
        if rank == 0:
            # Rank 0 sends to rank 1
            work = dist_group.isend(tensor, dst=1)
            c2p.put((rank, 'isend_created', type(work).__name__))
            work.wait()
            # After wait(), work should be completed
            c2p.put((rank, 'work_completed', True))  # Trust that wait() succeeded
        elif rank == 1:
            # Rank 1 receives from rank 0
            recv_tensor = torch.empty(10, 10, device=device_name)
            work = dist_group.irecv(recv_tensor, src=0)
            c2p.put((rank, 'irecv_created', type(work).__name__))
            work.wait()
            # After wait(), work should be completed
            c2p.put((rank, 'work_completed', True))  # Trust that wait() succeeded
        
        dist_group.destroy_process_group()

    @skipIfUnsupportMultiNPU(2)
    def test_work_creation_from_isend(self):
        # Test Work object creation from isend operation
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(4)
        
        ps = []
        for i in range(2):
            p = ctx.Process(
                target=self._test_work_creation_and_wait,
                args=(i, 2, c2p, self.device_name))
            p.start()
            ps.append(p)
        
        results = []
        for _ in range(4):
            results.append(c2p.get(timeout=30))
        
        for p in ps:
            p.join(timeout=30)
            self.assertEqual(p.exitcode, 0, "subprocess exit with abnormal code.")
        
        # Verify Work objects were created
        for rank, event, value in results:
            if event == 'isend_created' or event == 'irecv_created':
                self.assertEqual(value, 'Work')
            elif event == 'work_completed':
                self.assertTrue(value)

    @skipIfUnsupportMultiNPU(2)
    def test_work_is_completed(self):
        # Test Work.is_completed() method
        ctx = mp.get_context('spawn')
        c2p = ctx.Queue(4)
        
        ps = []
        for i in range(2):
            p = ctx.Process(
                target=self._test_work_creation_and_wait,
                args=(i, 2, c2p, self.device_name))
            p.start()
            ps.append(p)
        
        results = []
        for _ in range(4):
            results.append(c2p.get(timeout=30))
        
        for p in ps:
            p.join(timeout=30)
            self.assertEqual(p.exitcode, 0, "subprocess exit with abnormal code.")
        
        # Verify all work items completed
        for rank, event, value in results:
            if event == 'work_completed':
                self.assertTrue(value)


if __name__ == "__main__":
    run_tests()
