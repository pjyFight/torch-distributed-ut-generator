## **路径命名**：覆盖主技能的规定（去掉前缀 `torch.distributed` 再转下划线）。例：`torch.distributed.all_reduce` → **`_all_reduce`**。

---

## 多卡测试策略（分布式 API 优先生效）

生成或增补 **`torch.distributed`** 相关 UT 时，按以下规则判断是否使用多卡 HCCL 测试：

### 1. 纯 Python 工具类 API（可使用单进程 / 单卡 / gloo）

以下类别的 API **无需多卡**，可在单进程环境下测试：

- **纯工具函数**：如 `utils._get_root_modules`、`_get_device_handle`
- **装饰器与状态机**：如 `_composable.contract`、`_get_registry`、`_insert_module_state`
- **纯数据结构构造**：如 `_dtensor_spec.TensorMeta`、`_composable_state`

这些 API 的特征：
- 不涉及实际的跨进程通信
- 仅操作 Python 对象、状态管理、参数检查
- 不依赖底层通信后端（HCCL/NCCL/Gloo）

### 2. 其他分布式 API（**必须使用多卡 HCCL**）

**除上述纯工具类外，所有分布式相关 API 默认使用多卡 NPU + HCCL 测试**，包括但不限于：

- **集合通信操作**：`all_reduce`、`all_gather`、`broadcast`、`reduce_scatter` 等
- **P2P 通信**：`send`、`recv`、`isend`、`irecv`
- **进程组相关**：`ProcessGroup`、`new_group`、`init_process_group`、`destroy_process_group`
- **异步工作对象**：`Work`、`Work.wait`、`Work.get_future`
- **分布式张量**：`DTensor`、`_local_tensor`、分布式 shard/stride 相关
- **FSDP 相关**：`_named_parameters_with_duplicates`、FSDP 状态管理
- **设备网格**：`DeviceMesh`、`_get_device_handle` 在多设备场景的行为

**要求**：
- 使用 `torch.multiprocessing.spawn` 创建多进程
- 后端必须使用 `hccl`（通过 `transfer_to_npu` 会自动将 nccl 映射为 hccl）
- 测试方法必须加 `@skipIfUnsupportMultiNPU(n)` 装饰器（n ≥ 2）
- 每个进程初始化 `init_process_group(backend='hccl', ...)`

### 3. 特殊说明

| API 类别 | 测试方式 | 说明 |
|----------|----------|------|
| Work / Work.wait | **多卡 HCCL** | 虽然是异步句柄，但验证真实 HCCL 异步语义 |
| DTensor._local_tensor | **多卡 HCCL** | 分布式张量在多 rank 下的本地 tensor 获取 |
| TensorMeta | 单进程 | 纯数据结构构造 |
| contract / _get_registry | 单进程 | 装饰器注册机制，纯 Python |
| _get_root_modules | 单进程 | 工具函数 |

---

## 多卡测试模板（HCCL）

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU


def _init_dist_process(rank, world_size, fn, backend='hccl'):
    """Initialize distributed process with HCCL backend."""
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    # Initialize NPU device
    torch.npu.set_device(rank)

    # Initialize process group with HCCL
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    try:
        fn(rank, world_size)
    finally:
        dist.destroy_process_group()


def _test_collective_example(rank, world_size):
    """Test collective operation on each rank."""
    # Create tensor on current NPU
    tensor = torch.ones(10, device=f'npu:{rank}')

    # Perform collective operation
    work = dist.all_reduce(tensor, async_op=True)
    work.wait()

    # Verify result
    expected = torch.ones(10) * world_size
    assert tensor.shape == expected.shape, f"Shape mismatch on rank {rank}"


class TestDistributedCollective(TestCase):
    """Test cases for distributed collective operations with HCCL."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    @skipIfUnsupportMultiNPU(2)
    def test_all_reduce_multinpu(self):
        """Test all_reduce with HCCL backend on multiple NPUs."""
        world_size = 2
        mp.spawn(
            _init_dist_process,
            args=(world_size, _test_collective_example),
            nprocs=world_size,
            join=True
        )


if __name__ == "__main__":
    run_tests()
```

---

## 单进程测试模板（纯工具类）

```python
import unittest
import torch

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


class TestDistributedUtility(TestCase):
    """Test cases for distributed utility functions (single process)."""

    def setUp(self):
        super().setUp()
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    def test_utility_function(self):
        """Test pure Python utility without distributed initialization."""
        # Direct function call without init_process_group
        result = torch.distributed.utils._get_root_modules(...)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    run_tests()
```

---

## 自检清单

- [ ] 路径与文件名无 `.`，且已按本文「路径命名」（去掉前缀 `torch.distributed` 再转下划线）
- [ ] 已按上文 **「多卡测试策略」** 判断测试方式：
  - 纯工具类 → 单进程测试
  - 其他分布式 API → **多卡 HCCL 测试**
- [ ] **设备检查放在 `setUp` 中**：使用 `self.assertEqual(device_name, 'npu')` 检查，**不是 skip**
- [ ] 多卡测试使用 `mp.spawn` 创建多进程
- [ ] 多卡测试初始化 `init_process_group(backend='hccl', ...)`
- [ ] 多卡测试使用 `torch.npu.set_device(rank)` 绑定设备
- [ ] 凡 **需要 ≥2 卡** 的测试方法均带 `@skipIfUnsupportMultiNPU(n)`
- [ ] 仅改动 `test/`；头部中文 docstring 与覆盖表已按实填写
- [ ] unittest + TestCase；无 pytest；无数值精度断言
- [ ] NPU 环境执行后已写 `UT_REPORT.md` 或 `UT_EXECUTION_REPORT.md`
- [ ] **报告包含三类用例统计**：PASS / FAIL / SKIP
- [ ] **报告包含跳过用例分析表**：列出每个跳过用例的原因及合理性评估
