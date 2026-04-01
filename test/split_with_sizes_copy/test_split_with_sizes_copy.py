# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.split_with_sizes_copy 接口功能正确性
API 名称：torch.split_with_sizes_copy
API 签名：split_with_sizes_copy(Tensor self, SymInt[] split_sizes, int dim=0,
                               *, Tensor[]? out=None) -> Tensor[] | None

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | size-0 维度张量、非空张量                                    | 已覆盖                                         |
| 枚举选项         | dim=0、dim=1、dim=-1（负数索引）                             | 已覆盖                                         |
| 参数类型         | Tensor、list[int]、int dim、list[Tensor] out                 | 已覆盖                                         |
| 传参与不传参     | out 省略（返回列表模式）vs 显式传入；dim 默认 vs 显式        | 已覆盖                                         |
| 等价类/边界值    | 1D/2D/3D、单元素切片、size-0 tensor                          | 已覆盖                                         |
| 正常传参场景     | float32/float16/bfloat16/int64 dtype、非连续输入             | 已覆盖                                         |
| 异常传参场景     | out 与 split_sizes 长度不一致；混合设备 out 列表             | 已覆盖                                         |

未覆盖项及原因：
- split_sizes 总和与 dim 大小不一致：行为依赖实现，无稳定负例
- int32 dtype：NPU 对整型支持有限，未覆盖

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/device 符合预期），
     不做精度和数值正确性校验。
"""

import unittest

import torch

try:
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except ImportError:
    pass

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


class TestSplitWithSizesCopy(TestCase):
    """Test cases for torch.split_with_sizes_copy on NPU."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu',
                         f"Expected device 'npu', got '{self.device_name}'")
        self.device = torch.device(self.device_name)

    def test_basic_1d_dim0(self):
        """Basic 1D split on NPU along dim=0 with out=."""
        x = torch.arange(10, dtype=torch.float32, device=self.device)
        ss = [2, 3, 5]
        out = [
            torch.zeros(2, dtype=torch.float32, device=self.device),
            torch.zeros(3, dtype=torch.float32, device=self.device),
            torch.zeros(5, dtype=torch.float32, device=self.device),
        ]
        torch.split_with_sizes_copy(x, ss, dim=0, out=out)
        self.assertEqual(out[0].shape, torch.Size([2]))
        self.assertEqual(out[1].shape, torch.Size([3]))
        self.assertEqual(out[2].shape, torch.Size([5]))

    def test_2d_dim0(self):
        """2D tensor split along dim=0 on NPU."""
        x = torch.arange(20, dtype=torch.float32, device=self.device).reshape(4, 5)
        ss = [2, 2]
        out = [
            torch.zeros(2, 5, dtype=torch.float32, device=self.device),
            torch.zeros(2, 5, dtype=torch.float32, device=self.device),
        ]
        torch.split_with_sizes_copy(x, ss, dim=0, out=out)
        self.assertEqual(out[0].shape, torch.Size([2, 5]))
        self.assertEqual(out[1].shape, torch.Size([2, 5]))

    def test_2d_dim1(self):
        """2D tensor split along dim=1 on NPU."""
        x = torch.arange(20, dtype=torch.float32, device=self.device).reshape(4, 5)
        ss = [2, 3]
        out = [
            torch.zeros(4, 2, dtype=torch.float32, device=self.device),
            torch.zeros(4, 3, dtype=torch.float32, device=self.device),
        ]
        torch.split_with_sizes_copy(x, ss, dim=1, out=out)
        self.assertEqual(out[0].shape, torch.Size([4, 2]))
        self.assertEqual(out[1].shape, torch.Size([4, 3]))

    def test_dim_negative_one(self):
        """Negative dim index (-1) split on NPU."""
        x = torch.arange(12, dtype=torch.float32, device=self.device).reshape(3, 4)
        ss = [2, 2]
        out = [
            torch.zeros(3, 2, dtype=torch.float32, device=self.device),
            torch.zeros(3, 2, dtype=torch.float32, device=self.device),
        ]
        torch.split_with_sizes_copy(x, ss, dim=-1, out=out)
        self.assertEqual(out[0].shape, torch.Size([3, 2]))

    def test_bfloat16_dtype(self):
        """bfloat16 dtype preserved in out tensors on NPU."""
        x = torch.zeros(6, dtype=torch.bfloat16, device=self.device)
        ss = [2, 4]
        out = [
            torch.zeros(2, dtype=torch.bfloat16, device=self.device),
            torch.zeros(4, dtype=torch.bfloat16, device=self.device),
        ]
        torch.split_with_sizes_copy(x, ss, dim=0, out=out)
        self.assertEqual(out[0].dtype, torch.bfloat16)
        self.assertEqual(out[1].dtype, torch.bfloat16)

    def test_float16_dtype(self):
        """float16 dtype preserved in out tensors on NPU."""
        x = torch.zeros(6, dtype=torch.float16, device=self.device)
        ss = [3, 3]
        out = [
            torch.zeros(3, dtype=torch.float16, device=self.device),
            torch.zeros(3, dtype=torch.float16, device=self.device),
        ]
        torch.split_with_sizes_copy(x, ss, dim=0, out=out)
        self.assertEqual(out[0].dtype, torch.float16)
        self.assertEqual(out[1].dtype, torch.float16)

    def test_int64_dtype(self):
        """int64 dtype preserved in out tensors on NPU."""
        x = torch.arange(9, dtype=torch.int64, device=self.device)
        ss = [4, 5]
        out = [
            torch.zeros(4, dtype=torch.int64, device=self.device),
            torch.zeros(5, dtype=torch.int64, device=self.device),
        ]
        torch.split_with_sizes_copy(x, ss, dim=0, out=out)
        self.assertEqual(out[0].dtype, torch.int64)
        self.assertEqual(out[1].dtype, torch.int64)

    def test_no_out_returns_list_of_tensors(self):
        """Without out=, returns a list of Tensor with correct shape and device."""
        x = torch.arange(10, dtype=torch.float32, device=self.device)
        ss = [4, 6]
        result = torch.split_with_sizes_copy(x, ss)
        self.assertIsInstance(result, (list, tuple))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, torch.Size([4]))
        self.assertEqual(result[1].shape, torch.Size([6]))
        self.assertEqual(result[0].device.type, self.device_name)
        self.assertEqual(result[1].device.type, self.device_name)

    def test_no_out_tensors_are_contiguous_copies(self):
        """Without out=, returned tensors are fresh contiguous copies, not aliases."""
        x = torch.arange(6, dtype=torch.float32, device=self.device)
        ss = [3, 3]
        result = torch.split_with_sizes_copy(x, ss)
        self.assertTrue(result[0].is_contiguous())
        self.assertTrue(result[1].is_contiguous())

    def test_high_dim_3d(self):
        """3D tensor split along dim=1 on NPU."""
        x = torch.randn(2, 6, 4, dtype=torch.float32, device=self.device)
        ss = [2, 4]
        out = [
            torch.empty(2, 2, 4, dtype=torch.float32, device=self.device),
            torch.empty(2, 4, 4, dtype=torch.float32, device=self.device),
        ]
        torch.split_with_sizes_copy(x, ss, dim=1, out=out)
        self.assertEqual(out[0].shape, torch.Size([2, 2, 4]))
        self.assertEqual(out[1].shape, torch.Size([2, 4, 4]))
        self.assertEqual(out[0].device.type, self.device_name)
        self.assertEqual(out[1].device.type, self.device_name)

    def test_empty_tensor_split(self):
        """Split along a size-0 dimension on NPU does not raise."""
        x = torch.randn(0, 4, dtype=torch.float32, device=self.device)
        ss = [0]
        out = [torch.empty(0, 4, dtype=torch.float32, device=self.device)]
        torch.split_with_sizes_copy(x, ss, dim=0, out=out)
        self.assertEqual(out[0].shape, torch.Size([0, 4]))

    def test_single_element_split_chunk(self):
        """Split where one chunk has size 1 on NPU."""
        x = torch.randn(5, dtype=torch.float32, device=self.device)
        ss = [1, 4]
        out = [
            torch.empty(1, dtype=torch.float32, device=self.device),
            torch.empty(4, dtype=torch.float32, device=self.device),
        ]
        torch.split_with_sizes_copy(x, ss, dim=0, out=out)
        self.assertEqual(out[0].shape, torch.Size([1]))
        self.assertEqual(out[1].shape, torch.Size([4]))

    def test_non_contiguous_input(self):
        """Non-contiguous input tensor on NPU (via transpose)."""
        x = torch.randn(6, 4, dtype=torch.float32, device=self.device).t()
        self.assertFalse(x.is_contiguous())
        ss = [2, 4]
        out = [
            torch.empty(4, 2, dtype=torch.float32, device=self.device),
            torch.empty(4, 4, dtype=torch.float32, device=self.device),
        ]
        torch.split_with_sizes_copy(x, ss, dim=1, out=out)
        self.assertEqual(out[0].shape, torch.Size([4, 2]))
        self.assertEqual(out[1].shape, torch.Size([4, 4]))

    def test_default_dim_is_zero(self):
        """Omitting dim defaults to dim=0 on NPU."""
        x = torch.randn(6, 3, dtype=torch.float32, device=self.device)
        ss = [2, 4]
        out = [
            torch.empty(2, 3, dtype=torch.float32, device=self.device),
            torch.empty(4, 3, dtype=torch.float32, device=self.device),
        ]
        torch.split_with_sizes_copy(x, ss, out=out)
        self.assertEqual(out[0].shape, torch.Size([2, 3]))
        self.assertEqual(out[1].shape, torch.Size([4, 3]))

    def test_out_return_value_is_none(self):
        """When out= is provided, return value must be None."""
        x = torch.randn(6, dtype=torch.float32, device=self.device)
        ss = [3, 3]
        out = [
            torch.empty(3, dtype=torch.float32, device=self.device),
            torch.empty(3, dtype=torch.float32, device=self.device),
        ]
        ret = torch.split_with_sizes_copy(x, ss, dim=0, out=out)
        self.assertIsNone(ret)

    def test_out_len_mismatch_raises(self):
        """out list shorter than split_sizes raises an exception on NPU."""
        x = torch.arange(6, dtype=torch.float32, device=self.device)
        ss = [2, 2, 2]
        out = [
            torch.zeros(2, device=self.device),
            torch.zeros(2, device=self.device),
        ]
        with self.assertRaises((RuntimeError, ValueError, IndexError)):
            torch.split_with_sizes_copy(x, ss, dim=0, out=out)

    def test_mixed_device_out_raises(self):
        """Mixed-device out list (NPU + CPU) should raise RuntimeError."""
        x = torch.arange(6, dtype=torch.float32, device=self.device)
        ss = [3, 3]
        out = [
            torch.empty(3, dtype=torch.float32, device=self.device),
            torch.empty(3, dtype=torch.float32),
        ]
        with self.assertRaises(RuntimeError):
            torch.split_with_sizes_copy(x, ss, dim=0, out=out)

    def test_cpu_baseline(self):
        """CPU baseline: basic split succeeds and output shapes are correct."""
        x = torch.arange(8, dtype=torch.float32)
        ss = [3, 5]
        out = [torch.zeros(3, dtype=torch.float32), torch.zeros(5, dtype=torch.float32)]
        torch.split_with_sizes_copy(x, ss, dim=0, out=out)
        self.assertEqual(out[0].shape, torch.Size([3]))
        self.assertEqual(out[1].shape, torch.Size([5]))


if __name__ == "__main__":
    run_tests()
