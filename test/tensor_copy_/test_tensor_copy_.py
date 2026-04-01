# -*- coding: utf-8 -*-
"""
测试目的：验证 Tensor.copy_ 接口功能正确性
API 名称：Tensor.copy_
API 签名：Tensor.copy_(src: Tensor, non_blocking: bool = False) -> Tensor

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          | 空 tensor（size=0 维度）与正常 tensor                        | 已覆盖：test_copy_npu_empty_tensor             |
| 枚举选项         | non_blocking=True / False                                    | 已覆盖：test_copy_npu_non_blocking_false / true|
| 参数类型         | src 为 Tensor（NPU/CPU）；非 Tensor 类型触发异常             | 已覆盖：test_copy_npu_invalid_src_type         |
| 传参与不传参     | non_blocking 显式传入 vs 省略默认                            | 已覆盖：test_copy_npu_basic / non_blocking 系列|
| 等价类/边界值    | 0-dim（标量）、1D、2D、高维、空 tensor、大 tensor           | 已覆盖：test_copy_npu_scalar / 1d / 3d / etc.  |
| 正常传参场景     | 同设备同 dtype、同设备跨 dtype、跨设备（CPU→NPU/NPU→CPU）   | 已覆盖：test_copy_npu_dtype_cast / cross_device|
| 异常传参场景     | shape 不兼容触发 RuntimeError；非 Tensor 触发 TypeError      | 已覆盖：test_copy_npu_shape_mismatch / invalid |
| shape            | 0-dim / 1D / 2D / 3D+ / 空 tensor                           | 已覆盖                                         |
| dtype            | float32 / float16 / bfloat16 / int32 / int64                 | 已覆盖：test_copy_npu_supported_dtypes         |
| device           | NPU 主路径（>80%）+ CPU 基线（≤20%）                        | 已覆盖：16 NPU / 2 CPU = 88.9% NPU             |
| 可选参数         | non_blocking 默认 False 与显式 True                          | 已覆盖                                         |
| 连续性           | 非连续 tensor（transpose 后）作为 src                        | 已覆盖：test_copy_npu_non_contiguous_src       |
| 跨设备拷贝       | CPU→NPU、NPU→CPU（copy_ 支持跨设备）                        | 已覆盖：test_copy_npu_from_cpu / to_cpu        |
| 自身拷贝         | dst 与 src 为同一 tensor                                     | 已覆盖：test_copy_npu_self_copy                |
| 返回值           | copy_ 返回 self（即 dst 本身）                               | 已覆盖：test_copy_npu_returns_self             |

未覆盖项及原因：
- 无

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""

import torch
import torch_npu  # noqa: F401

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    import unittest
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


class TestTensorCopy(TestCase):
    """Test cases for Tensor.copy_ on NPU."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")
        self.device = torch.device(self.device_name)

    # ------------------------------------------------------------------ #
    # NPU tests  (target: >80% of all test methods)
    # ------------------------------------------------------------------ #

    def test_copy_npu_basic(self):
        """Basic copy_ on NPU: shape, dtype, and device are preserved."""
        dst = torch.zeros(4, 4, device=self.device)
        src = torch.ones(4, 4, device=self.device)
        dst.copy_(src)
        self.assertEqual(dst.shape, torch.Size([4, 4]))
        self.assertEqual(dst.dtype, torch.float32)
        self.assertEqual(dst.device.type, self.device_name)

    def test_copy_npu_returns_self(self):
        """copy_ returns self (the dst tensor)."""
        dst = torch.zeros(3, 3, device=self.device)
        src = torch.ones(3, 3, device=self.device)
        result = dst.copy_(src)
        self.assertIs(result, dst)

    def test_copy_npu_scalar(self):
        """0-dim (scalar) tensor copy on NPU."""
        dst = torch.tensor(0.0, device=self.device)
        src = torch.tensor(1.0, device=self.device)
        dst.copy_(src)
        self.assertEqual(dst.shape, torch.Size([]))
        self.assertEqual(dst.dtype, torch.float32)

    def test_copy_npu_empty_tensor(self):
        """Empty tensor (size-0 dim) copy does not raise on NPU."""
        dst = torch.zeros(0, 4, device=self.device)
        src = torch.ones(0, 4, device=self.device)
        result = dst.copy_(src)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, torch.Size([0, 4]))

    def test_copy_npu_1d(self):
        """1D tensor copy on NPU."""
        dst = torch.zeros(10, device=self.device)
        src = torch.ones(10, device=self.device)
        dst.copy_(src)
        self.assertEqual(dst.shape, torch.Size([10]))

    def test_copy_npu_2d(self):
        """2D tensor copy on NPU."""
        dst = torch.zeros(5, 6, device=self.device)
        src = torch.ones(5, 6, device=self.device)
        dst.copy_(src)
        self.assertEqual(dst.shape, torch.Size([5, 6]))

    def test_copy_npu_high_dim(self):
        """High-dimensional (4D) tensor copy on NPU."""
        dst = torch.zeros(2, 3, 4, 5, device=self.device)
        src = torch.ones(2, 3, 4, 5, device=self.device)
        dst.copy_(src)
        self.assertEqual(dst.shape, torch.Size([2, 3, 4, 5]))
        self.assertEqual(dst.device.type, self.device_name)

    def test_copy_npu_non_blocking_false(self):
        """copy_ with non_blocking=False (explicit) on NPU."""
        dst = torch.zeros(3, 3, device=self.device)
        src = torch.ones(3, 3, device=self.device)
        result = dst.copy_(src, non_blocking=False)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, torch.Size([3, 3]))

    def test_copy_npu_non_blocking_true(self):
        """copy_ with non_blocking=True on NPU."""
        dst = torch.zeros(3, 3, device=self.device)
        src = torch.ones(3, 3, device=self.device)
        result = dst.copy_(src, non_blocking=True)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, torch.Size([3, 3]))

    def test_copy_npu_supported_dtypes(self):
        """copy_ preserves dst dtype for all supported dtypes on NPU."""
        for dtype in [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64]:
            with self.subTest(dtype=dtype):
                dst = torch.zeros(4, 4, dtype=dtype, device=self.device)
                src = torch.ones(4, 4, dtype=dtype, device=self.device)
                dst.copy_(src)
                self.assertEqual(dst.dtype, dtype, f"dtype mismatch for {dtype}")
                self.assertEqual(dst.shape, torch.Size([4, 4]))

    def test_copy_npu_dtype_cast(self):
        """copy_ from float16 src into float32 dst performs implicit cast on NPU."""
        dst = torch.zeros(4, 4, dtype=torch.float32, device=self.device)
        src = torch.ones(4, 4, dtype=torch.float16, device=self.device)
        dst.copy_(src)
        # dst dtype must remain float32 after copy
        self.assertEqual(dst.dtype, torch.float32)
        self.assertEqual(dst.shape, torch.Size([4, 4]))

    def test_copy_npu_non_contiguous_src(self):
        """Non-contiguous src (transposed view) copy on NPU."""
        src = torch.randn(8, 8, device=self.device).t()  # transpose → non-contiguous
        self.assertFalse(src.is_contiguous())
        dst = torch.zeros(8, 8, device=self.device)
        result = dst.copy_(src)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, torch.Size([8, 8]))

    def test_copy_npu_self_copy(self):
        """Self-copy (dst and src are the same tensor) on NPU does not raise."""
        x = torch.randn(5, 5, device=self.device)
        result = x.copy_(x)
        self.assertIs(result, x)
        self.assertEqual(x.shape, torch.Size([5, 5]))

    def test_copy_npu_from_cpu(self):
        """Cross-device copy from CPU src into NPU dst (copy_ supports this)."""
        dst = torch.zeros(4, 4, device=self.device)
        src = torch.ones(4, 4)  # CPU tensor
        result = dst.copy_(src)
        self.assertEqual(result.device.type, self.device_name)
        self.assertEqual(result.shape, torch.Size([4, 4]))

    def test_copy_npu_shape_mismatch(self):
        """Incompatible shapes between dst and src raise a RuntimeError on NPU."""
        dst = torch.zeros(3, 4, device=self.device)
        src = torch.ones(2, 5, device=self.device)
        with self.assertRaises(RuntimeError):
            dst.copy_(src)

    def test_copy_npu_invalid_src_type(self):
        """Non-Tensor src raises TypeError on NPU."""
        dst = torch.zeros(3, 3, device=self.device)
        with self.assertRaises(TypeError):
            dst.copy_("not_a_tensor")

    # ------------------------------------------------------------------ #
    # CPU baseline  (target: ≤20% of all test methods)
    # ------------------------------------------------------------------ #

    def test_copy_cpu_baseline(self):
        """CPU baseline: copy_ returns self with correct shape and dtype."""
        dst = torch.zeros(4, 4)
        src = torch.ones(4, 4)
        result = dst.copy_(src)
        self.assertIs(result, dst)
        self.assertEqual(result.shape, torch.Size([4, 4]))
        self.assertEqual(result.dtype, torch.float32)

    def test_copy_npu_to_cpu(self):
        """Cross-device copy from NPU src into CPU dst (copy_ supports this)."""
        dst = torch.zeros(4, 4)  # CPU tensor
        src = torch.ones(4, 4, device=self.device)  # NPU tensor
        result = dst.copy_(src)
        self.assertEqual(result.device.type, 'cpu')
        self.assertEqual(result.shape, torch.Size([4, 4]))


if __name__ == "__main__":
    run_tests()
