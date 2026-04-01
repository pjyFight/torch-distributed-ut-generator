# -*- coding: utf-8 -*-
"""
测试目的：验证 torch._foreach_copy_ 接口功能正确性
API 名称：torch._foreach_copy_
API 签名：torch._foreach_copy_(self: List[Tensor], src: List[Tensor], non_blocking: bool = False) -> None

覆盖维度表：
| 覆盖维度         | 说明                                                                          | 覆盖情况                                                                         |
|------------------|-------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| 空/非空          | 空列表（self/src 均为 []）vs 非空列表                                         | 已覆盖：test_foreach_copy_npu_empty_list（空列表触发 RuntimeError），test_foreach_copy_npu_basic |
| 枚举选项         | non_blocking: False（默认）/ True                                             | 已覆盖：test_foreach_copy_npu_non_blocking_false, test_foreach_copy_npu_non_blocking_true |
| 参数类型         | self/src 均为 List[Tensor]；non_blocking 为 bool                              | 已覆盖：各 NPU 用例覆盖不同组合                                                  |
| 传参与不传参     | non_blocking 省略（使用默认值 False）vs 显式传入 True                         | 已覆盖                                                                           |
| 等价类/边界值    | 单元素列表、多元素列表、0-dim（标量）、1D、2D、高维（≥3D）、size-0 空 tensor | 已覆盖：test_foreach_copy_npu_scalar, test_foreach_copy_npu_empty_tensor 等      |
| shape            | 0-dim / 1D / 2D / 3D+ / 空 tensor (size-0)                                   | 已覆盖                                                                           |
| dtype            | float32 / float16 / bfloat16 / int32；以及 src/self dtype 不同时的隐式转换     | 已覆盖：test_foreach_copy_npu_dtypes, test_foreach_copy_npu_dtype_cast           |
| 连续性           | 非连续 src tensor（transpose 后的视图）                                       | 已覆盖：test_foreach_copy_npu_non_contiguous_src                                 |
| 混合设备输入     | self 在 NPU，src 在 CPU（native_functions.yaml device_check: NoCheck）        | 已覆盖：test_foreach_copy_npu_mixed_device（验证 fallback 行为）                 |
| 正常传参场景     | 基本拷贝、多张量批量拷贝、形状各异的张量组                                    | 已覆盖                                                                           |
| 异常传参场景     | self 与 src 列表长度不匹配                                                    | 已覆盖：test_foreach_copy_npu_length_mismatch                                    |

未覆盖项及原因：
- out-of-place 变体（torch._foreach_copy，无末尾下划线）：属于独立 API，不在本文件覆盖范围。
- bool dtype：Tensor.copy_ 语义下 bool 理论上支持，但 NPU 侧实际支持情况不明确，暂不覆盖。

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/device 符合预期），
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


class TestForeachCopy_(TestCase):
    """Test cases for torch._foreach_copy_ on NPU."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")
        self.device = torch.device(self.device_name)

    # ------------------------------------------------------------------ #
    # NPU tests  (target: >80% of all test methods)
    # ------------------------------------------------------------------ #

    def test_foreach_copy_npu_basic(self):
        """Basic in-place copy of float32 2D tensors on NPU: shapes and dtypes unchanged."""
        self_tensors = [
            torch.zeros(3, 4, device=self.device),
            torch.zeros(2, 5, device=self.device),
        ]
        src_tensors = [
            torch.randn(3, 4, device=self.device),
            torch.randn(2, 5, device=self.device),
        ]
        torch._foreach_copy_(self_tensors, src_tensors)
        for t, s in zip(self_tensors, src_tensors):
            self.assertEqual(t.shape, s.shape)
            self.assertEqual(t.dtype, torch.float32)
            self.assertEqual(t.device.type, self.device_name)

    def test_foreach_copy_npu_non_blocking_false(self):
        """Explicit non_blocking=False on NPU completes synchronously and returns None."""
        self_tensors = [torch.zeros(4, device=self.device)]
        src_tensors = [torch.randn(4, device=self.device)]
        torch._foreach_copy_(self_tensors, src_tensors, non_blocking=False)
        self.assertEqual(self_tensors[0].shape, torch.Size([4]))
        self.assertEqual(self_tensors[0].device.type, self.device_name)

    def test_foreach_copy_npu_non_blocking_true(self):
        """non_blocking=True on NPU does not raise; self tensors retain shape and device."""
        self_tensors = [
            torch.zeros(4, device=self.device),
            torch.zeros(6, device=self.device),
        ]
        src_tensors = [
            torch.randn(4, device=self.device),
            torch.randn(6, device=self.device),
        ]
        torch._foreach_copy_(self_tensors, src_tensors, non_blocking=True)
        for t in self_tensors:
            self.assertEqual(t.device.type, self.device_name)
            self.assertIsInstance(t, torch.Tensor)

    def test_foreach_copy_npu_single_tensor(self):
        """Single-element list on NPU: shape and dtype preserved."""
        self_tensors = [torch.zeros(5, 5, device=self.device)]
        src_tensors = [torch.randn(5, 5, device=self.device)]
        torch._foreach_copy_(self_tensors, src_tensors)
        self.assertEqual(self_tensors[0].shape, torch.Size([5, 5]))
        self.assertEqual(self_tensors[0].dtype, torch.float32)
        self.assertEqual(self_tensors[0].device.type, self.device_name)

    def test_foreach_copy_npu_empty_list(self):
        """Empty list inputs raise RuntimeError (at least one tensor required)."""
        with self.assertRaises(RuntimeError):
            torch._foreach_copy_([], [])

    def test_foreach_copy_npu_scalar(self):
        """0-dim (scalar) tensors in the list on NPU: shape remains []."""
        self_tensors = [
            torch.tensor(0.0, device=self.device),
            torch.tensor(0.0, device=self.device),
        ]
        src_tensors = [
            torch.tensor(1.0, device=self.device),
            torch.tensor(2.0, device=self.device),
        ]
        torch._foreach_copy_(self_tensors, src_tensors)
        for t in self_tensors:
            self.assertEqual(t.shape, torch.Size([]))
            self.assertEqual(t.device.type, self.device_name)

    def test_foreach_copy_npu_high_dim(self):
        """High-dimensional tensors (3D and 4D) on NPU: shape unchanged after copy."""
        self_tensors = [
            torch.zeros(2, 3, 4, device=self.device),
            torch.zeros(1, 2, 3, 4, device=self.device),
        ]
        src_tensors = [
            torch.randn(2, 3, 4, device=self.device),
            torch.randn(1, 2, 3, 4, device=self.device),
        ]
        torch._foreach_copy_(self_tensors, src_tensors)
        self.assertEqual(self_tensors[0].shape, torch.Size([2, 3, 4]))
        self.assertEqual(self_tensors[1].shape, torch.Size([1, 2, 3, 4]))
        for t in self_tensors:
            self.assertEqual(t.device.type, self.device_name)

    def test_foreach_copy_npu_empty_tensor(self):
        """size-0 tensors in the list on NPU: shape remains (0, 4)."""
        self_tensors = [torch.zeros(0, 4, device=self.device)]
        src_tensors = [torch.randn(0, 4, device=self.device)]
        torch._foreach_copy_(self_tensors, src_tensors)
        self.assertEqual(self_tensors[0].shape, torch.Size([0, 4]))
        self.assertIsInstance(self_tensors[0], torch.Tensor)

    def test_foreach_copy_npu_dtypes(self):
        """Supported dtypes (float32, float16, bfloat16, int32) on NPU: self retains dtype."""
        float_dtypes = [torch.float32, torch.float16, torch.bfloat16]
        for dtype in float_dtypes:
            self_t = [torch.zeros(4, 4, dtype=dtype, device=self.device)]
            src_t = [torch.randn(4, 4, dtype=dtype, device=self.device)]
            torch._foreach_copy_(self_t, src_t)
            self.assertEqual(self_t[0].dtype, dtype, f"dtype mismatch for {dtype}")
            self.assertEqual(self_t[0].device.type, self.device_name)

        # int32
        self_int = [torch.zeros(4, 4, dtype=torch.int32, device=self.device)]
        src_int = [torch.ones(4, 4, dtype=torch.int32, device=self.device)]
        torch._foreach_copy_(self_int, src_int)
        self.assertEqual(self_int[0].dtype, torch.int32)
        self.assertEqual(self_int[0].device.type, self.device_name)

    def test_foreach_copy_npu_dtype_cast(self):
        """Copy float32 src into float16 self on NPU: self dtype unchanged (implicit cast)."""
        self_tensors = [torch.zeros(4, dtype=torch.float16, device=self.device)]
        src_tensors = [torch.randn(4, dtype=torch.float32, device=self.device)]
        torch._foreach_copy_(self_tensors, src_tensors)
        self.assertEqual(self_tensors[0].dtype, torch.float16)
        self.assertEqual(self_tensors[0].shape, torch.Size([4]))
        self.assertEqual(self_tensors[0].device.type, self.device_name)

    def test_foreach_copy_npu_non_contiguous_src(self):
        """Non-contiguous (transposed) source tensors on NPU: copy does not raise."""
        base = torch.randn(8, 8, device=self.device)
        src = base.t()  # non-contiguous after transpose
        self.assertFalse(src.is_contiguous())
        self_tensors = [torch.zeros(8, 8, device=self.device)]
        torch._foreach_copy_(self_tensors, [src])
        self.assertEqual(self_tensors[0].shape, torch.Size([8, 8]))
        self.assertEqual(self_tensors[0].device.type, self.device_name)

    def test_foreach_copy_npu_varied_shapes(self):
        """Multiple tensors with different shapes in one call on NPU."""
        shapes = [(3,), (2, 4), (1, 3, 5)]
        self_tensors = [torch.zeros(*s, device=self.device) for s in shapes]
        src_tensors = [torch.randn(*s, device=self.device) for s in shapes]
        torch._foreach_copy_(self_tensors, src_tensors)
        for t, s in zip(self_tensors, shapes):
            self.assertEqual(t.shape, torch.Size(s))
            self.assertEqual(t.device.type, self.device_name)

    def test_foreach_copy_npu_return_value(self):
        """In-place op modifies self in-place; self tensors remain on NPU with correct shape."""
        self_tensors = [torch.zeros(3, device=self.device)]
        src_tensors = [torch.ones(3, device=self.device)]
        torch._foreach_copy_(self_tensors, src_tensors)
        self.assertEqual(self_tensors[0].shape, torch.Size([3]))
        self.assertEqual(self_tensors[0].device.type, self.device_name)

    def test_foreach_copy_npu_mixed_device(self):
        """self on NPU, src on CPU: device_check=NoCheck allows cross-device fallback."""
        # native_functions.yaml sets device_check: NoCheck for this op, meaning
        # the kernel falls back to slow path instead of raising for mixed devices.
        self_tensors = [torch.zeros(4, device=self.device)]
        src_tensors = [torch.randn(4)]  # CPU tensor
        torch._foreach_copy_(self_tensors, src_tensors)
        self.assertEqual(self_tensors[0].device.type, self.device_name)
        self.assertEqual(self_tensors[0].shape, torch.Size([4]))

    def test_foreach_copy_npu_length_mismatch(self):
        """Mismatched list lengths (len(self) != len(src)) should raise RuntimeError on NPU."""
        self_tensors = [
            torch.zeros(4, device=self.device),
            torch.zeros(4, device=self.device),
        ]
        src_tensors = [torch.randn(4, device=self.device)]  # only 1 element
        with self.assertRaises(RuntimeError):
            torch._foreach_copy_(self_tensors, src_tensors)

    # ------------------------------------------------------------------ #
    # CPU baseline  (target: ≤20% of all test methods)
    # ------------------------------------------------------------------ #

    def test_foreach_copy_cpu_baseline(self):
        """CPU baseline: in-place copy modifies self tensors with correct shape/dtype."""
        self_tensors = [torch.zeros(3, 4), torch.zeros(2, 5)]
        src_tensors = [torch.randn(3, 4), torch.randn(2, 5)]
        torch._foreach_copy_(self_tensors, src_tensors)
        expected_shapes = [torch.Size([3, 4]), torch.Size([2, 5])]
        for t, shape in zip(self_tensors, expected_shapes):
            self.assertEqual(t.shape, shape)
            self.assertEqual(t.dtype, torch.float32)


if __name__ == "__main__":
    run_tests()
