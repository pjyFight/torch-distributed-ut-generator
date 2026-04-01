# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.autograd.graph._MultiHandle 接口功能正确性
API 名称：torch.autograd.graph._MultiHandle
API 签名：
    class _MultiHandle(RemovableHandle):
        def __init__(self, handles: tuple[RemovableHandle, ...]) -> None
        def remove(self) -> None
        def __getstate__(self) -> tuple[RemovableHandle, ...]
        def __setstate__(self, state: tuple[RemovableHandle, ...]) -> None

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                                   |
|------------------|--------------------------------------------------------------|------------------------------------------------------------|
| 空/非空          | handles 参数为空元组与非空元组                               | 已覆盖：空元组、单个 handle、多个 handle                   |
| 枚举选项         | N/A（无枚举型入参）                                          | N/A                                                        |
| 参数类型         | handles 为 tuple[RemovableHandle, ...]                       | 已覆盖：空元组、单元素元组、多元素元组                     |
| 传参与不传参     | __init__ 必填参数 handles                                    | 已覆盖：传入不同长度的 tuple                               |
| 等价类/边界值    | 空 tuple、单个 handle、多个 handle                           | 已覆盖                                                     |
| 正常传参场景     | 构造、remove()、上下文管理器、pickle 序列化                  | 已覆盖（NPU 设备上执行为主，占比 >80%）                    |
| 异常传参场景     | N/A（无稳定文档化异常路径）                                  | 未覆盖，原因：_MultiHandle 为内部 API，无公开异常契约       |

未覆盖项及原因：
- 异常场景：_MultiHandle 是内部 API，无公开文档说明合法异常路径，
  避免编写脆弱负例测试。

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""
import pickle
import unittest

import torch
import torch_npu  # noqa: F401 — registers NPU backend

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        unittest.main(argv=sys.argv)


def _identity_hook(grad):
    """Module-level named function — required for pickle serialization (no lambdas)."""
    return grad


def _make_handle(device):
    """Create a tensor on the given device, register a hook, and return (handle, tensor)."""
    x = torch.randn(3, requires_grad=True, device=device)
    handle = x.register_hook(_identity_hook)
    return handle, x


class TestMultiHandle(TestCase):
    """Test cases for torch.autograd.graph._MultiHandle."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(
            self.device_name, 'npu',
            f"Expected device 'npu', got '{self.device_name}'"
        )
        self.device = torch.device(self.device_name)
        self.cls = torch.autograd.graph._MultiHandle

    # ------------------------------------------------------------------
    # Import / type checks
    # ------------------------------------------------------------------

    def test_class_exists(self):
        """_MultiHandle should be accessible from torch.autograd.graph."""
        self.assertTrue(
            hasattr(torch.autograd.graph, '_MultiHandle'),
            "_MultiHandle not found in torch.autograd.graph"
        )

    def test_class_is_type(self):
        """_MultiHandle must be a class (type)."""
        self.assertIsInstance(self.cls, type)

    def test_inherits_from_removable_handle(self):
        """_MultiHandle must subclass RemovableHandle."""
        mh = self.cls(())
        self.assertIsInstance(mh, torch.utils.hooks.RemovableHandle)

    # ------------------------------------------------------------------
    # Construction — NPU handles
    # ------------------------------------------------------------------

    def test_construct_with_empty_tuple_npu(self):
        """Constructing with an empty tuple on NPU should succeed."""
        mh = self.cls(())
        self.assertIsNotNone(mh)
        self.assertIsInstance(mh, self.cls)

    def test_construct_with_single_handle_npu(self):
        """Constructing with a single NPU handle should produce a valid _MultiHandle."""
        handle, _ = _make_handle(self.device)
        mh = self.cls((handle,))
        self.assertIsInstance(mh, self.cls)

    def test_construct_with_multiple_handles_npu(self):
        """Constructing with multiple NPU handles should succeed."""
        h1, _ = _make_handle(self.device)
        h2, _ = _make_handle(self.device)
        h3, _ = _make_handle(self.device)
        mh = self.cls((h1, h2, h3))
        self.assertIsInstance(mh, self.cls)

    def test_handles_attribute_matches_input_npu(self):
        """The .handles attribute should equal the tuple passed to __init__."""
        h1, _ = _make_handle(self.device)
        h2, _ = _make_handle(self.device)
        input_tuple = (h1, h2)
        mh = self.cls(input_tuple)
        self.assertEqual(mh.handles, input_tuple)

    # ------------------------------------------------------------------
    # remove() — NPU
    # ------------------------------------------------------------------

    def test_remove_empty_handles_npu(self):
        """remove() on an empty _MultiHandle should not raise."""
        mh = self.cls(())
        mh.remove()  # must not raise

    def test_remove_single_handle_npu(self):
        """remove() on a single-handle _MultiHandle should not raise."""
        handle, _ = _make_handle(self.device)
        mh = self.cls((handle,))
        mh.remove()

    def test_remove_multiple_handles_npu(self):
        """remove() on a multi-handle _MultiHandle should not raise."""
        h1, _ = _make_handle(self.device)
        h2, _ = _make_handle(self.device)
        mh = self.cls((h1, h2))
        mh.remove()

    def test_hook_inactive_after_remove_npu(self):
        """After remove(), the wrapped hook must not fire on subsequent backward passes."""
        counter = {"count": 0}

        def counting_hook(grad):
            counter["count"] += 1
            return grad

        x = torch.randn(3, requires_grad=True, device=self.device)
        raw_handle = x.register_hook(counting_hook)
        mh = self.cls((raw_handle,))

        (x * 2).sum().backward()
        count_before = counter["count"]

        mh.remove()
        x.grad = None
        (x * 3).sum().backward()
        self.assertEqual(counter["count"], count_before)

    # ------------------------------------------------------------------
    # Context manager (__enter__ / __exit__) — NPU
    # ------------------------------------------------------------------

    def test_context_manager_empty_npu(self):
        """Using an empty _MultiHandle as a context manager must not raise."""
        with self.cls(()):
            pass

    def test_context_manager_single_handle_npu(self):
        """Context manager with one NPU handle: __enter__ returns self."""
        handle, _ = _make_handle(self.device)
        with self.cls((handle,)) as mh:
            self.assertIsNotNone(mh)

    def test_context_manager_multiple_handles_npu(self):
        """Context manager with multiple NPU handles should not raise."""
        h1, _ = _make_handle(self.device)
        h2, _ = _make_handle(self.device)
        with self.cls((h1, h2)):
            pass

    def test_hook_inactive_after_context_exit_npu(self):
        """After exiting the context manager, hooks must no longer fire."""
        counter = {"count": 0}

        def counting_hook(grad):
            counter["count"] += 1
            return grad

        x = torch.randn(3, requires_grad=True, device=self.device)
        raw_handle = x.register_hook(counting_hook)

        with self.cls((raw_handle,)):
            pass

        count_after_exit = counter["count"]
        x.grad = None
        (x * 5).sum().backward()
        self.assertEqual(counter["count"], count_after_exit)

    # ------------------------------------------------------------------
    # Pickle serialization (__getstate__ / __setstate__)
    # ------------------------------------------------------------------

    def test_getstate_returns_tuple_empty(self):
        """__getstate__ on empty _MultiHandle must return a tuple."""
        mh = self.cls(())
        state = mh.__getstate__()
        self.assertIsInstance(state, tuple)

    def test_getstate_length_matches_handles_npu(self):
        """__getstate__ length must equal number of wrapped handles."""
        h1, _ = _make_handle(self.device)
        h2, _ = _make_handle(self.device)
        mh = self.cls((h1, h2))
        state = mh.__getstate__()
        self.assertIsInstance(state, tuple)
        self.assertEqual(len(state), 2)

    def test_setstate_restores_instance_npu(self):
        """__setstate__ must restore a valid _MultiHandle instance."""
        h1, _ = _make_handle(self.device)
        mh_original = self.cls((h1,))
        state = mh_original.__getstate__()

        mh_restored = self.cls.__new__(self.cls)
        mh_restored.__setstate__(state)
        self.assertIsInstance(mh_restored, self.cls)

    def test_pickle_round_trip_empty(self):
        """Empty _MultiHandle must survive a pickle round-trip."""
        mh = self.cls(())
        data = pickle.dumps(mh)
        mh2 = pickle.loads(data)
        self.assertIsInstance(mh2, self.cls)

    def test_pickle_round_trip_with_handles_npu(self):
        """Non-empty _MultiHandle (with NPU handles) must survive a pickle round-trip."""
        h1, _ = _make_handle(self.device)
        h2, _ = _make_handle(self.device)
        mh = self.cls((h1, h2))
        data = pickle.dumps(mh)
        mh2 = pickle.loads(data)
        self.assertIsInstance(mh2, self.cls)

    # ------------------------------------------------------------------
    # Backward integration on NPU — verify hook fires during backward
    # ------------------------------------------------------------------

    def test_hook_fires_before_remove_npu(self):
        """Hook wrapped in _MultiHandle must fire during backward before remove()."""
        counter = {"count": 0}

        def counting_hook(grad):
            counter["count"] += 1
            return grad

        x = torch.randn(3, requires_grad=True, device=self.device)
        raw_handle = x.register_hook(counting_hook)
        _mh = self.cls((raw_handle,))

        (x * 2).sum().backward()
        self.assertEqual(counter["count"], 1)

    # ------------------------------------------------------------------
    # CPU baseline (structural only — ≤20% of total tests)
    # ------------------------------------------------------------------

    def test_construct_cpu_baseline(self):
        """CPU baseline: construct _MultiHandle with CPU tensor hooks."""
        h_cpu, _ = _make_handle(torch.device('cpu'))
        mh = self.cls((h_cpu,))
        self.assertIsInstance(mh, self.cls)

    def test_remove_cpu_baseline(self):
        """CPU baseline: remove() should not raise for CPU handles."""
        h_cpu, _ = _make_handle(torch.device('cpu'))
        mh = self.cls((h_cpu,))
        mh.remove()


if __name__ == "__main__":
    run_tests()
