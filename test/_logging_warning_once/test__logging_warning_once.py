# -*- coding: utf-8 -*-
"""
测试目的：验证 torch._logging.warning_once 接口功能正确性
API 名称：torch._logging.warning_once
API 签名：warning_once(logger_obj, *args, **kwargs) -> None

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                               |
|------------------|--------------------------------------------------------------|--------------------------------------------------------|
| 空/非空          | 消息字符串为空串与非空串                                     | 已覆盖：空字符串、普通字符串、带 % 格式化占位符的字符串 |
| 枚举选项         | N/A（无枚举型入参）                                          | N/A                                                    |
| 参数类型         | logger_obj 为 logging.Logger；args 为字符串及格式化参数      | 已覆盖：Logger 实例，字符串 msg，带位置参数的格式化     |
| 传参与不传参     | 仅 msg 与附带格式化参数两种形式                              | 已覆盖                                                 |
| 等价类/边界值    | 相同 (logger, msg, args) 只触发一次；不同消息各自触发一次   | 已覆盖                                                 |
| 正常传参场景     | 正常消息调用不报错；重复调用仅发出一条警告                   | 已覆盖                                                 |
| 异常传参场景     | logger_obj 为 None 时引发 AttributeError                    | 已覆盖                                                 |

未覆盖项及原因：
- kwargs 路径：warning_once 将 kwargs 直接透传给 logger.warning；不同 kwargs 是否破坏去重与 lru_cache
  的 hash 策略相关，lru_cache 要求参数 hashable，可变 kwargs 会导致 TypeError，属于调用方使用错误，
  无需在正常功能测试中覆盖。

注意：本测试仅验证功能正确性（调用不报错、副作用符合预期），
     不做精度和数值正确性校验。
"""
import logging
import unittest
import unittest.mock

import torch
import torch_npu  # noqa: F401 — registers NPU backend

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests() -> None:
        unittest.main(argv=sys.argv)


class TestWarningOnce(TestCase):
    """Test cases for torch._logging.warning_once."""

    def setUp(self):
        super().setUp()
        # Verify NPU backend is registered (even for CPU-only APIs).
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")
        # Reset lru_cache before each test to isolate deduplication state.
        if hasattr(torch._logging.warning_once, 'cache_clear'):
            torch._logging.warning_once.cache_clear()
        self.logger = logging.getLogger('torch._logging_test')

    def tearDown(self):
        super().tearDown()
        # Always clear cache after each test to avoid cross-test pollution.
        if hasattr(torch._logging.warning_once, 'cache_clear'):
            torch._logging.warning_once.cache_clear()

    # ------------------------------------------------------------------
    # Basic callable / return type
    # ------------------------------------------------------------------

    def test_basic_call_no_error(self):
        """Normal call with a plain message should not raise."""
        with unittest.mock.patch.object(self.logger, 'warning'):
            torch._logging.warning_once(self.logger, "simple message")

    def test_returns_none(self):
        """warning_once should return None."""
        with unittest.mock.patch.object(self.logger, 'warning'):
            result = torch._logging.warning_once(self.logger, "return value test")
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # Deduplication (once semantics)
    # ------------------------------------------------------------------

    def test_emits_once_on_duplicate_calls(self):
        """Duplicate (logger, msg, args) should trigger logger.warning exactly once."""
        with unittest.mock.patch.object(self.logger, 'warning') as mock_warn:
            torch._logging.warning_once(self.logger, "duplicate message")
            torch._logging.warning_once(self.logger, "duplicate message")
            torch._logging.warning_once(self.logger, "duplicate message")
        self.assertEqual(mock_warn.call_count, 1)

    def test_emits_once_with_format_args(self):
        """Same message + same format args → exactly one emission."""
        with unittest.mock.patch.object(self.logger, 'warning') as mock_warn:
            torch._logging.warning_once(self.logger, "value is %s", "foo")
            torch._logging.warning_once(self.logger, "value is %s", "foo")
        self.assertEqual(mock_warn.call_count, 1)

    def test_different_messages_each_emit_once(self):
        """Different messages should each be emitted once independently."""
        with unittest.mock.patch.object(self.logger, 'warning') as mock_warn:
            torch._logging.warning_once(self.logger, "message A")
            torch._logging.warning_once(self.logger, "message B")
            torch._logging.warning_once(self.logger, "message A")  # duplicate of first
            torch._logging.warning_once(self.logger, "message B")  # duplicate of second
        self.assertEqual(mock_warn.call_count, 2)

    def test_different_format_args_trigger_separate_emissions(self):
        """Same format string with different args are distinct cache keys."""
        with unittest.mock.patch.object(self.logger, 'warning') as mock_warn:
            torch._logging.warning_once(self.logger, "val=%s", "x")
            torch._logging.warning_once(self.logger, "val=%s", "y")
        self.assertEqual(mock_warn.call_count, 2)

    # ------------------------------------------------------------------
    # Message content forwarded correctly
    # ------------------------------------------------------------------

    def test_message_forwarded_to_logger(self):
        """The exact message (and args) passed to warning_once reaches logger.warning."""
        with unittest.mock.patch.object(self.logger, 'warning') as mock_warn:
            torch._logging.warning_once(self.logger, "hello %s", "world")
        mock_warn.assert_called_once_with("hello %s", "world")

    def test_empty_string_message(self):
        """Empty string message is a valid call and is forwarded once."""
        with unittest.mock.patch.object(self.logger, 'warning') as mock_warn:
            torch._logging.warning_once(self.logger, "")
            torch._logging.warning_once(self.logger, "")
        self.assertEqual(mock_warn.call_count, 1)

    # ------------------------------------------------------------------
    # Cache isolation after cache_clear
    # ------------------------------------------------------------------

    def test_cache_clear_resets_dedup(self):
        """After cache_clear(), the same message can be emitted again."""
        if not hasattr(torch._logging.warning_once, 'cache_clear'):
            self.skipTest("warning_once has no cache_clear — skip cache isolation test")
        with unittest.mock.patch.object(self.logger, 'warning') as mock_warn:
            torch._logging.warning_once(self.logger, "re-emit test")
            torch._logging.warning_once.cache_clear()
            torch._logging.warning_once(self.logger, "re-emit test")
        self.assertEqual(mock_warn.call_count, 2)

    # ------------------------------------------------------------------
    # Different logger objects
    # ------------------------------------------------------------------

    def test_different_loggers_emit_independently(self):
        """Two different logger objects with the same message are different cache keys."""
        logger_a = logging.getLogger('torch._logging_test_A')
        logger_b = logging.getLogger('torch._logging_test_B')
        with unittest.mock.patch.object(logger_a, 'warning') as mock_a, \
                unittest.mock.patch.object(logger_b, 'warning') as mock_b:
            torch._logging.warning_once(logger_a, "shared message")
            torch._logging.warning_once(logger_b, "shared message")
        self.assertEqual(mock_a.call_count, 1)
        self.assertEqual(mock_b.call_count, 1)

    # ------------------------------------------------------------------
    # Exception path
    # ------------------------------------------------------------------

    def test_none_logger_raises(self):
        """Passing None as logger_obj should raise AttributeError."""
        with self.assertRaises((AttributeError, TypeError)):
            torch._logging.warning_once(None, "should fail")

    def test_non_logger_object_raises(self):
        """Passing an object without .warning() method should raise AttributeError."""
        with self.assertRaises(AttributeError):
            torch._logging.warning_once(object(), "should fail")


if __name__ == "__main__":
    run_tests()
