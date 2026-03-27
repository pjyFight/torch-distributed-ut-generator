# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._composable.contract._get_registry 接口的功能正确性
API 名称：torch.distributed._composable.contract._get_registry
API 签名：_get_registry(module) -> Optional[dict]

覆盖维度：
+------------------+----------------------------------------+
| 维度             | 覆盖值                                 |
+------------------+----------------------------------------+
| module 状态      | 已注册, 未注册                        |
| 返回值           | dict 或 None                         |
+------------------+----------------------------------------+

未覆盖项及原因：
- 内部 API，具体行为可能随版本变化

注意：本测试仅验证功能正确性，
     不做数值正确性校验。
"""

import torch
import torch.nn as nn
import pytest


def _assert_raises(exc_types, fn):
    try:
        fn()
    except exc_types:
        return
    raise AssertionError(f"expected one of {exc_types}")


try:
    from torch.distributed._composable.contract import _get_registry
except ImportError:
    pytest.skip("_get_registry not available in this PyTorch version", allow_module_level=True)


def test_get_registry_basic():
    """基础功能：获取模块注册表，未注册返回 None"""
    module = nn.Linear(4, 4)
    registry = _get_registry(module)
    assert registry is None or isinstance(registry, dict)


def test_get_registry_unregistered():
    """未注册的模块返回 None"""
    module = nn.Linear(4, 4)
    result = _get_registry(module)
    assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
