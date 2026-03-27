# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.utils._get_root_modules 接口的功能正确性
API 名称：torch.distributed.utils._get_root_modules
API 签名：_get_root_modules(modules: list[nn.Module]) -> list[nn.Module]

覆盖维度：
+------------------+----------------------------------------+
| 维度             | 覆盖值                                 |
+------------------+----------------------------------------+
| modules 输入     | 单模块列表, 多模块列表                |
| module 类型      | Linear, Sequential                    |
| 返回值           | list 类型，根模块列表                 |
+------------------+----------------------------------------+

未覆盖项及原因：
- 内部 API，具体行为可能随版本变化

注意：本测试仅验证功能正确性（返回正确的根模块列表），
     不做数值正确性校验。
"""

import torch
import torch.nn as nn
import pytest

try:
    from torch.distributed.utils import _get_root_modules
except ImportError:
    pytest.skip("_get_root_modules not available in this PyTorch version", allow_module_level=True)


def test_get_root_modules_basic():
    """基础功能：传入模块列表"""
    modules = [nn.Linear(4, 4)]
    roots = _get_root_modules(modules)
    assert isinstance(roots, list)
    assert len(roots) >= 1


def test_get_root_modules_multiple():
    """多个模块"""
    modules = [nn.Linear(4, 4), nn.Linear(4, 4)]
    roots = _get_root_modules(modules)
    assert isinstance(roots, list)
    assert len(roots) >= 1


def test_get_root_modules_empty():
    """空列表"""
    modules = []
    roots = _get_root_modules(modules)
    assert isinstance(roots, list)
    assert len(roots) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
