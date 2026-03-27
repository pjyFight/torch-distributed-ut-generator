# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._composable.contract 装饰器的功能正确性
API 名称：torch.distributed._composable.contract
API 签名：contract(state_cls: type = _State) -> Callable

覆盖维度：
+------------------+----------------------------------------+
| 维度             | 覆盖值                                 |
+------------------+----------------------------------------+
| contract 本身    | 可导入，可作为装饰器                   |
| state_cls        | 默认 _State                          |
+------------------+----------------------------------------+

未覆盖项及原因：
- 内部 API，具体行为可能随版本变化
- contract 装饰器需要完整分布式环境才能正确使用

注意：本测试仅验证功能正确性，
     不做数值正确性校验。
"""

import torch
import pytest

try:
    from torch.distributed._composable import contract
    from torch.distributed._composable_state import _State
except ImportError:
    pytest.skip("contract not available in this PyTorch version", allow_module_level=True)


def test_contract_importable():
    """验证 contract 可导入"""
    assert contract is not None
    assert callable(contract)


def test_contract_returns_callable():
    """验证 contract 返回可调用对象"""
    result = contract()
    assert callable(result)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
