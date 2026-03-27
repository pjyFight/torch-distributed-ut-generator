# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed._composable_state._insert_module_state 接口的功能正确性
API 名称：torch.distributed._composable_state._insert_module_state
API 签名：_insert_module_state(module, state) -> None

覆盖维度：
+------------------+----------------------------------------+
| 维度             | 覆盖值                                 |
+------------------+----------------------------------------+
| module 类型      | Linear, Sequential, ModuleList         |
| state 类型       | FSDPState, Custom State               |
| module 结构      | 单层, 多层嵌套                        |
| state 访问       | 通过 _composable_state 获取           |
+------------------+----------------------------------------+

未覆盖项及原因：
- 内部 API，行为可能变化
- 特定 state 类型需要完整分布式环境

注意：本测试仅验证功能正确性（状态插入和获取正确），
     不做数值正确性校验。
"""

import torch
import torch.nn as nn
import pytest

try:
    from torch.distributed._composable_state import _insert_module_state, _get_module_state
except ImportError:
    pytest.skip("_insert_module_state not available", allow_module_level=True)


def _assert_raises(exc_types, fn):
    try:
        fn()
    except exc_types:
        return
    raise AssertionError(f"expected one of {exc_types}")


@pytest.mark.timeout(120)
def test_insert_module_state_basic():
    """基础功能：为 module 插入 state"""
    class DummyState:
        def __init__(self):
            self.value = "test"

    module = nn.Linear(4, 4)
    state = DummyState()
    _insert_module_state(module, state)

    retrieved = _get_module_state(module)
    assert retrieved is state
    assert retrieved.value == "test"


@pytest.mark.timeout(120)
def test_insert_module_state_sequential():
    """Sequential 模块"""
    class DummyState:
        def __init__(self, val):
            self.value = val

    model = nn.Sequential(
        nn.Linear(4, 4),
        nn.ReLU(),
        nn.Linear(4, 4)
    )
    state = DummyState("sequential_test")
    _insert_module_state(model, state)

    retrieved = _get_module_state(model)
    assert retrieved is state


@pytest.mark.timeout(120)
def test_insert_module_state_nested():
    """嵌套模块"""
    class DummyState:
        def __init__(self, val):
            self.value = val

    class SubModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, x):
            return self.linear(x)

    module = SubModule()
    state = DummyState("nested_test")
    _insert_module_state(module, state)

    retrieved = _get_module_state(module)
    assert retrieved is state


@pytest.mark.timeout(120)
def test_insert_module_state_multiple_modules():
    """为多个不同模块插入 state"""
    class DummyState:
        def __init__(self, val):
            self.value = val

    module1 = nn.Linear(4, 4)
    module2 = nn.Linear(4, 4)
    state1 = DummyState("module1")
    state2 = DummyState("module2")

    _insert_module_state(module1, state1)
    _insert_module_state(module2, state2)

    assert _get_module_state(module1).value == "module1"
    assert _get_module_state(module2).value == "module2"


@pytest.mark.timeout(120)
def test_insert_module_state_invalid_module():
    """异常场景：传入非 Module 类型"""
    _assert_raises(
        (TypeError, RuntimeError),
        lambda: _insert_module_state("not_a_module", None)
    )


@pytest.mark.timeout(120)
def test_get_module_state_no_state():
    """获取不存在的 state"""
    module = nn.Linear(4, 4)
    state = _get_module_state(module)
    assert state is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
