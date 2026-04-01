## 框架类 API UT 补充规范

本文是 [SKILL.md](SKILL.md) 的延伸，适用于**非计算类、非分布式**的框架工具 API。

### 适用范围

| 适用（本文生效） | 不适用（回归主技能 SKILL.md） |
|-----------------|-------------------------------|
| `torch._logging.warning_once` | `torch.pow`、`torch.add` 等张量运算 |
| `torch.amp.autocast`（上下文管理器） | `torch.nn.functional.relu` 等神经网络算子 |
| `torch.autograd.grad`（梯度工具） | `torch.linalg.*`、`torch.fft.*` 等数学运算 |
| `torch.jit.is_scripting`、`torch.jit.script` | `torch.Tensor.to`、`torch.Tensor.view` 等 Tensor 方法 |
| `torch.fx.symbolic_trace` | |
| `torch.profiler.profile` | |
| `torch._C.*`、`torch.backends.*` 等底层状态/注册 API | |

**判断方法**：API 的主要职责是**处理张量数值**还是**管理框架状态/行为**？前者走主技能，后者走本文。

---

## API 分类与测试策略

框架类 API 分为两类：

| 类别 | 典型示例 | NPU 依赖 | 测试重点 |
|------|----------|----------|----------|
| **A. 纯 Python 工具** | `torch._logging.warning_once`、`torch.jit.is_scripting`、`torch.fx.symbolic_trace` | 无（结果不依赖硬件） | 返回值类型/内容、副作用（如日志输出、注册状态）、异常路径 |
| **B. 硬件感知工具** | `torch.amp.autocast`、`torch.autograd.grad`、`torch.profiler.profile`、`torch.set_grad_enabled` | 中等依赖 | 上下文管理器进出状态、NPU 上框架行为正确性、NPU vs CPU 结构一致性 |

**决策**：

```
API 是否需要张量在设备上执行 / 感知设备类型才能发挥作用？
    ├─ 否 → 类别 A（纯 Python 工具）
    └─ 是 → 类别 B（硬件感知工具）
```

---

## 测试模板

### 模板 A：纯 Python 工具类

适用于日志、JIT 查询、FX 图构造、注册机制等无硬件依赖的工具函数。

```python
# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.{api_name} 接口功能正确性
API 名称：torch.{api_name}
API 签名：{完整签名}

覆盖维度表：
| 覆盖维度         | 说明                                | 覆盖情况    |
|------------------|-------------------------------------|-------------|
| 基础调用         | 正常参数调用不报错、返回值类型正确  | 已覆盖      |
| 参数枚举/边界值  | 全部合法入参组合                    | 已覆盖      |
| 副作用验证       | 日志输出 / 状态变更等可观测副作用   | 按 API 填写 |
| 幂等性           | 多次调用结果一致（若有缓存机制）    | 按 API 填写 |
| 异常路径         | 非法参数触发预期异常                | 已覆盖      |

未覆盖项及原因：
- （按实际填写；若无则写「无」）

注意：本测试仅验证功能正确性（调用不报错、返回类型/副作用符合预期），
     不做精度和数值正确性校验。
"""
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


class Test{ApiClassName}(TestCase):
    """Test cases for torch.{api_name}."""

    def setUp(self):
        super().setUp()
        # Even for CPU-only APIs, verify the NPU backend is registered.
        device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")

    def test_{method}_basic(self):
        """Normal call returns expected type without error."""
        result = torch.{api_name}(...)
        self.assertIsInstance(result, {ExpectedType})

    def test_{method}_return_value(self):
        """Return value matches documented contract."""
        result = torch.{api_name}(...)
        self.assertEqual(result, {expected})

    def test_{method}_invalid_arg(self):
        """Invalid argument raises expected exception."""
        with self.assertRaises({ExpectedException}):
            torch.{api_name}({invalid_arg})


if __name__ == "__main__":
    run_tests()
```

---

### 模板 B：硬件感知工具类

适用于上下文管理器（autocast、no_grad）、autograd 工具、profiler 等需要感知设备的框架 API。

```python
# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.{api_name} 接口功能正确性
...（头部注释同主技能规范）
"""
import torch
import torch_npu  # noqa: F401

try:
    from torch_npu.testing.testcase import TestCase, run_tests
except ImportError:
    import sys
    from unittest import TestCase

    def run_tests():
        import unittest
        unittest.main(argv=sys.argv)


class Test{ApiClassName}(TestCase):
    """Test cases for torch.{api_name} with NPU awareness."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")
        self.device = torch.device(self.device_name)

    def tearDown(self):
        # Restore any global state modified during tests.
        super().tearDown()

    def test_{method}_npu(self):
        """Verify framework behavior on NPU."""
        # ... test body ...
        pass

    def test_{method}_state_change(self):
        """Verify observable state change (e.g., flag toggled by context manager)."""
        # e.g., assert torch.is_grad_enabled() changes inside/outside context.
        pass

    def test_{method}_npu_vs_cpu_structure(self):
        """NPU and CPU paths produce structurally identical results (shape/dtype only)."""
        # Assert shape, dtype, device — never numeric values.
        pass

    def test_{method}_invalid_arg(self):
        """Invalid argument raises expected exception."""
        with self.assertRaises((ValueError, RuntimeError)):
            torch.{api_name}({invalid_arg})


if __name__ == "__main__":
    run_tests()
```

---

## 各类 API 专项要点

### `torch._logging` / 日志类

- **捕获日志输出**：使用 `unittest.mock.patch` 或向 logger 添加临时 `logging.Handler`；**禁止** `caplog`（pytest 专属）。
- **缓存隔离**：`warning_once` 内部有去重缓存；在 `setUp` 中调用 `cache_clear()`（若有），或直接 patch 内部集合，避免用例间污染。
- **断言**：`self.assertIn(msg, records)`、`self.assertEqual(len(records), 1)` 等；非数值断言。

```python
import io
import logging
import unittest.mock

def setUp(self):
    super().setUp()
    device_name = torch._C._get_privateuse1_backend_name()
    self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")
    # Reset warning_once dedup cache before each test.
    if hasattr(torch._logging.warning_once, 'cache_clear'):
        torch._logging.warning_once.cache_clear()

def test_warning_once_emits_once(self):
    """Duplicate calls with the same message should emit only one log record."""
    logger = logging.getLogger('torch')
    with unittest.mock.patch.object(logger, 'warning') as mock_warn:
        torch._logging.warning_once(logger, "test message %s", "arg")
        torch._logging.warning_once(logger, "test message %s", "arg")
    self.assertEqual(mock_warn.call_count, 1)
```

### `torch.amp` / `torch.autocast`

- **禁止硬编码 `'cuda'`**：使用 `torch._C._get_privateuse1_backend_name()` 获取 device_type。
- **状态断言**：验证 `torch.is_autocast_enabled()` 在上下文内外的切换。
- **dtype 断言**：验证上下文内张量 `.dtype` 的转换，不做数值对比。

### `torch.autograd` 工具

- `grad` / `backward`：只断言梯度的 `shape`、`dtype`、`device`，**不断言数值**。
- `torch.autograd.Function` 子类：验证 `forward` 输出的 shape/dtype；`backward` 不报错即可。
- `torch.no_grad()` / `torch.set_grad_enabled()`：断言 `torch.is_grad_enabled()` 状态切换；务必在 `tearDown` 中还原。

### `torch.jit`

- `torch.jit.script` / `torch.jit.trace`：只验证脚本化后的可调用性与输出 shape/dtype。
- `torch.jit.is_scripting()`：在脚本和非脚本两条路径下分别断言返回值（`True` / `False`）。

### `torch.fx`

- `torch.fx.symbolic_trace`：验证返回 `GraphModule`、`graph.nodes` 非空、`str(graph)` 不报错。
- 不对具体节点名或算子顺序做硬编码断言（实现细节易变）。

### `torch.profiler`

- 以简单 NPU 算子（如 `torch.randn(..., device=npu).sum()`）作为被 profiling 的操作。
- 断言 `profiler.key_averages()` 不为空、事件总数 > 0。
- **不**对具体 kernel 名称做断言（NPU kernel 名与 CUDA 不同，且版本间可变）。

---

## 全局状态与用例隔离

框架类 API 常操作共享全局状态，须格外注意：

1. **`setUp` / `tearDown` 配对还原**：修改了全局标志（`torch.set_grad_enabled`、日志级别、后端注册）的测试方法，须在 `tearDown` 中还原。
2. **缓存类 API**：在 `setUp` 中调用 `cache_clear()`（`lru_cache`）或重置内部字典/集合。
3. **上下文管理器优先**：能用 `with` 块的场景优先使用，避免手动恢复状态。
4. **`unittest.mock.patch` 作用域**：使用 `with` 语句或 `@patch` 装饰器限定 patch 范围，确保每个测试方法结束后自动还原。

---

## 自检清单（框架类补充项）

以下条目需在主技能 [SKILL.md](SKILL.md) 自检基础上额外确认：

- [ ] **本文适用性确认**：目标 API 是框架工具类（非计算类），本文规范生效；计算类 API（如 `torch.pow`）回归主技能 SKILL.md
- [ ] 已按决策流程将 API 归入 **A（纯 Python 工具）/ B（硬件感知工具）**，并选用对应模板
- [ ] 类别 A：`setUp` 中保留设备名检查（`self.assertEqual(device_name, 'npu')`）
- [ ] 类别 B：`tearDown` 中已还原所有全局状态修改
- [ ] 类别 B：NPU vs CPU 结构对比只断言 shape / dtype，**不断言数值**
- [ ] 日志类：已用 `unittest.mock` 捕获输出；**无** `caplog`
- [ ] 缓存类：`setUp` 中已调用 `cache_clear()` 或等效重置，避免用例间污染
- [ ] 全局状态修改：`setUp`/`tearDown` 已成对还原，或使用 `with`/`@patch` 自动还原
- [ ] **无** pytest 全系 API（`pytest.mark`、`caplog`、`monkeypatch`、`pytest.fixture` 等）
- [ ] 无数值精度断言；仅断言 shape / dtype / device / 返回类型 / 副作用 / 异常
- [ ] 文件头部中文 docstring 与覆盖维度表已按实际用例填写
