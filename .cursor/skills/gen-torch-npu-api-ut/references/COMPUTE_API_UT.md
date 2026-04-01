## 计算类 API UT 补充规范

本文是 [SKILL.md](SKILL.md) 的延伸，适用于**以张量数值运算为主体**的计算类 API。

### 适用范围

| 适用（本文生效） | 不适用 |
|-----------------|--------|
| 逐元素运算：`torch.pow`、`torch.add`、`torch.mul`、`torch.sin` | 框架工具：`torch._logging.*`、`torch.amp.autocast` → [FRAMEWORK_API_UT.md](FRAMEWORK_API_UT.md) |
| 规约运算：`torch.sum`、`torch.mean`、`torch.max` | 分布式：`torch.distributed.*` → [DISTRIBUTED_API_UT.md](DISTRIBUTED_API_UT.md) |
| 矩阵/线性代数：`torch.matmul`、`torch.linalg.*` | |
| FFT：`torch.fft.*` | |
| 神经网络算子：`torch.nn.functional.*` | |
| 张量变形：`torch.reshape`、`torch.cat`、`torch.stack`、`torch.transpose` | |
| 张量创建：`torch.zeros`、`torch.ones`、`torch.randn`（带 device 参数） | |
| Tensor 方法：`Tensor.to`、`Tensor.view`、`Tensor.float` | |

**判断方法**：API 的主要职责是**对张量数据做数值变换或构造**？是 → 本文；否 → 按类别查对应文档。

---

## 覆盖维度

计算类 API 用例须以**入参全覆盖**为底线，重点覆盖以下维度：

| 维度 | 说明 | 优先级 |
|------|------|--------|
| **shape** | 0-dim（标量）、1D、2D、高维（≥3D）、空 tensor（size=0）、单元素 | 高 |
| **dtype** | 全部文档声明支持的 dtype（float32/float16/bfloat16/int32/int64/bool/complex 等） | 高 |
| **device** | NPU 主路径（>80%）+ CPU 基线（≤20%） | 高 |
| **参数枚举** | 所有离散参数的每个合法取值（如 `dim`、`keepdim`、`reduction` 等） | 高 |
| **可选参数** | 显式传入 vs 省略默认 | 中 |
| **in-place vs out-of-place** | 若 API 同时有 `tensor.op_()` 变体，需各覆盖 | 中 |
| **out= 参数** | 若支持 `out=` 参数，验证结果写入指定 tensor | 中 |
| **广播** | 不同 shape 的 tensor 间广播行为 | 中 |
| **连续性** | 非连续 tensor（slice/transpose 后的 view）作为输入 | 中 |
| **混合设备输入** | 多 Tensor 输入时 NPU+CPU 混用，验证异常处理 | 高 |
| **异常路径** | 不合法参数（错误 dtype、越界 dim、shape 不兼容等）触发预期异常 | 中 |

---

## 断言规范

| 允许断言 | 禁止断言 |
|----------|----------|
| `result.shape` | 张量数值（`torch.allclose`、逐元素 `assertEqual`） |
| `result.dtype` | 浮点误差容差（`atol`、`rtol`） |
| `result.device.type` | |
| `result.is_contiguous()` | |
| `isinstance(result, torch.Tensor)` | |
| 异常类型与消息（`assertRaises`、`assertRaisesRegex`） | |

---

## 测试模板

```python
# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.{api_name} 接口功能正确性
API 名称：torch.{api_name}
API 签名：{完整签名}

覆盖维度表：
| 覆盖维度         | 说明                              | 覆盖情况                                       |
|------------------|-----------------------------------|------------------------------------------------|
| shape            | 0-dim / 1D / 2D / ND / 空 tensor | 按实际填写：已覆盖 / 未覆盖及原因              |
| dtype            | float32 / float16 / ...           | 按实际填写                                     |
| device           | NPU >80%，CPU 基线 ≤20%           | 按实际填写                                     |
| 参数枚举         | dim / keepdim / ...               | 按实际填写                                     |
| 可选参数         | 显式传入 vs 省略默认              | 按实际填写                                     |
| in-place 变体    | op_() 变体（若有）                | 按实际填写                                     |
| out= 参数        | 结果写入指定 tensor（若有）       | 按实际填写                                     |
| 广播             | 不同 shape 广播                   | 按实际填写                                     |
| 连续性           | 非连续 tensor 输入                | 按实际填写                                     |
| 混合设备输入     | NPU+CPU 混用触发异常              | 按实际填写                                     |
| 异常路径         | 非法参数触发预期异常              | 按实际填写；无稳定异常路径则写未覆盖及原因     |

未覆盖项及原因：
- （按实际填写；若无则写「无」）

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


class Test{ApiClassName}(TestCase):
    """Test cases for torch.{api_name} on NPU."""

    def setUp(self):
        super().setUp()
        self.device_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(self.device_name, 'npu', f"Expected device 'npu', got '{self.device_name}'")
        self.device = torch.device(self.device_name)

    # ------------------------------------------------------------------ #
    # NPU tests  (target: >80% of all test methods)
    # ------------------------------------------------------------------ #

    def test_{api}_npu_basic(self):
        """Basic call on NPU with typical input returns correct shape and dtype."""
        x = torch.randn(4, 4, device=self.device)
        result = torch.{api_name}(x)
        self.assertEqual(result.shape, torch.Size([4, 4]))
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.device.type, self.device_name)

    def test_{api}_npu_dtype_float16(self):
        """float16 input on NPU produces float16 output."""
        x = torch.randn(4, 4, dtype=torch.float16, device=self.device)
        result = torch.{api_name}(x)
        self.assertEqual(result.dtype, torch.float16)

    def test_{api}_npu_scalar(self):
        """0-dim (scalar) tensor input on NPU."""
        x = torch.tensor(3.0, device=self.device)
        result = torch.{api_name}(x)
        self.assertEqual(result.shape, torch.Size([]))

    def test_{api}_npu_empty_tensor(self):
        """Empty tensor (size-0 dim) input on NPU does not raise."""
        x = torch.randn(0, 4, device=self.device)
        result = torch.{api_name}(x)
        self.assertIsInstance(result, torch.Tensor)

    def test_{api}_npu_high_dim(self):
        """High-dimensional input (≥3D) on NPU."""
        x = torch.randn(2, 3, 4, 5, device=self.device)
        result = torch.{api_name}(x)
        self.assertEqual(result.device.type, self.device_name)

    def test_{api}_npu_non_contiguous(self):
        """Non-contiguous tensor input on NPU."""
        x = torch.randn(8, 8, device=self.device).t()  # transpose → non-contiguous
        self.assertFalse(x.is_contiguous())
        result = torch.{api_name}(x)
        self.assertIsInstance(result, torch.Tensor)

    def test_{api}_npu_optional_param_default(self):
        """Call with optional param omitted uses default correctly."""
        x = torch.randn(4, 4, device=self.device)
        result = torch.{api_name}(x)  # no explicit optional params
        self.assertIsInstance(result, torch.Tensor)

    def test_{api}_npu_optional_param_explicit(self):
        """Call with optional param explicitly set."""
        x = torch.randn(4, 4, device=self.device)
        result = torch.{api_name}(x, {param}={value})
        self.assertIsInstance(result, torch.Tensor)

    def test_{api}_npu_mixed_device_input(self):
        """NPU + CPU mixed device inputs should raise RuntimeError."""
        x_npu = torch.randn(4, device=self.device)
        x_cpu = torch.randn(4)
        with self.assertRaises(RuntimeError):
            torch.{api_name}(x_npu, x_cpu)

    def test_{api}_npu_invalid_arg(self):
        """Invalid argument raises expected exception on NPU."""
        with self.assertRaises((ValueError, RuntimeError, TypeError)):
            torch.{api_name}({invalid_input}, device=self.device)

    # ------------------------------------------------------------------ #
    # CPU baseline  (target: ≤20% of all test methods)
    # ------------------------------------------------------------------ #

    def test_{api}_cpu_baseline(self):
        """CPU baseline: call succeeds and returns a Tensor of correct type."""
        x = torch.randn(4, 4)
        result = torch.{api_name}(x)
        self.assertIsInstance(result, torch.Tensor)


if __name__ == "__main__":
    run_tests()
```

---

## 各类计算 API 补充要点

### 带 `dim` 参数的规约运算（`torch.sum`、`torch.mean`、`torch.max` 等）

- 覆盖 `dim=0`、`dim=-1`、`dim=None`（全局规约）。
- 覆盖 `keepdim=True` 与 `keepdim=False`，断言输出 shape 变化正确。
- 越界 `dim` 值须触发 `IndexError` 或 `RuntimeError`。

### 二元运算（`torch.add`、`torch.pow`、`torch.mul` 等）

- 覆盖 **相同 shape** 的两个 tensor、**广播** shape（如 `(4,)` 与 `(3, 4)`）、**标量** 与 tensor 混用。
- 混合设备（NPU tensor + CPU tensor）须触发 `RuntimeError`。
- 若有 in-place 变体（`tensor.add_(...)`），须单独覆盖；断言操作后原 tensor 的 shape/dtype 不变。

### 矩阵运算（`torch.matmul`、`torch.mm`、`torch.bmm` 等）

- 覆盖 2D×2D、batch（3D）、向量×矩阵、矩阵×向量等合法维度组合。
- shape 不兼容（如 `(3, 4)` × `(5, 6)`）须触发 `RuntimeError`。

### 张量变形（`torch.reshape`、`torch.cat`、`torch.stack`、`torch.transpose`）

- 覆盖合法 shape 变换，断言输出 shape。
- 非法 shape（元素总数不匹配）须触发 `RuntimeError`。
- `torch.cat`：覆盖沿不同 `dim` 拼接；不同 device 的 tensor 列表须触发异常。

### 支持 `out=` 参数的 API

```python
def test_{api}_npu_out_param(self):
    """Result is written into the pre-allocated out tensor."""
    x = torch.randn(4, 4, device=self.device)
    out = torch.empty(4, 4, device=self.device)
    result = torch.{api_name}(x, out=out)
    # out tensor should be the same object as result.
    self.assertIs(result, out)
    self.assertEqual(out.shape, torch.Size([4, 4]))
```

### dtype 扩展覆盖

对于 dtype 支持范围较宽的 API，可用循环写法：

```python
def test_{api}_npu_supported_dtypes(self):
    """All documented dtypes produce output of matching dtype on NPU."""
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        x = torch.randn(4, 4, dtype=dtype, device=self.device)
        result = torch.{api_name}(x)
        self.assertEqual(result.dtype, dtype, f"dtype mismatch for {dtype}")
```

---

## 自检清单（计算类补充项）

以下条目需在主技能 [SKILL.md](SKILL.md) 自检基础上额外确认：

- [ ] 已确认 API 属于计算类（主职责为张量数值变换），本文规范生效
- [ ] shape 维度：已覆盖 0-dim / 1D / 2D / 高维 / 空 tensor
- [ ] dtype：已覆盖文档声明支持的全部 dtype，或在覆盖维度表中注明未覆盖原因
- [ ] 多 Tensor 输入：已添加 NPU+CPU 混合设备场景（应触发 `RuntimeError`）
- [ ] 有 in-place 变体的 API：已单独覆盖 `op_()` 方法
- [ ] 有 `out=` 参数的 API：已验证结果写入指定 tensor
- [ ] 有 `dim` 参数的 API：已覆盖正向 dim、负向 dim、None（全局）及越界 dim
- [ ] NPU 用例数 >80%，CPU 基线 ≤20%
- [ ] **无数值精度断言**；仅断言 shape / dtype / device / `is_contiguous` / 异常
- [ ] 无 pytest 全系 API
- [ ] 文件头部中文 docstring 与覆盖维度表已按实际用例填写
