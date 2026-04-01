---
name: gen-torch-npu-api-ut
description: >-
  为 torch_npu（PTA）场景生成与 ascend_pytorch/test 风格一致的 API 功能 UT（unittest + TestCase，禁止 pytest）。
  当用户要求生成 torch_npu API 功能用例、PTA API UT、NPU 适配层 torch API 测试时触发。
  Use when generating torch_npu functional unit tests, PTA API tests, or NPU-patched torch API UT.
---

# torch_npu（PTA）API 功能 UT 生成

## 触发条件

用户提出以下类似请求时激活本技能：

- 生成 **torch_npu / PTA** 的 **torch API** 功能用例、功能 UT
- 为某 **torch.xxx** 写 NPU 侧功能测试（读 `transfer_to_npu` 适配）
- 「按 ascend 风格」生成 **unittest** 的 API 测试文件

## 风格约定（与 ascend_pytorch/test 对齐）

**必须**以 `ascend_pytorch/test/` 下用例为范本（如 `test/npu/test_torch_npu.py`、`test/optim/test_optim.py`），保持统一：

- 使用 **`unittest`**：**禁止** `pytest` 全系 API（含 `pytest.mark`、`pytest.raises`、`pytest.skip`、`pytest.fixture`、`pytest.parametrize` 等）。
- 测试类**继承 `TestCase`**；测试方法名为 `test_*`。
- 优先使用 `from torch_npu.testing.testcase import TestCase, run_tests`；若需与 ascend 部分文件一致，可对照同目录是否使用 `torch.testing._internal.common_utils.TestCase`（以**目标目录最近邻**用例为准）。
- **设备检查统一放在 `setUp` 中**：使用 `torch._C._get_privateuse1_backend_name()` 检查设备类型，如果不是 `'npu'` 直接报错（`self.assertEqual` 或 `raise AssertionError`），**不要 skip**。示例：
  ```python
  def setUp(self):
      super().setUp()
      device_name = torch._C._get_privateuse1_backend_name()
      self.assertEqual(device_name, 'npu', f"Expected device 'npu', got '{device_name}'")
  ```
- 多卡用例：对 **≥2 块 NPU** 的测试方法使用 **`@skipIfUnsupportMultiNPU(n)`**（`from torch_npu.testing.common_distributed import skipIfUnsupportMultiNPU`），与 ascend 一致。分布式类 API 的多卡策略详见下文「专项文档路由」。
- 文件末尾：`if __name__ == "__main__": run_tests()`（无 `torch_npu` 时回退 `unittest.main`，与 `gen-distributed-ut` 技能中模式一致）。
- 断言：`self.assert*` / `self.assertRaises` / `self.assertRaisesRegex`；**禁止**以逐元素浮点对比做**数值精度**验收（见下文「禁止事项」）。
- **注释语言**：除文件顶部按模板编写的**头部注释**（简体中文 docstring / 覆盖维度表等）外，**代码中**其余注释（行内注释、`#` 说明、测试方法 docstring 若需编写）**统一使用英文**；避免正文代码块中英混用。

## 前置准备（强制执行顺序）

1. **确认目标 API 与类别**：向用户同时确认：
   - 要测试的 **torch API 全名**（如 `torch.linalg.vector_norm`）
   - 该 API 所属**类别**（必须由用户明确指定）：

   | 类别 | 说明 | 典型示例 |
   |------|------|----------|
   | **计算类** | 主职责为张量数值运算/变换 | `torch.pow`、`torch.matmul`、`torch.nn.functional.relu`、`torch.fft.fft` |
   | **框架类** | 主职责为框架状态管理/工具/行为控制 | `torch._logging.warning_once`、`torch.amp.autocast`、`torch.autograd.grad`、`torch.jit.script` |
   | **分布式类** | 属于 `torch.distributed` 命名空间 | `torch.distributed.all_reduce`、`torch.distributed.init_process_group` |

   若用户未提供类别，**必须主动询问**，不得自行猜测后直接生成。

2. **查阅 API 签名**：在 `pytorch/torch/` 下定位实现，读取**完整函数/方法签名**、参数列表、默认值与 docstring，列出全部参数。
3. **查阅 NPU 适配层**：打开 `ascend_pytorch/torch_npu/contrib/transfer_to_npu.py`，确认该 API 是否被 patch、映射关系（如 `cuda→npu`、`nccl→hccl`）及特殊行为。
4. **查阅现有测试**：优先读 `ascend_pytorch/test/` 中同类或同模块测试；不足时对照 `pytorch/test/`。

## 文件规范

### 输出路径与命名

```
test/{api_full_name_underscored}/test_{api_full_name_underscored}.py
```

- **`api_full_name_underscored`**：API 在 `torch` 命名空间下的路径，**去掉第一层包名 `torch.`** 后，将剩余段中的 `.` 全部替换为 `_`，注意保留大小写。
  - 例：`torch.tensor` → `tensor`
  - 例：`torch.nn.functional.relu` → `nn_functional_relu`
  - 例：`torch.linalg.vector_norm` → `linalg_vector_norm`
  - 例：`torch.Tensor.to` → `Tensor_to`
- 目录名与文件名**仅使用下划线**，**禁止点号**；**禁止**在文件名中使用 `.`。
- **每个 API 只生成一个**测试文件。`test/` 指**工作区根目录**下的 `test/`。
- 若 API 为 `Tensor` 等方法，路径命名以与用户确认的「逻辑 API 名」为准（如 `Tensor_add`），仍遵循无点号、下划线规则。

### 禁止修改范围

**仅允许**创建或修改 **`test/`** 目录下文件；**不得**改动 `pytorch/`、`ascend_pytorch/` 内任何源码。

## 文件头部注释（简体中文模板）

将下列模板置于文件最顶端 docstring，`{api_name}` 等为占位符，须替换为实际内容；**覆盖维度表**按真实用例填写，不得留空占位。

```python
# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.{api_name} 接口功能正确性
API 名称：torch.{api_name}
API 签名：{完整签名}

覆盖维度表：
| 覆盖维度         | 说明                                                         | 覆盖情况                                       |
|------------------|--------------------------------------------------------------|------------------------------------------------|
| 空/非空          |                          | 按 API 实际填写：已覆盖 / 未覆盖及原因          |
| 枚举选项         |                           | 按 API 实际填写                                |
| 参数类型         |           | 按 API 实际填写                                |
| 传参与不传参     |               | 按 API 实际填写                                |
| 等价类/边界值    |                   | 按 API 实际填写                                |
| 正常传参场景     |                         | 按 API 实际填写                                |
| 异常传参场景     |   | 按 API 实际填写；无稳定异常路径则写未覆盖及原因 |

未覆盖项及原因：
- （与上表「未覆盖」呼应，逐条说明；若无则写「无」或删除本列表项）
- ...（根据实际 API 补充）

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype/类型符合预期），
     不做精度和数值正确性校验。
"""
```

## 设备与 NPU/CPU 占比

- **设备类型字符串**：使用 `torch._C._get_privateuse1_backend_name()` 获取（加载 `torch_npu` 并完成注册后通常为 `"npu"`）。构造 `torch.device` 时与该字符串一致，避免写死 `"cuda"`（除非对照 ascend 中明确需要双端分支的范式）。
- **默认假设 NPU 可用**；用例主体在 NPU 上执行。
- 若 API **同时支持 CPU**：在**整套测试方法**中，**NPU 上执行的用例数量应 >80%**，**CPU 用例 ≤20%**（仅保留必要基线，如 dtype/shape 或文档声明的 CPU 路径）。

## 用例设计（接口功能一致性）

对「通过构造用例验证接口行为」类 API，以**入参全覆盖**为底线，在 docstring 表格中可追溯，包括但不限于：

- 某参数 **空 / 非空**（`None` 与合法非空值，按语义）
- 某参数的 **全部合法枚举或离散选项**
- 某参数 **声明支持的类型**（如 `int`、`list`、`Tensor`）
- **可选参数**：显式传入 vs 省略默认
- **等价类、边界值**（空 tensor、单元素、典型 shape 等）
- **正常路径**与**可稳定断言的异常路径**（无稳定异常则于表中注明未覆盖）
- **混合设备输入场景**：对于涉及多 Tensor 输入的 API，需补充 **NPU/CPU 混合设备输入**场景的测试，验证 API 对异构设备输入的处理行为（如是否抛出异常、是否正确处理等）

**禁止**只写一条 happy path。

## 禁止事项

- **禁止 pytest 全系 API**。
- **禁止**滥用 `unittest.expectedFailure`；非有明确跟踪缺陷勿用。
- **禁止**以容差对比浮点/整 tensor **数值正确性**；允许 `shape`、`dtype`、`device`、返回类型、`is_contiguous` 等**结构与类型**断言。

## 执行、修复与报告（NPU 环境）

在**具备 NPU** 的运行环境中，生成 UT 后**询问用户是否执行**。

- 若执行失败：定位原因、**仅修改 `test/` 下文件**修复后**重新执行**，直至通过或明确环境/上游限制并在 `UT_REPORT.md` 中记录。
- 每次完成「**生成 UT → 本机执行完毕**」后，**必须**落盘一份 **Markdown 报告**，不得仅口头汇报。

| 场景 | 路径 |
|------|------|
| 仅本次单个 API 的 UT | `test/{api_full_name_underscored}/UT_REPORT.md` |
| 本次跑完整个 `test/` 或需总表 | `test/UT_EXECUTION_REPORT.md`（可覆盖为最近一次全量结果） |

报告必须包含以下内容：
- **执行命令**：完整的测试执行命令
- **环境摘要**：Python 版本、PyTorch 版本、torch_npu 版本、NPU 设备信息、CANN 版本
- **测试结果表**：列出所有测试方法及其结果（PASS/FAIL/SKIP）
- **统计信息**：通过数、跳过数、失败数
- **跳过用例分析表**：列出所有被跳过的用例，包含跳过条件、跳过原因、合理性评估
- **失败栈摘要**：如有失败用例，提供关键错误信息
- **本次改动文件列表**

**报告格式示例**：
```markdown
## 测试结果
| 测试方法 | 结果 | 说明 |
|----------|------|------|
| test_xxx | PASS | 测试说明 |
| test_yyy | SKIP | 跳过原因说明 |

## 统计
- 通过: X
- 跳过: Y
- 失败: Z

## 跳过用例分析
| 测试方法 | 跳过条件 | 跳过原因 | 合理性评估 |
|----------|----------|----------|------------|
| test_xxx | device_count < 2 | 需要多卡环境 | 合理，使用 @skipIfUnsupportMultiNPU(2) |
```

## 专项文档路由

上述条款为公共基线。**在完成前置准备步骤 1（确认类别）后，立即按下表打开对应专项文档**，专项文档的规则优先于本文同名条款。

| 用户指定类别 | 必读专项文档 | 核心要求摘要 |
|-------------|-------------|-------------|
| **计算类** | [**COMPUTE_API_UT.md**](COMPUTE_API_UT.md) | shape/dtype/device 全覆盖；NPU >80%；混合设备输入；in-place/out= 变体 |
| **框架类** | [**FRAMEWORK_API_UT.md**](FRAMEWORK_API_UT.md) | 分 A（纯工具）/ B（硬件感知）两类；全局状态隔离；日志类禁用 caplog |
| **分布式类** | [**DISTRIBUTED_API_UT.md**](DISTRIBUTED_API_UT.md) | 除纯工具类外默认多卡 HCCL；配合 `@skipIfUnsupportMultiNPU` |

> **注意**：若用户在步骤 1 中未指定类别，**停止生成，先向用户询问类别**，确认后再继续。
