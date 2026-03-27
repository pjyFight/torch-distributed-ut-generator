---
name: gen-distributed-ut
description: 生成 PyTorch 分布式 API 的功能测试 UT 文件，同时兼容 GPU(NCCL) 和 NPU(HCCL) 环境。当用户要求生成 torch 分布式 API 功能测试文件、distributed UT、集合通信测试用例时触发。Use when generating distributed API test, collective communication test, or torch.distributed UT files.
---

# PyTorch 分布式 API 功能测试 UT 生成

## 触发条件

用户提出以下类似请求时激活本技能：
- "生成 torch 分布式 api 功能测试文件"
- "生成 all_reduce / broadcast / ... 的 UT"
- "生成 distributed 集合通信测试用例"

## 前置准备

1. **确认目标 API**：向用户确认要测试的 `torch.distributed` API 名称（如 `all_reduce`、`broadcast`、`all_gather` 等）。
2. **查阅 API 签名**：在 `pytorch/torch/distributed/distributed_c10d.py` 中读取目标 API 的完整函数签名、参数列表及文档字符串，提取所有参数及其默认值。
3. **查阅 NPU 适配层**：在 `ascend_pytorch/torch_npu/contrib/transfer_to_npu.py` 中确认该 API 是否被 patch，以及 patch 的具体行为（如 `nccl→hccl`、`cuda→npu`）。
4. **查阅现有测试**：分别在 `ascend_pytorch/test/distributed/` 和 `pytorch/test/distributed/` 中查找已有测试作为参考。

## 文件规范

### 输出路径

```
test/{api_full_name_underscored}/test_{api_full_name_underscored}.py
```

其中：
- `api_full_name_underscored` 指 API 全称（含模块路径）中将 `.` 替换为 `_` 后的结果  
  例如：`torch.distributed.utils._to_kwargs` → `torch_distributed_utils__to_kwargs`
- **目录名与文件名统一使用下划线，不使用点号**
- 每个 API 只生成一个测试文件。`test/` 是工作区根目录下的 test 目录。
- **禁止在测试文件名中使用 `.`**（会导致 `pytest` 以模块导入时解析失败）。

### 禁止修改范围

仅允许创建/修改 `test/` 目录下的文件，不得改动 `pytorch/` 或 `ascend_pytorch/` 的任何源码。

## 文件头部注释模板（简体中文）

```python
# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.{api_name} 接口在多进程分布式场景下的功能正确性
API 名称：torch.distributed.{api_name}
API 签名：{完整签名}

覆盖维度：
+------------------+----------------------------------------+
| 维度             | 覆盖值                                 |
+------------------+----------------------------------------+
| tensor dtype     | float32, bfloat16                      |
| async_op         | True, False                            |
| op (ReduceOp)    | SUM, AVG, MAX, MIN, ...                |
| group            | 默认组, 自定义子组                     |
| src/dst          | 0, 非0 rank                            |
| ...              | ...                                    |
+------------------+----------------------------------------+

未覆盖项及原因：
- float16: NPU HCCL 对 float16 集合通信支持有限，易产生精度溢出干扰功能验证
- int8/uint8: 部分集合操作不支持该类型
- ...（根据实际 API 补充）

注意：本测试仅验证功能正确性（调用不报错、输出 shape/dtype 符合预期），
     不做精度和数值正确性校验。
"""
```

## GPU / NPU 双环境兼容

### 环境检测与条件导入

文件顶部统一使用以下模式：

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest

# 环境检测
_IS_NPU = hasattr(torch, 'npu') and torch.npu.is_available()
_IS_CUDA = not _IS_NPU and torch.cuda.is_available()

if _IS_NPU:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu  # noqa: F401  自动 patch cuda→npu, nccl→hccl

DEVICE_TYPE = "npu" if _IS_NPU else ("cuda" if _IS_CUDA else "cpu")
BACKEND = "hccl" if _IS_NPU else ("nccl" if _IS_CUDA else "gloo")
```

### transfer_to_npu 未覆盖的 patch

`transfer_to_npu` 会自动将大部分 `cuda/nccl` 调用映射为 `npu/hccl`，但以下场景需要手动分支处理：

- `torch.npu.set_device(rank)` vs `torch.cuda.set_device(rank)`——transfer_to_npu 已 patch `torch.Tensor.cuda` 等，但在 spawn 子进程入口处建议显式写分支以确保 NPU 优先可用
- 如果 API 调用中需要显式指定 `device=torch.device("cuda", rank)`，改为 `torch.device(DEVICE_TYPE, rank)`
- 特定于 NPU 的环境变量（如 `HCCL_WHITELIST_DISABLE`）需在 NPU 分支中设置

```python
def _setup_device(rank):
    if _IS_NPU:
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        torch.npu.set_device(rank)
    elif _IS_CUDA:
        torch.cuda.set_device(rank)
```

### 首要保证 NPU 可执行

- 默认以 NPU 环境为主进行编写和测试
- GPU 路径作为兼容支持
- 若某个参数组合在 NPU 下不支持但 GPU 下支持，**不要**使用 `pytest.skip` 跳过；保留用例并让测试显式暴露真实失败/异常
- **NPU 用例占比要求**：在 NPU 可用环境下，实际执行的 NPU 路径用例占比应 **>= 70%**；CPU 用例仅保留少量基线校验（建议 10%~30%）。

## 测试架构

### 多进程模型（spawn）

所有分布式测试使用 `mp.spawn` 拉起，默认 **2 卡**（`world_size=2`）。仅在 API 语义确实需要时才使用 4/8 卡。

```python
WORLD_SIZE = 2

def _worker(rank, world_size, test_fn, *args):
    """每个 rank 的入口函数"""
    _setup_device(rank)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = _get_free_port()
    dist.init_process_group(backend=BACKEND, rank=rank, world_size=world_size)
    try:
        test_fn(rank, world_size, *args)
    finally:
        dist.destroy_process_group()

def _run_test(test_fn, world_size=WORLD_SIZE, *args):
    mp.spawn(_worker, args=(world_size, test_fn, *args), nprocs=world_size, join=True)
```

### 端口管理

使用动态端口避免冲突：

```python
import socket

def _get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return str(s.getsockname()[1])
```

**注意**：端口需要在 spawn 之前获取并通过环境变量或参数传递给子进程，确保所有 rank 使用同一端口。推荐在主进程中设置 `os.environ['MASTER_PORT']`，子进程继承。

### 超时控制

为每个 spawn 测试设置合理超时（默认 120 秒），避免死锁挂起：

```python
@pytest.mark.timeout(120)
def test_xxx():
    ...
```

需安装 `pytest-timeout`（`pip install pytest-timeout`）；若环境无该插件，可省略 `@pytest.mark.timeout` 或改用其他超时机制。

## 用例设计规范

### 接口功能一致性基础标准（强制）

对于“通过构造用例验证接口功能一致性”的 API，必须以**入参全覆盖**为基础，覆盖包括但不限于：

- **空/非空覆盖**：每个可空参数都要覆盖 `None` 与非 `None`（有效值）两类。
- **枚举选项全覆盖**：对离散选项参数（如 `"tanh"`、`"gaussian"`、`ReduceOp` 等）覆盖所有合法选项。
- **类型全覆盖**：对支持多类型的参数（如 `int/list/tuple/Tensor/...`）覆盖所有被接口声明支持的类型。
- **可选参数传/不传**：每个可选参数都要有“显式传参”和“不传参走默认值”两类用例。
- **正常/异常双场景**：同时覆盖合法输入（不报错、返回结构正确）与非法输入（明确断言异常类型）。

若存在“组合参数空间过大”的情况，必须基于以下方法裁剪但保持代表性覆盖：

- **等价类划分**：至少覆盖每个等价类 1 个代表值（含有效等价类与无效等价类）。
- **边界值分析**：覆盖边界点、边界内一点、边界外一点（如空张量、单元素、极小/极大 shape、最小/最大 rank）。
- **约束组合优先**：优先覆盖“最容易触发行为差异/错误”的参数组合（例如 dtype × device × async_op）。

禁止仅覆盖“主路径 happy path”；若参数具备明确定义域，必须在用例中体现定义域覆盖证据。

### 参数覆盖矩阵

针对每个 API，必须覆盖：

| 覆盖维度 | 说明 |
|---------|------|
| **参数传/不传** | 有默认值的参数分别测试传入和不传入 |
| **None / 非None** | 可选参数分别传 None 和有效值 |
| **Tensor dtype** | 至少覆盖 `torch.float32` 和 `torch.bfloat16` |
| **async_op** | True / False（若 API 支持） |
| **ReduceOp** | SUM / AVG / MAX / MIN 等（若 API 支持） |
| **group** | 默认组（WORLD）和自定义子组（`dist.new_group`） |
| **src / dst** | rank 0 和非 0 rank（若 API 支持） |
| **tensor shape** | 至少包含 1D、2D 场景，含空 tensor（`torch.empty(0)`） |

### 使用 pytest.parametrize

用 `@pytest.mark.parametrize` 组合参数维度，减少重复代码：

```python
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("async_op", [True, False])
def test_all_reduce_basic(dtype, async_op):
    def _test_fn(rank, world_size):
        tensor = torch.ones(4, 4, dtype=dtype, device=DEVICE_TYPE)
        work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=async_op)
        if async_op:
            work.wait()
        assert tensor.shape == (4, 4)
        assert tensor.dtype == dtype
    _run_test(_test_fn)
```

### CUDA / NPU 互斥设备：合并为单用例

若 API 入参为设备类型字符串（如 `"cuda"` / `"npu"`），且**同一环境通常只具备其中一种加速器**，不要拆成两个用例并分别 `pytest.skip`：

- **禁止**：`test_xxx_cuda()` 在 NPU 机上 skip、`test_xxx_npu()` 在 GPU 机上 skip，导致全量始终有一条无意义 skip。
- **推荐**：合并为一个用例，按当前环境只执行对应分支；仅当 CUDA 与 NPU 均不可用时再 `skip`。

环境检测与项目内约定保持一致（NPU 优先时 `_IS_CUDA` 与 `_IS_NPU` 互斥）：

```python
_IS_NPU = hasattr(torch, "npu") and torch.npu.is_available()
_IS_CUDA = not _IS_NPU and torch.cuda.is_available()

def test_xxx_supported_device():
    """按当前可用设备验证（CUDA 或 NPU，二选一）"""
    if _IS_CUDA:
        assert _do_thing("cuda")  # 或 _get_device_handle("cuda") 等
    elif _IS_NPU:
        assert _do_thing("npu")
    else:
        pytest.skip("No CUDA or NPU device available")
```

若与「返回类型 / 非空」类断言重复，只保留**一个**合并用例，勿再写第二个仅换设备分支的重复测试。

### NPU 优先用例配比（新增强制规则）

- 优先将主路径测试的 `target_device/device` 设为 `_primary_device()`（NPU 环境即 NPU）。
- 仅保留少量 CPU baseline 用例（如空输入、dtype 保持），避免 CPU 用例占比过高。
- 提交前统计并确保：`NPU 用例数 / 总用例数 >= 70%`（在 NPU 可用环境中）。

### 正常场景验证要点

- 调用不抛异常
- 返回值类型正确（同步返回 None，异步返回 Work 对象）
- 输出 tensor 的 shape、dtype 与预期一致
- Work 对象的 `wait()` 不抛异常、`is_completed()` 最终为 True
- **不做数值正确性校验**

### 异常场景（断言方式）

**原则**：异常类型必须被明确断言；在 **主进程、单进程** 测试中可使用 `pytest.raises`。

**`mp.spawn` 子进程内禁止使用 `pytest.raises` 上下文管理器**：子进程不是 pytest 执行上下文，`pytest.raises` 无法正确与父进程通信。应在 worker 内使用 `try/except` 或封装辅助函数，未捕获到预期异常时主动 `raise AssertionError`。

```python
def _assert_raises(exc_types, fn):
    try:
        fn()
    except exc_types:
        return
    raise AssertionError(f"expected one of {exc_types}, callable={fn!r}")

def test_all_reduce_invalid_tensor():
    def _test_fn(rank, world_size):
        _assert_raises(
            (TypeError, RuntimeError, AttributeError),
            lambda: dist.all_reduce("not_a_tensor"),
        )
    _run_test(_test_fn)
```

**与「必须使用 pytest.raises」规范的对应关系**：分布式 spawn 场景下，`_assert_raises` / `try-except` 与 `pytest.raises` **语义等价**（均显式校验异常类型）；仅在 **未使用 spawn、在主进程执行** 的异常用例（例如未 `init_process_group` 即调用 API）中，直接使用 `pytest.raises`。

常见异常场景：
- 传入非 Tensor 类型
- tensor 在错误 device 上（如 CPU tensor 用 NCCL/HCCL 后端）
- 各 rank 的 tensor shape 不一致（对要求一致的 API）
- 非法 src/dst rank（超出 world_size）
- 进程组未初始化就调用
- 传入不支持的 ReduceOp
- 参数类型不匹配（如期望 int 却传 list/str）
- 枚举值非法（如 mode 传未定义字符串）
- 越界值/边界外值（如负维度、非法索引、超范围 rank）

### 禁止事项

- **禁止** `pytest.xfail`
- **禁止** 数值精度校验（不对比具体数值结果）
- **禁止** 修改 `test/` 目录以外的文件

### 仅允许 pytest.skip 的场景

只有以下情况才允许使用 `pytest.skip`，且必须写清楚原因：

```python
if not _IS_NPU and not _IS_CUDA:
    pytest.skip("无可用 GPU/NPU 设备，跳过分布式测试")
```

**明确禁止**：
- 禁止因为“不支持 bfloat16 / bf16”而使用 `pytest.skip`
- 禁止任何 `if not supports_bfloat16: pytest.skip(...)` 形式

## 用例命名规范

```
test_{api_name}_basic                       # 基础功能，默认参数
test_{api_name}_dtype_{dtype}               # 不同数据类型
test_{api_name}_async                       # 异步操作
test_{api_name}_custom_group                # 自定义进程组
test_{api_name}_src_rank_{n}                # 不同源 rank
test_{api_name}_empty_tensor                # 空 tensor
test_{api_name}_invalid_{scenario}          # 异常场景
test_{api_name}_multi_op                    # 不同 ReduceOp
```

使用 parametrize 时，函数名可简化，参数自动生成用例 ID。

## 执行与覆盖率

### 执行测试

如果当前环境支持运行（有 GPU/NPU 设备），执行生成的 UT 文件：

```bash
cd test/{api_full_name_underscored}
python -m pytest test_{api_full_name_underscored}.py -v --tb=short 2>&1 | tee test_result.log
```

- 如果有 bug，立即修复后重新执行
- 确保全部用例 PASSED 或合理 SKIPPED
- 若存在多个 Python 解释器，**固定使用同一解释器路径**（如 `/usr/bin/python`），避免 NPU 检测结果不一致。

### 覆盖率报告

#### Python 实现的 API

```bash
# 先确认 API 函数所在模块路径
# 例如 torch.distributed.distributed_c10d
python -m pytest test_{api_full_name_underscored}.py -v \
    --cov=torch.distributed \
    --cov-report=term-missing \
    --cov-report=html:./coverage_report \
    2>&1 | tee coverage.log
```

覆盖率报告生成在 `test/{api_full_name_underscored}/coverage_report/` 目录下。

#### C++ pybind 实现的 API

使用 gcov/lcov（GCC）或 llvm-cov（Clang）方案：

```bash
# 1. 确认 PyTorch 是否以 coverage 模式编译（需 -fprofile-arcs -ftest-coverage）
# 2. 执行测试
python -m pytest test_{api_full_name_underscored}.py -v
# 3. 收集覆盖率
lcov --capture --directory /path/to/torch/build --output-file coverage.info
lcov --extract coverage.info '*/c10d/*' --output-file c10d_coverage.info
genhtml c10d_coverage.info --output-directory ./coverage_report
```

如果 PyTorch 未以 coverage 模式编译，在报告中说明并给出编译指引。

## 补充最佳实践

### 进程组生命周期

- 每个测试用例中 `init_process_group` 和 `destroy_process_group` 必须配对
- 使用 `try/finally` 确保异常时也能清理
- 自定义子组也需要正确销毁

### Work 对象测试（异步场景）

```python
work = dist.all_reduce(tensor, async_op=True)
assert isinstance(work, dist.Work)
work.wait()
assert work.is_completed()
```

### 空 tensor / 边界场景

- `torch.empty(0, dtype=...)` 测试零元素 tensor
- 单元素 tensor `torch.tensor([1.0])`
- 高维 tensor（4D+）


### 大 tensor 场景

至少包含一个 shape 较大的 tensor（如 `[1024, 1024]`）以验证非 trivial 数据量下的功能正确性。

### 混合 dtype 异常

测试同一集合操作中各 rank 使用不同 dtype 的异常行为（预期报错）。

## 检查清单

生成 UT 文件后，逐项验证：

- [ ] 文件路径为 `test/{api_full_name_underscored}/test_{api_full_name_underscored}.py`
- [ ] 目录名/文件名均使用下划线风格（API 全称中 `.` 全部替换为 `_`）
- [ ] 测试文件名不包含 `.`
- [ ] 头部注释包含测试目的、API 名称、覆盖维度表格、未覆盖项说明（简体中文）
- [ ] 条件导入 `torch_npu` 和 `transfer_to_npu`，NPU 优先
- [ ] 使用 `mp.spawn`，默认 2 卡
- [ ] 覆盖参数传/不传、None/非None、float32/bfloat16、正常/异常
- [ ] 每个参数满足“空/非空、类型、候选值、传/不传”四类覆盖要求（按接口实际支持范围）
- [ ] 使用等价类划分与边界值分析补充用例，并有对应场景
- [ ] 异常场景明确断言异常类型（spawn 子进程内用 `_assert_raises`/`try-except`，主进程单测可用 `pytest.raises`）
- [ ] 无 `pytest.xfail`
- [ ] `pytest.skip` 仅用于环境缺失（无可用 GPU/NPU 设备），且写明原因
- [ ] 不因 bfloat16/bf16 不支持而 `skip`，相关用例必须保留并实际执行
- [ ] 对「按设备类型字符串区分 cuda/npu」的 API，使用**单用例分支**覆盖当前环境，避免拆成两个用例导致常驻一条 skip
- [ ] 无数值正确性校验
- [ ] 进程组正确 init/destroy
- [ ] 端口使用动态分配
- [ ] 如环境可执行，已跑通测试并生成覆盖率报告
- [ ] 在 NPU 可用环境下，NPU 路径用例占比 >= 70%

## 参考路径

| 资源 | 路径 |
|------|------|
| API 签名 | `pytorch/torch/distributed/distributed_c10d.py` |
| transfer_to_npu | `ascend_pytorch/torch_npu/contrib/transfer_to_npu.py` |
| NPU 现有测试 | `ascend_pytorch/test/distributed/` |
| GPU 现有测试 | `pytorch/test/distributed/` |
| NPU 测试工具 | `ascend_pytorch/torch_npu/testing/common_distributed.py` |
| GPU 测试工具 | `pytorch/torch/testing/_internal/common_distributed.py` |

## 延伸阅读（渐进式披露）

生成具体 API 时按需阅读，避免在 SKILL 正文中堆叠过长内容：

- [api-reference.md](api-reference.md)：核心集合通信 / P2P / 对象通信 API 签名与 HCCL 注意事项速查
- [example-ut.md](example-ut.md)：`all_reduce` 完整示例 UT（含 spawn、参数化、异常辅助函数）
