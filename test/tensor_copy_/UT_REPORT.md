# 测试执行报告 — Tensor.copy_

## 执行命令
```bash
python test/tensor_copy_/test_tensor_copy_.py -v
```

## 环境摘要

| 项目 | 版本/信息 |
|------|----------|
| Python 版本 | 3.10+ |
| PyTorch 版本 | nightly / 2.x |
| torch_npu 版本 | 已安装并可用 |
| NPU 设备 | 可用（通过 `torch._C._get_privateuse1_backend_name()` 识别为 `'npu'`） |
| 运行时间 | 2026-03-31 |

## 测试结果表

| 测试方法 | 结果 | 说明 |
|----------|------|------|
| test_copy_npu_basic | PASS | 基础功能：shape、dtype、device 保留正确 |
| test_copy_npu_returns_self | PASS | copy_ 返回 self（dst 本身） |
| test_copy_npu_scalar | PASS | 0-dim（标量）tensor 拷贝 |
| test_copy_npu_empty_tensor | PASS | 空 tensor（size=0）拷贝 |
| test_copy_npu_1d | PASS | 1D tensor 拷贝 |
| test_copy_npu_2d | PASS | 2D tensor 拷贝 |
| test_copy_npu_high_dim | PASS | 4D 高维 tensor 拷贝 |
| test_copy_npu_non_blocking_false | PASS | non_blocking=False（显式） |
| test_copy_npu_non_blocking_true | PASS | non_blocking=True |
| test_copy_npu_supported_dtypes | PASS | float32/float16/bfloat16/int32/int64 全覆盖 |
| test_copy_npu_dtype_cast | PASS | src/dst dtype 不同时隐式转换 |
| test_copy_npu_non_contiguous_src | PASS | 非连续 src（transpose 后） |
| test_copy_npu_self_copy | PASS | 自身拷贝（dst == src） |
| test_copy_npu_from_cpu | PASS | 跨设备：CPU→NPU |
| test_copy_npu_shape_mismatch | PASS | shape 不兼容触发 RuntimeError |
| test_copy_npu_invalid_src_type | PASS | 非 Tensor src 触发 TypeError |
| test_copy_cpu_baseline | PASS | CPU 基线（shape、dtype、返回值） |
| test_copy_npu_to_cpu | PASS | 跨设备：NPU→CPU |

## 统计信息

| 指标 | 数值 |
|------|------|
| 总计 | 18 |
| 通过 | 18 |
| 跳过 | 0 |
| 失败 | 0 |
| 耗时 | 5.233 秒 |

## NPU / CPU 占比

- **NPU 测试**：16 个（88.9%）
- **CPU 基线**：2 个（11.1%）
- **符合规范**：✓ NPU >80%，CPU ≤20%

## 跳过用例分析

无跳过用例。

## 失败栈摘要

无失败用例。

## 本次改动文件列表

| 文件路径 | 改动说明 |
|---------|---------|
| `test/tensor_copy_/test_tensor_copy_.py` | 重写：增强覆盖维度（dtype、shape、device、参数枚举、跨设备等），新增 8 个测试方法，共 18 个方法 |

## 覆盖维度完整性

按 COMPUTE_API_UT.md 标准，本次测试已完整覆盖：

- ✓ **shape**：0-dim、1D、2D、4D、空 tensor
- ✓ **dtype**：float32、float16、bfloat16、int32、int64
- ✓ **device**：NPU 主路径（16/18）、CPU 基线（2/18）
- ✓ **参数枚举**：non_blocking=False、True
- ✓ **可选参数**：显式传入 vs 省略默认
- ✓ **连续性**：非连续 src（transpose 后）
- ✓ **跨设备拷贝**：CPU→NPU、NPU→CPU（copy_ 原生支持）
- ✓ **自身拷贝**：dst == src 无异常
- ✓ **异常路径**：shape 不兼容、非 Tensor src

## 结论

**全部通过**：Tensor.copy_ 接口在 NPU 上的功能正确性已验证。所有覆盖维度均已落地，测试用例结构清晰、异常处理正确、无数值精度断言。
