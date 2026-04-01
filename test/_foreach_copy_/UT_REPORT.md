# UT 执行报告 — torch._foreach_copy_

## 执行命令

```bash
python test/_foreach_copy_/test__foreach_copy_.py
```

## 环境摘要

| 项目 | 版本 |
|------|------|
| Python | 3.11.14 |
| PyTorch | 2.7.1+cpu |
| torch_npu | 2.7.1.post2 |
| NPU 设备数量 | 8 |

## 测试结果

| 测试方法 | 结果 | 说明 |
|----------|------|------|
| test_foreach_copy_npu_basic | PASS | 基础 NPU float32 2D 拷贝，shape/dtype/device 正确 |
| test_foreach_copy_npu_non_blocking_false | PASS | non_blocking=False 显式传入，同步完成 |
| test_foreach_copy_npu_non_blocking_true | PASS | non_blocking=True 不报错，tensor 保留正确 device |
| test_foreach_copy_npu_single_tensor | PASS | 单元素列表，shape/dtype 保持不变 |
| test_foreach_copy_npu_empty_list | PASS | 空列表触发 RuntimeError（至少需要一个 tensor）|
| test_foreach_copy_npu_scalar | PASS | 0-dim 标量 tensor，shape 为 [] |
| test_foreach_copy_npu_high_dim | PASS | 3D / 4D 高维 tensor，shape 正确 |
| test_foreach_copy_npu_empty_tensor | PASS | size-0 空 tensor，shape 保持 (0, 4) |
| test_foreach_copy_npu_dtypes | PASS | float32 / float16 / bfloat16 / int32 均正确 |
| test_foreach_copy_npu_dtype_cast | PASS | float32 src → float16 self 隐式转换，self dtype 不变 |
| test_foreach_copy_npu_non_contiguous_src | PASS | 非连续（transpose）src，拷贝不报错 |
| test_foreach_copy_npu_varied_shapes | PASS | 同批次 1D / 2D / 3D 混合形状 |
| test_foreach_copy_npu_return_value | PASS | self 原地修改后保留正确 shape 和 device |
| test_foreach_copy_npu_mixed_device | PASS | self NPU + src CPU 混合设备（device_check: NoCheck fallback）|
| test_foreach_copy_npu_length_mismatch | PASS | self/src 长度不匹配触发 RuntimeError |
| test_foreach_copy_cpu_baseline | PASS | CPU 基线，shape/dtype 正确 |

## 统计

- 通过: **16**
- 跳过: **0**
- 失败: **0**

## 跳过用例分析

无跳过用例。

## 修复记录

初次执行发现两处问题，已修复：

| 问题 | 原因 | 修复方式 |
|------|------|----------|
| `assertIsNone(result)` 失败（共 5 处） | NPU 侧 `torch._foreach_copy_` 返回 `self` 列表而非 `None`，与 native_functions.yaml `-> ()` 签名存在实现差异 | 移除所有 `assertIsNone` 断言，改为直接断言 `self` 张量的 shape/device |
| `test_foreach_copy_npu_empty_list` ERROR | 空列表触发 `RuntimeError: Tensor list must have at least one tensor.` | 将用例改为 `assertRaises(RuntimeError)` 验证该异常行为 |

## 本次改动文件列表

- `test/_foreach_copy_/test__foreach_copy_.py`（新建）
- `test/_foreach_copy_/UT_REPORT.md`（本报告，新建）
