# UT 执行报告 — torch.split_with_sizes_copy

## 执行命令

```bash
python test/split_with_sizes_copy/test_split_with_sizes_copy.py -v
```

## 环境摘要

| 项目 | 版本 |
|------|------|
| Python | 3.11.14 |
| PyTorch | 2.7.1+cpu |
| torch_npu | 2.7.1.post2 |
| NPU 设备数 | 8 |
| 执行日期 | 2026-03-31 |

## 测试结果

| 测试方法 | 结果 | 说明 |
|----------|------|------|
| test_basic_1d_dim0 | PASS | 1D NPU split with out=，dim=0 |
| test_2d_dim0 | PASS | 2D NPU split，dim=0 |
| test_2d_dim1 | PASS | 2D NPU split，dim=1 |
| test_dim_negative_one | PASS | 负数 dim=-1 |
| test_bfloat16_dtype | PASS | bfloat16 dtype 保持 |
| test_float16_dtype | PASS | float16 dtype 保持 |
| test_int64_dtype | PASS | int64 dtype 保持 |
| test_no_out_returns_list_of_tensors | PASS | 无 out= 时返回 list[Tensor] |
| test_no_out_tensors_are_contiguous_copies | PASS | 返回值为 contiguous 新张量 |
| test_high_dim_3d | PASS | 3D 张量，dim=1 |
| test_empty_tensor_split | PASS | size-0 空张量 |
| test_single_element_split_chunk | PASS | 单元素 chunk（size=1） |
| test_non_contiguous_input | PASS | 非连续输入（transpose 后） |
| test_default_dim_is_zero | PASS | 省略 dim 走默认 dim=0 |
| test_out_return_value_is_none | PASS | out= 传入时返回 None |
| test_out_len_mismatch_raises | PASS | out 长度与 split_sizes 不一致抛异常 |
| test_mixed_device_out_raises | PASS | NPU+CPU 混合 out 触发 RuntimeError |
| test_cpu_baseline | PASS | CPU 基线验证 |

## 统计

- 通过：**18**
- 跳过：**0**
- 失败：**0**

## 跳过用例分析

无跳过用例。

## 失败栈摘要

无失败用例。

> 执行过程中发现一处测试逻辑 bug：`test_non_contiguous_input` 初版使用 `randn(4, 6).t()` 得到 shape `(6, 4)`，对 dim=1（大小=4）传入 `split_sizes=[2,4]`（和为 6），导致 RuntimeError。
> 修正为 `randn(6, 4).t()` 得到 `(4, 6)`，dim=1 大小=6，split_sizes=[2,4] 求和匹配，修复后通过。

## 本次改动文件列表

- `test/split_with_sizes_copy/test_split_with_sizes_copy.py`（补充 11 个测试方法，兼容上游新目录命名）
- `test/split_with_sizes_copy/UT_REPORT.md`（本文件，新增）
