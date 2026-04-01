# UT 执行报告 — torch.autograd.graph._MultiHandle

## 执行命令

```bash
python -m unittest discover -s test/autograd_graph__MultiHandle -p "test_autograd_graph__MultiHandle.py" -v
```

## 环境摘要

| 项目 | 版本 |
|------|------|
| Python | 3.11.14 |
| PyTorch | 2.7.1+cpu |
| torch_npu | 2.7.1.post2 |
| CANN | 8.5.1 (V100R001C25SPC002B220) |
| NPU 设备数 | 8 |

## 测试结果

| 测试方法 | 结果 | 说明 |
|----------|------|------|
| test_class_exists | PASS | _MultiHandle 可从 torch.autograd.graph 访问 |
| test_class_is_type | PASS | _MultiHandle 为 type 类型 |
| test_inherits_from_removable_handle | PASS | 继承自 RemovableHandle |
| test_construct_with_empty_tuple_npu | PASS | 空 tuple 构造成功（NPU） |
| test_construct_with_single_handle_npu | PASS | 单 handle 构造成功（NPU） |
| test_construct_with_multiple_handles_npu | PASS | 多 handle 构造成功（NPU） |
| test_handles_attribute_matches_input_npu | PASS | .handles 属性与传入 tuple 一致（NPU） |
| test_remove_empty_handles_npu | PASS | 空 handles remove() 不报错（NPU） |
| test_remove_single_handle_npu | PASS | 单 handle remove() 不报错（NPU） |
| test_remove_multiple_handles_npu | PASS | 多 handle remove() 不报错（NPU） |
| test_hook_inactive_after_remove_npu | PASS | remove() 后 hook 不再触发（NPU backward 验证） |
| test_context_manager_empty_npu | PASS | 空 _MultiHandle 作上下文管理器不报错（NPU） |
| test_context_manager_single_handle_npu | PASS | 单 handle 上下文管理器返回 self（NPU） |
| test_context_manager_multiple_handles_npu | PASS | 多 handle 上下文管理器不报错（NPU） |
| test_hook_inactive_after_context_exit_npu | PASS | with 块退出后 hook 不再触发（NPU backward 验证） |
| test_getstate_returns_tuple_empty | PASS | 空 _MultiHandle __getstate__ 返回 tuple |
| test_getstate_length_matches_handles_npu | PASS | __getstate__ 长度与 handles 数量一致（NPU） |
| test_setstate_restores_instance_npu | PASS | __setstate__ 还原出合法实例（NPU） |
| test_pickle_round_trip_empty | PASS | 空 _MultiHandle pickle round-trip 成功 |
| test_pickle_round_trip_with_handles_npu | PASS | 非空 _MultiHandle pickle round-trip 成功（NPU） |
| test_hook_fires_before_remove_npu | PASS | backward 期间 hook 正常触发（NPU） |
| test_construct_cpu_baseline | PASS | CPU baseline：构造不报错 |
| test_remove_cpu_baseline | PASS | CPU baseline：remove() 不报错 |

## 统计

- 通过: **23**
- 跳过: **0**
- 失败: **0**

## NPU/CPU 用例占比

- NPU 用例: 21 / 23 ≈ **91%**
- CPU 用例: 2 / 23 ≈ **9%**（满足 ≤20% 要求）

## 跳过用例分析

无跳过用例。

## 本次改动文件列表

- `test/autograd_graph__MultiHandle/test_autograd_graph__MultiHandle.py`（新增）
- `test/autograd_graph__MultiHandle/UT_REPORT.md`（新增）
