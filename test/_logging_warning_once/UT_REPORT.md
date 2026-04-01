# UT 执行报告 — torch._logging.warning_once

## 执行命令

```bash
python -m unittest discover -s test/_logging_warning_once -p "test__logging_warning_once.py" -v
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
| test_basic_call_no_error | PASS | 正常调用不报错 |
| test_returns_none | PASS | 返回值为 None |
| test_emits_once_on_duplicate_calls | PASS | 重复调用仅 emit 一次 |
| test_emits_once_with_format_args | PASS | 带格式化参数的重复调用仅 emit 一次 |
| test_different_messages_each_emit_once | PASS | 不同消息各自独立计数 |
| test_different_format_args_trigger_separate_emissions | PASS | 相同格式串不同参数视为不同缓存键 |
| test_message_forwarded_to_logger | PASS | 消息及参数原样转发给 logger.warning |
| test_empty_string_message | PASS | 空字符串消息合法且仅 emit 一次 |
| test_cache_clear_resets_dedup | PASS | cache_clear() 后同消息可再次 emit |
| test_different_loggers_emit_independently | PASS | 不同 logger 实例各自独立去重 |
| test_none_logger_raises | PASS | None 作为 logger 触发 AttributeError/TypeError |
| test_non_logger_object_raises | PASS | 无 .warning() 方法的对象触发 AttributeError |

## 统计

- 通过: **12**
- 跳过: **0**
- 失败: **0**

## 跳过用例分析

无跳过用例。

## 本次改动文件列表

- `test/_logging_warning_once/test__logging_warning_once.py`（新增）
- `test/_logging_warning_once/UT_REPORT.md`（新增）
