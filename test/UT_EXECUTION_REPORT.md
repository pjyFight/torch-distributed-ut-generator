# UT 执行结果报告（最新）

## 执行信息
- 执行时间: 2026-03-26
- 执行目录: `/home/p00845783/pta_ut`
- 执行命令: `pytest -q test`
- 总耗时: `360.06s` (`0:06:00`)
- 退出码: `0`

## 结果总览
- 总用例数: `93`
- 通过: `93`
- 失败: `0`
- 跳过: `0`
- 告警: `5`
- 通过率: `100%` (`93/93`)

## 告警信息（摘要）
- `torch_npu` 相关告警:
  - `ImportWarning`（`transfer_to_npu.py:362`）
  - `RuntimeWarning`（`transfer_to_npu.py:291`）
- `PytestUnknownMarkWarning: Unknown pytest.mark.timeout`
  - `test/tensor_copy_/test_tensor_copy_.py`
  - `test/torch_distributed__composable_state__insert_module_state/test_torch_distributed__composable_state__insert_module_state.py`
  - `test/torch_split_with_sizes_copy/test_torch_split_with_sizes_copy.py`

## 原始统计行
`93 passed, 0 skipped, 5 warnings in 360.06s (0:06:00)`
