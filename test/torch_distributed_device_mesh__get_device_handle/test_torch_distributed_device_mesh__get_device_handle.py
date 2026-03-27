# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.device_mesh._get_device_handle 接口的功能正确性
API 名称：torch.distributed.device_mesh._get_device_handle
API 签名：_get_device_handle(device_type: str) -> Optional[module]

覆盖维度：
+------------------+----------------------------------------+
| 维度             | 覆盖值                                 |
+------------------+----------------------------------------+
| device_type      | "cuda", "npu", "invalid"             |
| 返回值           | DeviceHandle 对象或 None               |
+------------------+----------------------------------------+

未覆盖项及原因：
- 内部 API，具体行为可能随版本变化
- 特定设备类型需要对应硬件支持

注意：本测试仅验证功能正确性，
     不做数值正确性校验。
"""

import torch
import pytest

_IS_NPU = hasattr(torch, 'npu') and torch.npu.is_available()
_IS_CUDA = not _IS_NPU and torch.cuda.is_available()

if _IS_NPU:
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401


try:
    from torch.distributed.device_mesh import _get_device_handle
except ImportError:
    pytest.skip("_get_device_handle not available in this PyTorch version", allow_module_level=True)


def test_get_device_handle_supported_device():
    """获取当前可用设备（CUDA 或 NPU）的 device handle"""
    if _IS_CUDA:
        assert _get_device_handle("cuda") is not None
    elif _IS_NPU:
        assert _get_device_handle("npu") is not None
    else:
        pytest.skip("No CUDA or NPU device available")


def test_get_device_handle_invalid():
    """无效 device_type 返回 None"""
    result = _get_device_handle("invalid_device")
    assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
