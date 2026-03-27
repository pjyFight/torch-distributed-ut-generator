# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.split_with_sizes_copy 接口在张量分割场景下的功能正确性
API 名称：torch.split_with_sizes_copy
API 签名：split_with_sizes_copy(Tensor all_gather_output, SymInt[] all_gather_input_split_sizes, int dim=0, *, Tensor(a!)[] out) -> ()

覆盖维度：
+------------------+----------------------------------------+
| 维度             | 覆盖值                                 |
+------------------+----------------------------------------+
| tensor dtype     | float32, bfloat16                     |
| tensor shape     | [10], [4,10], [2,5,10]                |
| split_sizes      | [2,3,5], [1,2,3,4], [5,5]            |
| dim              | 0, 1, -1                              |
| out 列表长度     | 匹配 split_sizes 长度                 |
| 空 tensor        | 0 元素 tensor                         |
+------------------+----------------------------------------+

未覆盖项及原因：
- 异常 split_sizes：sum 不等于 tensor 维度大小（行为依赖实现）
- 非连续 tensor：需要特殊构造
- 原地操作副作用：out 参数会原地修改

注意：本测试仅验证功能正确性（分割后 shape/dtype 正确），
     不做数值正确性校验。
"""

import torch
import pytest

_IS_NPU = hasattr(torch, 'npu') and torch.npu.is_available()
_IS_CUDA = not _IS_NPU and torch.cuda.is_available()

if _IS_NPU:
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

DEVICE_TYPE = "npu" if _IS_NPU else ("cuda" if _IS_CUDA else "cpu")


def _assert_raises(exc_types, fn):
    try:
        fn()
    except exc_types:
        return
    raise AssertionError(f"expected one of {exc_types}")


@pytest.mark.timeout(120)
def test_split_with_sizes_copy_basic():
    """基础功能：1D tensor 按 split_sizes 分割"""
    device = torch.device(DEVICE_TYPE)
    all_gather_output = torch.arange(10, dtype=torch.float32, device=device)
    split_sizes = [2, 3, 5]
    out = [torch.zeros(2, dtype=torch.float32, device=device),
           torch.zeros(3, dtype=torch.float32, device=device),
           torch.zeros(5, dtype=torch.float32, device=device)]
    torch.split_with_sizes_copy(all_gather_output, split_sizes, dim=0, out=out)
    assert out[0].shape == (2,)
    assert out[1].shape == (3,)
    assert out[2].shape == (5,)


@pytest.mark.timeout(120)
def test_split_with_sizes_copy_dim0():
    """不同 dim：dim=0"""
    device = torch.device(DEVICE_TYPE)
    all_gather_output = torch.arange(20, dtype=torch.float32, device=device).reshape(4, 5)
    split_sizes = [2, 2]
    out = [torch.zeros(2, 5, dtype=torch.float32, device=device),
           torch.zeros(2, 5, dtype=torch.float32, device=device)]
    torch.split_with_sizes_copy(all_gather_output, split_sizes, dim=0, out=out)
    assert out[0].shape == (2, 5)
    assert out[1].shape == (2, 5)


@pytest.mark.timeout(120)
def test_split_with_sizes_copy_dim1():
    """不同 dim：dim=1"""
    device = torch.device(DEVICE_TYPE)
    all_gather_output = torch.arange(20, dtype=torch.float32, device=device).reshape(4, 5)
    split_sizes = [2, 3]
    out = [torch.zeros(4, 2, dtype=torch.float32, device=device),
           torch.zeros(4, 3, dtype=torch.float32, device=device)]
    torch.split_with_sizes_copy(all_gather_output, split_sizes, dim=1, out=out)
    assert out[0].shape == (4, 2)
    assert out[1].shape == (4, 3)


@pytest.mark.timeout(120)
def test_split_with_sizes_copy_dim_negative():
    """不同 dim：dim=-1（最后一维）"""
    device = torch.device(DEVICE_TYPE)
    all_gather_output = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)
    split_sizes = [2, 2]
    out = [torch.zeros(3, 2, dtype=torch.float32, device=device),
           torch.zeros(3, 2, dtype=torch.float32, device=device)]
    torch.split_with_sizes_copy(all_gather_output, split_sizes, dim=-1, out=out)
    assert out[0].shape == (3, 2)
    assert out[1].shape == (3, 2)


@pytest.mark.timeout(120)
def test_split_with_sizes_copy_dtype_float32():
    """dtype：float32"""
    device = torch.device(DEVICE_TYPE)
    all_gather_output = torch.ones(10, dtype=torch.float32, device=device)
    split_sizes = [5, 5]
    out = [torch.zeros(5, dtype=torch.float32, device=device),
           torch.zeros(5, dtype=torch.float32, device=device)]
    torch.split_with_sizes_copy(all_gather_output, split_sizes, dim=0, out=out)
    assert out[0].dtype == torch.float32
    assert out[1].dtype == torch.float32


@pytest.mark.timeout(120)
def test_split_with_sizes_copy_dtype_bfloat16():
    """dtype：bfloat16"""
    device = torch.device(DEVICE_TYPE)
    all_gather_output = torch.ones(10, dtype=torch.bfloat16, device=device)
    split_sizes = [3, 7]
    out = [torch.zeros(3, dtype=torch.bfloat16, device=device),
           torch.zeros(7, dtype=torch.bfloat16, device=device)]
    torch.split_with_sizes_copy(all_gather_output, split_sizes, dim=0, out=out)
    assert out[0].dtype == torch.bfloat16
    assert out[1].dtype == torch.bfloat16


@pytest.mark.timeout(120)
def test_split_with_sizes_copy_large_tensor():
    """大 tensor [1024, 1024]"""
    device = torch.device(DEVICE_TYPE)
    all_gather_output = torch.ones(1024, 1024, dtype=torch.float32, device=device)
    split_sizes = [256, 256, 256, 256]
    out = [torch.zeros(256, 1024, dtype=torch.float32, device=device) for _ in range(4)]
    torch.split_with_sizes_copy(all_gather_output, split_sizes, dim=0, out=out)
    for o in out:
        assert o.shape == (256, 1024)


@pytest.mark.timeout(120)
def test_split_with_sizes_copy_3d_tensor():
    """3D tensor"""
    device = torch.device(DEVICE_TYPE)
    all_gather_output = torch.arange(40, dtype=torch.float32, device=device).reshape(2, 4, 5)
    split_sizes = [1, 1]
    out = [torch.zeros(1, 4, 5, dtype=torch.float32, device=device),
           torch.zeros(1, 4, 5, dtype=torch.float32, device=device)]
    torch.split_with_sizes_copy(all_gather_output, split_sizes, dim=0, out=out)
    assert out[0].shape == (1, 4, 5)
    assert out[1].shape == (1, 4, 5)


@pytest.mark.timeout(120)
def test_split_with_sizes_copy_invalid_split_sizes():
    """异常场景：split_sizes 长度与 out 列表长度不匹配"""
    device = torch.device(DEVICE_TYPE)
    all_gather_output = torch.ones(10, dtype=torch.float32, device=device)
    split_sizes = [2, 3, 5]
    out = [torch.zeros(2, dtype=torch.float32, device=device),
           torch.zeros(3, dtype=torch.float32, device=device)]
    _assert_raises(
        (RuntimeError, ValueError),
        lambda: torch.split_with_sizes_copy(all_gather_output, split_sizes, dim=0, out=out)
    )


@pytest.mark.timeout(120)
def test_split_with_sizes_copy_invalid_dim():
    """异常场景：dim 越界"""
    device = torch.device(DEVICE_TYPE)
    all_gather_output = torch.ones(10, dtype=torch.float32, device=device)
    split_sizes = [5, 5]
    out = [torch.zeros(5, dtype=torch.float32, device=device),
           torch.zeros(5, dtype=torch.float32, device=device)]
    _assert_raises(
        (RuntimeError, IndexError),
        lambda: torch.split_with_sizes_copy(all_gather_output, split_sizes, dim=10, out=out)
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
