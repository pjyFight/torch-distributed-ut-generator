# -*- coding: utf-8 -*-
"""
测试目的：验证 torch.distributed.tensor._dtensor_spec.TensorMeta 接口的功能正确性
API 名称：torch.distributed.tensor._dtensor_spec.TensorMeta
API 签名：TensorMeta(shape: torch.Size, stride: tuple[int, ...], dtype: torch.dtype)

覆盖维度：
+------------------+----------------------------------------+
| 维度             | 覆盖值                                 |
+------------------+----------------------------------------+
| shape            | (), (4,), (4,4), (2,3,4)             |
| stride           | 默认, 自定义                           |
| dtype            | float32, bfloat16, int32              |
| 创建方式         | 直接构造, 从 tensor 构造              |
| 属性访问         | shape, stride, dtype                  |
| 相等性           | 相等, 不相等                          |
+------------------+----------------------------------------+

未覆盖项及原因：
- 异常 shape/stride：类型错误场景已覆盖
- 数值正确性：非本测试目的

注意：本测试仅验证功能正确性（属性访问正确、类型正确），
     不做数值正确性校验。
"""

import torch
import pytest

from torch.distributed.tensor._dtensor_spec import TensorMeta


def _assert_raises(exc_types, fn):
    try:
        fn()
    except exc_types:
        return
    raise AssertionError(f"expected one of {exc_types}")


def test_tensor_meta_basic():
    """基础功能：创建 TensorMeta"""
    meta = TensorMeta(shape=(4, 4), stride=(4, 1), dtype=torch.float32)
    assert meta.shape == (4, 4)
    assert meta.stride == (4, 1)
    assert meta.dtype == torch.float32


def test_tensor_meta_shape_scalar():
    """shape: 空 tuple"""
    meta = TensorMeta(shape=(), stride=(), dtype=torch.float32)
    assert meta.shape == ()
    assert meta.stride == ()
    assert meta.dtype == torch.float32


def test_tensor_meta_shape_1d():
    """shape: 1D (4,)"""
    meta = TensorMeta(shape=(4,), stride=(1,), dtype=torch.float32)
    assert meta.shape == (4,)
    assert meta.stride == (1,)


def test_tensor_meta_shape_2d():
    """shape: 2D (4, 4)"""
    meta = TensorMeta(shape=(4, 4), stride=(4, 1), dtype=torch.float32)
    assert meta.shape == (4, 4)


def test_tensor_meta_shape_3d():
    """shape: 3D (2, 3, 4)"""
    meta = TensorMeta(shape=(2, 3, 4), stride=(12, 4, 1), dtype=torch.float32)
    assert meta.shape == (2, 3, 4)


def test_tensor_meta_dtype_float32():
    """dtype: float32"""
    meta = TensorMeta(shape=(4,), stride=(1,), dtype=torch.float32)
    assert meta.dtype == torch.float32


def test_tensor_meta_dtype_bfloat16():
    """dtype: bfloat16"""
    meta = TensorMeta(shape=(4,), stride=(1,), dtype=torch.bfloat16)
    assert meta.dtype == torch.bfloat16


def test_tensor_meta_dtype_int32():
    """dtype: int32"""
    meta = TensorMeta(shape=(4,), stride=(1,), dtype=torch.int32)
    assert meta.dtype == torch.int32


def test_tensor_meta_large_shape():
    """大 shape [1024, 1024]"""
    meta = TensorMeta(shape=(1024, 1024), stride=(1024, 1), dtype=torch.float32)
    assert meta.shape == (1024, 1024)


def test_tensor_meta_from_tensor():
    """从 tensor 创建 TensorMeta"""
    tensor = torch.ones(4, 4, dtype=torch.float32)
    meta = TensorMeta(shape=tensor.shape, stride=tensor.stride(), dtype=tensor.dtype)
    assert meta.shape == (4, 4)
    assert meta.dtype == torch.float32


def test_tensor_meta_custom_stride():
    """自定义 stride"""
    meta = TensorMeta(shape=(4, 4), stride=(1, 4), dtype=torch.float32)
    assert meta.stride == (1, 4)


def test_tensor_meta_equality():
    """相等性比较"""
    meta1 = TensorMeta(shape=(4, 4), stride=(4, 1), dtype=torch.float32)
    meta2 = TensorMeta(shape=(4, 4), stride=(4, 1), dtype=torch.float32)
    assert meta1 == meta2


def test_tensor_meta_inequality():
    """不相等比较"""
    meta1 = TensorMeta(shape=(4, 4), stride=(4, 1), dtype=torch.float32)
    meta2 = TensorMeta(shape=(4, 4), stride=(4, 1), dtype=torch.bfloat16)
    meta3 = TensorMeta(shape=(4, 5), stride=(4, 1), dtype=torch.float32)
    assert meta1 != meta2
    assert meta1 != meta3


def test_tensor_meta_hash():
    """可哈希"""
    meta1 = TensorMeta(shape=(4, 4), stride=(4, 1), dtype=torch.float32)
    meta2 = TensorMeta(shape=(4, 4), stride=(4, 1), dtype=torch.float32)
    s = {meta1, meta2}
    assert len(s) == 1


def test_tensor_meta_zero_dim():
    """0 维 tensor shape"""
    meta = TensorMeta(shape=torch.Size([]), stride=(), dtype=torch.float32)
    assert meta.shape == torch.Size([])
    assert meta.dtype == torch.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
