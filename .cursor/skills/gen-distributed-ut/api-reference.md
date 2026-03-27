# PyTorch 分布式核心 API 签名参考

> 来源：`pytorch_gpu/torch/distributed/distributed_c10d.py` (v2.7.1)

## 集合通信 API

### all_reduce

```python
def all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| tensor | Tensor | 必填 | in-place 操作 |
| op | ReduceOp | SUM | SUM/AVG/MAX/MIN/PRODUCT/BAND/BOR/BXOR/PREMUL_SUM |
| group | ProcessGroup | None(WORLD) | 进程组 |
| async_op | bool | False | 是否异步 |

### broadcast

```python
def broadcast(tensor, src=None, group=None, async_op=False)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| tensor | Tensor | 必填 | 广播数据 |
| src | int | None | 源 rank（None 时 group 须有默认） |
| group | ProcessGroup | None(WORLD) | 进程组 |
| async_op | bool | False | 是否异步 |

### reduce

```python
def reduce(tensor, dst=None, op=ReduceOp.SUM, group=None, async_op=False)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| tensor | Tensor | 必填 | in-place，结果在 dst rank |
| dst | int | None | 目标 rank |
| op | ReduceOp | SUM | 归约操作 |
| group | ProcessGroup | None(WORLD) | 进程组 |
| async_op | bool | False | 是否异步 |

### all_gather

```python
def all_gather(tensor_list, tensor, group=None, async_op=False)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| tensor_list | list[Tensor] | 必填 | 输出列表，长度=world_size |
| tensor | Tensor | 必填 | 本 rank 的输入 |
| group | ProcessGroup | None(WORLD) | 进程组 |
| async_op | bool | False | 是否异步 |

### all_gather_into_tensor

```python
def all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| output_tensor | Tensor | 必填 | 输出 tensor (world_size * input_size) |
| input_tensor | Tensor | 必填 | 本 rank 输入 |
| group | ProcessGroup | None(WORLD) | 进程组 |
| async_op | bool | False | 是否异步 |

### gather

```python
def gather(tensor, gather_list=None, dst=None, group=None, async_op=False)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| tensor | Tensor | 必填 | 本 rank 输入 |
| gather_list | list[Tensor] | None | 仅 dst rank 需要提供 |
| dst | int | None | 目标 rank |
| group | ProcessGroup | None(WORLD) | 进程组 |
| async_op | bool | False | 是否异步 |

### scatter

```python
def scatter(tensor, scatter_list=None, src=None, group=None, async_op=False)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| tensor | Tensor | 必填 | 接收 tensor |
| scatter_list | list[Tensor] | None | 仅 src rank 提供 |
| src | int | None | 源 rank |
| group | ProcessGroup | None(WORLD) | 进程组 |
| async_op | bool | False | 是否异步 |

### reduce_scatter

```python
def reduce_scatter(output, input_list, op=ReduceOp.SUM, group=None, async_op=False)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| output | Tensor | 必填 | 本 rank 的输出 |
| input_list | list[Tensor] | 必填 | 各 rank 的输入列表 |
| op | ReduceOp | SUM | 归约操作 |
| group | ProcessGroup | None(WORLD) | 进程组 |
| async_op | bool | False | 是否异步 |

### reduce_scatter_tensor

```python
def reduce_scatter_tensor(output, input, op=ReduceOp.SUM, group=None, async_op=False)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| output | Tensor | 必填 | 输出（大小 = input / world_size） |
| input | Tensor | 必填 | 输入 |
| op | ReduceOp | SUM | 归约操作 |
| group | ProcessGroup | None(WORLD) | 进程组 |
| async_op | bool | False | 是否异步 |

### all_to_all

```python
def all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| output_tensor_list | list[Tensor] | 必填 | 输出列表 |
| input_tensor_list | list[Tensor] | 必填 | 输入列表 |
| group | ProcessGroup | None(WORLD) | 进程组 |
| async_op | bool | False | 是否异步 |

### all_to_all_single

```python
def all_to_all_single(output, input, output_split_sizes=None, input_split_sizes=None, group=None, async_op=False)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| output | Tensor | 必填 | 输出 |
| input | Tensor | 必填 | 输入 |
| output_split_sizes | list[int] | None | 各 rank 输出分割大小 |
| input_split_sizes | list[int] | None | 各 rank 输入分割大小 |
| group | ProcessGroup | None(WORLD) | 进程组 |
| async_op | bool | False | 是否异步 |

### barrier

```python
def barrier(group=GroupMember.WORLD, async_op=False, device_ids=None)
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| group | ProcessGroup | WORLD | 进程组 |
| async_op | bool | False | 是否异步 |
| device_ids | list[int] | None | 设备 ID 列表 |

## 点对点通信 API

### send / recv

```python
def send(tensor, dst=None, group=None, tag=0)
def recv(tensor, src=None, group=None, tag=0)
```

### isend / irecv

```python
def isend(tensor, dst=None, group=None, tag=0)
def irecv(tensor, src=None, group=None, tag=0)
```

## 对象集合通信 API

### broadcast_object_list

```python
def broadcast_object_list(object_list, src=None, group=None, device=None)
```

### all_gather_object

```python
def all_gather_object(object_list, obj, group=None)
```

## 进程组管理 API

### init_process_group

```python
def init_process_group(backend=None, init_method=None, timeout=None, world_size=-1, rank=-1, store=None, group_name=None, pg_options=None, device_id=None)
```

### destroy_process_group

```python
def destroy_process_group(group=None)
```

### new_group

```python
def new_group(ranks=None, timeout=None, backend=None, pg_options=None, use_local_synchronization=False)
```

## NPU(HCCL) 后端注意事项

1. HCCL 后端不支持 `BAND/BOR/BXOR` ReduceOp
2. HCCL 对 `gather` / `scatter` 在部分场景有限制
3. `all_to_all_single` 的 split_sizes 在 HCCL 上需要所有 rank 一致
4. 空 tensor 操作在 HCCL 上可能行为不同
5. `device_ids` 参数在 `barrier` 中对 HCCL 无意义
6. `tag` 参数在 HCCL 后端下不生效
