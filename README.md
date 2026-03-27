# PyTorch Distributed Unit Test Generator

自动化生成 PyTorch 分布式 API 的功能测试用例，兼容 GPU(NCCL) 和 NPU(HCCL) 环境。

## 子模块

本项目依赖以下子模块：

| 子模块 | 仓库 |
|--------|------|
| `ascend_pytorch` | https://gitcode.com/Ascend/pytorch.git |
| `pytorch` | https://github.com/pytorch/pytorch.git |

## 初始化

```bash
git submodule update --init --recursive
```

## 特性

- **双环境兼容**：自动检测并适配 GPU (NCCL) 和 NPU (HCCL)
- **参数覆盖**：完整覆盖 API 的参数空间（类型、枚举、可选/必传、空/非空等）
- **多进程测试**：基于 `mp.spawn` 的分布式测试框架
- **自动端口管理**：动态分配端口避免冲突
- **异常验证**：明确断言异常类型
- **NPU 优先**：优先保障 NPU 环境下的测试覆盖

## 生成测试

使用 `/gen-distributed-ut` 技能生成指定 API 的测试用例。

## 执行测试

```bash
cd test/<api_name>
python -m pytest test_<api_name>.py -v
```

## 测试报告

测试执行结果见 [test/UT_EXECUTION_REPORT.md](test/UT_EXECUTION_REPORT.md)

## License

本项目遵循上游 PyTorch 项目的许可证。
