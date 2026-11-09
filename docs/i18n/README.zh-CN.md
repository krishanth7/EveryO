# EveryO — 开放式神经计算框架

**语言：**简体中文 · [English](../../README.md) · [全部语言](../../README.md#documentation-in-10-languages)

EveryO 是面向神经网络研究、教学与可复现实验的开源 Python 框架。其 NumPy 核心实现了张量、自动微分、神经网络层、优化器以及训练流程。CUDA、TensorFlow 和 ONNX 均为可选集成。

## 快速开始

需要 Python 3.9–3.12。在仓库根目录执行：

```bash
python -m pip install -e .
everyo doctor
```

## 可运行示例

```python
import everyo as eo

x = eo.tensor([2.0], requires_grad=True)
y = x * x
y.backward()
print(x.grad)
```

**预期输出：**

```text
[4.]
```

该结果表示 `x = 2` 时 `x²` 的导数。模型训练示例请参阅 [examples](../../examples/)。

## 技术范围

- **核心能力：**NumPy、自动微分、神经网络层、优化和序列化。
- **验证方式：**回归测试、数值校验和 CPU 基准测试。
- **可选集成：**CUDA 加速、TensorFlow 对照测试和 ONNX 导出。
- **限制：**尚未获得独立的生产环境认证。请查看[资格评估报告](../production-qualification-2026-11-09.md)。

[英文文档](../../README.md)是权威技术参考。
