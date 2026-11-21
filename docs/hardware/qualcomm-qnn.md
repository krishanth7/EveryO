# Qualcomm QNN & Snapdragon Acceleration

The Qualcomm Neural Processing SDK (QNN) and Qualcomm AI Engine direct SDK target dedicated Hexagon NPUs and Adreno GPUs on Snapdragon platforms (e.g. Snapdragon X Elite, Snapdragon 8cx).

## Supported Hardware

- Snapdragon X Elite / Snapdragon X Plus (Windows on ARM laptops).
- Snapdragon 8 Gen 1/2/3/4 (Android / Embedded Linux).

## Detection

EveryO detects Qualcomm NPU hardware by probing the `QNN_SDK_ROOT` environment variable and inspecting ONNX Runtime's `QNNExecutionProvider`:

```python
import everyo as eo

status = eo.backends.get_backend_status()["qnn"]
print(f"QNN: {status['status']}")
```
