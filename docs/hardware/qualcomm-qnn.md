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

## Qualification Tier & Implementation Status

- **Qualification Tier**: **Device detection only** (Snapdragon on Windows/Linux).
- **Compute Kernels**: **Not implemented**. QNN graph execution kernels are tracked for Phase 3.
- **Execution & Fallback Behavior**: Detection-only support does NOT execute compute operations. Requesting `qnn` safely falls back to CPU when `strict=False`, or raises `EveryODeviceError` when `strict=True`. It is never auto-selected by `eo.device("auto")`.
