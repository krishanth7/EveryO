# Microsoft DirectML Acceleration

Microsoft DirectML provides low-level GPU acceleration across all DirectX 12 Feature Level 12_0 compatible graphics hardware on Windows.

## Hardware Support

DirectML is vendor-independent and operates on:
- AMD Radeon GPUs (RDNA, Vega, Polaris).
- Intel Arc and Iris Xe GPUs.
- NVIDIA GeForce and RTX GPUs.
- Qualcomm Adreno GPUs on Snapdragon Windows laptops.

## Requirements

- **Operating System**: Windows 10 (version 1903+) or Windows 11.
- **Runtime**: `onnxruntime-directml` (`pip install onnxruntime-directml`).

## Detection

```python
import everyo as eo

dml_status = eo.backends.get_backend_status()["directml"]
print(f"DirectML Available: {dml_status['available']}")
```

EveryO inspects adapter capabilities rather than assuming availability merely because Windows is installed.

## Qualification Tier & Implementation Status

- **Qualification Tier**: **Device detection only** (Windows).
- **Compute Kernels**: **Not implemented**. DirectML graph dispatch kernels are tracked for Phase 3.
- **Unsupported Environments**: Non-Windows operating systems (Linux, macOS) are **Unsupported** for DirectML.
- **Execution & Fallback Behavior**: Detection-only support does NOT execute compute operations. Requesting `directml` safely falls back to CPU when `strict=False`, or raises `EveryODeviceError` when `strict=True`. It is never auto-selected by `eo.device("auto")`.
