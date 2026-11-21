# Apple Metal & MPS Acceleration

Apple Metal provides low-overhead hardware acceleration on macOS across Apple Silicon (M1, M2, M3, M4) integrated GPUs and Apple Neural Engine.

## Requirements

- **Operating System**: macOS 12.0 Monterey, macOS 13 Ventura, macOS 14 Sonoma, macOS 15 Sequoia.
- **Hardware**: Apple Silicon (arm64 architecture).
- **Runtime**: macOS native `Metal.framework` and `MetalPerformanceShaders.framework`.

## Detection

On non-macOS platforms (Windows, Linux), EveryO immediately reports `Metal is only available on macOS.` without raising errors. On macOS, EveryO queries `system_profiler SPDisplaysDataType` and Apple Silicon architecture flags.

```python
import everyo as eo

status = eo.backends.get_backend_status()["metal"]
print(f"Metal Status: {status['status']} - {status['reason']}")
```

## Qualification Tier & Implementation Status

- **Qualification Tier**: **Device detection only** (macOS).
- **Compute Kernels**: **Not implemented**. Metal Performance Shaders (MPS) kernels are tracked for Phase 3.
- **Unsupported Environments**: Non-macOS operating systems (Windows, Linux) are **Unsupported** for Metal.
- **Execution & Fallback Behavior**: Detection-only support does NOT execute compute operations. Requesting `metal` safely falls back to CPU when `strict=False`, or raises `EveryODeviceError` when `strict=True`. It is never auto-selected by `eo.device("auto")`.
