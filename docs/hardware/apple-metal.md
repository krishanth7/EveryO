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
