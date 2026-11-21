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
