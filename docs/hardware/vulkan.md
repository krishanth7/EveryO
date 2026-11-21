# Vulkan Compute Acceleration

Vulkan is a cross-platform graphics and compute standard by the Khronos Group. It serves as a vendor-independent GPU compute path across Windows, Linux, and Android.

## Hardware Support

Any GPU supporting Vulkan 1.2 or 1.3:
- AMD Radeon (GCN, RDNA).
- Intel HD, Iris, and Arc Graphics.
- NVIDIA GeForce (Maxwell and newer).
- ARM Mali and Qualcomm Adreno.

## Detection

EveryO inspects `vulkaninfo --summary` to discover available physical devices without loading graphics drivers unnecessarily:

```python
import everyo as eo

vulkan_devices = eo.backends.detect_all().get("vulkan", [])
for dev in vulkan_devices:
    print(f"Vulkan Device: {dev.name}")
```

## Qualification

Vulkan device detection is operational and verified. Compute execution via SPIR-V shaders is tracked for subsequent phases.
