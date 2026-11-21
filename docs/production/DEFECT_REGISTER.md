# EveryO Production Qualification Defect Register

## 1. Defect Classification System

The Defect Register tracks bugs, portability issues, security risks, and architectural constraints identified during the Phase 21 qualification programme.

### Severity Definitions

- **Critical**: Security vulnerability, data corruption, or unhandled process termination.
- **High**: Functional failure affecting an entire platform or operating system.
- **Medium**: Test portability defect, performance regression, or missing compliance standard.
- **Low**: Non-breaking ergonomic defect or missing non-critical endpoint.
- **Architectural Limitation**: Documented design boundary or pending multi-phase roadmap item.

---

## 2. Tracked Defect Ledger

| Defect ID | Severity | Component | Summary & Root Cause | Mitigation & Resolution | Status |
|---|---|---|---|---|---|
| `DEF-P21-001` | **High** | `everyo.backends.detection` | macOS `detect_system_memory()` failed because `os.sysconf("SC_AVPHYS_PAGES")` is unsupported on macOS Darwin, returning `0` free memory. | Implemented native macOS memory discovery using `sysctl hw.memsize` and `vm_stat` parsing (free + inactive + speculative pages) with dynamic Mach page sizes (4KB Intel, 16KB Apple Silicon). | **RESOLVED** |
| `DEF-P21-002` | **Medium** | `tests/test_memory_detection.py` | Patching `ctypes.windll.kernel32.GlobalMemoryStatusEx` on Linux/macOS raised `AttributeError: module 'ctypes' has no attribute 'windll'`. | Refactored test to inject a controlled `types.SimpleNamespace` test double into `ctypes.windll` using `create=True`, enabling universal test execution across all OSes. | **RESOLVED** |
| `DEF-P21-003` | **Critical** | `everyo.backends` | Risk of conflating driver detection with operational compute execution (e.g., claiming ROCm support merely because `rocm-smi` is on PATH). | Enforced strict five-tier qualification taxonomy. Backends without kernels report `is_available() == False` and are strictly excluded from `eo.device("auto")`. | **RESOLVED** |
| `DEF-P21-004` | **Medium** | `everyo.security` | Lack of machine-readable Software Bill of Materials (SBOM) for enterprise procurement and supply chain auditing. | Implemented `everyo.security.sbom` and `scripts/generate_sbom.py` generating standard CycloneDX v1.5 JSON and SPDX v2.3 JSON. | **RESOLVED** |
| `DEF-P21-005` | **Low** | `everyo.serving` | HTTP serving lacked Kubernetes-standard `/healthz` and `/readyz` probe routes, requiring non-standard probe configuration. | Added `/healthz`, `/ready`, and `/readyz` routes to `CPUModelServer._handler_factory()`, preserving 100% backward compatibility with `/health`. | **RESOLVED** |
| `DEF-P21-006` | **Architectural Limitation** | `everyo.backends` | Non-CUDA hardware backends (ROCm/HIP, Intel SYCL, DirectML, Apple Metal) support device detection only and have no compute kernels. | Backends fall back safely and observably to the CPU reference baseline. Compute kernels scheduled for implementation in Phase 22. | **BLOCKED / ROADMAP** |

---

## 3. Defect Metrics & Resolution Summary

- **Total Tracked Defects**: 6
- **Resolved**: 5 (83.3%)
- **Blocked / Accepted Architectural Limitations**: 1 (16.7%)
- **Open Unhandled Defects**: 0 (0.0%)
