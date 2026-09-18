// EveryO CUDA kernels — public C++ interface.
//
// Host entry points preserve the original NumPy-facing API. Device-suffixed
// entry points accept already-resident CUDA pointers so chained operations do
// not repeat host/device transfers.
//
// All entry points throw std::runtime_error on invalid arguments or CUDA
// failures, which pybind11 converts into Python exceptions.

#ifndef EVERYO_CUDA_H_
#define EVERYO_CUDA_H_

#include <cstddef>
#include <string>
#include <vector>

namespace everyo {

// Default threads per block. 256 is a good general-purpose choice: it is a
// multiple of the 32-thread warp size and leaves room for several resident
// blocks per streaming multiprocessor.
constexpr int kBlockSize = 256;

// Tile width for the shared-memory matmul kernel (16 x 16 = 256 threads).
constexpr int kTileWidth = 16;

// ---------------------------------------------------------------------------
// Device interrogation
// ---------------------------------------------------------------------------

// Number of CUDA devices visible to this process. Returns 0 (never throws)
// when no driver or device is present, so Python can probe safely.
int DeviceCount();

// CUDA runtime version reported by the driver, e.g. 12030 for 12.3.
int RuntimeVersion();

// Human-readable properties of one device: name, compute capability, memory.
struct DeviceProperties {
  int index = 0;
  std::string name;
  int major = 0;
  int minor = 0;
  std::size_t total_memory = 0;
  int multiprocessor_count = 0;
};

DeviceProperties GetDeviceProperties(int index);

// ---------------------------------------------------------------------------
// Element-wise kernels
// ---------------------------------------------------------------------------

// out[i] = a[i] + b[i] for i in [0, n).
void VectorAdd(const float* a, const float* b, float* out, std::size_t n);
void VectorAddDevice(const float* a, const float* b, float* out, std::size_t n);

// out[i] = a[i] * b[i] for i in [0, n).
void VectorMultiply(const float* a, const float* b, float* out, std::size_t n);
void VectorMultiplyDevice(const float* a, const float* b, float* out, std::size_t n);

// out[i] = max(x[i], 0) for i in [0, n).
void Relu(const float* x, float* out, std::size_t n);
void ReluDevice(const float* x, float* out, std::size_t n);

// ---------------------------------------------------------------------------
// Linear algebra and reductions
// ---------------------------------------------------------------------------

// Row-major matrix product: out (m x n) = a (m x k) * b (k x n).
void MatMul(const float* a, const float* b, float* out, int m, int k, int n);
void MatMulDevice(const float* a, const float* b, float* out, int m, int k, int n);

// Sum of every element of x, computed with a two-stage tree reduction.
float Sum(const float* x, std::size_t n);
float SumDevice(const float* x, std::size_t n);

}  // namespace everyo

#endif  // EVERYO_CUDA_H_
