// Element-wise addition and multiplication.

#include <cuda_runtime.h>

#include <stdexcept>

#include "everyo_cuda.h"
#include "common.cuh"

namespace everyo {
namespace {

__global__ void VectorAddKernel(const float* a, const float* b, float* out,
                                std::size_t n) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  // Bounds check: the grid is rounded up, so the last block overshoots.
  if (index < n) {
    out[index] = a[index] + b[index];
  }
}

__global__ void VectorMultiplyKernel(const float* a, const float* b, float* out,
                                     std::size_t n) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < n) {
    out[index] = a[index] * b[index];
  }
}

// Shared host-side driver for both element-wise binary kernels.
template <typename Kernel>
void RunBinary(Kernel kernel, const float* a, const float* b, float* out,
               std::size_t n) {
  if (n == 0) {
    return;
  }
  if (a == nullptr || b == nullptr || out == nullptr) {
    throw std::runtime_error("Received a null pointer for an element-wise kernel.");
  }

  const std::size_t bytes = n * sizeof(float);
  detail::DeviceBuffer device_a(bytes);
  detail::DeviceBuffer device_b(bytes);
  detail::DeviceBuffer device_out(bytes);

  EVERYO_CUDA_CHECK(cudaMemcpy(device_a.get(), a, bytes, cudaMemcpyHostToDevice));
  EVERYO_CUDA_CHECK(cudaMemcpy(device_b.get(), b, bytes, cudaMemcpyHostToDevice));

  kernel<<<detail::GridSize(n, kBlockSize), kBlockSize>>>(
      device_a.get(), device_b.get(), device_out.get(), n);
  EVERYO_CUDA_CHECK_KERNEL();

  EVERYO_CUDA_CHECK(
      cudaMemcpy(out, device_out.get(), bytes, cudaMemcpyDeviceToHost));
}

}  // namespace

void VectorAdd(const float* a, const float* b, float* out, std::size_t n) {
  RunBinary(VectorAddKernel, a, b, out, n);
}

void VectorMultiply(const float* a, const float* b, float* out, std::size_t n) {
  RunBinary(VectorMultiplyKernel, a, b, out, n);
}

}  // namespace everyo
