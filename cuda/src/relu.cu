// ReLU activation.

#include <cuda_runtime.h>

#include <stdexcept>

#include "everyo_cuda.h"
#include "common.cuh"

namespace everyo {
namespace {

__global__ void ReluKernel(const float* x, float* out, std::size_t n) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < n) {
    const float value = x[index];
    out[index] = value > 0.0f ? value : 0.0f;
  }
}

}  // namespace

void Relu(const float* x, float* out, std::size_t n) {
  if (n == 0) {
    return;
  }
  if (x == nullptr || out == nullptr) {
    throw std::runtime_error("Relu received a null pointer.");
  }

  const std::size_t bytes = n * sizeof(float);
  detail::DeviceBuffer device_x(bytes);
  detail::DeviceBuffer device_out(bytes);

  EVERYO_CUDA_CHECK(cudaMemcpy(device_x.get(), x, bytes, cudaMemcpyHostToDevice));

  ReluKernel<<<detail::GridSize(n, kBlockSize), kBlockSize>>>(
      device_x.get(), device_out.get(), n);
  EVERYO_CUDA_CHECK_KERNEL();

  EVERYO_CUDA_CHECK(
      cudaMemcpy(out, device_out.get(), bytes, cudaMemcpyDeviceToHost));
}

}  // namespace everyo
