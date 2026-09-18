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

}  // namespace

void VectorAdd(const float* a, const float* b, float* out, std::size_t n) {
  if (n == 0) return;
  const std::size_t bytes = n * sizeof(float);
  detail::DeviceBuffer da(bytes), db(bytes), dout(bytes);
  EVERYO_CUDA_CHECK(cudaMemcpy(da.get(), a, bytes, cudaMemcpyHostToDevice));
  EVERYO_CUDA_CHECK(cudaMemcpy(db.get(), b, bytes, cudaMemcpyHostToDevice));
  VectorAddDevice(da.get(), db.get(), dout.get(), n);
  EVERYO_CUDA_CHECK(cudaMemcpy(out, dout.get(), bytes, cudaMemcpyDeviceToHost));
}

void VectorMultiply(const float* a, const float* b, float* out, std::size_t n) {
  if (n == 0) return;
  const std::size_t bytes = n * sizeof(float);
  detail::DeviceBuffer da(bytes), db(bytes), dout(bytes);
  EVERYO_CUDA_CHECK(cudaMemcpy(da.get(), a, bytes, cudaMemcpyHostToDevice));
  EVERYO_CUDA_CHECK(cudaMemcpy(db.get(), b, bytes, cudaMemcpyHostToDevice));
  VectorMultiplyDevice(da.get(), db.get(), dout.get(), n);
  EVERYO_CUDA_CHECK(cudaMemcpy(out, dout.get(), bytes, cudaMemcpyDeviceToHost));
}

void VectorAddDevice(const float* a, const float* b, float* out, std::size_t n) {
  if (n == 0) return;
  VectorAddKernel<<<detail::GridSize(n, kBlockSize), kBlockSize>>>(a, b, out, n);
  EVERYO_CUDA_CHECK_KERNEL();
}

void VectorMultiplyDevice(const float* a, const float* b, float* out, std::size_t n) {
  if (n == 0) return;
  VectorMultiplyKernel<<<detail::GridSize(n, kBlockSize), kBlockSize>>>(a, b, out, n);
  EVERYO_CUDA_CHECK_KERNEL();
}

}  // namespace everyo
