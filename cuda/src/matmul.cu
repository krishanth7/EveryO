// Tiled matrix multiplication.
//
// The kernel stages kTileWidth x kTileWidth tiles of both operands in shared
// memory. Each element loaded into shared memory is reused kTileWidth times,
// which is what makes the kernel worth writing over a naive global-memory
// version: it cuts global loads by roughly that factor.

#include <cuda_runtime.h>

#include <stdexcept>

#include "everyo_cuda.h"
#include "common.cuh"

namespace everyo {
namespace {

__global__ void MatMulKernel(const float* a, const float* b, float* out, int m,
                             int k, int n) {
  __shared__ float tile_a[kTileWidth][kTileWidth];
  __shared__ float tile_b[kTileWidth][kTileWidth];

  const int row = blockIdx.y * kTileWidth + threadIdx.y;
  const int column = blockIdx.x * kTileWidth + threadIdx.x;
  float accumulator = 0.0f;

  const int tiles = (k + kTileWidth - 1) / kTileWidth;
  for (int tile = 0; tile < tiles; ++tile) {
    const int a_column = tile * kTileWidth + threadIdx.x;
    const int b_row = tile * kTileWidth + threadIdx.y;

    // Out-of-range elements are zeroed so partial tiles contribute nothing.
    tile_a[threadIdx.y][threadIdx.x] =
        (row < m && a_column < k) ? a[row * k + a_column] : 0.0f;
    tile_b[threadIdx.y][threadIdx.x] =
        (b_row < k && column < n) ? b[b_row * n + column] : 0.0f;

    __syncthreads();

    for (int index = 0; index < kTileWidth; ++index) {
      accumulator += tile_a[threadIdx.y][index] * tile_b[index][threadIdx.x];
    }

    // Every thread must finish reading the tiles before they are overwritten.
    __syncthreads();
  }

  if (row < m && column < n) {
    out[row * n + column] = accumulator;
  }
}

}  // namespace

void MatMul(const float* a, const float* b, float* out, int m, int k, int n) {
  if (m <= 0 || k <= 0 || n <= 0) {
    throw std::runtime_error("MatMul dimensions must all be positive.");
  }
  if (a == nullptr || b == nullptr || out == nullptr) {
    throw std::runtime_error("MatMul received a null pointer.");
  }

  const std::size_t a_bytes = static_cast<std::size_t>(m) * k * sizeof(float);
  const std::size_t b_bytes = static_cast<std::size_t>(k) * n * sizeof(float);
  const std::size_t out_bytes = static_cast<std::size_t>(m) * n * sizeof(float);

  detail::DeviceBuffer device_a(a_bytes);
  detail::DeviceBuffer device_b(b_bytes);
  detail::DeviceBuffer device_out(out_bytes);

  EVERYO_CUDA_CHECK(cudaMemcpy(device_a.get(), a, a_bytes, cudaMemcpyHostToDevice));
  EVERYO_CUDA_CHECK(cudaMemcpy(device_b.get(), b, b_bytes, cudaMemcpyHostToDevice));

  const dim3 block(kTileWidth, kTileWidth);
  const dim3 grid((n + kTileWidth - 1) / kTileWidth,
                  (m + kTileWidth - 1) / kTileWidth);
  MatMulKernel<<<grid, block>>>(device_a.get(), device_b.get(), device_out.get(),
                                m, k, n);
  EVERYO_CUDA_CHECK_KERNEL();

  EVERYO_CUDA_CHECK(
      cudaMemcpy(out, device_out.get(), out_bytes, cudaMemcpyDeviceToHost));
}

}  // namespace everyo
