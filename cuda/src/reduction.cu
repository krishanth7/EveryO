// Full-array sum via a two-stage tree reduction.
//
// Stage 1: each block reduces its slice in shared memory to one partial sum.
// Stage 2: the (small) array of partial sums is reduced on the host, which
// avoids a second kernel launch and the atomics that would make the result
// non-deterministic.

#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>

#include "everyo_cuda.h"
#include "common.cuh"

namespace everyo {
namespace {

__global__ void SumKernel(const float* x, float* partials, std::size_t n) {
  extern __shared__ float scratch[];

  const unsigned int tid = threadIdx.x;
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  scratch[tid] = (index < n) ? x[index] : 0.0f;
  __syncthreads();

  // Tree reduction within the block; stride halves each iteration.
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partials[blockIdx.x] = scratch[0];
  }
}

}  // namespace

float Sum(const float* x, std::size_t n) {
  if (n == 0) {
    return 0.0f;
  }
  if (x == nullptr) {
    throw std::runtime_error("Sum received a null pointer.");
  }

  const std::size_t bytes = n * sizeof(float);
  detail::DeviceBuffer device_x(bytes);
  EVERYO_CUDA_CHECK(cudaMemcpy(device_x.get(), x, bytes, cudaMemcpyHostToDevice));
  return SumDevice(device_x.get(), n);
}

float SumDevice(const float* x, std::size_t n) {
  if (n == 0) return 0.0f;
  const unsigned int blocks = detail::GridSize(n, kBlockSize);
  detail::DeviceBuffer device_partials(blocks * sizeof(float));

  const std::size_t shared_bytes = kBlockSize * sizeof(float);
  SumKernel<<<blocks, kBlockSize, shared_bytes>>>(x, device_partials.get(), n);
  EVERYO_CUDA_CHECK_KERNEL();

  std::vector<float> partials(blocks);
  EVERYO_CUDA_CHECK(cudaMemcpy(partials.data(), device_partials.get(),
                               blocks * sizeof(float), cudaMemcpyDeviceToHost));

  // Accumulate in double precision: summing many float partials in float
  // loses accuracy quickly for large inputs.
  double total = 0.0;
  for (float value : partials) {
    total += static_cast<double>(value);
  }
  return static_cast<float>(total);
}

}  // namespace everyo
