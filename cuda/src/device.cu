// Device interrogation helpers.

#include <cuda_runtime.h>

#include "everyo_cuda.h"
#include "common.cuh"

namespace everyo {

int DeviceCount() {
  int count = 0;
  // Deliberately not checked: a machine with no driver must report 0 rather
  // than raise, so that Python can probe availability safely.
  cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess) {
    cudaGetLastError();  // clear the sticky error
    return 0;
  }
  return count;
}

int RuntimeVersion() {
  int version = 0;
  EVERYO_CUDA_CHECK(cudaRuntimeGetVersion(&version));
  return version;
}

DeviceProperties GetDeviceProperties(int index) {
  if (index < 0 || index >= DeviceCount()) {
    throw std::runtime_error("CUDA device index out of range.");
  }
  cudaDeviceProp properties{};
  EVERYO_CUDA_CHECK(cudaGetDeviceProperties(&properties, index));

  DeviceProperties result;
  result.index = index;
  result.name = properties.name;
  result.major = properties.major;
  result.minor = properties.minor;
  result.total_memory = properties.totalGlobalMem;
  result.multiprocessor_count = properties.multiProcessorCount;
  return result;
}

}  // namespace everyo
