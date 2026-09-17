// Shared error-handling helpers for the EveryO CUDA kernels.

#ifndef EVERYO_CUDA_COMMON_CUH_
#define EVERYO_CUDA_COMMON_CUH_

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace everyo {
namespace detail {

// Throws std::runtime_error with the CUDA error name and the originating call.
inline void Check(cudaError_t status, const char* expression, const char* file,
                  int line) {
  if (status != cudaSuccess) {
    std::ostringstream message;
    message << "CUDA error in " << file << ":" << line << " during '"
            << expression << "': " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
  }
}

// RAII wrapper so device memory is released even when a later call throws.
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
    if (bytes_ == 0) {
      return;
    }
    cudaError_t status = cudaMalloc(&pointer_, bytes_);
    if (status != cudaSuccess) {
      std::ostringstream message;
      message << "Failed to allocate " << bytes_
              << " bytes on the CUDA device: " << cudaGetErrorString(status);
      throw std::runtime_error(message.str());
    }
  }

  ~DeviceBuffer() {
    if (pointer_ != nullptr) {
      // Nothing useful can be done if this fails during stack unwinding.
      cudaFree(pointer_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  float* get() const { return static_cast<float*>(pointer_); }
  std::size_t bytes() const { return bytes_; }

 private:
  void* pointer_ = nullptr;
  std::size_t bytes_ = 0;
};

// Number of blocks needed to cover n elements with the given block size.
inline unsigned int GridSize(std::size_t n, int block_size) {
  return static_cast<unsigned int>((n + block_size - 1) / block_size);
}

}  // namespace detail
}  // namespace everyo

#define EVERYO_CUDA_CHECK(expression) \
  ::everyo::detail::Check((expression), #expression, __FILE__, __LINE__)

// Checks both the launch configuration error and any error raised while the
// kernel ran. cudaDeviceSynchronize() makes asynchronous faults observable here
// instead of surfacing at an unrelated call site later.
#define EVERYO_CUDA_CHECK_KERNEL()               \
  do {                                           \
    EVERYO_CUDA_CHECK(cudaGetLastError());       \
    EVERYO_CUDA_CHECK(cudaDeviceSynchronize());  \
  } while (0)

#endif  // EVERYO_CUDA_COMMON_CUH_
