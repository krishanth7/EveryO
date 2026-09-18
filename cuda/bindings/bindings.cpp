// pybind11 bindings exposing the EveryO CUDA kernels to Python as
// everyo._everyo_cuda.
//
// The bindings are intentionally thin: they validate shapes and dtypes, hand
// raw pointers to the kernels in cuda/src, and let pybind11 translate C++
// exceptions into Python ones. All arrays are float32 and C-contiguous.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "everyo_cuda.h"

namespace py = pybind11;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

void CheckCuda(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

class DeviceTensor {
 public:
  explicit DeviceTensor(const FloatArray& array) {
    shape_.reserve(array.ndim());
    for (py::ssize_t i = 0; i < array.ndim(); ++i) shape_.push_back(array.shape(i));
    size_ = static_cast<std::size_t>(array.size());
    CheckCuda(cudaMalloc(&data_, size_ * sizeof(float)));
    CheckCuda(cudaMemcpy(data_, array.data(), size_ * sizeof(float), cudaMemcpyHostToDevice));
  }

  explicit DeviceTensor(std::vector<py::ssize_t> shape) : shape_(std::move(shape)) {
    size_ = 1;
    for (auto dimension : shape_) size_ *= static_cast<std::size_t>(dimension);
    CheckCuda(cudaMalloc(&data_, size_ * sizeof(float)));
  }

  ~DeviceTensor() { if (data_) cudaFree(data_); }
  DeviceTensor(const DeviceTensor&) = delete;
  DeviceTensor& operator=(const DeviceTensor&) = delete;

  float* data() const { return static_cast<float*>(data_); }
  std::size_t size() const { return size_; }
  const std::vector<py::ssize_t>& shape() const { return shape_; }

  FloatArray numpy() const {
    FloatArray output(shape_);
    CheckCuda(cudaMemcpy(output.mutable_data(), data_, size_ * sizeof(float), cudaMemcpyDeviceToHost));
    return output;
  }

 private:
  void* data_ = nullptr;
  std::size_t size_ = 0;
  std::vector<py::ssize_t> shape_;
};

using DeviceTensorPtr = std::shared_ptr<DeviceTensor>;

void RequireSameShape(const DeviceTensorPtr& a, const DeviceTensorPtr& b) {
  if (a->shape() != b->shape()) throw std::invalid_argument("Device tensor shapes must match.");
}

DeviceTensorPtr DeviceAdd(const DeviceTensorPtr& a, const DeviceTensorPtr& b) {
  RequireSameShape(a, b);
  auto out = std::make_shared<DeviceTensor>(a->shape());
  everyo::VectorAddDevice(a->data(), b->data(), out->data(), a->size());
  return out;
}

DeviceTensorPtr DeviceMultiply(const DeviceTensorPtr& a, const DeviceTensorPtr& b) {
  RequireSameShape(a, b);
  auto out = std::make_shared<DeviceTensor>(a->shape());
  everyo::VectorMultiplyDevice(a->data(), b->data(), out->data(), a->size());
  return out;
}

DeviceTensorPtr DeviceRelu(const DeviceTensorPtr& x) {
  auto out = std::make_shared<DeviceTensor>(x->shape());
  everyo::ReluDevice(x->data(), out->data(), x->size());
  return out;
}

DeviceTensorPtr DeviceMatMul(const DeviceTensorPtr& a, const DeviceTensorPtr& b) {
  if (a->shape().size() != 2 || b->shape().size() != 2 || a->shape()[1] != b->shape()[0])
    throw std::invalid_argument("Device matmul requires compatible 2-D tensors.");
  auto out = std::make_shared<DeviceTensor>(
      std::vector<py::ssize_t>{a->shape()[0], b->shape()[1]});
  everyo::MatMulDevice(a->data(), b->data(), out->data(),
                       static_cast<int>(a->shape()[0]), static_cast<int>(a->shape()[1]),
                       static_cast<int>(b->shape()[1]));
  return out;
}

void RequireRank(const FloatArray& array, int rank, const char* name) {
  if (array.ndim() != rank) {
    std::ostringstream message;
    message << name << " must be a " << rank << "-D array, but it has "
            << array.ndim() << " dimension(s).";
    throw std::invalid_argument(message.str());
  }
}

void RequireSameSize(const FloatArray& a, const FloatArray& b) {
  if (a.size() != b.size()) {
    std::ostringstream message;
    message << "Element-wise kernels need arrays of equal length, got "
            << a.size() << " and " << b.size() << ".";
    throw std::invalid_argument(message.str());
  }
}

FloatArray VectorAdd(FloatArray a, FloatArray b) {
  RequireRank(a, 1, "a");
  RequireRank(b, 1, "b");
  RequireSameSize(a, b);

  FloatArray out(a.size());
  everyo::VectorAdd(a.data(), b.data(), out.mutable_data(),
                    static_cast<std::size_t>(a.size()));
  return out;
}

FloatArray VectorMultiply(FloatArray a, FloatArray b) {
  RequireRank(a, 1, "a");
  RequireRank(b, 1, "b");
  RequireSameSize(a, b);

  FloatArray out(a.size());
  everyo::VectorMultiply(a.data(), b.data(), out.mutable_data(),
                         static_cast<std::size_t>(a.size()));
  return out;
}

FloatArray Relu(FloatArray x) {
  RequireRank(x, 1, "x");

  FloatArray out(x.size());
  everyo::Relu(x.data(), out.mutable_data(), static_cast<std::size_t>(x.size()));
  return out;
}

FloatArray MatMul(FloatArray a, FloatArray b) {
  RequireRank(a, 2, "a");
  RequireRank(b, 2, "b");

  const int m = static_cast<int>(a.shape(0));
  const int k = static_cast<int>(a.shape(1));
  const int n = static_cast<int>(b.shape(1));
  if (static_cast<int>(b.shape(0)) != k) {
    std::ostringstream message;
    message << "Cannot multiply matrices with shapes (" << m << ", " << k
            << ") and (" << b.shape(0) << ", " << n
            << "). Expected the inner dimensions to match.";
    throw std::invalid_argument(message.str());
  }

  FloatArray out({m, n});
  everyo::MatMul(a.data(), b.data(), out.mutable_data(), m, k, n);
  return out;
}

float Sum(FloatArray x) {
  RequireRank(x, 1, "x");
  return everyo::Sum(x.data(), static_cast<std::size_t>(x.size()));
}

py::dict DeviceProperties(int index) {
  const everyo::DeviceProperties properties = everyo::GetDeviceProperties(index);
  py::dict result;
  result["index"] = properties.index;
  result["name"] = properties.name;
  result["compute_capability"] =
      std::to_string(properties.major) + "." + std::to_string(properties.minor);
  result["total_memory"] = properties.total_memory;
  result["multiprocessor_count"] = properties.multiprocessor_count;
  return result;
}

}  // namespace

PYBIND11_MODULE(_everyo_cuda, module) {
  module.doc() =
      "EveryO CUDA kernels with NumPy entry points and persistent device tensors.";

  module.def("device_count", &everyo::DeviceCount,
             "Number of visible CUDA devices (0 when none).");
  module.def("runtime_version", &everyo::RuntimeVersion,
             "CUDA runtime version, e.g. 12030 for 12.3.");
  module.def("device_properties", &DeviceProperties, py::arg("index") = 0,
             "Properties of one CUDA device as a dictionary.");

  module.def("vector_add", &VectorAdd, py::arg("a"), py::arg("b"),
             "Element-wise addition of two 1-D float32 arrays.");
  module.def("vector_multiply", &VectorMultiply, py::arg("a"), py::arg("b"),
             "Element-wise multiplication of two 1-D float32 arrays.");
  module.def("relu", &Relu, py::arg("x"),
             "ReLU activation over a 1-D float32 array.");
  module.def("matmul", &MatMul, py::arg("a"), py::arg("b"),
             "Matrix product of two 2-D float32 arrays.");
  module.def("sum", &Sum, py::arg("x"),
             "Sum of every element of a 1-D float32 array.");

  py::class_<DeviceTensor, DeviceTensorPtr>(module, "DeviceTensor")
      .def(py::init<const FloatArray&>())
      .def_property_readonly("shape", &DeviceTensor::shape)
      .def_property_readonly("size", &DeviceTensor::size)
      .def("numpy", &DeviceTensor::numpy);
  module.def("device_add", &DeviceAdd);
  module.def("device_multiply", &DeviceMultiply);
  module.def("device_relu", &DeviceRelu);
  module.def("device_matmul", &DeviceMatMul);
  module.def("device_sum", [](const DeviceTensorPtr& x) {
    return everyo::SumDevice(x->data(), x->size());
  });
}
