// pybind11 bindings exposing the EveryO CUDA kernels to Python as
// everyo._everyo_cuda.
//
// The bindings are intentionally thin: they validate shapes and dtypes, hand
// raw pointers to the kernels in cuda/src, and let pybind11 translate C++
// exceptions into Python ones. All arrays are float32 and C-contiguous.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>
#include <stdexcept>
#include <string>

#include "everyo_cuda.h"

namespace py = pybind11;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

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
      "EveryO CUDA kernels. Every function takes and returns float32 NumPy "
      "arrays; host/device transfers happen inside each call.";

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
}
