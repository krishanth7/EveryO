#include <cstddef>

// Optional standalone C++ library for ctypes; no Python or GPU dependencies.
// Build manually, e.g. on Linux:
// c++ -O3 -std=c++17 -fPIC -shared cpu_cpp/everyo_axpy.cpp -o cpu_cpp/libeveryo_axpy.so
extern "C" void everyo_axpy_f64(std::size_t n, double alpha, const double* x,
                                  const double* y, double* out) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = alpha * x[i] + y[i];
    }
}
