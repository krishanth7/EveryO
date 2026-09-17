# Build the optional EveryO CUDA extension.
#
#   cmake -S cuda -B build/cuda -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/cuda --parallel
#
# The resulting module is copied next to the Python package so that
# `import everyo._everyo_cuda` finds it. Building is entirely optional:
# EveryO runs on NumPy when the extension is absent.

cmake_minimum_required(VERSION 3.18)

project(everyo_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Target the architectures of the GPUs present on the build machine; override
# with -DCMAKE_CUDA_ARCHITECTURES=75;86 for cross-compilation.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# pybind11 is located through its own CMake package, falling back to asking the
# active interpreter where its headers live (the usual pip install layout).
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_LOOKUP_RESULT
  )
  if(NOT PYBIND11_LOOKUP_RESULT EQUAL 0)
    message(FATAL_ERROR
      "pybind11 was not found. Install it with 'pip install pybind11' and "
      "re-run cmake.")
  endif()
  find_package(pybind11 CONFIG REQUIRED PATHS "${PYBIND11_CMAKE_DIR}")
endif()

pybind11_add_module(_everyo_cuda
  bindings/bindings.cpp
  src/device.cu
  src/vector_add.cu
  src/relu.cu
  src/matmul.cu
  src/reduction.cu
)

target_include_directories(_everyo_cuda PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(_everyo_cuda PRIVATE CUDA::cudart)

set_target_properties(_everyo_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

# Install next to the Python package so the import path resolves.
set(EVERYO_PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../everyo")
add_custom_command(TARGET _everyo_cuda POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E copy
          "$<TARGET_FILE:_everyo_cuda>"
          "${EVERYO_PACKAGE_DIR}/$<TARGET_FILE_NAME:_everyo_cuda>"
  COMMENT "Copying the CUDA extension into the everyo package"
)
