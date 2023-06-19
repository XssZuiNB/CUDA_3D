#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdint.h>

#include <memory>
#include <stdexcept>

template <typename T>
std::shared_ptr<T> make_device_copy(T obj) {
  T* d_data;
  auto res = cudaMalloc(&d_data, sizeof(T));
  if (res != cudaSuccess)
    throw std::runtime_error("cudaMalloc failed status: " + res);
  cudaMemcpy(d_data, &obj, sizeof(T), cudaMemcpyHostToDevice);
  return std::shared_ptr<T>(d_data, [](T* data) { cudaFree(data); });
}

template <typename T>
std::shared_ptr<T> alloc_dev(int elements) {
  T* d_data;
  auto res = cudaMalloc(&d_data, sizeof(T) * elements);
  if (res != cudaSuccess)
    throw std::runtime_error("cudaMalloc failed status: " + res);
  return std::shared_ptr<T>(d_data, [](T* p) { cudaFree(p); });
}
