#include <cuda.h>
#include <cuda_runtime_api.h>

#include <stdexcept>

#include "cuda_util.cuh"

/*
template <typename T>
std::shared_ptr<T> make_device_copy(T obj) {
  T* d_data;
  auto res = cudaMalloc(&d_data, sizeof(T));

  if (res != cudaSuccess)
    throw std::runtime_error("cudaMalloc failed status: " + res);

  cudaMemcpy(d_data, &obj, sizeof(T), cudaMemcpyHostToDevice);

  return std::shared_ptr<T>(d_data, [](T* data) { cudaFree(data); });
}
*/
template <typename T> bool alloc_dev(std::shared_ptr<T> &cuda_ptr, int elements)
{
    auto err = ::cudaSuccess;

    T *d_data;
    if (err = cudaMalloc(&d_data, sizeof(T) * elements) != ::cudaSuccess)
        return false;

    cuda_ptr = std::shared_ptr<T>(d_data, [](T *p) { cudaFree(p); });

    return true;
}
