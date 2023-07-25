#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>

#include <memory>
#include <stdexcept>

__forceinline__ static int div_up(int total, int grain)
{
    return (total + grain - 1) / grain;
}

template <typename T> static inline bool make_device_copy(std::shared_ptr<T> &cuda_ptr, T obj)
{
    T *d_data;
    auto err = cudaMalloc(&d_data, sizeof(T));

    if (err != cudaSuccess)
        return false;

    cudaMemcpy(d_data, &obj, sizeof(T), cudaMemcpyHostToDevice);

    cuda_ptr = std::shared_ptr<T>(d_data, [](T *data) { cudaFree(data); });

    return true;
}

template <typename T> static inline bool alloc_device(std::shared_ptr<T> &cuda_ptr, int elements)
{
    T *d_data;
    auto err = cudaMalloc(&d_data, sizeof(T) * elements);
    if (err != ::cudaSuccess)
        return false;

    cuda_ptr = std::shared_ptr<T>(d_data, [](T *p) { cudaFree(p); });

    return true;
}

__forceinline__ static void check_cuda_error(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " +
                                 cudaGetErrorString(err));
    }
}

template <typename T>
void print_device_vector(const thrust::device_vector<T> &d_vec, const std::string &what)
{
    std::cout << what << ": ";
    thrust::copy(d_vec.begin(), d_vec.end(), std::ostream_iterator<T>(std::cout, " "));
    std::cout << std::endl;
}
