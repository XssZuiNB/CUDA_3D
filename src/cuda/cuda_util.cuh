#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <memory>

static inline int div_up(int total, int grain)
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

template <typename T> static inline bool alloc_dev(std::shared_ptr<T> &cuda_ptr, int elements)
{
    T *d_data;
    auto err = cudaMalloc(&d_data, sizeof(T) * elements);
    if (err != ::cudaSuccess)
        return false;

    cuda_ptr = std::shared_ptr<T>(d_data, [](T *p) { cudaFree(p); });

    return true;
}