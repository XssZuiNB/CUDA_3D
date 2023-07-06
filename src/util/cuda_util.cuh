#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <stdexcept>

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

template <typename T> static inline bool alloc_device(std::shared_ptr<T> &cuda_ptr, int elements)
{
    T *d_data;
    auto err = cudaMalloc(&d_data, sizeof(T) * elements);
    if (err != ::cudaSuccess)
        return false;

    cuda_ptr = std::shared_ptr<T>(d_data, [](T *p) { cudaFree(p); });

    return true;
}

inline static void check_cuda_error(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " +
                                 cudaGetErrorString(err));
    }
}

inline static void cuda_print_devices()
{
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    }
}