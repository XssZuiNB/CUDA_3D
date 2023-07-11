#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <iostream>

#include "gpu_check.hpp"

__global__ void __kernel_cudaWarmUpGPU()
{
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    ind = ind + 1;
}

bool cuda_warm_up_gpu(uint8_t device_num)
{
    auto err = cudaSetDevice(device_num);
    if (err != cudaSuccess)
        return false;

    __kernel_cudaWarmUpGPU<<<1, 1>>>();

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        return false;

    return true;
}

void cuda_print_devices()
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