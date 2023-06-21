#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>

#include <memory>

#include <stdint.h>

#include "cuda_point_cloud.cuh"
#include "cuda_util.cuh"

__global__ void kernel_cudaWarmUpGPU()
{
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    ind = ind + 1;
}

cudaError_t cudaWarmUpGPU()
{
    kernel_cudaWarmUpGPU<<<1, 1>>>();
    cudaDeviceSynchronize();
    return cudaGetLastError();
}

static inline int divUp(int total, int grain)
{
    return (total + grain - 1) / grain;
}

template <typename T> bool make_device_copy(std::shared_ptr<T> &cuda_ptr, T obj)
{
    T *d_data;
    auto err = cudaMalloc(&d_data, sizeof(T));

    if (err != cudaSuccess)
        return false;

    cudaMemcpy(d_data, &obj, sizeof(T), cudaMemcpyHostToDevice);

    cuda_ptr = std::shared_ptr<T>(d_data, [](T *data) { cudaFree(data); });

    return true;
}

template <typename T> bool alloc_dev(std::shared_ptr<T> &cuda_ptr, int elements)
{
    T *d_data;
    auto err = cudaMalloc(&d_data, sizeof(T) * elements);
    if (err != ::cudaSuccess)
        return false;

    cuda_ptr = std::shared_ptr<T>(d_data, [](T *p) { cudaFree(p); });

    return true;
}

__device__ static void __transform_point_to_point(float to_point[3], const float from_point[3],
                                                  const gca::extrinsics &extrin)
{
    to_point[0] = extrin.rotation[0] * from_point[0] + extrin.rotation[1] * from_point[1] +
                  extrin.rotation[2] * from_point[2] + extrin.translation[0];
    to_point[1] = extrin.rotation[3] * from_point[0] + extrin.rotation[4] * from_point[1] +
                  extrin.rotation[5] * from_point[2] + extrin.translation[1];
    to_point[2] = extrin.rotation[6] * from_point[0] + extrin.rotation[7] * from_point[1] +
                  extrin.rotation[8] * from_point[2] + extrin.translation[2];
}

__device__ static void __depth_uv_to_xyz(const int uv[2], const float depth, float xyz[3],
                                         const float depth_scale,
                                         const gca::intrinsics &depth_intrin)
{
    auto z = depth * depth_scale;
    xyz[2] = z;
    xyz[0] = (uv[0] - depth_intrin.cx) * z / depth_intrin.fx;
    xyz[1] = (uv[1] - depth_intrin.cy) * z / depth_intrin.fy;
}

__device__ static void __xyz_to_color_uv(const float xyz[3], float uv[2],
                                         const gca::intrinsics &color_intrin)
{

    uv[0] = (xyz[0] * color_intrin.fx / xyz[2]) + color_intrin.cx;
    uv[1] = (xyz[1] * color_intrin.fy / xyz[2]) + color_intrin.cy;
}

__global__ void __kernel_make_pointcloud(gca::point_t *point_set_out, const uint32_t width,
                                         const uint32_t height, const uint16_t *depth_data,
                                         const uint8_t *color_data,
                                         const gca::intrinsics *depth_intrin,
                                         const gca::intrinsics *color_intrin,
                                         const gca::extrinsics *depth_to_color_extrin,
                                         const float depth_scale)
{
    int depth_x = blockIdx.x * blockDim.x + threadIdx.x;
    int depth_y = blockIdx.y * blockDim.y + threadIdx.y;

    int depth_pixel_index = depth_y * width + depth_x;

    if (depth_x >= 0 && depth_x < width && depth_y >= 0 && depth_y < height)
    {
        int depth_uv[2] = {depth_x, depth_y};
        float depth_xyz[3];

        const uint16_t depth_value = depth_data[depth_pixel_index];
        if (depth_value == 0)
        {
            point_set_out[depth_pixel_index] = {0};
            return;
        }

        __depth_uv_to_xyz(depth_uv, depth_value, depth_xyz, depth_scale, *depth_intrin);

        float color_xyz[3];
        __transform_point_to_point(color_xyz, depth_xyz, *depth_to_color_extrin);

        float color_uv[2];

        __xyz_to_color_uv(color_xyz, color_uv, *color_intrin);

        const int target_x = color_uv[0] + 0.5f;
        const int target_y = color_uv[1] + 0.5f;

        if (target_x >= 0 && target_x < width && target_y >= 0 && target_y < height)
        {
            gca::point_t p;
            p.x = depth_xyz[0];
            p.y = -depth_xyz[1];
            p.z = -depth_xyz[2];

            const int color_offset = 3 * depth_pixel_index;
            p.b = color_data[color_offset + 0];
            p.g = color_data[color_offset + 1];
            p.r = color_data[color_offset + 2];

            point_set_out[depth_pixel_index] = p;
        }
    }
}

bool gpu_make_point_set(gca::point_t *result, uint32_t width, const uint32_t height,
                        const uint16_t *depth_data, const uint8_t *color_data,
                        const gca::intrinsics &depth_intrin, const gca::intrinsics &color_intrin,
                        const gca::extrinsics &depth_to_color_extrin, const float depth_scale)
{
    cudaSetDevice(0);

    std::shared_ptr<gca::intrinsics> depth_intrin_ptr;
    std::shared_ptr<gca::intrinsics> color_intrin_ptr;
    std::shared_ptr<gca::extrinsics> depth_to_color_extrin_ptr;

    if (!make_device_copy(depth_intrin_ptr, depth_intrin))
        return false;
    if (!make_device_copy(color_intrin_ptr, color_intrin))
        return false;
    if (!make_device_copy(depth_to_color_extrin_ptr, depth_to_color_extrin))
        return false;

    auto depth_pixel_count = width * height;
    auto depth_byte_size = sizeof(uint16_t) * depth_pixel_count;
    auto color_byte_size = sizeof(uint8_t) * depth_pixel_count * 3;

    std::shared_ptr<uint16_t> depth_frame_ptr;
    std::shared_ptr<uint8_t> color_frame_ptr;

    if (!alloc_dev(depth_frame_ptr, depth_pixel_count))
        return false;
    cudaMemcpy(depth_frame_ptr.get(), depth_data, depth_byte_size, cudaMemcpyHostToDevice);

    if (!alloc_dev(color_frame_ptr, depth_pixel_count * 3))
        return false;
    cudaMemcpy(color_frame_ptr.get(), color_data, color_byte_size, cudaMemcpyHostToDevice);

    std::shared_ptr<gca::point_t> result_ptr;
    if (!alloc_dev(result_ptr, depth_pixel_count))
        return false;

    // config threads
    dim3 threads(16, 16);
    dim3 depth_blocks(divUp(width, threads.x), divUp(height, threads.y));

    __kernel_make_pointcloud<<<depth_blocks, threads>>>(
        result_ptr.get(), width, height, depth_frame_ptr.get(), color_frame_ptr.get(),
        depth_intrin_ptr.get(), color_intrin_ptr.get(), depth_to_color_extrin_ptr.get(),
        depth_scale);

    cudaDeviceSynchronize();

    cudaMemcpy(result, result_ptr.get(), sizeof(gca::point_t) * depth_pixel_count,
               cudaMemcpyDeviceToHost);

    return true;
}
