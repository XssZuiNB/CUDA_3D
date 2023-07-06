#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <stdint.h>
#include <stdio.h>

#include "camera/camera_param.hpp"
#include "cuda_container/cuda_container.hpp"
#include "geometry/type.hpp"
#include "util/cuda_util.cuh"

#include "cuda_point_cloud.cuh"

__global__ void __kernel_cudaWarmUpGPU()
{
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    ind = ind + 1;
}

cudaError_t cuda_warm_up_gpu(uint8_t device_num)
{
    cudaSetDevice(device_num);

    __kernel_cudaWarmUpGPU<<<1, 1>>>();
    cudaDeviceSynchronize();
    return cudaGetLastError();
}

__device__ static void __transform_point_to_point(float to_point[3], const float from_point[3],
                                                  const gca::extrinsics &extrin)
{
    to_point[0] = extrin.rotation[0] * from_point[0] + extrin.rotation[3] * from_point[1] +
                  extrin.rotation[6] * from_point[2] + extrin.translation[0];
    to_point[1] = extrin.rotation[1] * from_point[0] + extrin.rotation[4] * from_point[1] +
                  extrin.rotation[7] * from_point[2] + extrin.translation[1];
    to_point[2] = extrin.rotation[2] * from_point[0] + extrin.rotation[5] * from_point[1] +
                  extrin.rotation[8] * from_point[2] + extrin.translation[2];
}

__device__ static void __depth_uv_to_xyz(const float uv[2], const float depth, float xyz[3],
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
    __shared__ gca::intrinsics depth_intrin_shared;
    __shared__ gca::intrinsics color_intrin_shared;
    __shared__ gca::extrinsics depth_to_color_extrin_shared;

    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        depth_intrin_shared = *depth_intrin;
        color_intrin_shared = *color_intrin;
        depth_to_color_extrin_shared = *depth_to_color_extrin;
    }

    __syncthreads();

    int depth_x = blockIdx.x * blockDim.x + threadIdx.x;
    int depth_y = blockIdx.y * blockDim.y + threadIdx.y;

    int depth_pixel_index = depth_y * width + depth_x;

    // Shared memory or texture memory loading of depth_data and color_data

    if (depth_x >= 0 && depth_x < width && depth_y >= 0 && depth_y < height)
    {
        // Extract depth value
        const uint16_t depth_value = depth_data[depth_pixel_index];

        if (depth_value == 0)
        {
            point_set_out[depth_pixel_index] = {0};
            return;
        }

        // Calculate depth_uv and depth_xyz
        float depth_uv[2] = {depth_x - 0.5f, depth_y - 0.5f};
        float depth_xyz[3];
        __depth_uv_to_xyz(depth_uv, depth_value, depth_xyz, depth_scale, depth_intrin_shared);

        // Calculate color_xyz
        float color_xyz[3];
        __transform_point_to_point(color_xyz, depth_xyz, depth_to_color_extrin_shared);

        // Calculate color_uv
        float color_uv[2];
        __xyz_to_color_uv(color_xyz, color_uv, color_intrin_shared);

        const int target_x = static_cast<int>(color_uv[0] + 0.5f);
        const int target_y = static_cast<int>(color_uv[1] + 0.5f);

        if (target_x >= 0 && target_x < width && target_y >= 0 && target_y < height)
        {
            gca::point_t p;
            p.x = depth_xyz[0];
            p.y = -depth_xyz[1];
            p.z = -depth_xyz[2];

            const int color_index = 3 * (target_y * width + target_x);
            p.b = color_data[color_index + 0];
            p.g = color_data[color_index + 1];
            p.r = color_data[color_index + 2];

            point_set_out[depth_pixel_index] = p;
        }
    }
}

cudaError_t gpu_make_point_set(gca::point_t *result, uint32_t width, const uint32_t height,
                               const gca::cuda_depth_frame &depth_data,
                               const gca::cuda_color_frame &color_data,
                               const gca::intrinsics &depth_intrin,
                               const gca::intrinsics &color_intrin,
                               const gca::extrinsics &depth_to_color_extrin,
                               const float depth_scale)
{
    std::shared_ptr<gca::intrinsics> depth_intrin_ptr;
    std::shared_ptr<gca::intrinsics> color_intrin_ptr;
    std::shared_ptr<gca::extrinsics> depth_to_color_extrin_ptr;
    std::shared_ptr<gca::point_t> result_ptr;

    auto depth_pixel_count = width * height;

    auto result_byte_size = sizeof(gca::point_t) * depth_pixel_count;

    if (!make_device_copy(depth_intrin_ptr, depth_intrin))
        return cudaGetLastError();
    if (!make_device_copy(color_intrin_ptr, color_intrin))
        return cudaGetLastError();
    if (!make_device_copy(depth_to_color_extrin_ptr, depth_to_color_extrin))
        return cudaGetLastError();

    if (!alloc_dev(result_ptr, result_byte_size))
        return cudaGetLastError();

    dim3 threads(32, 32);
    dim3 depth_blocks(div_up(width, threads.x), div_up(height, threads.y));

    __kernel_make_pointcloud<<<depth_blocks, threads>>>(
        result_ptr.get(), width, height, depth_data.data(), color_data.data(),
        depth_intrin_ptr.get(), color_intrin_ptr.get(), depth_to_color_extrin_ptr.get(),
        depth_scale);

    cudaDeviceSynchronize();

    cudaMemcpy(result, result_ptr.get(), result_byte_size, cudaMemcpyDeviceToHost);

    return cudaGetLastError();
}
