#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdint.h>

#include <memory>
#include <stdexcept>

#include "cuda_util.cuh"
#include "type.hpp"

__device__ static void cuda_transform_point_to_point(float to_point[3],
                                                     const gca::extrinsics *extrin,
                                                     const float from_point[3])
{
    to_point[0] = extrin->rotation[0] * from_point[0] + extrin->rotation[1] * from_point[1] +
                  extrin->rotation[2] * from_point[2] + extrin->translation[0];
    to_point[1] = extrin->rotation[3] * from_point[0] + extrin->rotation[4] * from_point[1] +
                  extrin->rotation[5] * from_point[2] + extrin->translation[1];
    to_point[2] = extrin->rotation[6] * from_point[0] + extrin->rotation[7] * from_point[1] +
                  extrin->rotation[8] * from_point[2] + extrin->translation[2];
}

/*
__global__ void kernel_color_to_depth(uint8_t *aligned_out, const uint32_t width,
                                      const uint32_t height const uint16_t *depth_in,
                                      const uint8_t *color_in,
                                      const gca::extrinsics *depth_to_color,
                                      const float depth_scale)
{
    int depth_x = blockIdx.x * blockDim.x + threadIdx.x;

    int depth_y = blockIdx.y * blockDim.y + threadIdx.y;

    int depth_pixel_index = depth_y * width + depth_x;

    if (depth_x >= 0 && depth_x < width && depth_y >= 0 && depth_y < height)
    {
        float uv[2] = {depth_x, depth_y};
        float xyz[3];

        const float depth_value = depth_in[depth_pixel_index] * depth_scale;

        if (depth_value == 0)
            return;

        cuda_deproject_pixel_to_point_with_distortion(depth_intrin, uv, xyz, depth_value);

        float target_xyz[3];
        cuda_transform_point_to_point(target_xyz, depth_to_color, xyz);

        if (target_xyz[2] <= 0.f)
        {
            return;
        }

        float xy_for_projection[2];
        xy_for_projection[0] = target_xyz[0] / target_xyz[2];
        xy_for_projection[1] = target_xyz[1] / target_xyz[2];

        float target_uv[2] = {-1.f, -1.f};

        cuda_project_pixel_to_point_with_distortion(color_intrin, xy_for_projection, target_uv);

        const int target_x = target_uv[0] + 0.5f;
        const int target_y = target_uv[1] + 0.5f;

        if (target_x >= 0 && target_x < color_intrin->width && target_y >= 0 &&
            target_y < color_intrin->height)
        {
            const int from_offset = 3 * depth_pixel_index;
            const int to_offset = 3 * (target_y * color_intrin->width + target_x);

            aligned_out[from_offset + 0] = color_in[to_offset + 0];
            aligned_out[from_offset + 1] = color_in[to_offset + 1];
            aligned_out[from_offset + 2] = color_in[to_offset + 2];
        }
    }
}

void cuda_k4a_align::align_color_to_depth(uint8_t *aligned_out, const uint16_t *depth_in,
                                          const uint8_t *color_in, float depth_scale,
                                          const k4a_calibration_t &calibration)
{
    cuda_align::cuda_intrinsics depth_intrinsic(calibration.depth_camera_calibration);
    cuda_align::cuda_intrinsics color_intrinsic(calibration.color_camera_calibration);
    cuda_align::cuda_extrinsics depth_to_color(
        calibration.extrinsics[K4A_CALIBRATION_TYPE_DEPTH][K4A_CALIBRATION_TYPE_COLOR]);

    const int depth_pixel_count = depth_intrinsic.width * depth_intrinsic.height;
    const int color_pixel_count = color_intrinsic.width * color_intrinsic.height;
    const int aligned_pixel_count = depth_pixel_count;

    const int depth_byte_size = depth_pixel_count * sizeof(uint16_t);
    const int color_byte_size = color_pixel_count * sizeof(uint8_t) * 3;
    const int aligned_byte_size = aligned_pixel_count * sizeof(uint8_t) * 3;

    // allocate and copy objects to cuda device memory
    if (!d_depth_intrinsics)
        d_depth_intrinsics = make_device_copy(depth_intrinsic);
    if (!d_color_intrinsics)
        d_color_intrinsics = make_device_copy(color_intrinsic);
    if (!d_depth_color_extrinsics)
        d_depth_color_extrinsics = make_device_copy(depth_to_color);

    if (!d_depth_in)
        d_depth_in = alloc_dev<uint16_t>(depth_pixel_count);
    cudaMemcpy(d_depth_in.get(), depth_in, depth_byte_size, cudaMemcpyHostToDevice);
    if (!d_color_in)
        d_color_in = alloc_dev<uint8_t>(color_pixel_count * 3);
    cudaMemcpy(d_color_in.get(), color_in, color_byte_size, cudaMemcpyHostToDevice);
    if (!d_aligned_out)
        d_aligned_out = alloc_dev<uint8_t>(aligned_byte_size);
    cudaMemset(d_aligned_out.get(), 0, aligned_byte_size);

    // config threads
    dim3 threads(CUDA_THREADS_PER_BLOCK, CUDA_THREADS_PER_BLOCK);
    dim3 depth_blocks(divUp(depth_intrinsic.width, threads.x),
                      divUp(depth_intrinsic.height, threads.y));

    kernel_color_to_depth<<<depth_blocks, threads>>>(
        d_aligned_out.get(), d_depth_in.get(), d_color_in.get(), d_depth_intrinsics.get(),
        d_color_intrinsics.get(), d_depth_color_extrinsics.get(), depth_scale);

    cudaDeviceSynchronize();

    cudaMemcpy(aligned_out, d_aligned_out.get(), aligned_byte_size, cudaMemcpyDeviceToHost);
}
*/