#include "camera/camera_param.hpp"
#include "cuda_container/cuda_container.hpp"
#include "geometry/cuda_point_cloud_factory.cuh"
#include "geometry/geometry_util.cuh"
#include "geometry/type.hpp"
#include "util/cuda_util.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>

namespace gca
{
/************************************ CUDA bilateral_filter  ************************************/

__forceinline__ __device__ static float __gaussian(float x, float sigma)
{
    return exp(-(x * x) / (2 * sigma * sigma));
}

__device__ static float __bilateral_filter(const uint16_t *input, uint32_t input_width,
                                           uint32_t input_height, int index_x, int index_y,
                                           int filter_radius, float sigma_space, float sigma_depth)
{
    float sum_weight = 0.0f;
    float sum = 0.0f;

    for (int dy = -filter_radius; dy <= filter_radius; ++dy)
    {
        for (int dx = -filter_radius; dx <= filter_radius; ++dx)
        {
            int nx = index_x + dx;
            int ny = index_y + dy;

            if (nx >= 0 && nx < input_width && ny >= 0 && ny < input_height)
            {
                float weight = __gaussian(sqrtf(dx * dx + dy * dy), sigma_space) *
                               __gaussian(abs(input[index_y * input_width + index_x] -
                                              input[ny * input_width + nx]),
                                          sigma_depth);

                sum_weight += weight;
                sum += weight * input[ny * input_width + nx];
            }
        }
    }

    return sum / sum_weight;
}

/****************** Create point cloud from rgbd, include invalid point remove ******************/

__forceinline__ __device__ static void __transform_point_to_point(float to_point[3],
                                                                  const float from_point[3],
                                                                  const gca::extrinsics &extrin)
{
    to_point[0] = extrin.rotation[0] * from_point[0] + extrin.rotation[3] * from_point[1] +
                  extrin.rotation[6] * from_point[2] + extrin.translation[0];
    to_point[1] = extrin.rotation[1] * from_point[0] + extrin.rotation[4] * from_point[1] +
                  extrin.rotation[7] * from_point[2] + extrin.translation[1];
    to_point[2] = extrin.rotation[2] * from_point[0] + extrin.rotation[5] * from_point[1] +
                  extrin.rotation[8] * from_point[2] + extrin.translation[2];
}

__forceinline__ __device__ static void __depth_uv_to_xyz(const float uv[2], const float depth,
                                                         float xyz[3],
                                                         const gca::intrinsics &depth_intrin)
{
    auto z = depth;
    xyz[2] = z;
    xyz[0] = (uv[0] - depth_intrin.cx) * z / depth_intrin.fx;
    xyz[1] = (uv[1] - depth_intrin.cy) * z / depth_intrin.fy;
}

__forceinline__ __device__ static void __xyz_to_color_uv(const float xyz[3], float uv[2],
                                                         const gca::intrinsics &color_intrin)
{
    uv[0] = (xyz[0] * color_intrin.fx / xyz[2]) + color_intrin.cx;
    uv[1] = (xyz[1] * color_intrin.fy / xyz[2]) + color_intrin.cy;
}

__global__ static void __kernel_make_pointcloud_Z16_BGR8(
    gca::point_t *point_set_out, const uint32_t width, const uint32_t height,
    const uint16_t *depth_frame_data, const uint8_t *color_frame_data,
    const gca::intrinsics *depth_intrin_ptr, const gca::intrinsics *color_intrin_ptr,
    const gca::extrinsics *depth_to_color_extrin_ptr, const float depth_scale, float threshold_min,
    float threshold_max, bool if_bilateral_filter = false)
{

    __shared__ gca::intrinsics depth_intrin_shared;
    __shared__ gca::intrinsics color_intrin_shared;
    __shared__ gca::extrinsics depth_to_color_extrin_shared;

    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        depth_intrin_shared = *depth_intrin_ptr;
        color_intrin_shared = *color_intrin_ptr;
        depth_to_color_extrin_shared = *depth_to_color_extrin_ptr;
    }

    __syncthreads();

    auto depth_x = blockIdx.x * blockDim.x + threadIdx.x;
    auto depth_y = blockIdx.y * blockDim.y + threadIdx.y;
    auto depth_pixel_index = depth_y * width + depth_x;

    if (depth_x < width && depth_y < height)
    {
        float depth_value;
        if (if_bilateral_filter)
            depth_value =
                __bilateral_filter(depth_frame_data, width, height, depth_x, depth_y, 7, 1, 50) *
                depth_scale;
        else
            depth_value = __ldg(&depth_frame_data[depth_pixel_index]) * depth_scale;

        if (depth_value <= 0.0 || depth_value < threshold_min || depth_value > threshold_max)
        {
            point_set_out[depth_pixel_index].property = gca::point_property::invalid;
            return;
        }

        float depth_uv[2] = {depth_x - 0.5f, depth_y - 0.5f};
        float depth_xyz[3];
        __depth_uv_to_xyz(depth_uv, depth_value, depth_xyz, depth_intrin_shared);

        float color_xyz[3];
        __transform_point_to_point(color_xyz, depth_xyz, depth_to_color_extrin_shared);

        float color_uv[2];
        __xyz_to_color_uv(color_xyz, color_uv, color_intrin_shared);

        const int target_x = static_cast<int>(color_uv[0] + 0.5f);
        const int target_y = static_cast<int>(color_uv[1] + 0.5f);
        if (target_x < 0 || target_x >= width || target_y < 0 || target_y >= height)
        {
            point_set_out[depth_pixel_index].property = gca::point_property::invalid;
            return;
        }

        gca::point_t p;

        p.coordinates.x = depth_xyz[0];
        p.coordinates.y = depth_xyz[1];
        p.coordinates.z = depth_xyz[2];

        const int color_index = 3 * (target_y * width + target_x);
        p.color.b = float(__ldg(&color_frame_data[color_index + 0])) / 255;
        p.color.g = float(__ldg(&color_frame_data[color_index + 1])) / 255;
        p.color.r = float(__ldg(&color_frame_data[color_index + 2])) / 255;

        p.property = gca::point_property::inactive;

        point_set_out[depth_pixel_index] = p;
    }
}

/************************* For Debug, not be used in point cloud class **************************/
bool cuda_make_point_cloud(std::vector<gca::point_t> &result,
                           const gca::cuda_depth_frame &cuda_depth_container,
                           const gca::cuda_color_frame &cuda_color_container,
                           const gca::cuda_camera_param &param, float threshold_min_in_meter,
                           float threshold_max_in_meter)
{
    auto depth_intrin_ptr = param.get_depth_intrinsics_ptr();
    auto color_intrin_ptr = param.get_color_intrinsics_ptr();
    auto depth2color_extrin_ptr = param.get_depth2color_extrinsics_ptr();
    auto width = param.get_width();
    auto height = param.get_height();
    auto depth_scale = param.get_depth_scale();

    if (!depth_intrin_ptr || !color_intrin_ptr || !depth2color_extrin_ptr || !width || !height)
        return false;

    if (depth_scale - 0.0 < 0.0001)
        return false;

    auto depth_pixel_count = width * height;
    auto result_byte_size = sizeof(gca::point_t) * depth_pixel_count;

    std::shared_ptr<gca::point_t> result_ptr;
    if (!alloc_device(result_ptr, result_byte_size))
        return false;

    dim3 threads(32, 32);
    dim3 depth_blocks(div_up(width, threads.x), div_up(height, threads.y));

    __kernel_make_pointcloud_Z16_BGR8<<<depth_blocks, threads>>>(
        result_ptr.get(), width, height, cuda_depth_container.get_depth_frame_vec().data().get(),
        cuda_color_container.get_color_frame_vec().data().get(), depth_intrin_ptr, color_intrin_ptr,
        depth2color_extrin_ptr, depth_scale, threshold_min_in_meter, threshold_max_in_meter);

    if (cudaDeviceSynchronize() != cudaSuccess)
        return false;

    cudaMemcpy(result.data(), result_ptr.get(), result_byte_size, cudaMemcpyDefault);

    return true;
}

/************************** This overload is used in point cloud class **************************/
::cudaError_t cuda_make_point_cloud(thrust::device_vector<gca::point_t> &result,
                                    const gca::cuda_depth_frame &cuda_depth_container,
                                    const gca::cuda_color_frame &cuda_color_container,
                                    const gca::cuda_camera_param &param,
                                    float threshold_min_in_meter, float threshold_max_in_meter)
{
    auto depth_intrin_ptr = param.get_depth_intrinsics_ptr();
    auto color_intrin_ptr = param.get_color_intrinsics_ptr();
    auto depth2color_extrin_ptr = param.get_depth2color_extrinsics_ptr();
    auto width = param.get_width();
    auto height = param.get_height();
    auto depth_scale = param.get_depth_scale();
    auto depth_frame_cuda_ptr =
        thrust::raw_pointer_cast(cuda_depth_container.get_depth_frame_vec().data());
    auto color_frame_cuda_ptr =
        thrust::raw_pointer_cast(cuda_color_container.get_color_frame_vec().data());

    if (!depth_intrin_ptr || !color_intrin_ptr || !depth2color_extrin_ptr || !width || !height)
        return ::cudaErrorInvalidValue;

    if (depth_scale - 0.0 < 0.00001)
        return ::cudaErrorInvalidValue;

    auto depth_pixel_count = width * height;
    result.resize(depth_pixel_count);

    dim3 threads(32, 32);
    dim3 depth_blocks(div_up(width, threads.x), div_up(height, threads.y));

    __kernel_make_pointcloud_Z16_BGR8<<<depth_blocks, threads>>>(
        thrust::raw_pointer_cast(result.data()), width, height, depth_frame_cuda_ptr,
        color_frame_cuda_ptr, depth_intrin_ptr, color_intrin_ptr, depth2color_extrin_ptr,
        depth_scale, threshold_min_in_meter,
        threshold_max_in_meter); // didnt use bilateral filter, later maybe a compare to see if it
                                 // is needed

    auto err = cudaDeviceSynchronize();
    if (err != ::cudaSuccess)
        return err;

    remove_invalid_points(result);
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
        return err;

    return ::cudaSuccess;
}
} // namespace gca
