#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <stdint.h>
#include <stdio.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/gather.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/sort.h>

#include "camera/camera_param.hpp"
#include "cuda_container/cuda_container.hpp"
#include "geometry/cuda_point_cloud_factory.cuh"
#include "geometry/geometry_util.cuh"
#include "geometry/type.hpp"
#include "util/cuda_util.cuh"

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

#pragma unroll
    for (int dy = -filter_radius; dy <= filter_radius; ++dy)
    {
#pragma unroll
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

    int depth_x = blockIdx.x * blockDim.x + threadIdx.x;
    int depth_y = blockIdx.y * blockDim.y + threadIdx.y;
    int depth_pixel_index = depth_y * width + depth_x;

    // Shared memory or texture memory loading of depth_frame_data and color_frame_data

    if (depth_x >= 0 && depth_x < width && depth_y >= 0 && depth_y < height)
    {
        float depth_value;
        // Extract depth value
        if (if_bilateral_filter)
            depth_value =
                __bilateral_filter(depth_frame_data, width, height, depth_x, depth_y, 7, 1, 50) *
                depth_scale;
        else
            depth_value = depth_frame_data[depth_pixel_index] * depth_scale;

        if (depth_value <= 0.0 || depth_value < threshold_min || depth_value > threshold_max)
        {
            point_set_out[depth_pixel_index].property = gca::point_property::invalid;
            return;
        }

        // Calculate depth_uv and depth_xyz
        float depth_uv[2] = {depth_x - 0.5f, depth_y - 0.5f};
        float depth_xyz[3];
        __depth_uv_to_xyz(depth_uv, depth_value, depth_xyz, depth_intrin_shared);

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

            p.coordinates.x = depth_xyz[0];
            p.coordinates.y = depth_xyz[1];
            p.coordinates.z = depth_xyz[2];

            const int color_index = 3 * (target_y * width + target_x);
            p.b = color_frame_data[color_index + 0];
            p.g = color_frame_data[color_index + 1];
            p.r = color_frame_data[color_index + 2];

            p.property = gca::point_property::inactive;

            point_set_out[depth_pixel_index] = p;
        }
        else
        {
            point_set_out[depth_pixel_index].property = gca::point_property::invalid;
            return;
        }
    }
}

/******************************* Functor check if a point is valid ******************************/
struct check_is_valid_point_functor
{
    __host__ __device__ __forceinline__ bool operator()(gca::point_t p)
    {
        return p.property != gca::point_property::invalid;
    }
};

__forceinline__ static void remove_invalid_points(thrust::device_vector<gca::point_t> &result)
{
    thrust::device_vector<gca::point_t> temp(result.size());
    auto new_size = thrust::copy_if(result.begin(), result.end(), temp.begin(),
                                    check_is_valid_point_functor()) -
                    temp.begin();
    temp.resize(new_size);

    result.swap(temp);
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
        result_ptr.get(), width, height, cuda_depth_container.data(), cuda_color_container.data(),
        depth_intrin_ptr, color_intrin_ptr, depth2color_extrin_ptr, depth_scale,
        threshold_min_in_meter, threshold_max_in_meter);

    if (cudaDeviceSynchronize() != cudaSuccess)
        return false;

    cudaMemcpy(result.data(), result_ptr.get(), result_byte_size, cudaMemcpyDefault);

    return true;
}

/************************** This overload is used in point cloud class **************************/
bool cuda_make_point_cloud(thrust::device_vector<gca::point_t> &result,
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
    result.resize(depth_pixel_count);

    dim3 threads(32, 32);
    dim3 depth_blocks(div_up(width, threads.x), div_up(height, threads.y));

    __kernel_make_pointcloud_Z16_BGR8<<<depth_blocks, threads>>>(
        result.data().get(), width, height, cuda_depth_container.data(),
        cuda_color_container.data(), depth_intrin_ptr, color_intrin_ptr, depth2color_extrin_ptr,
        depth_scale, threshold_min_in_meter, threshold_max_in_meter, true);

    if (cudaDeviceSynchronize() != ::cudaSuccess)
        return false;

    remove_invalid_points(result);

    return true;
}

/****************** Voxel grid Downsampling with Eigen Vector3f as coordinates ******************/
/**************************** Useful functors for thrust algorithms  ****************************/

struct add_points_functor
{
    __device__ gca::point_t operator()(const gca::point_t &first, const gca::point_t &second)
    {

        return gca::point_t{.coordinates{
                                .x = first.coordinates.x + second.coordinates.x,
                                .y = first.coordinates.y + second.coordinates.y,
                                .z = first.coordinates.z + second.coordinates.z,
                            },
                            .r = first.r + second.r,
                            .g = first.g + second.g,
                            .b = first.b + second.b,
                            .property = gca::point_property::inactive};
    }
};

struct compute_points_mean_functor
{
    compute_points_mean_functor(const uint32_t min_points_n_threshold)
        : m_threshold(min_points_n_threshold)
    {
    }

    const uint32_t m_threshold;

    __device__ gca::point_t operator()(const gca::point_t &points_sum, const uint32_t n)
    {
        if (n <= m_threshold)
        {
            return gca::point_t{.property = gca::point_property::invalid};
        }

        return gca::point_t{.coordinates{
                                .x = points_sum.coordinates.x / n,
                                .y = points_sum.coordinates.y / n,
                                .z = points_sum.coordinates.z / n,
                            },
                            .r = points_sum.r / n,
                            .g = points_sum.g / n,
                            .b = points_sum.b / n,
                            .property = gca::point_property::inactive};
    }
};

/*********************************** Voxel grid down sampling ***********************************/

::cudaError_t cuda_voxel_grid_downsample(thrust::device_vector<gca::point_t> &result_points,
                                         const thrust::device_vector<gca::point_t> &src_points,
                                         const float3 &voxel_grid_min_bound, const float voxel_size,
                                         const uint32_t min_points_num_in_one_voxel)
{
    auto n_points = src_points.size();
    if (result_points.size() != n_points)
    {
        return ::cudaErrorInvalidValue;
    }

    thrust::device_vector<int3> keys(n_points);

    thrust::transform(src_points.begin(), src_points.end(), keys.begin(),
                      compute_voxel_key_functor(voxel_grid_min_bound, voxel_size));
    auto err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    thrust::device_vector<gca::point_t> sorted_point_cloud(n_points);
    thrust::device_vector<size_t> index_vec(n_points);
    thrust::sequence(index_vec.begin(), index_vec.end());
    thrust::sort_by_key(keys.begin(), keys.end(), index_vec.begin(), compare_voxel_key_functor());
    thrust::gather(index_vec.begin(), index_vec.end(), src_points.begin(),
                   sorted_point_cloud.begin());
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    /* The following line of sorting code is only for memory. RIP
       man.... By using it the sorting time is 5ms. But this is
       too slow for me. So i made a vector which contains only the
       index of the points and only let it be sorted. And than I
       tried to get every point by using the src_points vector and
       sorted index. This really works and the whole process runs
       less than 1.5 ms. I'm very happy about it!!! */
    // thrust::sort_by_key(keys.begin(), keys.end(),
    // sorted_point_cloud.begin(), compare_voxel_key_functor());

    thrust::device_vector<uint32_t> points_counter_per_voxel(n_points, 1);
    thrust::device_vector<uint32_t> result_points_counter(n_points);

    auto end_iter_of_points =
        thrust::reduce_by_key(keys.begin(), keys.end(), sorted_point_cloud.begin(),
                              thrust::make_discard_iterator(), result_points.begin(),
                              voxel_key_equal_functor(), add_points_functor())
            .second;
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    auto end_iter_of_points_counter =
        thrust::reduce_by_key(keys.begin(), keys.end(), points_counter_per_voxel.begin(),
                              thrust::make_discard_iterator(), result_points_counter.begin(),
                              voxel_key_equal_functor())
            .second;
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    auto new_n_points = end_iter_of_points - result_points.begin();
    if (new_n_points != (end_iter_of_points_counter - result_points_counter.begin()))
    {
        return ::cudaErrorInvalidValue;
    }

    result_points.resize(new_n_points);
    result_points_counter.resize(new_n_points);

    thrust::transform(result_points.begin(), result_points.end(), result_points_counter.begin(),
                      result_points.begin(),
                      compute_points_mean_functor(min_points_num_in_one_voxel));
    if (err != ::cudaSuccess)
    {
        return err;
    }

    remove_invalid_points(result_points);

    return ::cudaSuccess;
}
} // namespace gca
