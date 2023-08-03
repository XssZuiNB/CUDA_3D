#include "geometry/cuda_voxel_grid_down_sample.cuh"
#include "geometry/geometry_util.cuh"
#include "geometry/type.hpp"
#include "util/cuda_util.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

namespace gca
{
/**************************** Useful functors for thrust algorithms  ****************************/

struct compute_voxel_key_functor
{
    compute_voxel_key_functor(const float3 &voxel_grid_min_bound, const float voxel_size)
        : m_voxel_grid_min_bound(voxel_grid_min_bound)
        , m_voxel_size(voxel_size)
    {
    }

    const float3 m_voxel_grid_min_bound;
    const float m_voxel_size;

    __forceinline__ __device__ int3 operator()(const gca::point_t &point)
    {
        int3 ref_coord;
        ref_coord.x =
            __float2int_rd((point.coordinates.x - m_voxel_grid_min_bound.x) / m_voxel_size);
        ref_coord.y =
            __float2int_rd((point.coordinates.y - m_voxel_grid_min_bound.y) / m_voxel_size);
        ref_coord.z =
            __float2int_rd((point.coordinates.z - m_voxel_grid_min_bound.z) / m_voxel_size);
        return ref_coord;
    }
};

struct compare_voxel_key_functor : public thrust::binary_function<int3, int3, bool>
{
    __forceinline__ __device__ bool operator()(const int3 &lhs, const int3 &rhs) const
    {
        if (lhs.x != rhs.x)
            return lhs.x < rhs.x;

        else if (lhs.y != rhs.y)
            return lhs.y < rhs.y;

        else if (lhs.z != rhs.z)
            return lhs.z < rhs.z;

        return false;
    }
};

struct voxel_key_equal_functor : public thrust::binary_function<int3, int3, bool>
{
    __forceinline__ __device__ bool operator()(const int3 &lhs, const int3 &rhs) const
    {
        return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
    }
};

struct add_points_functor
{
    __forceinline__ __device__ gca::point_t operator()(const gca::point_t &first,
                                                       const gca::point_t &second)
    {
        return gca::point_t{.coordinates{
                                .x = first.coordinates.x + second.coordinates.x,
                                .y = first.coordinates.y + second.coordinates.y,
                                .z = first.coordinates.z + second.coordinates.z,
                            },
                            .color{
                                .r = first.color.r + second.color.r,
                                .g = first.color.g + second.color.g,
                                .b = first.color.b + second.color.b,
                            },
                            .property = gca::point_property::inactive};
    }
};

struct compute_points_mean_functor
{
    __forceinline__ __device__ gca::point_t operator()(const gca::point_t &points_sum,
                                                       const gca::counter_t n)
    {
        return gca::point_t{.coordinates{
                                .x = points_sum.coordinates.x / n,
                                .y = points_sum.coordinates.y / n,
                                .z = points_sum.coordinates.z / n,
                            },
                            .color{
                                .r = points_sum.color.r / n,
                                .g = points_sum.color.g / n,
                                .b = points_sum.color.b / n,
                            },
                            .property = gca::point_property::inactive};
    }
};

/*********************************** Voxel grid down sampling ***********************************/

::cudaError_t cuda_voxel_grid_downsample(thrust::device_vector<gca::point_t> &result_points,
                                         const thrust::device_vector<gca::point_t> &src_points,
                                         const float3 &voxel_grid_min_bound, const float voxel_size,
                                         ::cudaStream_t stream)
{
    auto exec_policy = thrust::cuda::par.on(stream);

    auto n_points = src_points.size();
    if (result_points.size() != n_points)
    {
        return ::cudaErrorInvalidValue;
    }

    thrust::device_vector<int3> keys(n_points);
    thrust::transform(exec_policy, src_points.begin(), src_points.end(), keys.begin(),
                      compute_voxel_key_functor(voxel_grid_min_bound, voxel_size));
    auto err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    thrust::device_vector<gca::index_t> index_vec(n_points);
    thrust::sequence(exec_policy, index_vec.begin(), index_vec.end());
    thrust::sort_by_key(exec_policy, keys.begin(), keys.end(), index_vec.begin(),
                        compare_voxel_key_functor());
    auto get_point_with_sorted_index_iter =
        thrust::make_permutation_iterator(src_points.begin(), index_vec.begin());
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    auto end_iter_of_points =
        thrust::reduce_by_key(exec_policy, keys.begin(), keys.end(),
                              get_point_with_sorted_index_iter, thrust::make_discard_iterator(),
                              result_points.begin(), voxel_key_equal_functor(),
                              add_points_functor())
            .second;
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    thrust::device_vector<gca::counter_t> points_counter_per_voxel(n_points, 1);
    auto end_iter_of_points_counter =
        thrust::reduce_by_key(exec_policy, keys.begin(), keys.end(),
                              points_counter_per_voxel.begin(), thrust::make_discard_iterator(),
                              points_counter_per_voxel.begin(), voxel_key_equal_functor())
            .second;
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    auto new_n_points = end_iter_of_points - result_points.begin();
    if (new_n_points != (end_iter_of_points_counter - points_counter_per_voxel.begin()))
    {
        return ::cudaErrorInvalidValue;
    }

    result_points.resize(new_n_points);
    points_counter_per_voxel.resize(new_n_points);

    thrust::transform(exec_policy, result_points.begin(), result_points.end(),
                      points_counter_per_voxel.begin(), result_points.begin(),
                      compute_points_mean_functor());
    if (err != ::cudaSuccess)
    {
        return err;
    }

    cudaStreamSynchronize(stream);

    return ::cudaSuccess;
}
} // namespace gca
