#include "cuda_voxel_grid_down_sample.cuh"

#include "geometry/geometry_util.cuh"
#include "geometry/type.hpp"
#include "util/cuda_util.cuh"
#include "util/math.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

namespace gca
{
/**************************** Useful functors for thrust algorithms  ****************************/

struct compute_voxel_key_functor
{
    compute_voxel_key_functor(const float3 voxel_grid_min_bound, const float voxel_size)
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

struct compare_voxel_key_functor
{
    __forceinline__ __device__ bool operator()(const int3 &lhs, const int3 &rhs) const
    {
        if (lhs.x != rhs.x)
            return lhs.x < rhs.x;

        if (lhs.y != rhs.y)
            return lhs.y < rhs.y;

        return lhs.z < rhs.z;
    }
};

struct voxel_key_equal_functor
{
    __forceinline__ __device__ bool operator()(const int3 &lhs, const int3 &rhs) const
    {
        return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
    }
};

struct add_and_count_points_functor
{
    __forceinline__ __device__ thrust::tuple<gca::point_t, gca::counter_t> operator()(
        const thrust::tuple<gca::point_t, gca::counter_t> &fir,
        const thrust::tuple<gca::point_t, gca::counter_t> &sec)
    {
        auto fir_pts = thrust::get<0>(fir);
        auto sec_pts = thrust::get<0>(sec);
        return thrust::make_tuple(gca::point_t{fir_pts.coordinates + sec_pts.coordinates,
                                               fir_pts.color + sec_pts.color,
                                               gca::point_property::inactive},
                                  thrust::get<1>(fir) + thrust::get<1>(sec));
    }
};

struct compute_points_mean_functor
{
    __forceinline__ __device__ gca::point_t operator()(const gca::point_t &points_sum,
                                                       const gca::counter_t n)
    {
        return gca::point_t{points_sum.coordinates / n, points_sum.color / n,
                            gca::point_property::inactive};
    }
};

/*********************************** Voxel grid down sampling ***********************************/

thrust::device_vector<gca::point_t> cuda_voxel_grid_downsample(
    const thrust::device_vector<gca::point_t> &src_points, const float3 &voxel_grid_min_bound,
    const float voxel_size)
{
    auto n_points = src_points.size();

    thrust::device_vector<int3> keys(n_points);
    thrust::transform(src_points.begin(), src_points.end(), keys.begin(),
                      compute_voxel_key_functor(voxel_grid_min_bound, voxel_size));

    thrust::device_vector<gca::index_t> index_vec(n_points);
    thrust::sequence(index_vec.begin(), index_vec.end());

    thrust::sort_by_key(keys.begin(), keys.end(), index_vec.begin(), compare_voxel_key_functor());

    auto get_point_with_sorted_index_iter =
        thrust::make_permutation_iterator(src_points.begin(), index_vec.begin());

    thrust::device_vector<gca::counter_t> points_counter_per_voxel(n_points);
    thrust::device_vector<gca::point_t> result_points(n_points);

    auto zip_in = thrust::make_zip_iterator(thrust::make_tuple(
        get_point_with_sorted_index_iter, thrust::make_constant_iterator<gca::counter_t>(1)));
    auto zip_out = thrust::make_zip_iterator(
        thrust::make_tuple(result_points.begin(), points_counter_per_voxel.begin()));

    auto new_n_points =
        thrust::reduce_by_key(keys.begin(), keys.end(), zip_in, keys.begin(), zip_out,
                              voxel_key_equal_functor(), add_and_count_points_functor())
            .first -
        keys.begin();

    result_points.resize(new_n_points);

    thrust::transform(result_points.begin(), result_points.end(), points_counter_per_voxel.begin(),
                      result_points.begin(), compute_points_mean_functor());

    return result_points;
}
} // namespace gca
