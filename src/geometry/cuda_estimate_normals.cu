#include "geometry/cuda_estimate_normals.cuh"
#include "geometry/cuda_nn_search.cuh"
#include "geometry/type.hpp"
#include "util/cuda_util.cuh"
#include "util/numeric.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

namespace gca
{
struct fill_keys_functor
{
    fill_keys_functor(thrust::device_vector<gca::index_t> &keys)
        : m_keys_ptr(thrust::raw_pointer_cast(keys.data()))
    {
    }

    gca::index_t *m_keys_ptr;

    __forceinline__ __device__ void operator()(
        const thrust::pair<gca::index_t, gca::counter_t> &pair)
    {
        auto start_idx = pair.first;
        m_keys_ptr[start_idx] = 1;
    }
};

struct sum_points_functor
{
    sum_points_functor(const thrust::device_vector<gca::point_t> &points,
                       const thrust::device_vector<gca::index_t> &all_neighbors_idx)
        : m_points_ptr(thrust::raw_pointer_cast(points.data()))
        , m_all_neighbors_idx_ptr(thrust::raw_pointer_cast(all_neighbors_idx.data()))
    {
    }

    const gca::point_t *m_points_ptr;
    const gca::index_t *m_all_neighbors_idx_ptr;

    __forceinline__ __device__ float3
    operator()(const thrust::pair<gca::index_t, gca::counter_t> &pair)
    {
        auto idx = __ldg(&m_all_neighbors_idx_ptr[pair.first]);
        auto sum_x = __ldg(&(m_points_ptr[idx].coordinates.x));
        auto sum_y = __ldg(&(m_points_ptr[idx].coordinates.x));
        auto sum_z = __ldg(&(m_points_ptr[idx].coordinates.x));

        for (gca::counter_t i = 0; i < pair.second; i++)
        {
            auto idx = __ldg(&m_all_neighbors_idx_ptr[pair.first + i]);
        }
    }
};

::cudaError_t cuda_estimate_normals(thrust::device_vector<float3> &result_normals,
                                    const thrust::device_vector<gca::point_t> &points,
                                    const float3 min_bound, const float3 max_bound,
                                    const float search_radius)
{
    auto n_points = points.size();
    if (result_normals.size() != n_points)
    {
        result_normals.resize(n_points);
    }

    thrust::device_vector<gca::index_t> all_neighbors;
    thrust::device_vector<thrust::pair<gca::index_t, gca::counter_t>>
        pair_neighbors_begin_idx_and_count;

    auto err = cuda_search_radius_neighbors(all_neighbors, pair_neighbors_begin_idx_and_count,
                                            points, min_bound, max_bound, search_radius);
    if (err != ::cudaSuccess)
    {
        return err;
    }

    thrust::device_vector<float3> point_coordinates_mean(n_points);
    thrust::transform(pair_neighbors_begin_idx_and_count.begin(),
                      pair_neighbors_begin_idx_and_count.end(), point_coordinates_mean.begin(),
                      sum_points_functor(points, all_neighbors));
    /*
    thrust::device_vector<gca::index_t> keys_(all_neighbors.size(), 0);
    thrust::device_vector<gca::index_t> keys(all_neighbors.size());
    thrust::for_each(pair_neighbors_begin_idx_and_count.begin() + 1,
                     pair_neighbors_begin_idx_and_count.end(), fill_keys_functor(keys_));
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    thrust::inclusive_scan(keys_.begin(), keys_.end(), keys.begin());
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    thrust::reduce_by_key();
    */
    return ::cudaSuccess;
}
} // namespace gca
