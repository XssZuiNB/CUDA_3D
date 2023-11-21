#pragma once

#include "geometry/type.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

namespace gca
{
::cudaError_t cuda_nn_search(thrust::device_vector<gca::index_t> &result_nn_idx_in_R,
                             const thrust::device_vector<gca::point_t> &points_Q,
                             const thrust::device_vector<gca::point_t> &points_R,
                             const float3 min_bound, const float3 max_bound,
                             const float search_radius);

::cudaError_t cuda_search_radius_neighbors(
    thrust::device_vector<gca::index_t> &result_radius_neighbor_idxs_in_R,
    thrust::device_vector<thrust::pair<gca::index_t, gca::counter_t>>
        &result_pair_neighbors_begin_idx_and_count,
    const thrust::device_vector<gca::point_t> &points_Q,
    const thrust::device_vector<gca::point_t> &points_R, const float3 min_bound,
    const float3 max_bound, const float search_radius);

__forceinline__ ::cudaError_t cuda_search_radius_neighbors(
    thrust::device_vector<gca::index_t> &result_radius_neighbor_idxs,
    thrust::device_vector<thrust::pair<gca::index_t, gca::counter_t>>
        &result_pair_neighbors_begin_idx_and_count,
    const thrust::device_vector<gca::point_t> &points, const float3 min_bound,
    const float3 max_bound, const float search_radius)
{
    return cuda_search_radius_neighbors(result_radius_neighbor_idxs,
                                        result_pair_neighbors_begin_idx_and_count, points, points,
                                        min_bound, max_bound, search_radius);
}

::cudaError_t cuda_grid_radius_outliers_removal(
    thrust::device_vector<gca::point_t> &result_points,
    const thrust::device_vector<gca::point_t> &src_points, const float3 min_bound,
    const float3 max_bound, const float grid_cell_side_len, const float radius,
    const gca::counter_t min_neighbors_in_radius);
} // namespace gca