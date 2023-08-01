#pragma once

#include "geometry/type.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

namespace gca
{
::cudaError_t cuda_nn_search(
    thrust::device_vector<thrust::pair<gca::index_t, gca::index_t>> &result_index_pair,
    const thrust::device_vector<gca::point_t> &points_Q,
    const thrust::device_vector<gca::point_t> &points_R, float3 min_bound, const float3 max_bound,
    const float search_radius);

::cudaError_t cuda_grid_radius_outliers_removal(
    thrust::device_vector<gca::point_t> &result_points,
    const thrust::device_vector<gca::point_t> &src_points, const float3 min_bound,
    const float3 max_bound, const float radius, const gca::counter_t min_neighbors_in_radius);
} // namespace gca