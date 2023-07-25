#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "geometry/type.hpp"

namespace gca
{
cudaError_t cuda_grid_radius_outliers_removal(thrust::device_vector<gca::point_t> &result_points,
                                              const thrust::device_vector<gca::point_t> &src_points,
                                              const float3 min_bound, const float3 max_bound,
                                              const float radius,
                                              const gca::counter_t min_neighbors_in_radius);
}