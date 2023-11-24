#pragma once

#include "geometry/type.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

namespace gca
{
::cudaError_t cuda_compute_color_gradient(thrust::device_vector<float3> &result,
                                          const thrust::device_vector<gca::point_t> &pts,
                                          const thrust::device_vector<float3> &normals,
                                          const float3 min_bound, const float3 max_bound,
                                          const float search_radius);
}
