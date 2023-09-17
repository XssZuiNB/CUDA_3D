#pragma once

#include "geometry/type.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>

namespace gca
{
::cudaError_t cuda_estimate_normals(thrust::device_vector<float3> &result_normals,
                                    const thrust::device_vector<gca::point_t> &points,
                                    const float3 min_bound, const float3 max_bound,
                                    const float search_radius);
} // namespace gca