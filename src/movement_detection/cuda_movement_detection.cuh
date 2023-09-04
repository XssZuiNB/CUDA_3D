#pragma once

#include "geometry/type.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

namespace gca
{
::cudaError_t cuda_movement_detection(thrust::device_vector<gca::point_t> &result,
                                      const thrust::device_vector<gca::point_t> &pts_this_frame,
                                      const thrust::device_vector<gca::point_t> &pts_last_frame,
                                      const float3 min_bound, const float3 max_bound,
                                      const float geometry_constraint,
                                      const float color_constraint);
}