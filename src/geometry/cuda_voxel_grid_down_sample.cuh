#pragma once

#include "geometry/type.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

namespace gca
{
thrust::device_vector<gca::point_t> cuda_voxel_grid_downsample(
    const thrust::device_vector<gca::point_t> &src_points, const float3 &voxel_grid_min_bound,
    const float voxel_size);
}
