#pragma once

#include "camera/camera_param.hpp"
#include "cuda_container/cuda_container.hpp"
#include "geometry/type.hpp"
#include "util/cuda_util.cuh"

#include <memory>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

namespace gca
{
::cudaError_t cuda_make_point_cloud(thrust::device_vector<gca::point_t> &result,
                                    const gca::cuda_depth_frame &cuda_depth_container,
                                    const gca::cuda_color_frame &cuda_color_container,
                                    const gca::cuda_camera_param &param,
                                    float threshold_min_in_meter, float threshold_max_in_meter,
                                    ::cudaStream_t stream = cudaStreamDefault);
} // namespace gca
