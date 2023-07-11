#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>
#include <stdint.h>

#include "camera/camera_param.hpp"
#include "cuda_container/cuda_container.hpp"
#include "geometry/type.hpp"
#include "util/cuda_util.cuh"

bool gpu_make_point_set(gca::point_t *result, const gca::cuda_depth_frame &cuda_depth_container,
                        const gca::cuda_color_frame &cuda_color_container,
                        const gca::cuda_camera_param &param, float threshold_min_in_meter = 0.0,
                        float threshold_max_in_meter = 100.0);