#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>
#include <stdint.h>

#include "camera/camera_param.hpp"
#include "cuda_container/cuda_container.hpp"
#include "geometry/type.hpp"
#include "util/cuda_util.cuh"

cudaError_t cuda_warm_up_gpu(uint8_t device_num);

bool gpu_make_point_set(gca::point_t *result, uint32_t width, const uint32_t height,
                        const gca::cuda_depth_frame &cuda_depth_container,
                        const gca::cuda_color_frame &cuda_color_container,
                        const gca::intrinsics &depth_intrin, const gca::intrinsics &color_intrin,
                        const gca::extrinsics &depth_to_color_extrin, const float depth_scale);
