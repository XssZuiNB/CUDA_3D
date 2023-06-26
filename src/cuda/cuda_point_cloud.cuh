#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "type.hpp"

cudaError_t cuda_warm_up_gpu(uint8_t device_num);

bool gpu_make_point_set(gca::point_t *result, uint32_t width, const uint32_t height,
                        const uint16_t *depth_data, const uint8_t *color_data,
                        const gca::intrinsics &depth_intrin, const gca::intrinsics &color_intrin,
                        const gca::extrinsics &depth_to_color_extrin, const float depth_scale);
