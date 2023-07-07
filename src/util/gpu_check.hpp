#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

void cuda_print_devices();
bool cuda_warm_up_gpu(uint8_t device_num);
