#pragma once

#include "geometry/type.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

namespace gca
{
::cudaError_t cuda_compute_res_and_mean_res(
    thrust::device_vector<float> &residuals, float &mean_residual,
    const thrust::device_vector<gca::point_t> &pts_src_frame,
    const thrust::device_vector<gca::point_t> &pts_tgt_frame,
    const thrust::device_vector<gca::index_t> &nn_idxs, const float geometry_constraint,
    const float geometry_weight, const float photometry_weight);

::cudaError_t cuda_moving_objects_seg(thrust::device_vector<gca::point_t> &output,
                                      gca::counter_t n_clusters,
                                      const thrust::device_vector<gca::index_t> &clusters,
                                      const thrust::device_vector<gca::point_t> &pts_src,
                                      const thrust::device_vector<float> &residuals);
} // namespace gca