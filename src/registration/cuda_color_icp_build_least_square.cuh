#pragma once

#include "geometry/type.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

namespace gca
{
::cudaError_t cuda_compute_residual_color_icp(
    thrust::device_vector<thrust::pair<float, float>> &result_rg_rc_pair,
    const thrust::device_vector<gca::point_t> &src_points,
    const thrust::device_vector<gca::point_t> &tgt_points,
    const thrust::device_vector<float3> &tgt_normals,
    const thrust::device_vector<float3> &tgt_color_gradient,
    const thrust::device_vector<gca::index_t> &nn_src_tgt, const float lambda);

::cudaError_t cuda_compute_residual_color_icp(
    thrust::device_vector<float> &result_rg_plus_rc,
    const thrust::device_vector<gca::point_t> &src_points,
    const thrust::device_vector<gca::point_t> &tgt_points,
    const thrust::device_vector<float3> &tgt_normals,
    const thrust::device_vector<float3> &tgt_color_gradient,
    const thrust::device_vector<gca::index_t> &nn_src_tgt, const float lambda);

::cudaError_t cuda_compute_rmse_color_icp(float &result_rmse,
                                          const thrust::device_vector<gca::point_t> &src_points,
                                          const thrust::device_vector<gca::point_t> &tgt_points,
                                          const thrust::device_vector<float3> &tgt_normals,
                                          const thrust::device_vector<float3> &tgt_color_gradient,
                                          const thrust::device_vector<gca::index_t> &nn_src_tgt,
                                          const float lambda);

::cudaError_t cuda_build_gauss_newton_color_icp(
    mat6x6 &JTJ, mat6x1 &JTr, float &RMSE, const thrust::device_vector<gca::point_t> &src_points,
    const thrust::device_vector<gca::point_t> &tgt_points,
    const thrust::device_vector<float3> &tgt_normals,
    const thrust::device_vector<float3> &tgt_color_gradient,
    const thrust::device_vector<gca::index_t> &nn_src_tgt, const float lambda);
}