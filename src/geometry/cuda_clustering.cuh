#pragma once

#include "geometry/type.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace gca
{
::cudaError_t cuda_euclidean_clustering(std::vector<thrust::host_vector<gca::index_t>> &clusters,
                                        const thrust::device_vector<gca::point_t> &points,
                                        const float3 min_bound, const float3 max_bound,
                                        const float cluster_tolerance,
                                        const gca::counter_t min_cluster_size,
                                        const gca::counter_t max_cluster_size);

::cudaError_t cuda_local_convex_segmentation(std::vector<thrust::host_vector<gca::index_t>> &objs,
                                             const thrust::device_vector<gca::point_t> &points,
                                             const thrust::device_vector<float3> &normals,
                                             const float3 min_bound, const float3 max_bound,
                                             const float cluster_tolerance,
                                             const gca::counter_t min_cluster_size,
                                             const gca::counter_t max_cluster_size);

::cudaError_t cuda_residual_based_segmentation(std::vector<thrust::host_vector<gca::index_t>> &objs,
                                               const thrust::device_vector<gca::point_t> &points,
                                               const thrust::device_vector<float> &abs_residual,
                                               const float3 min_bound, const float3 max_bound,
                                               const float cluster_tolerance,
                                               const gca::counter_t min_cluster_size,
                                               const gca::counter_t max_cluster_size);

void cuda_remove_plane(thrust::device_vector<gca::point_t> &result_points,
                       const std::vector<float> &plane_model,
                       const thrust::device_vector<gca::point_t> &src_points);
} // namespace gca
