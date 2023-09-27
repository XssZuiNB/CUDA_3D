#include "geometry/cuda_nn_search.cuh"
#include "geometry/type.hpp"
#include "util/math.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

namespace gca
{
__forceinline__ __device__ float color_intensity(gca::color3 color)
{
    return (0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b) * 255.0f;
}

struct compute_residual_functor
{
    compute_residual_functor(const thrust::device_vector<gca::point_t> &pts_tgt_frame,
                             const thrust::device_vector<gca::index_t> &nn_idx_in_last_frame,
                             const float geometry_constraint, const float weight_geometry,
                             const float weight_photometry)
        : m_pts_tgt_frame_ptr(thrust::raw_pointer_cast(pts_tgt_frame.data()))
        , m_nn_idx_ptr(thrust::raw_pointer_cast(nn_idx_in_last_frame.data()))
        , m_geometry_constraint(geometry_constraint)
        , m_weight_geometry(weight_geometry)
        , m_weight_photometry(weight_photometry)
    {
    }

    const gca::point_t *m_pts_tgt_frame_ptr;
    const gca::index_t *m_nn_idx_ptr;
    const float m_geometry_constraint;
    const float m_weight_geometry;
    const float m_weight_photometry;

    __forceinline__ __device__ float operator()(const gca::point_t &p, gca::index_t i) const
    {
        auto nn_idx = __ldg(&m_nn_idx_ptr[i]);

        if (nn_idx != -1)
        {
            auto nn = m_pts_tgt_frame_ptr[nn_idx];

            auto geometry_redidual = norm(p.coordinates - nn.coordinates);

            auto intensity_redidual = (color_intensity(p.color) - color_intensity(nn.color));

            return (m_weight_geometry * geometry_redidual +
                    m_weight_photometry * intensity_redidual);
        }

        return m_weight_geometry * m_geometry_constraint;
    }
};

/* This moving objects detection algorithm assume that camera is not moving */
::cudaError_t cuda_compute_res_and_mean_res(
    thrust::device_vector<float> &residuals, float &mean_residual,
    const thrust::device_vector<gca::point_t> &pts_src_frame,
    const thrust::device_vector<gca::point_t> &pts_tgt_frame,
    const thrust::device_vector<gca::index_t> &nn_idxs, const float geometry_constraint,
    const float geometry_weight, const float photometry_weight)
{
    auto n_points = pts_src_frame.size();

    if (residuals.size() != n_points)
        residuals.resize(n_points);

    thrust::transform(pts_src_frame.begin(), pts_src_frame.end(),
                      thrust::make_counting_iterator<gca::index_t>(0), residuals.begin(),
                      compute_residual_functor(pts_tgt_frame, nn_idxs, geometry_constraint,
                                               geometry_weight, photometry_weight));
    auto err = cudaGetLastError();
    if (err != ::cudaSuccess)
        return err;

    auto mean_residual_total =
        thrust::reduce(residuals.begin(), residuals.end(), 0.0f, thrust::plus<float>()) / n_points;

    err = cudaGetLastError();
    if (err != ::cudaSuccess)
        return err;

    return cudaSuccess;
}

::cudaError_t cuda_moving_objects_seg(thrust::device_vector<gca::point_t> &output,
                                      gca::counter_t n_clusters,
                                      const thrust::device_vector<gca::index_t> &clusters,
                                      const thrust::device_vector<gca::point_t> &pts_src,
                                      const thrust::device_vector<float> &residuals)
{
    auto n_points = pts_src.size();
    if (n_points != clusters.size() || n_points != residuals.size())
        return ::cudaErrorInvalidValue;

    if (n_points != output.size())
        output.resize(n_points);

    thrust::copy(pts_src.begin(), pts_src.end(), output.begin());

    thrust::device_vector<float> cluster_residuals(n_clusters, 0.0f);
    thrust::device_vector<gca::counter_t> n_points_in_cluster(n_clusters, 0);

    auto f = [&](thrust::tuple<gca::index_t, float> cluster_and_res) {
        auto cluster_num = thrust::get<0>(cluster_and_res);
        auto res = thrust::get<1>(cluster_and_res);
        cluster_residuals[cluster_num] += res;
        n_points_in_cluster[cluster_num] += 1;
    };
}
} // namespace gca
