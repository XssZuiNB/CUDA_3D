#include "geometry/cuda_nn_search.cuh"
#include "geometry/type.hpp"
#include "util/cuda_util.cuh"
#include "util/math.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

namespace gca
{
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

            auto intensity_redidual = abs(p.color.to_intensity() - nn.color.to_intensity());

            return (m_weight_geometry * geometry_redidual +
                    m_weight_photometry * intensity_redidual);
        }

        return m_weight_geometry * m_geometry_constraint;
    }
};

struct reduce_sum_cluster_residual_and_size_functor
{
    reduce_sum_cluster_residual_and_size_functor(thrust::device_vector<float> &cluster_residuals,
                                                 thrust::device_vector<gca::counter_t> &n,
                                                 const float mean_residual_over_all)
        : m_cluster_residual_ptr(thrust::raw_pointer_cast(cluster_residuals.data()))
        , m_n_pts_ptr(thrust::raw_pointer_cast(n.data()))
        , m_mean_r_over_all(mean_residual_over_all)

    {
    }

    float *m_cluster_residual_ptr;
    gca::counter_t *m_n_pts_ptr;
    const float m_mean_r_over_all;

    __forceinline__ __device__ void operator()(
        const thrust::tuple<gca::index_t, float> &cluster_and_res)
    {
        auto cluster_num = thrust::get<0>(cluster_and_res);
        if (cluster_num != -1)
        {
            auto res = thrust::get<1>(cluster_and_res);
            auto robust_res = cauchy_kernel(res);
            atomicAdd(&m_n_pts_ptr[cluster_num], 1);
            atomicAdd(&m_cluster_residual_ptr[cluster_num], robust_res);
        }
    }

private:
    __forceinline__ __device__ float cauchy_kernel(float res)
    {
        auto c_square = 0.25 * (m_mean_r_over_all * m_mean_r_over_all);
        // c^2/2 should be devided later, it is omitted here for simplification
        return logf(1.0f + res * res / c_square);
    }
};

struct compute_cluster_score_functor
{
    compute_cluster_score_functor(const float mean_residual_over_all, const float var)
        : m_mean_r_over_all(mean_residual_over_all)
        , m_sigma(sqrtf(var))
    {
    }

    const float m_mean_r_over_all;
    const float m_sigma;
    const float m_weight = 1.5;
    const float repr_res = m_mean_r_over_all + m_sigma;

    __forceinline__ __device__ float operator()(
        const thrust::tuple<float, gca::counter_t> &sum_cluster_residual_and_size)
    {
        auto res_ = thrust::get<0>(sum_cluster_residual_and_size) /
                    thrust::get<1>(sum_cluster_residual_and_size);
        auto robust_thres =
            logf(1.0f + repr_res * repr_res / (0.25 * m_mean_r_over_all * m_mean_r_over_all));

        return res_ / robust_thres;
    }
};

struct segment_moving_cluster_functor
{
    segment_moving_cluster_functor(const thrust::device_vector<float> &cluster_scores)
        : m_cluster_scores_ptr(thrust::raw_pointer_cast(cluster_scores.data()))
    {
    }

    const float *m_cluster_scores_ptr;

    __forceinline__ __device__ gca::point_t operator()(
        const thrust::tuple<gca::index_t, gca::point_t> &cluster_point_tuple)
    {
        auto cluster = thrust::get<0>(cluster_point_tuple);
        auto p = thrust::get<1>(cluster_point_tuple);
        auto score = (m_cluster_scores_ptr[cluster]);

        if (score >= 0.6f)
        {
            p.color.r = 1.0f;
            p.color.g = 0.0f;
            p.color.b = 0.0f;
        }

        return p;
    }
};

struct compute_var_functor
{
    compute_var_functor(float mean_residual_over_all)
        : m_mean_residual_over_all(mean_residual_over_all)
    {
    }

    const float m_mean_residual_over_all;

    __forceinline__ __device__ float operator()(float x)
    {
        return (x - m_mean_residual_over_all) * (x - m_mean_residual_over_all);
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

    mean_residual =
        thrust::reduce(residuals.begin(), residuals.end(), 0.0f, thrust::plus<float>()) / n_points;

    std::cout << "mean: " << mean_residual << std::endl;

    err = cudaGetLastError();
    if (err != ::cudaSuccess)
        return err;

    return cudaSuccess;
}

::cudaError_t cuda_moving_objects_seg(thrust::device_vector<gca::point_t> &output,
                                      const gca::counter_t n_clusters,
                                      const thrust::device_vector<gca::index_t> &clusters,
                                      const thrust::device_vector<gca::point_t> &pts_src,
                                      const thrust::device_vector<float> &residuals,
                                      const float mean_residual_over_all)
{
    auto n_points = pts_src.size();
    if (n_points != clusters.size() || n_points != residuals.size())
        return ::cudaErrorInvalidValue;

    if (n_points != output.size())
        output.resize(n_points);

    auto var = thrust::transform_reduce(residuals.begin(), residuals.end(),
                                        compute_var_functor(mean_residual_over_all), 0.0f,
                                        thrust::plus<float>()) /
               n_points;

    thrust::device_vector<float> cluster_residuals(n_clusters, 0.0f);
    thrust::device_vector<gca::counter_t> n_points_in_cluster(n_clusters, 0);

    thrust::for_each(
        thrust::make_zip_iterator(thrust::make_tuple(clusters.begin(), residuals.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(clusters.end(), residuals.end())),
        reduce_sum_cluster_residual_and_size_functor(cluster_residuals, n_points_in_cluster,
                                                     mean_residual_over_all));
    auto err = cudaGetLastError();
    if (err != ::cudaSuccess)
        return err;

    print_device_vector(cluster_residuals, "res: ");
    print_device_vector(n_points_in_cluster, "num: ");

    thrust::device_vector<float> cluster_scores(n_clusters);
    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(cluster_residuals.begin(),
                                                                   n_points_in_cluster.begin())),
                      thrust::make_zip_iterator(
                          thrust::make_tuple(cluster_residuals.end(), n_points_in_cluster.end())),
                      cluster_scores.begin(),
                      compute_cluster_score_functor(mean_residual_over_all, var));
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
        return err;

    print_device_vector(cluster_scores, "score: ");

    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(clusters.begin(), pts_src.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(clusters.end(), pts_src.end())),
        output.begin(), segment_moving_cluster_functor(cluster_scores));
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
        return err;

    return ::cudaSuccess;
}
} // namespace gca
