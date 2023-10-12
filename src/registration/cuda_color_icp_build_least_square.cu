#include "cuda_color_icp_build_least_square.cuh"

#include "util/math.cuh"

#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>

namespace gca
{
struct compute_residual_pair_functor
{
    compute_residual_pair_functor(const thrust::device_vector<gca::point_t> &tgt_points,
                                  const thrust::device_vector<float3> &tgt_normals,
                                  const thrust::device_vector<float3> &tgt_color_gradient,
                                  const float lambda)
        : m_tgt_points_ptr(thrust::raw_pointer_cast(tgt_points.data()))
        , m_tgt_normals_ptr(thrust::raw_pointer_cast(tgt_normals.data()))
        , m_tgt_color_gradient_ptr(thrust::raw_pointer_cast(tgt_color_gradient.data()))
        , m_sqrt_lambda(sqrtf(lambda))
        , m_sqrt_1_minus_lambda(sqrtf(1.0f - lambda))
    {
    }

    const gca::point_t *m_tgt_points_ptr;
    const float3 *m_tgt_normals_ptr;
    const float3 *m_tgt_color_gradient_ptr;
    const float m_sqrt_lambda;
    const float m_sqrt_1_minus_lambda;

    __forceinline__ __device__ thrust::pair<float, float> operator()(
        thrust::tuple<gca::point_t, gca::index_t> &pts_and_nn)
    {
        auto pts_src = thrust::get<0>(pts_and_nn);
        auto nn_idx_tgt = thrust::get<1>(pts_and_nn);

        auto nn_pts_tgt = m_tgt_points_ptr[nn_idx_tgt];
        auto normal = m_tgt_normals_ptr[nn_idx_tgt];
        auto color_gradient_tgt = m_tgt_color_gradient_ptr[nn_idx_tgt];

        auto r_ = dot((pts_src.coordinates - nn_pts_tgt.coordinates), normal);
        auto rg = m_sqrt_lambda * r_; // geometric

        auto indensity_pts_src = pts_src.color.to_intensity();
        auto indensity_pts_tgt = nn_pts_tgt.color.to_intensity();
        // project src point onto the tangent plane of taget point
        auto proj_coordinates = pts_src.coordinates - r_ * normal;

        auto indensity_proj = dot(color_gradient_tgt, (proj_coordinates - nn_pts_tgt.coordinates));

        auto rc = m_sqrt_1_minus_lambda * (indensity_pts_src - indensity_proj);

        return thrust::make_pair(rg, rc);
    }
};

::cudaError_t cuda_compute_residual(
    thrust::device_vector<thrust::pair<float, float>> &result_rg_rc_pair,
    const thrust::device_vector<gca::point_t> &src_points,
    const thrust::device_vector<gca::point_t> &tgt_points,
    const thrust::device_vector<float3> &tgt_normals,
    const thrust::device_vector<float3> &tgt_color_gradient,
    const thrust::device_vector<gca::index_t> &nn_src_tgt, const float lambda)
{
    if (lambda > 1.0f)
    {
        return ::cudaErrorInvalidValue;
    }

    auto n_points_src = src_points.size();
    if (nn_src_tgt.size() != n_points_src)
    {
        return ::cudaErrorInvalidValue;
    }

    if (tgt_points.size() != tgt_normals.size() || tgt_points.size() != tgt_color_gradient.size())
    {
        return ::cudaErrorInvalidValue;
    }

    if (result_rg_rc_pair.size() != n_points_src)
    {
        result_rg_rc_pair.resize(n_points_src);
    }

    auto zipped_begin =
        thrust::make_zip_iterator(thrust::make_tuple(src_points.begin(), nn_src_tgt.begin()));

    auto zipped_end =
        thrust::make_zip_iterator(thrust::make_tuple(src_points.end(), nn_src_tgt.end()));

    thrust::transform(
        zipped_begin, zipped_end, result_rg_rc_pair.begin(),
        compute_residual_pair_functor(tgt_points, tgt_normals, tgt_color_gradient, lambda));
    auto err = cudaGetLastError();
    return err;
}
} // namespace gca
