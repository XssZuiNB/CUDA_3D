#include "cuda_color_icp_build_least_square.cuh"

#include "util/math.cuh"

#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/pair.h>
#include <thrust/transform_reduce.h>
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
        , m_sqrt_lambda_geometry(sqrtf(lambda))
        , m_sqrt_lambda_color(sqrtf(1.0f - lambda))
    {
    }

    const gca::point_t *m_tgt_points_ptr;
    const float3 *m_tgt_normals_ptr;
    const float3 *m_tgt_color_gradient_ptr;
    const float m_sqrt_lambda_geometry;
    const float m_sqrt_lambda_color;

    __forceinline__ __device__ thrust::pair<float, float> operator()(
        const thrust::tuple<gca::point_t, gca::index_t> &pts_and_nn) const
    {
        auto pts_src = thrust::get<0>(pts_and_nn);
        auto nn_idx_tgt = thrust::get<1>(pts_and_nn);

        if (nn_idx_tgt < 0)
        {
            return thrust::make_pair(0.0f, 0.0f);
        }

        auto nn_pts_tgt = m_tgt_points_ptr[nn_idx_tgt];
        auto normal = m_tgt_normals_ptr[nn_idx_tgt];
        auto color_gradient_tgt = m_tgt_color_gradient_ptr[nn_idx_tgt];

        auto rg_ = dot((pts_src.coordinates - nn_pts_tgt.coordinates), normal);
        auto rg_weighted = m_sqrt_lambda_geometry * rg_; // geometric

        auto indensity_pts_src = pts_src.color.to_intensity();
        auto indensity_pts_tgt = nn_pts_tgt.color.to_intensity();
        // project src point onto the tangent plane of taget point
        auto proj_coordinates = pts_src.coordinates - rg_ * normal;
        // get projected intensity
        auto indensity_proj = indensity_pts_tgt +
                              dot(color_gradient_tgt, (proj_coordinates - nn_pts_tgt.coordinates));
        auto rc_weighted = m_sqrt_lambda_color * (indensity_proj - indensity_pts_src); // color

        return thrust::make_pair(rg_weighted, rc_weighted);
    }
};

struct compute_residual_functor : compute_residual_pair_functor
{
    compute_residual_functor(const thrust::device_vector<gca::point_t> &tgt_points,
                             const thrust::device_vector<float3> &tgt_normals,
                             const thrust::device_vector<float3> &tgt_color_gradient,
                             const float lambda)
        : compute_residual_pair_functor(tgt_points, tgt_normals, tgt_color_gradient, lambda)
    {
    }

    __forceinline__ __device__ float operator()(
        const thrust::tuple<gca::point_t, gca::index_t> &pts_and_nn) const
    {
        const auto &pair = compute_residual_pair_functor::operator()(pts_and_nn);
        return pair.first + pair.second;
    }
};

struct compute_rmse_functor
{
    compute_rmse_functor(const thrust::device_vector<gca::point_t> &tgt_points,
                         const thrust::device_vector<float3> &tgt_normals,
                         const thrust::device_vector<float3> &tgt_color_gradient,
                         const float lambda)
        : m_tgt_points_ptr(thrust::raw_pointer_cast(tgt_points.data()))
        , m_tgt_normals_ptr(thrust::raw_pointer_cast(tgt_normals.data()))
        , m_tgt_color_gradient_ptr(thrust::raw_pointer_cast(tgt_color_gradient.data()))
        , m_lambda_geometry(lambda)
        , m_lambda_color(1.0f - lambda)
    {
    }

    const gca::point_t *m_tgt_points_ptr;
    const float3 *m_tgt_normals_ptr;
    const float3 *m_tgt_color_gradient_ptr;
    const float m_lambda_geometry;
    const float m_lambda_color;

    __forceinline__ __device__ float operator()(
        const thrust::tuple<gca::point_t, gca::index_t> &pts_and_nn) const
    {
        auto pts_src = thrust::get<0>(pts_and_nn);
        auto nn_idx_tgt = thrust::get<1>(pts_and_nn);

        if (nn_idx_tgt < 0)
        {
            return 0.0f;
        }

        auto nn_pts_tgt = m_tgt_points_ptr[nn_idx_tgt];
        auto normal = m_tgt_normals_ptr[nn_idx_tgt];
        auto color_gradient_tgt = m_tgt_color_gradient_ptr[nn_idx_tgt];

        auto rg_ = dot((pts_src.coordinates - nn_pts_tgt.coordinates), normal);
        auto rg_square_weighted = m_lambda_geometry * rg_ * rg_;

        auto indensity_pts_src = pts_src.color.to_intensity();
        auto indensity_pts_tgt = nn_pts_tgt.color.to_intensity();
        // project src point onto the tangent plane of taget point
        auto proj_coordinates = pts_src.coordinates - rg_ * normal;
        // get projected intensity
        auto indensity_proj = indensity_pts_tgt +
                              dot(color_gradient_tgt, (proj_coordinates - nn_pts_tgt.coordinates));
        auto rc_ = indensity_proj - indensity_pts_src;
        auto rc_square_weighted = m_lambda_color * rc_ * rc_;

        return rg_square_weighted + rc_square_weighted;
    }
};

struct compute_JTJ_and_JTr_functor
{
    compute_JTJ_and_JTr_functor(const thrust::device_vector<gca::point_t> &tgt_points,
                                const thrust::device_vector<float3> &tgt_normals,
                                const thrust::device_vector<float3> &tgt_color_gradient,
                                const float lambda)
        : m_tgt_points_ptr(thrust::raw_pointer_cast(tgt_points.data()))
        , m_tgt_normals_ptr(thrust::raw_pointer_cast(tgt_normals.data()))
        , m_tgt_color_gradient_ptr(thrust::raw_pointer_cast(tgt_color_gradient.data()))
        , m_sqrt_lambda_geometry(sqrtf(lambda))
        , m_sqrt_lambda_color(sqrtf(1.0f - lambda))
    {
    }

    const gca::point_t *m_tgt_points_ptr;
    const float3 *m_tgt_normals_ptr;
    const float3 *m_tgt_color_gradient_ptr;
    const float m_sqrt_lambda_geometry;
    const float m_sqrt_lambda_color;

    __forceinline__ __device__ float operator()()
    {
    }

private:
    void compute_jacobian_and_residual(
        mat6x1 &J_geometry, mat6x1 &J_color, float &r_geometry, float &r_color,
        const thrust::tuple<gca::point_t, gca::index_t> &pts_and_nn) const
    {
        auto pts_src = thrust::get<0>(pts_and_nn);
        auto nn_idx_tgt = thrust::get<1>(pts_and_nn);

        if (nn_idx_tgt < 0)
        {
            J_geometry.set_zero();
            J_color.set_zero();
            r_geometry = 0;
            r_color = 0;
            return;
        }

        auto nn_pts_tgt = m_tgt_points_ptr[nn_idx_tgt];
        auto normal = m_tgt_normals_ptr[nn_idx_tgt];
        auto color_gradient_tgt = m_tgt_color_gradient_ptr[nn_idx_tgt];

        // r_geometry without weight, this is also needed in color part, so compute it here firstly.
        auto r_geometry_ = dot((pts_src.coordinates - nn_pts_tgt.coordinates), normal);

        // geometric
        // jacobian
        mat3x1 J_geo_rotation(m_sqrt_lambda_geometry * cross(pts_src.coordinates, normal));
        mat3x1 J_geo_tranlation(m_sqrt_lambda_geometry * normal);
        J_geometry.set_block<3, 1>(J_geo_rotation, 0, 0);
        J_geometry.set_block<3, 1>(J_geo_tranlation, 3, 0);

        // residual
        r_geometry = m_sqrt_lambda_geometry * r_geometry_;

        // color
        // jacobian
        // paper equation 28, 29
        mat3x1 normal_as_mat(normal);

        // J_f(s)
        mat3x3 df_div_ds(float3x3::get_identity());

        // from equation 9: df(s)/ds = I - n * nT
        df_div_ds = df_div_ds - normal_as_mat * normal_as_mat.get_transpose();

        // from equation 8: dC_p(u)/du = d_pT -> equation 29 = d_pT * (I - n * nT) * J_s(xi)
        mat3x1 color_gradient_tgt_as_mat(color_gradient_tgt);

        //  d_pT * (I - n * nT) is called here as dc_x_df
        float3 dc_x_df(color_gradient_tgt_as_mat.get_transpose() * df_div_ds);

        // derivation of the formula J_s(xi) see Masterthesis von Shengsi Xu
        mat3x1 J_color_fir(m_sqrt_lambda_color * cross(pts_src.coordinates, dc_x_df));
        mat3x1 J_color_sec(m_sqrt_lambda_color * dc_x_df);
        J_color.set_block<3, 1>(J_color_fir, 0, 0);
        J_color.set_block<3, 1>(J_color_sec, 0, 0);
        // residual
        auto indensity_pts_src = pts_src.color.to_intensity();
        auto indensity_pts_tgt = nn_pts_tgt.color.to_intensity();
        // project src point onto the tangent plane of taget point
        auto proj_coordinates = pts_src.coordinates - r_geometry_ * normal;
        // get projected intensity
        auto indensity_proj = indensity_pts_tgt +
                              dot(color_gradient_tgt, (proj_coordinates - nn_pts_tgt.coordinates));
        auto rc_weighted = m_sqrt_lambda_color * (indensity_proj - indensity_pts_src); // color
    }
};

::cudaError_t cuda_compute_residual_color_icp(
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

    return ::cudaSuccess;
}

::cudaError_t cuda_compute_residual_color_icp(
    thrust::device_vector<float> &result_rg_plus_rc,
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

    if (result_rg_plus_rc.size() != n_points_src)
    {
        result_rg_plus_rc.resize(n_points_src);
    }

    auto zipped_begin =
        thrust::make_zip_iterator(thrust::make_tuple(src_points.begin(), nn_src_tgt.begin()));

    auto zipped_end =
        thrust::make_zip_iterator(thrust::make_tuple(src_points.end(), nn_src_tgt.end()));

    thrust::transform(
        zipped_begin, zipped_end, result_rg_plus_rc.begin(),
        compute_residual_functor(tgt_points, tgt_normals, tgt_color_gradient, lambda));

    return ::cudaSuccess;
}

::cudaError_t cuda_compute_rmse_color_icp(float &result_rmse,
                                          const thrust::device_vector<gca::point_t> &src_points,
                                          const thrust::device_vector<gca::point_t> &tgt_points,
                                          const thrust::device_vector<float3> &tgt_normals,
                                          const thrust::device_vector<float3> &tgt_color_gradient,
                                          const thrust::device_vector<gca::index_t> &nn_src_tgt,
                                          const float lambda)
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

    auto zipped_begin =
        thrust::make_zip_iterator(thrust::make_tuple(src_points.begin(), nn_src_tgt.begin()));

    auto zipped_end =
        thrust::make_zip_iterator(thrust::make_tuple(src_points.end(), nn_src_tgt.end()));

    result_rmse = thrust::transform_reduce(
        zipped_begin, zipped_end,
        compute_rmse_functor(tgt_points, tgt_normals, tgt_color_gradient, lambda), 0.0f,
        thrust::plus<float>());

    return ::cudaSuccess;
}

::cudaError_t cuda_build_gauss_newton_color_icp(
    mat6x6 &JTJ, mat6x1 &JTr, const thrust::device_vector<gca::point_t> &src_points,
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
}
} // namespace gca
