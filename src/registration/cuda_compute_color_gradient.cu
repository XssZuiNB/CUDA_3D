#include "cuda_compute_color_gradient.cuh"

#include "geometry/cuda_nn_search.cuh"
#include "util/math.cuh"

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace gca
{
__forceinline__ __device__ void ldlt(const mat3x3 &A, mat3x3 &L, mat3x3 &D)
{
    L.set_identity();
    D.set_zero();

    D(0, 0) = A(0, 0);
    L(1, 0) = A(1, 0) / D(0, 0);
    L(2, 0) = A(2, 0) / D(0, 0);
    D(1, 1) = A(1, 1) - L(1, 0) * L(1, 0) * D(0, 0);
    L(2, 1) = (A(2, 1) - L(1, 0) * L(2, 0) * D(0, 0)) / D(1, 1);
    D(2, 2) = A(2, 2) - L(2, 0) * L(2, 0) * D(0, 0) - L(2, 1) * L(2, 1) * D(1, 1);
}

// Ly = b
__forceinline__ __device__ mat3x1 solve_lower(const mat3x3 &L, const mat3x1 &b)
{
    mat3x1 y;
    y(0) = b(0);
    y(1) = b(1) - L(1, 0) * y(0);
    y(2) = b(2) - L(2, 0) * y(0) - L(2, 1) * y(1);
    return y;
}

// Dz = y
__forceinline__ __device__ mat3x1 solve_diagonal(const mat3x3 &D, const mat3x1 &y)
{
    mat3x1 z;
    z(0) = y(0) / D(0, 0);
    z(1) = y(1) / D(1, 1);
    z(2) = y(2) / D(2, 2);
    return z;
}

// L^Tx = z
__forceinline__ __device__ mat3x1 solve_upper(const mat3x3 &L, const mat3x1 &z)
{
    mat3x1 x;
    x(2) = z(2);
    x(1) = z(1) - L(2, 1) * x(2);
    x(0) = z(0) - L(1, 0) * x(1) - L(2, 0) * x(2);
    return x;
}

struct compute_color_gradient_functor
{
    compute_color_gradient_functor(
        const thrust::device_vector<gca::point_t> &pts,
        const thrust::device_vector<float3> &normals,
        const thrust::device_vector<gca::index_t> &all_neighbors,
        const thrust::device_vector<thrust::pair<gca::index_t, gca::counter_t>>
            &pair_neighbors_begin_idx_and_count)
        : m_pts_ptr(thrust::raw_pointer_cast(pts.data()))
        , m_normals_ptr(thrust::raw_pointer_cast(normals.data()))
        , m_all_neighbors_ptr(thrust::raw_pointer_cast(all_neighbors.data()))
        , m_neighbors_begin_idx_and_count_ptr(
              thrust::raw_pointer_cast(pair_neighbors_begin_idx_and_count.data()))
    {
    }

    const gca::point_t *m_pts_ptr;
    const float3 *m_normals_ptr;
    const gca::index_t *m_all_neighbors_ptr;
    const thrust::pair<gca::index_t, gca::counter_t> *m_neighbors_begin_idx_and_count_ptr;

    __forceinline__ __device__ float3 operator()(gca::index_t idx) const
    {
        const auto &pts(m_pts_ptr[idx]);
        const auto &normal(m_normals_ptr[idx]);
        const auto begin_idx(__ldg(&(m_neighbors_begin_idx_and_count_ptr[idx].first)));
        const auto knn(__ldg(&(m_neighbors_begin_idx_and_count_ptr[idx].second)));

        if (knn < 5)
        {
            return make_float3(0.0f, 0.0f, 0.0f);
        }

        float intensity = pts.color.to_intensity();

        mat3x3 AtA;
        mat3x1 Atb;
        AtA.set_zero();
        Atb.set_zero();

        /* equation 10 least square L(d_p)' = 0 => (f(p') - p)T * d_p = C(p') - C(P)
         * -> Ax = b => ATA * x = ATb
         * => (f(p') - p) * (f(p') - p)T * d_p = (f(p') - p) * (C(p') - C(P))
         * => d_p = ((f(p') - p) * (f(p') - p)T) ^ -1 * ((f(p') - p) * (C(p') - C(P))) */
        for (gca::index_t i = 0; i < knn; ++i)
        {
            const int nn_idx = __ldg(&m_all_neighbors_ptr[begin_idx + i]);
            // find itself, continue
            if (nn_idx == idx)
                continue;

            const auto &nn_pts = m_pts_ptr[nn_idx];
            const auto nn_pts_proj_coordinates =
                nn_pts.coordinates - dot(nn_pts.coordinates - pts.coordinates, normal) * normal;

            float nn_intensity = nn_pts.color.to_intensity();

            mat3x1 vec_pp_p(nn_pts_proj_coordinates - pts.coordinates);
            AtA += vec_pp_p * vec_pp_p.get_transpose();
            Atb += vec_pp_p * (nn_intensity - intensity);
        }

        const mat3x1 n_mat(normal);
        // orthogonal constraint  (after equation 10 in paper)
        // Goal is to give an addition term d_pT * n_p = 0 for least square to optimize
        // Because this color gradient should be on the tangent plane of the point
        // This term makes convergence faster!
        AtA += (knn - 1) * (knn - 1) * n_mat * n_mat.get_transpose();
        // Atb += (knn - 1) * n_mat; // This makes slower convergence...
        AtA(0, 0) += 1.0e-6f;
        AtA(1, 1) += 1.0e-6f;
        AtA(2, 2) += 1.0e-6f;

        mat3x3 L, D;
        ldlt(AtA, L, D);
        auto y = solve_lower(L, Atb);
        auto z = solve_diagonal(D, y);
        auto x = solve_upper(L, z);

        // inverse is numerically unstable, could cause problem sometimes.
        // const float3 x(AtA.get_inverse() * Atb);
        return x;
    }
};

::cudaError_t cuda_compute_color_gradient(thrust::device_vector<float3> &result,
                                          const thrust::device_vector<gca::point_t> &pts,
                                          const thrust::device_vector<float3> &normals,
                                          const float3 min_bound, const float3 max_bound,
                                          const float search_radius)
{
    auto n_points = pts.size();
    if (n_points != normals.size())
    {
        return ::cudaErrorInvalidValue;
    }

    thrust::device_vector<gca::index_t> all_neighbors;
    thrust::device_vector<thrust::pair<gca::index_t, gca::counter_t>>
        pair_neighbors_begin_idx_and_count;

    auto err = cuda_search_radius_neighbors(all_neighbors, pair_neighbors_begin_idx_and_count, pts,
                                            min_bound, max_bound, search_radius);
    if (err != ::cudaSuccess)
    {
        return err;
    }

    if (result.size() != n_points)
    {
        result.resize(n_points);
    }

    auto func = compute_color_gradient_functor(pts, normals, all_neighbors,
                                               pair_neighbors_begin_idx_and_count);

    thrust::transform(thrust::make_counting_iterator<gca::index_t>(0),
                      thrust::make_counting_iterator<gca::index_t>(n_points), result.begin(), func);

    return ::cudaSuccess;
}
} // namespace gca