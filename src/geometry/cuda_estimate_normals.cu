#include "cuda_estimate_normals.cuh"

#include "geometry/cuda_nn_search.cuh"
#include "geometry/type.hpp"
#include "util/cuda_util.cuh"
#include "util/fast_eigen_val_vec.cuh"
#include "util/math.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

namespace gca
{
struct fill_start_keys_functor
{
    fill_start_keys_functor(const thrust::device_vector<thrust::pair<gca::index_t, gca::counter_t>>
                                &pair_neighbors_begin_idx_and_count,
                            thrust::device_vector<gca::index_t> &keys)
        : m_pair_ptr(thrust::raw_pointer_cast(pair_neighbors_begin_idx_and_count.data()))
        , m_keys_ptr(thrust::raw_pointer_cast(keys.data()))
    {
    }

    const thrust::pair<gca::index_t, gca::counter_t> *m_pair_ptr;
    gca::index_t *m_keys_ptr;

    __forceinline__ __device__ void operator()(const gca::counter_t idx)
    {
        auto start_idx = __ldg((&m_pair_ptr[idx].first));
        auto n = __ldg((&m_pair_ptr[idx].second));
        for (gca::counter_t i = 0; i < n; i++)
        {
            m_keys_ptr[start_idx + i] = idx;
        }
    }
};

struct compute_cumulant_functor
{
    compute_cumulant_functor(const thrust::device_vector<gca::point_t> &points)
        : m_points_ptr(thrust::raw_pointer_cast(points.data()))
    {
    }

    const gca::point_t *m_points_ptr;

    __forceinline__ __device__ mat9x1 operator()(gca::index_t idx) const
    {
        mat9x1 cm;
        if (idx < 0)
        {
            cm.set_zero();
            return cm;
        }

        const auto point_x = __ldg(&m_points_ptr[idx].coordinates.x);
        const auto point_y = __ldg(&m_points_ptr[idx].coordinates.y);
        const auto point_z = __ldg(&m_points_ptr[idx].coordinates.z);

        cm(0) = point_x;
        cm(1) = point_y;
        cm(2) = point_z;
        cm(3) = point_x * point_x;
        cm(4) = point_x * point_y;
        cm(5) = point_x * point_z;
        cm(6) = point_y * point_y;
        cm(7) = point_y * point_z;
        cm(8) = point_z * point_z;

        return cm;
    }
};

struct compute_normal_functor
{
    compute_normal_functor(){};

    __device__ float3
    operator()(const thrust::tuple<mat9x1, thrust::pair<gca::index_t, gca::counter_t>> &tuple) const
    {
        auto count = thrust::get<1>(tuple).second;
        if (count < 3)
        {
            return make_float3(0.0f, 0.0f, 1.0f);
        }

        auto cum = thrust::get<0>(tuple);
        auto cumulants = cum / (float)count;

        mat3x3 covariance;
        covariance(0, 0) = cumulants(3) - cumulants(0) * cumulants(0);
        covariance(1, 1) = cumulants(6) - cumulants(1) * cumulants(1);
        covariance(2, 2) = cumulants(8) - cumulants(2) * cumulants(2);
        covariance(0, 1) = cumulants(4) - cumulants(0) * cumulants(1);
        covariance(1, 0) = covariance(0, 1);
        covariance(0, 2) = cumulants(5) - cumulants(0) * cumulants(2);
        covariance(2, 0) = covariance(0, 2);
        covariance(1, 2) = cumulants(7) - cumulants(1) * cumulants(2);
        covariance(2, 1) = covariance(1, 2);

        auto eigen_result = fast_eigen_compute_3x3_symm(covariance);

        auto eigen_pair_0 = thrust::get<0>(eigen_result);
        auto eigen_pair_1 = thrust::get<1>(eigen_result);
        auto eigen_pair_2 = thrust::get<2>(eigen_result);

        float3 normal;
        if (eigen_pair_0.first < eigen_pair_1.first && eigen_pair_0.first < eigen_pair_2.first)
        {
            normal = eigen_pair_0.second;
        }
        else if (eigen_pair_1.first < eigen_pair_0.first && eigen_pair_1.first < eigen_pair_2.first)
        {
            normal = eigen_pair_1.second;
        }
        else
            normal = eigen_pair_2.second;

        return (norm(normal) > 0.0f) ? normal : make_float3(0.0f, 0.0f, 1.0f);
    }
};

struct align_normal_direction_functor
{
    align_normal_direction_functor()
        : m_orientation_reference(make_float3(0.0f, 0.0f, -1.0f))
    {
    }

    float3 m_orientation_reference;

    __device__ void operator()(float3 &normal) const
    {
        if (norm3df(normal.x, normal.y, normal.z) == 0.0f)
        {
            normal = make_float3(0.0f, 0.0f, 1.0f);
        }
        else if (dot(normal, m_orientation_reference) < 0.0f)
        {
            normal *= -1.0f;
        }
    }
};

::cudaError_t cuda_estimate_normals(thrust::device_vector<float3> &result_normals,
                                    const thrust::device_vector<gca::point_t> &points,
                                    const float3 min_bound, const float3 max_bound,
                                    const float search_radius)
{
    auto n_points = points.size();
    if (result_normals.size() != n_points)
    {
        result_normals.resize(n_points);
    }

    thrust::device_vector<gca::index_t> all_neighbors;
    thrust::device_vector<thrust::pair<gca::index_t, gca::counter_t>>
        pair_neighbors_begin_idx_and_count;

    auto err = cuda_search_radius_neighbors(all_neighbors, pair_neighbors_begin_idx_and_count,
                                            points, min_bound, max_bound, search_radius);
    if (err != ::cudaSuccess)
    {
        check_cuda_error(err, __FILE__, __LINE__);
        return err;
    }

    thrust::counting_iterator<gca::counter_t> counter_iter(0);
    thrust::device_vector<gca::index_t> keys(all_neighbors.size());
    thrust::for_each(counter_iter, counter_iter + pair_neighbors_begin_idx_and_count.size(),
                     fill_start_keys_functor(pair_neighbors_begin_idx_and_count, keys));

    thrust::device_vector<mat9x1> cumulants_sum(n_points);
    auto compute_cumulant_iter =
        thrust::make_transform_iterator(all_neighbors.begin(), compute_cumulant_functor(points));
    thrust::reduce_by_key(keys.begin(), keys.end(), compute_cumulant_iter,
                          thrust::make_discard_iterator(), cumulants_sum.begin());

    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(
                          cumulants_sum.begin(), pair_neighbors_begin_idx_and_count.begin())),
                      thrust::make_zip_iterator(thrust::make_tuple(
                          cumulants_sum.end(), pair_neighbors_begin_idx_and_count.end())),
                      result_normals.begin(), compute_normal_functor());

    // This function could be used but now its not needed.
    /*
        thrust::for_each(result_normals.begin(), result_normals.end(),
                         align_normal_direction_functor());
    */
    return ::cudaSuccess;
}
} // namespace gca
