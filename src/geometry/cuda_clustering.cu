#include "cuda_clustering.cuh"

#include "geometry/cuda_nn_search.cuh"
#include "geometry/geometry_util.cuh"
#include "util/cuda_util.cuh"
#include "util/math.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/find.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <atomic>
#include <cmath>
#include <future>
#include <vector>

namespace gca
{
::cudaError_t cuda_euclidean_clustering(std::vector<thrust::host_vector<gca::index_t>> &clusters,
                                        const thrust::device_vector<gca::point_t> &points,
                                        const float3 min_bound, const float3 max_bound,
                                        const float cluster_tolerance,
                                        const gca::counter_t min_cluster_size,
                                        const gca::counter_t max_cluster_size)
{
    if (min_cluster_size < 0 || max_cluster_size <= 0 || max_cluster_size < min_cluster_size)
    {
        return ::cudaErrorInvalidValue;
    }

    thrust::device_vector<gca::index_t> all_neighbors;
    auto n_points = points.size();
    thrust::device_vector<thrust::pair<gca::index_t, gca::counter_t>>
        pair_neighbors_begin_idx_and_count(n_points);

    auto err = cuda_search_radius_neighbors(all_neighbors, pair_neighbors_begin_idx_and_count,
                                            points, min_bound, max_bound, cluster_tolerance);
    if (err != ::cudaSuccess)
    {
        check_cuda_error(err, __FILE__, __LINE__);
        return err;
    }

    thrust::host_vector<gca::index_t> all_neighbors_host(all_neighbors);
    thrust::host_vector<thrust::pair<gca::index_t, gca::counter_t>>
        pair_neighbors_begin_idx_and_count_host(pair_neighbors_begin_idx_and_count);

    std::vector<uint8_t> visited(n_points, 0); // DO NOT use vector<bool>!!!
    gca::index_t cluster = 0;
    clusters.clear();

    for (gca::index_t i = 0; i < n_points; ++i)
    {
        if (visited[i])
        {
            continue;
        }

        std::vector<gca::index_t> seed_queue;
        seed_queue.reserve(max_cluster_size);
        gca::index_t sq_idx = 0;
        seed_queue.push_back(i);

        visited[i] = 1;

        // std::vector<gca::index_t> false_negative;

        while (sq_idx < static_cast<gca::index_t>(seed_queue.size()) &&
               static_cast<gca::index_t>(seed_queue.size()) < max_cluster_size)
        {

            auto this_p = seed_queue[sq_idx];
            auto neighbor_begin_idx = pair_neighbors_begin_idx_and_count_host[this_p].first;
            auto n_neighbors = pair_neighbors_begin_idx_and_count_host[this_p].second;

            for (gca::index_t j = 0; j < n_neighbors; ++j)
            {
                auto neighbor = all_neighbors_host[neighbor_begin_idx + j];
                if (visited[neighbor])
                    continue;

                visited[neighbor] = 1;
                seed_queue.push_back(neighbor);
            }

            ++sq_idx;
        }

        if (seed_queue.size() >= min_cluster_size)
        {
            thrust::host_vector<gca::index_t> obj(std::move(seed_queue));
            clusters.push_back(std::move(obj));
            ++cluster;
        }
    }

    return ::cudaSuccess;
}

::cudaError_t cuda_local_convex_segmentation(std::vector<thrust::host_vector<gca::index_t>> &objs,
                                             const thrust::device_vector<gca::point_t> &points,
                                             const thrust::device_vector<float3> &normals,
                                             const float3 min_bound, const float3 max_bound,
                                             const float cluster_tolerance,
                                             const gca::counter_t min_cluster_size,
                                             const gca::counter_t max_cluster_size)
{
    if (min_cluster_size < 0 || max_cluster_size <= 0 || max_cluster_size < min_cluster_size)
    {
        return ::cudaErrorInvalidValue;
    }

    thrust::device_vector<gca::index_t> all_neighbors;
    auto n_points = points.size();
    thrust::device_vector<thrust::pair<gca::index_t, gca::counter_t>>
        pair_neighbors_begin_idx_and_count(n_points);

    auto err = cuda_search_radius_neighbors(all_neighbors, pair_neighbors_begin_idx_and_count,
                                            points, min_bound, max_bound, cluster_tolerance);
    if (err != ::cudaSuccess)
    {
        check_cuda_error(err, __FILE__, __LINE__);
        return err;
    }

    thrust::host_vector<gca::index_t> all_neighbors_host(all_neighbors);
    thrust::host_vector<thrust::pair<gca::index_t, gca::counter_t>>
        pair_neighbors_begin_idx_and_count_host(pair_neighbors_begin_idx_and_count);

    thrust::host_vector<gca::point_t> points_host(points);
    thrust::host_vector<float3> normals_host(normals);

    std::vector<gca::index_t> cluster_num_of_pts(n_points, -1);
    std::vector<uint8_t> visited(n_points, 0); // DO NOT use vector<bool>!!!
    gca::index_t cluster = 0;
    objs.clear();

    auto check_local_convex = [&](gca::index_t p1_idx, gca::index_t p2_idx) -> bool {
        /*paper */
        float3 d = points_host[p1_idx].coordinates - points_host[p2_idx].coordinates;
        float d_norm = norm(d);
        d = d / d_norm;

        const float3 &n1 = normals_host[p1_idx];
        const float3 &n2 = normals_host[p2_idx];

        bool condition1 = dot((n1 - n2), d) > 0 || dot(n1, n2) >= cosf(M_PI * (10 / 180)); // 10
                                                                                           // grad

        float3 s = cross(n1, n2);
        float angle_d_s = acos(dot(d, s));
        float theta = min(angle_d_s, M_PI - angle_d_s);
        float theta_thres = (M_PI / 3) / (1 + exp(-0.25 * (acos(dot(n1, n2)) - 0.14 * M_PI)));

        bool condition2 = theta > theta_thres;
        /*
        float3 d = points_host[p2_idx].coordinates -
        points_host[p1_idx].coordinates;
        float d_norm = norm(d);

        float3 n1 = normals_host[p1_idx];
        float3 n2 = normals_host[p2_idx];


        bool condition1 = (dot(n1, d) <= d_norm * cosf(M_PI / 2.0f - 0.05f)) &&
                          (dot(n2, -d) <= d_norm * cosf(M_PI / 2.0f - 0.05f)) &&
                          (abs(dot(cross(n1, d), n2)) <= 0.3f * d_norm) &&
                          (abs(dot(cross(n2, -d), n1)) <= 0.3f * d_norm);

        bool condition2 = dot(n1, n2) >= 1 - d_norm * cosf(M_PI / 2.0f - 0.3f);
        */
        return (condition1 && condition2);
    };

    for (gca::index_t i = 0; i < n_points; ++i)
    {
        if (visited[i])
        {
            continue;
        }

        std::vector<gca::index_t> seed_queue;
        seed_queue.reserve(max_cluster_size);
        gca::index_t sq_idx = 0;
        seed_queue.push_back(i);

        visited[i] = 1;
        cluster_num_of_pts[i] = cluster;

        // std::vector<gca::index_t> false_negative;

        while (sq_idx < static_cast<gca::index_t>(seed_queue.size()) &&
               static_cast<gca::index_t>(seed_queue.size()) < max_cluster_size)
        {

            auto this_p = seed_queue[sq_idx];
            auto neighbor_begin_idx = pair_neighbors_begin_idx_and_count_host[this_p].first;
            auto n_neighbors = pair_neighbors_begin_idx_and_count_host[this_p].second;

            for (gca::index_t j = 0; j < n_neighbors; ++j)
            {
                auto neighbor = all_neighbors_host[neighbor_begin_idx + j];
                if (visited[neighbor])
                    continue;

                if (!check_local_convex(this_p, neighbor))
                {

                    if (distance(points_host[this_p].color, points_host[neighbor].color) < 0.01)
                    {
                        visited[neighbor] = 1;
                        seed_queue.push_back(neighbor);
                    }

                    continue;
                }

                visited[neighbor] = 1;
                seed_queue.push_back(neighbor);
                cluster_num_of_pts[neighbor] = cluster;
            }

            ++sq_idx;
        }

        // seed_queue.insert(seed_queue.end(),
        // std::make_move_iterator(false_negative.begin()),std::make_move_iterator(false_negative.end()));
        if (seed_queue.size() >= min_cluster_size)
        {
            thrust::host_vector<gca::index_t> obj(std::move(seed_queue));
            objs.push_back(std::move(obj));
            ++cluster;
        }
    }

    return ::cudaSuccess;
}

::cudaError_t cuda_residual_based_segmentation(std::vector<thrust::host_vector<gca::index_t>> &objs,
                                               const thrust::device_vector<gca::point_t> &points,
                                               const thrust::device_vector<float> &abs_residual,
                                               const float3 min_bound, const float3 max_bound,
                                               const float cluster_tolerance,
                                               const gca::counter_t min_cluster_size,
                                               const gca::counter_t max_cluster_size)
{
    if (min_cluster_size < 0 || max_cluster_size <= 0 || max_cluster_size < min_cluster_size)
    {
        return ::cudaErrorInvalidValue;
    }

    thrust::device_vector<gca::index_t> all_neighbors;
    auto n_points = points.size();
    thrust::device_vector<thrust::pair<gca::index_t, gca::counter_t>>
        pair_neighbors_begin_idx_and_count(n_points);

    auto err = cuda_search_radius_neighbors(all_neighbors, pair_neighbors_begin_idx_and_count,
                                            points, min_bound, max_bound, cluster_tolerance);
    if (err != ::cudaSuccess)
    {
        check_cuda_error(err, __FILE__, __LINE__);
        return err;
    }

    thrust::host_vector<gca::index_t> all_neighbors_host(all_neighbors);
    thrust::host_vector<thrust::pair<gca::index_t, gca::counter_t>>
        pair_neighbors_begin_idx_and_count_host(pair_neighbors_begin_idx_and_count);

    thrust::host_vector<float> res_host(abs_residual);

    std::vector<gca::index_t> cluster_num_of_pts(n_points, -1);
    std::vector<uint8_t> visited(n_points, 0); // DO NOT use vector<bool>!!!
    gca::index_t cluster = 0;
    objs.clear();

    auto check_residual_diff = [&](gca::index_t p1_idx, gca::index_t p2_idx) -> bool {
        return abs(res_host[p1_idx] - res_host[p2_idx]) < 0.003;
    };

    for (gca::index_t i = 0; i < n_points; ++i)
    {
        if (visited[i])
        {
            continue;
        }

        std::vector<gca::index_t> seed_queue;
        seed_queue.reserve(max_cluster_size);
        gca::index_t sq_idx = 0;
        seed_queue.push_back(i);

        visited[i] = 1;
        cluster_num_of_pts[i] = cluster;

        while (sq_idx < static_cast<gca::index_t>(seed_queue.size()) &&
               static_cast<gca::index_t>(seed_queue.size()) < max_cluster_size)
        {

            auto this_p = seed_queue[sq_idx];
            auto neighbor_begin_idx = pair_neighbors_begin_idx_and_count_host[this_p].first;
            auto n_neighbors = pair_neighbors_begin_idx_and_count_host[this_p].second;

            for (gca::index_t j = 0; j < n_neighbors; ++j)
            {
                auto neighbor = all_neighbors_host[neighbor_begin_idx + j];
                if (visited[neighbor])
                    continue;

                if (!check_residual_diff(this_p, neighbor))
                {
                    continue;
                }

                visited[neighbor] = 1;
                seed_queue.push_back(neighbor);
            }

            ++sq_idx;
        }

        if (seed_queue.size() >= min_cluster_size)
        {
            thrust::host_vector<gca::index_t> obj(std::move(seed_queue));
            objs.push_back(std::move(obj));
            ++cluster;
        }
    }

    return ::cudaSuccess;
}

struct remove_plane_functor
{
    remove_plane_functor(float A, float B, float C, float D, float constraint)
        : m_A(A)
        , m_B(B)
        , m_C(C)
        , m_D(D)
        , m_coef(sqrtf(A * A + B * B + C * C))
        , m_constraint(constraint)
    {
    }

    float m_A;
    float m_B;
    float m_C;
    float m_D;
    float m_coef;
    float m_constraint;

    __forceinline__ __device__ bool operator()(const gca::point_t &pts)
    {
        return ((abs(m_A * pts.coordinates.x + m_B * pts.coordinates.y + m_C * pts.coordinates.z +
                     m_D) /
                 m_coef) > m_constraint);
    }
};

void cuda_remove_plane(thrust::device_vector<gca::point_t> &result_points,
                       const std::vector<float> &plane_model,
                       const thrust::device_vector<gca::point_t> &src_points)
{
    float A(plane_model[0]);
    float B(plane_model[1]);
    float C(plane_model[2]);
    float D(plane_model[3]);

    result_points.resize(src_points.size());

    auto end = thrust::copy_if(src_points.begin(), src_points.end(), result_points.begin(),
                               remove_plane_functor(A, B, C, D, 0.015));

    result_points.resize(end - result_points.begin());
}
} // namespace gca
