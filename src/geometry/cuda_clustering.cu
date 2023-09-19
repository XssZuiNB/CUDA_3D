#include "geometry/cuda_nn_search.cuh"
#include "util/cuda_util.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/find.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <vector>

namespace gca
{
struct fill_point_in_this_cluster_vec_functor
{
    fill_point_in_this_cluster_vec_functor(
        thrust::device_vector<uint> &parallel_queue,
        thrust::device_vector<uint> &point_in_this_cluster_vec,
        const thrust::device_vector<gca::index_t> &all_neighbors,
        const thrust::device_vector<thrust::pair<gca::index_t, gca::counter_t>>
            &pair_neighbors_begin_idx_and_count,
        const thrust::device_vector<uint> &visited, std::shared_ptr<bool> if_done)
        : m_parallel_queue_ptr(thrust::raw_pointer_cast(parallel_queue.data()))
        , m_point_in_this_cluster_vec_ptr(
              thrust::raw_pointer_cast(point_in_this_cluster_vec.data()))
        , m_all_neighbors_ptr(thrust::raw_pointer_cast(all_neighbors.data()))
        , m_pair_neighbors_begin_idx_and_count_ptr(
              thrust::raw_pointer_cast(pair_neighbors_begin_idx_and_count.data()))
        , m_visited_ptr(thrust::raw_pointer_cast(visited.data()))
        , m_if_done(if_done.get())
    {
    }

    uint *m_parallel_queue_ptr;
    uint *m_point_in_this_cluster_vec_ptr;
    const gca::index_t *m_all_neighbors_ptr;
    const thrust::pair<gca::index_t, gca::counter_t> *m_pair_neighbors_begin_idx_and_count_ptr;
    const uint *m_visited_ptr;
    bool *m_if_done;

    __forceinline__ __device__ void operator()(gca::counter_t i)
    {
        if (m_parallel_queue_ptr[i])
        {
            m_parallel_queue_ptr[i] = 0; // dequeue
            m_point_in_this_cluster_vec_ptr[i] = 1;

            /* Find its all neighbors */
            auto neighbor_begin_idx = __ldg(&(m_pair_neighbors_begin_idx_and_count_ptr[i].first));
            auto n_neighbors = __ldg(&(m_pair_neighbors_begin_idx_and_count_ptr[i].second));

            for (auto j = 0; j < n_neighbors; j++)
            {
                auto neighbor_idx = __ldg(&(m_all_neighbors_ptr[neighbor_begin_idx + j]));
                if (__ldg(&(m_visited_ptr[neighbor_idx])) == 0)
                {
                    *m_if_done = false;
                    m_parallel_queue_ptr[neighbor_idx] = 1; // enqueue all the neighbors
                    // mark its all neighbors as should be processed
                    m_point_in_this_cluster_vec_ptr[neighbor_idx] = 1;
                }
            }
        }
    }
};

struct set_as_visited_and_set_cluster_functor
{
    set_as_visited_and_set_cluster_functor(thrust::device_vector<uint> &visited,
                                           gca::index_t cluster)
        : m_visited(thrust::raw_pointer_cast(visited.data()))
        , m_cluster(cluster)
    {
    }

    uint *m_visited;
    gca::index_t m_cluster;

    __forceinline__ __device__ gca::index_t operator()(uint if_clustered, gca::counter_t i)
    {
        if (if_clustered)
        {
            m_visited[i] = 1;
            return m_cluster;
        }

        return -1;
    }
};

struct check_if_queue_empty_functor
{
    __forceinline__ __device__ uint operator()(uint fir, uint sec)
    {
        return fir || sec;
    }
};

::cudaError_t cuda_euclidean_clustering(thrust::device_vector<gca::index_t> &cluster_of_point,
                                        gca::counter_t &total_clusters,
                                        const thrust::device_vector<gca::point_t> &points,
                                        const float3 min_bound, const float3 max_bound,
                                        const float cluster_tolerance,
                                        const gca::counter_t min_cluster_size,
                                        const gca::counter_t max_cluster_size)
{
    if (min_cluster_size <= 0 || max_cluster_size <= 0 || max_cluster_size < min_cluster_size)
    {
        return ::cudaErrorInvalidValue;
    }

    auto n_points = points.size();
    if (cluster_of_point.size() != n_points)
    {
        cluster_of_point.resize(n_points);
    }

    thrust::device_vector<gca::index_t> all_neighbors;
    thrust::device_vector<thrust::pair<gca::index_t, gca::counter_t>>
        pair_neighbors_begin_idx_and_count;

    auto err = cuda_search_radius_neighbors(all_neighbors, pair_neighbors_begin_idx_and_count,
                                            points, min_bound, max_bound, cluster_tolerance);
    if (err != ::cudaSuccess)
    {
        return err;
    }

    thrust::device_vector<uint> visited(n_points, 0);
    std::vector<uint> visited_host(n_points, 0);
    gca::index_t cluster = 0;
    const bool bool_true = true;
    while (true)
    {
        auto find_not_visited = thrust::find(visited.begin(), visited.end(), 0);
        if (find_not_visited == visited.end())
        {
            break;
        }

        thrust::device_vector<uint> parallel_queue(n_points, 0);
        // points should be processed
        thrust::device_vector<uint> point_in_this_cluster(n_points, 0);

        auto i = find_not_visited - visited.begin();
        // enqueue, dequeue in functor "fill_point_in_this_cluster_vec_functor"
        parallel_queue[i] = 1;

        bool if_done = false;
        std::shared_ptr<bool> if_done_gpu;
        if (!make_device_copy(if_done_gpu, true))
            return cudaErrorMemoryAllocation;

        while (!if_done)
        {
            cudaMemcpy(if_done_gpu.get(), &bool_true, sizeof(bool), cudaMemcpyHostToDevice);

            thrust::for_each(thrust::make_counting_iterator<gca::counter_t>(0),
                             thrust::make_counting_iterator<gca::counter_t>(n_points),
                             fill_point_in_this_cluster_vec_functor(
                                 parallel_queue, point_in_this_cluster, all_neighbors,
                                 pair_neighbors_begin_idx_and_count, visited, if_done_gpu));
            err = cudaGetLastError();
            if (err != ::cudaSuccess)
            {
                return err;
            }

            cudaMemcpy(&if_done, if_done_gpu.get(), sizeof(bool), cudaMemcpyDeviceToHost);

            thrust::transform(point_in_this_cluster.begin(), point_in_this_cluster.end(),
                              thrust::make_counting_iterator<gca::counter_t>(0),
                              cluster_of_point.begin(),
                              set_as_visited_and_set_cluster_functor(visited, cluster));
            err = cudaGetLastError();
            if (err != ::cudaSuccess)
            {
                return err;
            }
        }

        thrust::copy(visited.begin(), visited.end(), visited_host.begin());

        cluster++;
    }

    total_clusters = cluster;

    return ::cudaSuccess;
}

::cudaError_t cuda_euclidean_clustering(std::vector<gca::index_t> &cluster_of_point,
                                        gca::counter_t &total_clusters,
                                        const thrust::device_vector<gca::point_t> &points,
                                        const float3 min_bound, const float3 max_bound,
                                        const float cluster_tolerance,
                                        const gca::counter_t min_cluster_size,
                                        const gca::counter_t max_cluster_size)
{
    if (min_cluster_size <= 0 || max_cluster_size <= 0 || max_cluster_size < min_cluster_size)
    {
        return ::cudaErrorInvalidValue;
    }

    auto n_points = points.size();
    if (cluster_of_point.size() != n_points)
    {
        cluster_of_point.resize(n_points);
    }

    thrust::device_vector<gca::index_t> all_neighbors;
    thrust::device_vector<thrust::pair<gca::index_t, gca::counter_t>>
        pair_neighbors_begin_idx_and_count(n_points);

    auto err = cuda_search_radius_neighbors(all_neighbors, pair_neighbors_begin_idx_and_count,
                                            points, min_bound, max_bound, cluster_tolerance);
    if (err != ::cudaSuccess)
    {
        return err;
    }

    std::vector<gca::index_t> all_neighbors_host(all_neighbors.size());
    std::vector<thrust::pair<gca::index_t, gca::counter_t>> pair_neighbors_begin_idx_and_count_host(
        n_points);

    thrust::copy(all_neighbors.begin(), all_neighbors.end(), all_neighbors_host.begin());
    thrust::copy(pair_neighbors_begin_idx_and_count.begin(),
                 pair_neighbors_begin_idx_and_count.end(),
                 pair_neighbors_begin_idx_and_count_host.begin());

    std::vector<uint8_t> visited(n_points, 0); // DO NOT use vector<bool>!!!
    gca::index_t cluster = 0;

    for (gca::index_t i = 0; i < n_points; i++)
    {
        if (visited[i])
        {
            continue;
        }

        std::vector<gca::index_t> seed_queue;
        gca::index_t sq_idx = 0;
        seed_queue.push_back(i);

        visited[i] = 1;

        while (sq_idx < static_cast<gca::index_t>(seed_queue.size()))
        {

            auto this_p = seed_queue[sq_idx];
            auto neighbor_begin_idx = pair_neighbors_begin_idx_and_count_host[this_p].first;
            auto n_neighbors = pair_neighbors_begin_idx_and_count_host[this_p].second;

            for (gca::index_t j = 0; j < n_neighbors; j++)
            {
                auto neighbor = all_neighbors_host[neighbor_begin_idx + j];
                if (visited[neighbor])
                    continue;

                visited[neighbor] = 1;
                seed_queue.push_back(neighbor);
            }

            sq_idx++;
        }

        if (seed_queue.size() >= min_cluster_size && seed_queue.size() <= max_cluster_size)
        {
            for (const auto &neighbor : seed_queue)
            {
                cluster_of_point[neighbor] = cluster;
            }
            cluster++;
        }
        else
        {
            for (const auto &neighbor : seed_queue)
            {
                cluster_of_point[neighbor] = -1;
            }
        }
    }
    total_clusters = cluster;

    return ::cudaSuccess;
}

::cudaError_t cuda_euclidean_over_segmentation(std::vector<gca::index_t> &cluster_of_point,
                                               gca::counter_t &total_clusters,
                                               const thrust::device_vector<gca::point_t> &points,
                                               const float3 min_bound, const float3 max_bound,
                                               const float cluster_tolerance,
                                               const gca::counter_t min_cluster_size,
                                               const gca::counter_t max_cluster_size)
{
    if (min_cluster_size <= 0 || max_cluster_size <= 0 || max_cluster_size < min_cluster_size)
    {
        return ::cudaErrorInvalidValue;
    }

    auto n_points = points.size();
    if (cluster_of_point.size() != n_points)
    {
        cluster_of_point.resize(n_points);
    }

    thrust::device_vector<gca::index_t> all_neighbors;
    thrust::device_vector<thrust::pair<gca::index_t, gca::counter_t>>
        pair_neighbors_begin_idx_and_count(n_points);

    auto err = cuda_search_radius_neighbors(all_neighbors, pair_neighbors_begin_idx_and_count,
                                            points, min_bound, max_bound, cluster_tolerance);
    if (err != ::cudaSuccess)
    {
        return err;
    }

    std::vector<gca::index_t> all_neighbors_host(all_neighbors.size());
    std::vector<thrust::pair<gca::index_t, gca::counter_t>> pair_neighbors_begin_idx_and_count_host(
        n_points);

    thrust::copy(all_neighbors.begin(), all_neighbors.end(), all_neighbors_host.begin());
    thrust::copy(pair_neighbors_begin_idx_and_count.begin(),
                 pair_neighbors_begin_idx_and_count.end(),
                 pair_neighbors_begin_idx_and_count_host.begin());

    std::vector<uint8_t> visited(n_points, 0); // DO NOT use vector<bool>!!!
    gca::index_t cluster = 0;

    for (gca::index_t i = 0; i < n_points; i++)
    {
        if (visited[i])
        {
            continue;
        }

        std::vector<gca::index_t> seed_queue;
        gca::index_t sq_idx = 0;
        seed_queue.push_back(i);

        visited[i] = 1;

        while (sq_idx < static_cast<gca::index_t>(seed_queue.size()))
        {

            auto this_p = seed_queue[sq_idx];
            auto neighbor_begin_idx = pair_neighbors_begin_idx_and_count_host[this_p].first;
            auto n_neighbors = pair_neighbors_begin_idx_and_count_host[this_p].second;

            for (gca::index_t j = 0; j < n_neighbors; j++)
            {
                auto neighbor = all_neighbors_host[neighbor_begin_idx + j];
                if (visited[neighbor])
                    continue;

                visited[neighbor] = 1;
                seed_queue.push_back(neighbor);
            }

            sq_idx++;
        }

        if (seed_queue.size() >= min_cluster_size && seed_queue.size() <= max_cluster_size)
        {
            for (const auto &neighbor : seed_queue)
            {
                cluster_of_point[neighbor] = cluster;
            }
            cluster++;
        }
        else
        {
            for (const auto &neighbor : seed_queue)
            {
                cluster_of_point[neighbor] = -1;
            }
        }
    }
    total_clusters = cluster;

    return ::cudaSuccess;
}

} // namespace gca
