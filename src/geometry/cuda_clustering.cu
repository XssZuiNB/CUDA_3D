#include "cuda_clustering.cuh"

#include "geometry/cuda_nn_search.cuh"
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
#include <future>
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
        check_cuda_error(err, __FILE__, __LINE__);
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

    for (gca::index_t i = 0; i < n_points; ++i)
    {
        if (visited[i])
            if (visited[i])
            {
                continue;
            }

        std::vector<gca::index_t> seed_queue;
        seed_queue.reserve(max_cluster_size);
        gca::index_t sq_idx = 0;
        seed_queue.push_back(i);

        visited[i] = 1;

        while (sq_idx < static_cast<gca::index_t>(seed_queue.size()))
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

        if (seed_queue.size() >= min_cluster_size && seed_queue.size() <= max_cluster_size)
        {
            for (const auto &neighbor : seed_queue)
            {
                cluster_of_point[neighbor] = cluster;
            }
            ++cluster;
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

::cudaError_t cuda_local_convex_segmentation(std::vector<thrust::host_vector<gca::index_t>> &objs,
                                             const thrust::device_vector<gca::point_t> &points,
                                             const thrust::device_vector<float3> &normals,
                                             const float3 min_bound, const float3 max_bound,
                                             const float cluster_tolerance,
                                             const gca::counter_t min_cluster_size,
                                             const gca::counter_t max_cluster_size)
{
    auto start = std::chrono::steady_clock::now();
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

    std::vector<uint8_t> visited(n_points, 0); // DO NOT use vector<bool>!!!
    gca::index_t cluster = 0;
    objs.clear();
    
    auto check_local_convex = [&](gca::index_t p1_idx, gca::index_t p2_idx) -> bool {
        /*paper */
        float3 d = points_host[p2_idx].coordinates - points_host[p1_idx].coordinates;
        float d_norm = norm(d);

        float3 &n1 = normals_host[p1_idx];
        float3 &n2 = normals_host[p2_idx];

        /********************************************************/
        bool condition1 = (dot(n1, d) <= d_norm * cosf(M_PI / 2.0f - 0.1f)) &&
                          (dot(n2, -d) <= d_norm * cosf(M_PI / 2.0f - 0.1f)) &&
                          ((dot(cross(n1, d), n2)) <= 0.3 * d_norm) &&
                          ((dot(cross(n2, -d), n1)) <= 0.3 * d_norm);

        bool condition2 = dot(n1, n2) >= 1 - d_norm * cosf(M_PI / 2.0f - 0.03f);

        return (condition1 || condition2);
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

                if (check_local_convex(this_p, neighbor))
                {
                    visited[neighbor] = 1;
                    seed_queue.push_back(neighbor);
                }
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
    auto end = std::chrono::steady_clock::now();
    std::cout << " time in milliseconds: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms"
              << std::endl;
    return ::cudaSuccess;
}

::cudaError_t cuda_local_convex_segmentation_async(
    std::vector<thrust::host_vector<gca::index_t>> &objs,
    const thrust::device_vector<gca::point_t> &points, const thrust::device_vector<float3> &normals,
    const float3 min_bound, const float3 max_bound, const float cluster_tolerance,
    const gca::counter_t min_cluster_size, const gca::counter_t max_cluster_size)
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

    std::vector<uint8_t> visited(n_points, 0); // DO NOT use vector<bool>!!!
    std::atomic<gca::index_t> cluster = 0;

    objs.clear();

    auto check_local_convex = [&](gca::index_t p1_idx, gca::index_t p2_idx) -> bool {
        /*paper */
        float3 d = points_host[p2_idx].coordinates - points_host[p1_idx].coordinates;
        double d_norm = norm(d);

        float3 &n1 = normals_host[p1_idx];
        float3 &n2 = normals_host[p2_idx];

        /******************************************************* 0.5 is 10 deg.*/
        bool condition1 = (dot(n1, d) <= d_norm * cos(M_PI / 2 - 0.02)) &&
                          (dot(n2, -d) <= d_norm * cos(M_PI / 2 - 0.02));

        bool condition2 = dot(n1, n2) >= 1 - d_norm * cos(M_PI / 2 - 0.02);

        return condition1 || condition2;
    };

    /* Multi threading for new cluster */
    std::mutex result_vec_mutex;
    static constexpr uint8_t max_threads = 1;
    std::future<void> futures[max_threads];
    std::atomic<bool> async_done[max_threads] = {false};
    auto make_new_cluster_async = [&](gca::index_t pts_idx, gca::index_t furture_idx) -> void {
        if (__sync_bool_compare_and_swap(&visited[pts_idx], 0, 1)) // only on gcc!!!
        {
            std::vector<gca::index_t> seed_queue;
            seed_queue.reserve(max_cluster_size);
            gca::index_t sq_idx = 0;
            seed_queue.push_back(pts_idx);

            while (sq_idx < static_cast<gca::index_t>(seed_queue.size()))
            {

                auto this_p = seed_queue[sq_idx];
                auto neighbor_begin_idx = pair_neighbors_begin_idx_and_count_host[this_p].first;
                auto n_neighbors = pair_neighbors_begin_idx_and_count_host[this_p].second;

                for (gca::index_t j = 0; j < n_neighbors; ++j)
                {
                    auto neighbor = all_neighbors_host[neighbor_begin_idx + j];
                    if (__sync_bool_compare_and_swap(&visited[neighbor], 1, 1))
                        continue;

                    if (!check_local_convex(this_p, neighbor))
                        continue;

                    if (__sync_bool_compare_and_swap(&visited[neighbor], 0, 1))
                    {
                        seed_queue.push_back(neighbor);
                    }
                }

                ++sq_idx;
            }

            if (seed_queue.size() >= min_cluster_size && seed_queue.size() <= max_cluster_size)
            {
                std::lock_guard<std::mutex> lock(result_vec_mutex);
                objs.push_back(thrust::host_vector<gca::index_t>(seed_queue));
                cluster.fetch_add(1);
            }

            async_done[furture_idx].store(true);
        }
    };

    auto start = std::chrono::steady_clock::now();
    for (gca::index_t i = 0; i < n_points; ++i)
    {
        if (__sync_bool_compare_and_swap(&visited[i], 0, 1))
        {
            std::vector<gca::index_t> seed_queue;
            seed_queue.reserve(max_cluster_size);
            gca::index_t sq_idx = 0;
            seed_queue.push_back(i);

            while (sq_idx < static_cast<gca::index_t>(seed_queue.size()))
            {
                auto this_p = seed_queue[sq_idx];
                auto neighbor_begin_idx = pair_neighbors_begin_idx_and_count_host[this_p].first;
                auto n_neighbors = pair_neighbors_begin_idx_and_count_host[this_p].second;

                for (gca::index_t j = 0; j < n_neighbors; ++j)
                {
                    auto neighbor = all_neighbors_host[neighbor_begin_idx + j];
                    if (__sync_bool_compare_and_swap(&visited[neighbor], 1, 1))
                        continue;

                    if (!check_local_convex(this_p, neighbor))
                    {
                        for (gca::index_t f_idx = 0; f_idx < max_threads; ++f_idx)
                        {
                            auto &f = futures[f_idx];
                            if (f.valid() && async_done[f_idx].load())
                            {
                                f.get();
                                f = std::async(make_new_cluster_async, neighbor, f_idx);
                                async_done[f_idx].store(false);
                            }
                            else if (!f.valid())
                            {
                                f = std::async(make_new_cluster_async, neighbor, f_idx);
                                async_done[f_idx].store(false);
                            }
                        }
                        continue;
                    }

                    if (__sync_bool_compare_and_swap(&visited[neighbor], 0, 1))
                    {
                        seed_queue.push_back(neighbor);
                    }
                }

                ++sq_idx;
            }

            if (seed_queue.size() >= min_cluster_size && seed_queue.size() <= max_cluster_size)
            {
                std::lock_guard<std::mutex> lock(result_vec_mutex);
                objs.push_back(thrust::host_vector<gca::index_t>(seed_queue));
                cluster.fetch_add(1);
            }
        }
    }
    for (auto &f : futures)
    {
        if (f.valid())
        {
            f.get();
        }
    }

    auto end = std::chrono::steady_clock::now();

    std::cout << " time in milliseconds: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms"
              << std::endl;
    return ::cudaSuccess;
}
} // namespace gca
