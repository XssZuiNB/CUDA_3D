#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include "geometry/cuda_grid_radius_outliers_removal.cuh"
#include "geometry/geometry_util.cuh"
#include "geometry/type.hpp"
#include "util/cuda_util.cuh"

namespace gca
{
/*functors*/

struct compute_grid_cell_functor
{
    compute_grid_cell_functor(const float3 &grid_cells_min_bound, const float grid_cell_size)
        : m_grid_cells_min_bound(grid_cells_min_bound)
        , m_grid_cell_size(grid_cell_size)
    {
    }

    const float3 m_grid_cells_min_bound;
    const float m_grid_cell_size;

    __forceinline__ __device__ int3 operator()(const gca::point_t &point) const
    {
        int3 grid_cell;
        grid_cell.x =
            __float2int_rd((point.coordinates.x - m_grid_cells_min_bound.x) / m_grid_cell_size);
        grid_cell.y =
            __float2int_rd((point.coordinates.y - m_grid_cells_min_bound.y) / m_grid_cell_size);
        grid_cell.z =
            __float2int_rd((point.coordinates.z - m_grid_cells_min_bound.z) / m_grid_cell_size);

        return grid_cell;
    }
};

struct compute_grid_cell_index_functor
{
    compute_grid_cell_index_functor(const float3 &grid_cells_min_bound, const float grid_cell_size,
                                    const gca::counter_t number_of_grid_cells_y,
                                    const gca::counter_t number_of_grid_cells_z)
        : m_compute_grid_cell(grid_cells_min_bound, grid_cell_size)
        , m_number_of_grid_cells_y(number_of_grid_cells_y)
        , m_number_of_grid_cells_z(number_of_grid_cells_z)
    {
    }

    const compute_grid_cell_functor m_compute_grid_cell;
    const gca::counter_t m_number_of_grid_cells_y;
    const gca::counter_t m_number_of_grid_cells_z;

    __forceinline__ __device__ gca::index_t operator()(const gca::point_t &point) const
    {
        auto grid_cell = m_compute_grid_cell(point);

        return grid_cell.x * m_number_of_grid_cells_y * m_number_of_grid_cells_z +
               grid_cell.y * m_number_of_grid_cells_z + grid_cell.z;
    }
};

struct get_begin_index_of_keys_functor
{
    __forceinline__ __device__ gca::index_t operator()(const gca::index_t begin,
                                                       const gca::index_t __)
    {
        return begin;
    }
};

struct fill_grid_cells_functor
{
    fill_grid_cells_functor(thrust::device_vector<gca::grid_cell_t> &result)
        : m_result(result.data().get())
    {
    }

    gca::grid_cell_t *m_result;

    __forceinline__ __device__ void operator()(
        thrust::tuple<gca::index_t, gca::index_t, gca::counter_t> grid_cell_tuple) const
    {
        gca::index_t index = thrust::get<0>(grid_cell_tuple);
        m_result[index].start_index = thrust::get<1>(grid_cell_tuple);
        m_result[index].points_number = thrust::get<2>(grid_cell_tuple);
    }
};

struct check_if_enough_radius_nn_functor
{
    check_if_enough_radius_nn_functor(
        const thrust::device_vector<gca::point_t> &src_points,
        const thrust::device_vector<gca::index_t> &src_points_sorted_idxs,
        const thrust::device_vector<gca::grid_cell_t> &grid_cells_vec,
        const float3 &grid_cells_min_bound, const gca::counter_t n_grid_cells_x,
        const gca::counter_t n_grid_cells_y, const gca::counter_t n_grid_cells_z,
        const float grid_cell_size, const float search_radius,
        const gca::counter_t min_neighbors_in_radius)

        : m_src_points_ptr(thrust::raw_pointer_cast(src_points.data()))
        , m_src_points_sorted_indecies_ptr(thrust::raw_pointer_cast(src_points_sorted_idxs.data()))
        , m_grid_cells_ptr(thrust::raw_pointer_cast(grid_cells_vec.data()))
        , m_grid_cells_min_bound(grid_cells_min_bound)
        , m_n_grid_cells_x(n_grid_cells_x)
        , m_n_grid_cells_y(n_grid_cells_y)
        , m_n_grid_cells_z(n_grid_cells_z)
        , m_grid_cell_size(grid_cell_size)
        , m_search_radius(search_radius)
        , m_min_neighbors_in_radius(min_neighbors_in_radius)
        , m_compute_grid_cell(grid_cells_min_bound, grid_cell_size)
    {
    }

    const gca::point_t *m_src_points_ptr;
    const gca::index_t *m_src_points_sorted_indecies_ptr;
    const gca::grid_cell_t *m_grid_cells_ptr;
    const float3 m_grid_cells_min_bound;
    const gca::counter_t m_n_grid_cells_x;
    const gca::counter_t m_n_grid_cells_y;
    const gca::counter_t m_n_grid_cells_z;
    const float m_grid_cell_size;
    const float m_search_radius;
    const gca::counter_t m_min_neighbors_in_radius;
    const compute_grid_cell_functor m_compute_grid_cell;

    __forceinline__ __device__ bool operator()(const gca::point_t &point)
    {
        auto grid_cell = m_compute_grid_cell(point);
        /* 1. Devide a grid cell into 8 little cubes
         * 2. Check in which cube the point is
         * 3. Which other grid cells should be considered depend on the point's position
         * 4. Because of this max. only 8 grid cells need to be considered. Comparing to the
         *    other grid cell approach implementation (27 grid cells) ist more than 3 times less
         *    iterations in for loop.
         * 5. The result of this implementation shows more than 3 times faster than other CUDA
         *    implementation and more than 100 times faster than PCL CPU implementaion.
         * 6. This implementation is first version and still not perfect, there are also
         *    possibilities to optimize it.
         */
        auto min_x_this_grid_cell = m_grid_cell_size * grid_cell.x + m_grid_cells_min_bound.x;
        gca::index_t ix_begin =
            (grid_cell.x == 0 || (point.coordinates.x - min_x_this_grid_cell) >= m_search_radius)
                ? 0
                : -1;
        gca::index_t ix_end = (grid_cell.x == (m_n_grid_cells_x - 1) ||
                               (point.coordinates.x - min_x_this_grid_cell) < m_search_radius)
                                  ? 1
                                  : 2;

        auto min_y_this_grid_cell = m_grid_cell_size * grid_cell.y + m_grid_cells_min_bound.y;
        gca::index_t iy_begin =
            (grid_cell.y == 0 || (point.coordinates.y - min_y_this_grid_cell) >= m_search_radius)
                ? 0
                : -1;
        gca::index_t iy_end = (grid_cell.y == (m_n_grid_cells_y - 1) ||
                               (point.coordinates.y - min_y_this_grid_cell) < m_search_radius)
                                  ? 1
                                  : 2;

        auto min_z_this_grid_cell = m_grid_cell_size * grid_cell.z + m_grid_cells_min_bound.z;
        gca::index_t iz_begin =
            (grid_cell.z == 0 || (point.coordinates.z - min_z_this_grid_cell) >= m_search_radius)
                ? 0
                : -1;
        gca::index_t iz_end = (grid_cell.z == (m_n_grid_cells_z - 1) ||
                               (point.coordinates.z - min_z_this_grid_cell) < m_search_radius)
                                  ? 1
                                  : 2;

        auto idx_this_grid_cell = grid_cell.x * m_n_grid_cells_y * m_n_grid_cells_z +
                                  grid_cell.y * m_n_grid_cells_z + grid_cell.z;

        gca::counter_t n_neighbors_in_radius = 0;

        for (auto i = ix_begin; i < ix_end; i++)
        {
            for (auto j = iy_begin; j < iy_end; j++)
            {
                for (auto k = iz_begin; k < iz_end; k++)
                {
                    auto idx_neighbor_grid_cell =
                        idx_this_grid_cell +
                        (i * m_n_grid_cells_y * m_n_grid_cells_z + j * m_n_grid_cells_z + k);
                    auto neighbor_grid_cell = m_grid_cells_ptr[idx_neighbor_grid_cell];
                    auto n_points = neighbor_grid_cell.points_number;
                    if (n_points == 0)
                        continue;

                    for (auto offset = 0; offset < n_points; offset++)
                    {
                        auto idx_of_sorted_points_indecies =
                            neighbor_grid_cell.start_index + offset;
                        auto idx_of_neighbor_point =
                            m_src_points_sorted_indecies_ptr[idx_of_sorted_points_indecies];
                        auto neighbor_point = m_src_points_ptr[idx_of_neighbor_point];

                        auto euclidean_distance_square =
                            (neighbor_point.coordinates.x - point.coordinates.x) *
                                (neighbor_point.coordinates.x - point.coordinates.x) +
                            (neighbor_point.coordinates.y - point.coordinates.y) *
                                (neighbor_point.coordinates.y - point.coordinates.y) +
                            (neighbor_point.coordinates.z - point.coordinates.z) *
                                (neighbor_point.coordinates.z - point.coordinates.z);

                        if (euclidean_distance_square < m_search_radius * m_search_radius)
                        {
                            n_neighbors_in_radius += 1;
                        }

                        if (n_neighbors_in_radius > m_min_neighbors_in_radius)
                        {
                            return true;
                        }
                    }
                }
            }
        }

        return false;
    }
};

::cudaError_t cuda_grid_radius_outliers_removal(
    thrust::device_vector<gca::point_t> &result_points,
    const thrust::device_vector<gca::point_t> &src_points, const float3 min_bound,
    const float3 max_bound, const float search_radius, const gca::counter_t min_neighbors_in_radius)
{
    /* 1. build and fill grid cells */
    auto grid_cell_size = search_radius * 2;
    auto n_grid_cells_x =
        static_cast<gca::counter_t>((max_bound.x - min_bound.x) / grid_cell_size) + 1;
    auto n_grid_cells_y =
        static_cast<gca::counter_t>((max_bound.y - min_bound.y) / grid_cell_size) + 1;
    auto n_grid_cells_z =
        static_cast<gca::counter_t>((max_bound.z - min_bound.z) / grid_cell_size) + 1;

    auto grid_cells_n = n_grid_cells_x * n_grid_cells_y * n_grid_cells_z;
    thrust::device_vector<gca::grid_cell_t> grid_cells_vec(grid_cells_n); // build grid cells

    auto n_points = src_points.size();
    if (result_points.size() != n_points)
    {
        return ::cudaErrorInvalidValue;
    }

    thrust::device_vector<gca::index_t> grid_cell_idxs(n_points);
    thrust::transform(
        src_points.begin(), src_points.end(), grid_cell_idxs.begin(),
        compute_grid_cell_index_functor(min_bound, grid_cell_size, n_grid_cells_y, n_grid_cells_z));
    auto err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    thrust::device_vector<gca::index_t> src_points_sorted_idxs(n_points);
    thrust::sequence(src_points_sorted_idxs.begin(), src_points_sorted_idxs.end());
    thrust::sort_by_key(grid_cell_idxs.begin(), grid_cell_idxs.end(),
                        src_points_sorted_idxs.begin());
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    thrust::device_vector<gca::index_t> idxs_of_sorted_idxs(n_points);
    thrust::sequence(idxs_of_sorted_idxs.begin(), idxs_of_sorted_idxs.end());
    auto end_iter_idx_of_sorted_idx =
        thrust::reduce_by_key(grid_cell_idxs.begin(), grid_cell_idxs.end(),
                              idxs_of_sorted_idxs.begin(), thrust::make_discard_iterator(),
                              idxs_of_sorted_idxs.begin(), thrust::equal_to<gca::index_t>(),
                              get_begin_index_of_keys_functor())
            .second;
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    thrust::device_vector<gca::counter_t> points_counter_per_grid_cell(n_points, 1);
    auto end_iter_pair_grid_cell_idxs_and_counter = thrust::reduce_by_key(
        grid_cell_idxs.begin(), grid_cell_idxs.end(), points_counter_per_grid_cell.begin(),
        grid_cell_idxs.begin(), points_counter_per_grid_cell.begin(),
        thrust::equal_to<gca::index_t>());
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    auto unique_grid_cells_n =
        end_iter_pair_grid_cell_idxs_and_counter.first - grid_cell_idxs.begin();

    if (unique_grid_cells_n != end_iter_idx_of_sorted_idx - idxs_of_sorted_idxs.begin())
    {
        return ::cudaErrorInvalidValue;
    }

    thrust::for_each(thrust::make_zip_iterator(
                         thrust::make_tuple(grid_cell_idxs.begin(), idxs_of_sorted_idxs.begin(),
                                            points_counter_per_grid_cell.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(
                         grid_cell_idxs.begin() + unique_grid_cells_n,
                         idxs_of_sorted_idxs.begin() + unique_grid_cells_n,
                         points_counter_per_grid_cell.begin() + unique_grid_cells_n)),
                     fill_grid_cells_functor(grid_cells_vec));
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    /* 2. For each point, compute euclidean distances between the point and all the points in
     * neighbor grids to check if they are in the radius of the point. Euclidean distance
     * requires sqrt, which might be slow, so here instead it square number should be used */
    auto end_iter_result_points = thrust::copy_if(
        src_points.begin(), src_points.end(), result_points.begin(),
        check_if_enough_radius_nn_functor(src_points, src_points_sorted_idxs, grid_cells_vec,
                                          min_bound, n_grid_cells_x, n_grid_cells_y, n_grid_cells_z,
                                          grid_cell_size, search_radius, min_neighbors_in_radius));
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    result_points.resize(end_iter_result_points - result_points.begin());
    return ::cudaSuccess;
}
} // namespace gca
