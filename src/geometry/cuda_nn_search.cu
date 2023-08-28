#include "geometry/cuda_nn_search.cuh"
#include "geometry/geometry_util.cuh"
#include "geometry/type.hpp"
#include "util/cuda_util.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/pair.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

namespace gca
{

/* Grid cell structure */
struct grid_cell_t
{
    index_t start_index = -1;
    counter_t points_number = 0;
};

inline std::ostream &operator<<(std::ostream &os, const grid_cell_t &cell)
{
    os << "Start Index: " << cell.start_index << ", Points Number: " << cell.points_number << "\n";
    return os;
}

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
        : m_result(thrust::raw_pointer_cast(result.data()))
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

struct nn_search_functor
{
    nn_search_functor(const thrust::device_vector<gca::point_t> &points_R,
                      const thrust::device_vector<gca::index_t> &points_sorted_idxs_R,
                      const thrust::device_vector<gca::grid_cell_t> &grid_cells_R,
                      const float3 &grid_cells_min_bound, const gca::counter_t n_grid_cells_x,
                      const gca::counter_t n_grid_cells_y, const gca::counter_t n_grid_cells_z,
                      const float search_radius)
        : m_points_R_ptr(thrust::raw_pointer_cast(points_R.data()))
        , m_points_sorted_idxs_R_ptr(thrust::raw_pointer_cast(points_sorted_idxs_R.data()))
        , m_grid_cells_R_ptr(thrust::raw_pointer_cast(grid_cells_R.data()))
        , m_grid_cells_min_bound(grid_cells_min_bound)
        , m_n_grid_cells_x(n_grid_cells_x)
        , m_n_grid_cells_y(n_grid_cells_y)
        , m_n_grid_cells_z(n_grid_cells_z)
        , m_search_radius(search_radius)
        , m_search_radius_square(search_radius * search_radius)
        , m_compute_grid_cell(grid_cells_min_bound, search_radius)
    {
    }

    const gca::point_t *m_points_R_ptr;
    const gca::index_t *m_points_sorted_idxs_R_ptr;
    const gca::grid_cell_t *m_grid_cells_R_ptr;
    const float3 m_grid_cells_min_bound;
    const gca::counter_t m_n_grid_cells_x;
    const gca::counter_t m_n_grid_cells_y;
    const gca::counter_t m_n_grid_cells_z;
    const float m_search_radius;
    const float m_search_radius_square;
    const compute_grid_cell_functor m_compute_grid_cell;

    __forceinline__ __device__ gca::index_t operator()(const gca::point_t &point)
    {
        /* 1. Devide a grid cell into 8 little cubes
         * 2. Check in which cube the point is
         * 3. Which other grid cells should be considered depend on the point's position
         * 4. Because of this max. only 8 grid cells need to be considered. Comparing to the
         *    other grid cell approach implementation (27 grid cells) ist more than 3 times less
         *    iterations in for loop.
         * 5. The result of this implementation shows more than 3 times faster than other CUDA
         *    implementation and more than 100 times faster than PCL CPU implementaion.
         */
        auto grid_cell = m_compute_grid_cell(point);

        auto idx_this_grid_cell = grid_cell.x * m_n_grid_cells_y * m_n_grid_cells_z +
                                  grid_cell.y * m_n_grid_cells_z + grid_cell.z;

        float min_distance_square = m_search_radius_square;
        gca::index_t nn_idx_in_R = -1;

        for (auto i = -1; i < 2; i++)
        {
            for (auto j = -1; j < 2; j++)
            {
                for (auto k = -1; k < 2; k++)
                {
                    auto idx_neighbor_grid_cell =
                        idx_this_grid_cell +
                        (i * m_n_grid_cells_y * m_n_grid_cells_z + j * m_n_grid_cells_z + k);

                    auto n_points =
                        __ldg(&(m_grid_cells_R_ptr[idx_neighbor_grid_cell].points_number));
                    if (n_points == 0)
                        continue;

                    auto start_index =
                        __ldg(&(m_grid_cells_R_ptr[idx_neighbor_grid_cell].start_index));
                    if (start_index == -1)
                        continue;

                    for (auto offset = 0; offset < n_points; offset++)
                    {
                        auto idx_of_sorted_points_indecies = start_index + offset;
                        auto idx_of_neighbor_point =
                            __ldg(&m_points_sorted_idxs_R_ptr[idx_of_sorted_points_indecies]);

                        auto neighbor_point_x =
                            __ldg(&(m_points_R_ptr[idx_of_neighbor_point].coordinates.x));
                        auto neighbor_point_y =
                            __ldg(&(m_points_R_ptr[idx_of_neighbor_point].coordinates.y));
                        auto neighbor_point_z =
                            __ldg(&(m_points_R_ptr[idx_of_neighbor_point].coordinates.z));

                        auto diff_x = neighbor_point_x - point.coordinates.x;
                        auto diff_y = neighbor_point_y - point.coordinates.y;
                        auto diff_z = neighbor_point_z - point.coordinates.z;

                        auto euclidean_distance_square =
                            diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;

                        if (euclidean_distance_square < min_distance_square)
                        {
                            min_distance_square = euclidean_distance_square;
                            nn_idx_in_R = idx_of_neighbor_point;
                        }
                    }
                }
            }
        }

        return nn_idx_in_R;
    }
};

struct nn_search_radius_step1_functor
{
    nn_search_radius_step1_functor(const thrust::device_vector<gca::point_t> &points_R,
                                   const thrust::device_vector<gca::index_t> &points_sorted_idxs_R,
                                   const thrust::device_vector<gca::grid_cell_t> &grid_cells_R,
                                   const float3 &grid_cells_min_bound,
                                   const gca::counter_t n_grid_cells_x,
                                   const gca::counter_t n_grid_cells_y,
                                   const gca::counter_t n_grid_cells_z, const float search_radius)
        : m_points_R_ptr(thrust::raw_pointer_cast(points_R.data()))
        , m_points_sorted_idxs_R_ptr(thrust::raw_pointer_cast(points_sorted_idxs_R.data()))
        , m_grid_cells_R_ptr(thrust::raw_pointer_cast(grid_cells_R.data()))
        , m_grid_cells_min_bound(grid_cells_min_bound)
        , m_n_grid_cells_x(n_grid_cells_x)
        , m_n_grid_cells_y(n_grid_cells_y)
        , m_n_grid_cells_z(n_grid_cells_z)
        , m_search_radius(search_radius)
        , m_search_radius_square(search_radius * search_radius)
        , m_compute_grid_cell(grid_cells_min_bound, search_radius)
    {
    }

    const gca::point_t *m_points_R_ptr;
    const gca::index_t *m_points_sorted_idxs_R_ptr;
    const gca::grid_cell_t *m_grid_cells_R_ptr;
    const float3 m_grid_cells_min_bound;
    const gca::counter_t m_n_grid_cells_x;
    const gca::counter_t m_n_grid_cells_y;
    const gca::counter_t m_n_grid_cells_z;
    const float m_search_radius;
    const float m_search_radius_square;
    const compute_grid_cell_functor m_compute_grid_cell;

    __forceinline__ __device__ gca::counter_t operator()(const gca::point_t &point)
    {
        auto grid_cell = m_compute_grid_cell(point);

        auto idx_this_grid_cell = grid_cell.x * m_n_grid_cells_y * m_n_grid_cells_z +
                                  grid_cell.y * m_n_grid_cells_z + grid_cell.z;

        gca::counter_t n_neighbors_in_radius = 0;

        for (auto i = -1; i < 2; i++)
        {
            for (auto j = -1; j < 2; j++)
            {
                for (auto k = -1; k < 2; k++)
                {
                    auto idx_neighbor_grid_cell =
                        idx_this_grid_cell +
                        (i * m_n_grid_cells_y * m_n_grid_cells_z + j * m_n_grid_cells_z + k);

                    auto n_points =
                        __ldg(&(m_grid_cells_R_ptr[idx_neighbor_grid_cell].points_number));
                    if (n_points == 0)
                        continue;

                    auto start_index =
                        __ldg(&(m_grid_cells_R_ptr[idx_neighbor_grid_cell].start_index));

                    for (auto offset = 0; offset < n_points; offset++)
                    {
                        auto idx_of_sorted_points_indecies = start_index + offset;
                        auto idx_of_neighbor_point =
                            __ldg(&m_points_sorted_idxs_R_ptr[idx_of_sorted_points_indecies]);

                        auto neighbor_point_x =
                            __ldg(&(m_points_R_ptr[idx_of_neighbor_point].coordinates.x));
                        auto neighbor_point_y =
                            __ldg(&(m_points_R_ptr[idx_of_neighbor_point].coordinates.y));
                        auto neighbor_point_z =
                            __ldg(&(m_points_R_ptr[idx_of_neighbor_point].coordinates.z));

                        auto diff_x = neighbor_point_x - point.coordinates.x;
                        auto diff_y = neighbor_point_y - point.coordinates.y;
                        auto diff_z = neighbor_point_z - point.coordinates.z;

                        auto euclidean_distance_square =
                            diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;

                        gca::counter_t condition =
                            (euclidean_distance_square < m_search_radius * m_search_radius);
                        n_neighbors_in_radius += condition;
                    }
                }
            }
        }

        return n_neighbors_in_radius;
    }
};

struct nn_search_radius_step2_functor
{
    nn_search_radius_step2_functor(
        thrust::device_vector<gca::index_t> &result_radius_neighbor_idxs_in_R,
        const thrust::device_vector<gca::point_t> &points_R,
        const thrust::device_vector<gca::index_t> &points_sorted_idxs_R,
        const thrust::device_vector<gca::grid_cell_t> &grid_cells_R,
        const float3 &grid_cells_min_bound, const gca::counter_t n_grid_cells_x,
        const gca::counter_t n_grid_cells_y, const gca::counter_t n_grid_cells_z,
        const float search_radius)
        : m_result_radius_neighbor_idxs_in_R_ptr(
              thrust::raw_pointer_cast(result_radius_neighbor_idxs_in_R.data()))
        , m_points_R_ptr(thrust::raw_pointer_cast(points_R.data()))
        , m_points_sorted_idxs_R_ptr(thrust::raw_pointer_cast(points_sorted_idxs_R.data()))
        , m_grid_cells_R_ptr(thrust::raw_pointer_cast(grid_cells_R.data()))
        , m_grid_cells_min_bound(grid_cells_min_bound)
        , m_n_grid_cells_x(n_grid_cells_x)
        , m_n_grid_cells_y(n_grid_cells_y)
        , m_n_grid_cells_z(n_grid_cells_z)
        , m_search_radius(search_radius)
        , m_search_radius_square(search_radius * search_radius)
        , m_compute_grid_cell(grid_cells_min_bound, search_radius)
    {
    }

    gca::index_t *m_result_radius_neighbor_idxs_in_R_ptr;
    const gca::point_t *m_points_R_ptr;
    const gca::index_t *m_points_sorted_idxs_R_ptr;
    const gca::grid_cell_t *m_grid_cells_R_ptr;
    const float3 m_grid_cells_min_bound;
    const gca::counter_t m_n_grid_cells_x;
    const gca::counter_t m_n_grid_cells_y;
    const gca::counter_t m_n_grid_cells_z;
    const float m_search_radius;
    const float m_search_radius_square;
    const compute_grid_cell_functor m_compute_grid_cell;

    __forceinline__ __device__ thrust::pair<gca::index_t, gca::counter_t> operator()(
        const thrust::tuple<gca::point_t, gca::index_t, gca::counter_t> &tuple)
    {
        auto point = thrust::get<0>(tuple);
        auto grid_cell = m_compute_grid_cell(point);

        auto idx_this_grid_cell = grid_cell.x * m_n_grid_cells_y * m_n_grid_cells_z +
                                  grid_cell.y * m_n_grid_cells_z + grid_cell.z;

        auto idx_in_result = thrust::get<1>(tuple);
        auto n_neighbors_in_radius = thrust::get<2>(tuple);

        for (auto i = -1; i < 2; i++)
        {
            for (auto j = -1; j < 2; j++)
            {
                for (auto k = -1; k < 2; k++)
                {
                    auto idx_neighbor_grid_cell =
                        idx_this_grid_cell +
                        (i * m_n_grid_cells_y * m_n_grid_cells_z + j * m_n_grid_cells_z + k);

                    auto n_points =
                        __ldg(&(m_grid_cells_R_ptr[idx_neighbor_grid_cell].points_number));
                    if (n_points == 0)
                        continue;

                    auto start_index =
                        __ldg(&(m_grid_cells_R_ptr[idx_neighbor_grid_cell].start_index));

                    for (auto offset = 0; offset < n_points; offset++)
                    {
                        auto idx_of_sorted_points_indecies = start_index + offset;
                        auto idx_of_neighbor_point =
                            __ldg(&m_points_sorted_idxs_R_ptr[idx_of_sorted_points_indecies]);

                        auto neighbor_point_x =
                            __ldg(&(m_points_R_ptr[idx_of_neighbor_point].coordinates.x));
                        auto neighbor_point_y =
                            __ldg(&(m_points_R_ptr[idx_of_neighbor_point].coordinates.y));
                        auto neighbor_point_z =
                            __ldg(&(m_points_R_ptr[idx_of_neighbor_point].coordinates.z));

                        auto diff_x = neighbor_point_x - point.coordinates.x;
                        auto diff_y = neighbor_point_y - point.coordinates.y;
                        auto diff_z = neighbor_point_z - point.coordinates.z;

                        auto euclidean_distance_square =
                            diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;

                        if (euclidean_distance_square < m_search_radius * m_search_radius)
                        {
                            m_result_radius_neighbor_idxs_in_R_ptr[idx_in_result] =
                                idx_of_neighbor_point;
                            idx_in_result++;
                            n_neighbors_in_radius--;
                        }

                        if (n_neighbors_in_radius == 0)
                        {
                            return thrust::make_pair(thrust::get<1>(tuple), thrust::get<2>(tuple));
                        }
                    }
                }
            }
        }

        return thrust::make_pair(thrust::get<1>(tuple), thrust::get<2>(tuple));
    }
};

struct check_if_enough_radius_nn_functor
{
    check_if_enough_radius_nn_functor(
        const thrust::device_vector<gca::point_t> &src_points,
        const thrust::device_vector<gca::index_t> &src_points_sorted_idxs,
        const thrust::device_vector<gca::grid_cell_t> &grid_cells_vec,
        const float3 grid_cells_min_bound, const gca::counter_t n_grid_cells_x,
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

    __forceinline__ __device__ gca::point_t operator()(const gca::point_t &point)
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
         */
        auto min_x_this_grid_cell =
            __fmaf_rn(m_grid_cell_size, grid_cell.x, m_grid_cells_min_bound.x);
        gca::index_t ix_begin =
            ((point.coordinates.x - min_x_this_grid_cell) >= m_search_radius) ? 0 : -1;
        gca::index_t ix_end =
            ((point.coordinates.x - min_x_this_grid_cell) < m_search_radius) ? 1 : 2;

        auto min_y_this_grid_cell =
            __fmaf_rn(m_grid_cell_size, grid_cell.y, m_grid_cells_min_bound.y);
        gca::index_t iy_begin =
            ((point.coordinates.y - min_y_this_grid_cell) >= m_search_radius) ? 0 : -1;
        gca::index_t iy_end =
            ((point.coordinates.y - min_y_this_grid_cell) < m_search_radius) ? 1 : 2;

        auto min_z_this_grid_cell =
            __fmaf_rn(m_grid_cell_size, grid_cell.z, m_grid_cells_min_bound.z);
        gca::index_t iz_begin =
            ((point.coordinates.z - min_z_this_grid_cell) >= m_search_radius) ? 0 : -1;
        gca::index_t iz_end =
            ((point.coordinates.z - min_z_this_grid_cell) < m_search_radius) ? 1 : 2;

        auto idx_this_grid_cell = grid_cell.x * m_n_grid_cells_y * m_n_grid_cells_z +
                                  grid_cell.y * m_n_grid_cells_z + grid_cell.z;

        gca::counter_t n_neighbors_in_radius = 0;
        auto new_point = point;

        for (auto i = ix_begin; i < ix_end; i++)
        {
            for (auto j = iy_begin; j < iy_end; j++)
            {
                for (auto k = iz_begin; k < iz_end; k++)
                {
                    auto idx_neighbor_grid_cell =
                        idx_this_grid_cell +
                        (i * m_n_grid_cells_y * m_n_grid_cells_z + j * m_n_grid_cells_z + k);

                    auto n_points =
                        __ldg(&(m_grid_cells_ptr[idx_neighbor_grid_cell].points_number));
                    if (n_points == 0)
                        continue;

                    auto start_index =
                        __ldg(&(m_grid_cells_ptr[idx_neighbor_grid_cell].start_index));

                    for (auto offset = 0; offset < n_points; offset++)
                    {
                        auto idx_of_sorted_points_indecies = start_index + offset;
                        auto idx_of_neighbor_point =
                            __ldg(&m_src_points_sorted_indecies_ptr[idx_of_sorted_points_indecies]);

                        auto neighbor_point_x =
                            __ldg(&(m_src_points_ptr[idx_of_neighbor_point].coordinates.x));
                        auto neighbor_point_y =
                            __ldg(&(m_src_points_ptr[idx_of_neighbor_point].coordinates.y));
                        auto neighbor_point_z =
                            __ldg(&(m_src_points_ptr[idx_of_neighbor_point].coordinates.z));

                        auto diff_x = neighbor_point_x - point.coordinates.x;
                        auto diff_y = neighbor_point_y - point.coordinates.y;
                        auto diff_z = neighbor_point_z - point.coordinates.z;

                        auto euclidean_distance_square =
                            diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;

                        gca::counter_t condition =
                            (euclidean_distance_square < m_search_radius * m_search_radius);
                        n_neighbors_in_radius += condition;
                    }
                }
            }
        }
        if (n_neighbors_in_radius <= m_min_neighbors_in_radius)
        {
            new_point.property = gca::point_property::invalid;
        }
        return new_point;
    }
};

/* variables for query points are named as *_Q, and for reference points as *_R */
::cudaError_t cuda_nn_search(thrust::device_vector<gca::index_t> &result_nn_idx_in_R,
                             const thrust::device_vector<gca::point_t> &points_Q,
                             const thrust::device_vector<gca::point_t> &points_R, float3 min_bound,
                             const float3 max_bound, const float search_radius)
{
    auto n_points_Q = points_Q.size();
    auto n_points_R = points_R.size();
    if (result_nn_idx_in_R.size() != n_points_Q)
    {
        result_nn_idx_in_R.resize(n_points_Q);
    }
    /* 1. Prepare resources for point clouds */
    auto grid_cell_size = search_radius;
    auto n_grid_cells_x =
        static_cast<gca::counter_t>((max_bound.x - min_bound.x) / grid_cell_size) + 1;
    auto n_grid_cells_y =
        static_cast<gca::counter_t>((max_bound.y - min_bound.y) / grid_cell_size) + 1;
    auto n_grid_cells_z =
        static_cast<gca::counter_t>((max_bound.z - min_bound.z) / grid_cell_size) + 1;

    auto grid_cells_n = n_grid_cells_x * n_grid_cells_y * n_grid_cells_z;
    if (grid_cells_n < 1)
    {
        return ::cudaErrorInvalidValue;
    }

    /* 2. compute the grid cell info */
    thrust::device_vector<gca::index_t> grid_cell_idxs_R(n_points_R);
    thrust::transform(
        points_R.begin(), points_R.end(), grid_cell_idxs_R.begin(),
        compute_grid_cell_index_functor(min_bound, grid_cell_size, n_grid_cells_y, n_grid_cells_z));
    auto err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    thrust::device_vector<gca::index_t> points_sorted_idxs_R(n_points_R);
    thrust::sequence(points_sorted_idxs_R.begin(), points_sorted_idxs_R.end());
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    thrust::sort_by_key(grid_cell_idxs_R.begin(), grid_cell_idxs_R.end(),
                        points_sorted_idxs_R.begin());
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    thrust::device_vector<gca::index_t> idxs_of_sorted_points_R(n_points_R);
    thrust::sequence(idxs_of_sorted_points_R.begin(), idxs_of_sorted_points_R.end());
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    auto end_iter_idxs_of_sorted_points_R =
        thrust::reduce_by_key(grid_cell_idxs_R.begin(), grid_cell_idxs_R.end(),
                              idxs_of_sorted_points_R.begin(), thrust::make_discard_iterator(),
                              idxs_of_sorted_points_R.begin(), thrust::equal_to<gca::index_t>(),
                              get_begin_index_of_keys_functor())
            .second;
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    thrust::device_vector<gca::counter_t> points_counter_per_grid_cell_R(n_points_R, 1);
    auto end_iter_pair_grid_cell_idxs_and_counter_R = thrust::reduce_by_key(
        grid_cell_idxs_R.begin(), grid_cell_idxs_R.end(), points_counter_per_grid_cell_R.begin(),
        grid_cell_idxs_R.begin(), points_counter_per_grid_cell_R.begin(),
        thrust::equal_to<gca::index_t>());
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    auto unique_grid_cells_n_R =
        end_iter_pair_grid_cell_idxs_and_counter_R.first - grid_cell_idxs_R.begin();
    if (unique_grid_cells_n_R != end_iter_idxs_of_sorted_points_R - idxs_of_sorted_points_R.begin())
    {
        return ::cudaErrorInvalidValue;
    }

    /* 3. fill the grid cell info into grid cells */
    thrust::device_vector<gca::grid_cell_t> grid_cells_R(grid_cells_n);
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(
                         grid_cell_idxs_R.begin(), idxs_of_sorted_points_R.begin(),
                         points_counter_per_grid_cell_R.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(
                         grid_cell_idxs_R.begin() + unique_grid_cells_n_R,
                         idxs_of_sorted_points_R.begin() + unique_grid_cells_n_R,
                         points_counter_per_grid_cell_R.begin() + unique_grid_cells_n_R)),
                     fill_grid_cells_functor(grid_cells_R));
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    /* 4. For each point, compute euclidean distances between the point and all the points in
     * neighbor grids to check if they are in the radius of the point. Euclidean distance
     * requires sqrt, which might be slow, so here instead it square number should be used */
    auto nn_search_func =
        nn_search_functor(points_R, points_sorted_idxs_R, grid_cells_R, min_bound, n_grid_cells_x,
                          n_grid_cells_y, n_grid_cells_z, search_radius);
    thrust::transform(points_Q.begin(), points_Q.end(), result_nn_idx_in_R.begin(), nn_search_func);
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    return ::cudaSuccess;
}

/* variables for query points are named as *_Q, and for reference points as *_R */
::cudaError_t cuda_search_radius_neighbors(
    thrust::device_vector<gca::index_t> &result_radius_neighbor_idxs_in_R,
    thrust::device_vector<thrust::pair<gca::index_t, gca::counter_t>>
        &result_pair_neighbors_begin_idx_and_count,
    const thrust::device_vector<gca::point_t> &points_Q,
    const thrust::device_vector<gca::point_t> &points_R, float3 min_bound, const float3 max_bound,
    const float search_radius)
{
    auto n_points_Q = points_Q.size();
    auto n_points_R = points_R.size();

    /* 1. Prepare resources for point clouds */
    auto grid_cell_size = search_radius;
    auto n_grid_cells_x =
        static_cast<gca::counter_t>((max_bound.x - min_bound.x) / grid_cell_size) + 1;
    auto n_grid_cells_y =
        static_cast<gca::counter_t>((max_bound.y - min_bound.y) / grid_cell_size) + 1;
    auto n_grid_cells_z =
        static_cast<gca::counter_t>((max_bound.z - min_bound.z) / grid_cell_size) + 1;

    auto grid_cells_n = n_grid_cells_x * n_grid_cells_y * n_grid_cells_z;
    if (grid_cells_n < 1)
    {
        return ::cudaErrorInvalidValue;
    }

    /* 2. compute the grid cell info */
    thrust::device_vector<gca::index_t> grid_cell_idxs_R(n_points_R);
    thrust::transform(
        points_R.begin(), points_R.end(), grid_cell_idxs_R.begin(),
        compute_grid_cell_index_functor(min_bound, grid_cell_size, n_grid_cells_y, n_grid_cells_z));
    auto err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    thrust::device_vector<gca::index_t> points_sorted_idxs_R(n_points_R);
    thrust::sequence(points_sorted_idxs_R.begin(), points_sorted_idxs_R.end());
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    thrust::sort_by_key(grid_cell_idxs_R.begin(), grid_cell_idxs_R.end(),
                        points_sorted_idxs_R.begin());
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    thrust::device_vector<gca::index_t> idxs_of_sorted_points_R(n_points_R);
    thrust::sequence(idxs_of_sorted_points_R.begin(), idxs_of_sorted_points_R.end());
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    auto end_iter_idxs_of_sorted_points_R =
        thrust::reduce_by_key(grid_cell_idxs_R.begin(), grid_cell_idxs_R.end(),
                              idxs_of_sorted_points_R.begin(), thrust::make_discard_iterator(),
                              idxs_of_sorted_points_R.begin(), thrust::equal_to<gca::index_t>(),
                              get_begin_index_of_keys_functor())
            .second;
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    thrust::device_vector<gca::counter_t> points_counter_per_grid_cell_R(n_points_R, 1);
    auto end_iter_pair_grid_cell_idxs_and_counter_R = thrust::reduce_by_key(
        grid_cell_idxs_R.begin(), grid_cell_idxs_R.end(), points_counter_per_grid_cell_R.begin(),
        grid_cell_idxs_R.begin(), points_counter_per_grid_cell_R.begin(),
        thrust::equal_to<gca::index_t>());
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    auto unique_grid_cells_n_R =
        end_iter_pair_grid_cell_idxs_and_counter_R.first - grid_cell_idxs_R.begin();
    if (unique_grid_cells_n_R != end_iter_idxs_of_sorted_points_R - idxs_of_sorted_points_R.begin())
    {
        return ::cudaErrorInvalidValue;
    }

    /* 3. fill the grid cell info into grid cells */
    thrust::device_vector<gca::grid_cell_t> grid_cells_R(grid_cells_n);
    thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(
                         grid_cell_idxs_R.begin(), idxs_of_sorted_points_R.begin(),
                         points_counter_per_grid_cell_R.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(
                         grid_cell_idxs_R.begin() + unique_grid_cells_n_R,
                         idxs_of_sorted_points_R.begin() + unique_grid_cells_n_R,
                         points_counter_per_grid_cell_R.begin() + unique_grid_cells_n_R)),
                     fill_grid_cells_functor(grid_cells_R));
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    /* 4. First step: Compute the neighbors number of every points */
    thrust::device_vector<gca::counter_t> num_of_neighbors(n_points_Q);
    auto step1_func = nn_search_radius_step1_functor(points_R, points_sorted_idxs_R, grid_cells_R,
                                                     min_bound, n_grid_cells_x, n_grid_cells_y,
                                                     n_grid_cells_z, search_radius);
    thrust::transform(points_Q.begin(), points_Q.end(), num_of_neighbors.begin(), step1_func);
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    /* 5. For Each point, compute start index of its neighbors in the result neighbors vector */
    thrust::device_vector<gca::index_t> neighbor_idxs_starts(n_points_Q);
    thrust::exclusive_scan(num_of_neighbors.begin(), num_of_neighbors.end(),
                           neighbor_idxs_starts.begin());
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    /* 6. compute total neighbors */
    gca::counter_t total_neighbors =
        thrust::reduce(num_of_neighbors.begin(), num_of_neighbors.end());
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    if (result_pair_neighbors_begin_idx_and_count.size() != n_points_Q)
    {
        result_pair_neighbors_begin_idx_and_count.resize(n_points_Q);
    }
    result_radius_neighbor_idxs_in_R.resize(total_neighbors);

    /* 7. Second step: Re-do the NN search and copy all the neighbor indicies into result vector */
    auto zipped_iter_begin = thrust::make_zip_iterator(thrust::make_tuple(
        points_Q.begin(), neighbor_idxs_starts.begin(), num_of_neighbors.begin()));

    auto zipped_iter_end = thrust::make_zip_iterator(
        thrust::make_tuple(points_Q.end(), neighbor_idxs_starts.end(), num_of_neighbors.end()));

    auto step2_func = nn_search_radius_step2_functor(
        result_radius_neighbor_idxs_in_R, points_R, points_sorted_idxs_R, grid_cells_R, min_bound,
        n_grid_cells_x, n_grid_cells_y, n_grid_cells_z, search_radius);

    thrust::transform(zipped_iter_begin, zipped_iter_end,
                      result_pair_neighbors_begin_idx_and_count.begin(), step2_func);
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    return ::cudaSuccess;
}

::cudaError_t cuda_grid_radius_outliers_removal(
    thrust::device_vector<gca::point_t> &result_points,
    const thrust::device_vector<gca::point_t> &src_points, const float3 min_bound,
    const float3 max_bound, const float grid_cell_side_len, const float search_radius,
    const gca::counter_t min_neighbors_in_radius)
{
    auto n_points = src_points.size();
    if (result_points.size() != n_points)
    {
        result_points.resize(n_points);
    }
    /* 1. build and fill grid cells */
    auto grid_cell_size = grid_cell_side_len;
    auto n_grid_cells_x =
        static_cast<gca::counter_t>((max_bound.x - min_bound.x) / grid_cell_size) + 1;
    auto n_grid_cells_y =
        static_cast<gca::counter_t>((max_bound.y - min_bound.y) / grid_cell_size) + 1;
    auto n_grid_cells_z =
        static_cast<gca::counter_t>((max_bound.z - min_bound.z) / grid_cell_size) + 1;

    auto grid_cells_n = n_grid_cells_x * n_grid_cells_y * n_grid_cells_z;
    if (grid_cells_n < 1)
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
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    thrust::sort_by_key(grid_cell_idxs.begin(), grid_cell_idxs.end(),
                        src_points_sorted_idxs.begin());
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    thrust::device_vector<gca::index_t> idxs_of_sorted_points(n_points);
    thrust::sequence(idxs_of_sorted_points.begin(), idxs_of_sorted_points.end());
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    auto end_iter_idx_of_sorted_idx =
        thrust::reduce_by_key(grid_cell_idxs.begin(), grid_cell_idxs.end(),
                              idxs_of_sorted_points.begin(), thrust::make_discard_iterator(),
                              idxs_of_sorted_points.begin(), thrust::equal_to<gca::index_t>(),
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

    if (unique_grid_cells_n != end_iter_idx_of_sorted_idx - idxs_of_sorted_points.begin())
    {
        return ::cudaErrorInvalidValue;
    }

    thrust::device_vector<gca::grid_cell_t> grid_cells_vec(grid_cells_n); // build grid cells
    thrust::for_each(thrust::make_zip_iterator(
                         thrust::make_tuple(grid_cell_idxs.begin(), idxs_of_sorted_points.begin(),
                                            points_counter_per_grid_cell.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(
                         grid_cell_idxs.begin() + unique_grid_cells_n,
                         idxs_of_sorted_points.begin() + unique_grid_cells_n,
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

    auto check_radius_func = check_if_enough_radius_nn_functor(
        src_points, src_points_sorted_idxs, grid_cells_vec, min_bound, n_grid_cells_x,
        n_grid_cells_y, n_grid_cells_z, grid_cell_size, search_radius, min_neighbors_in_radius);
    auto end_iter_result_points = thrust::transform(src_points.begin(), src_points.end(),
                                                    result_points.begin(), check_radius_func);
    err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        return err;
    }

    gca::remove_invalid_points(result_points);

    return ::cudaSuccess;
}
} // namespace gca
