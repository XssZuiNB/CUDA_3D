#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/gather.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include "geometry/geometry_util.cuh"
#include "geometry/type.hpp"

namespace gca
{
/*functor*/
struct compute_grid_cell_index_functor
{
    compute_grid_cell_index_functor(const float3 &grid_cells_min_bound, const float grid_cell_size,
                                    const size_t number_of_grid_cell_y,
                                    const size_t number_of_grid_cell_z)
        : m_grid_cells_min_bound(grid_cells_min_bound)
        , m_grid_cell_size(grid_cell_size)
        , m_number_of_grid_cell_y(number_of_grid_cell_y)
        , m_number_of_grid_cell_z(number_of_grid_cell_z)
    {
    }

    const float3 m_grid_cells_min_bound;
    const float m_grid_cell_size;
    const size_t m_number_of_grid_cell_y;
    const size_t m_number_of_grid_cell_z;

    __forceinline__ __device__ size_t operator()(const gca::point_t &point)
    {
        auto index_x =
            __float2ull_rd((point.coordinates.x - m_grid_cells_min_bound.x) / m_grid_cell_size);
        auto index_y =
            __float2ull_rd((point.coordinates.y - m_grid_cells_min_bound.x) / m_grid_cell_size);
        auto index_z =
            __float2ull_rd((point.coordinates.z - m_grid_cells_min_bound.x) / m_grid_cell_size);

        return index_x * m_number_of_grid_cell_y * m_number_of_grid_cell_z +
               index_y * m_number_of_grid_cell_z + index_z;
    }
};

cudaError_t grid_search_radius(thrust::device_vector<uint32_t> &result,
                               const thrust::device_vector<gca::point_t> &query_points,
                               const thrust::device_vector<gca::point_t> &reference_points,
                               float grid_cell_size)
{
    auto min_max_tuple_query_points = cuda_compute_min_max_bound(query_points);
    auto min_max_tuple_reference_points = cuda_compute_min_max_bound(reference_points);

    float min_x = min(thrust::get<0>(min_max_tuple_query_points).x,
                      thrust::get<0>(min_max_tuple_reference_points).x) -
                  0.5 * grid_cell_size;
    float min_y = min(thrust::get<0>(min_max_tuple_query_points).y,
                      thrust::get<0>(min_max_tuple_reference_points).y) -
                  0.5 * grid_cell_size;
    float min_z = min(thrust::get<0>(min_max_tuple_query_points).z,
                      thrust::get<0>(min_max_tuple_reference_points).z) -
                  0.5 * grid_cell_size;

    float max_x = max(thrust::get<1>(min_max_tuple_query_points).x,
                      thrust::get<1>(min_max_tuple_reference_points).x) +
                  0.5 * grid_cell_size;
    float max_y = max(thrust::get<1>(min_max_tuple_query_points).y,
                      thrust::get<1>(min_max_tuple_reference_points).y) +
                  0.5 * grid_cell_size;
    float max_z = max(thrust::get<1>(min_max_tuple_query_points).z,
                      thrust::get<1>(min_max_tuple_reference_points).z) +
                  0.5 * grid_cell_size;

    auto number_of_grid_cell_x = static_cast<size_t>((max_x - min_x) / grid_cell_size) + 1;
    auto number_of_grid_cell_y = static_cast<size_t>((max_y - min_y) / grid_cell_size) + 1;
    auto number_of_grid_cell_z = static_cast<size_t>((max_z - min_z) / grid_cell_size) + 1;
    auto grid_cells_n = number_of_grid_cell_x * number_of_grid_cell_y * number_of_grid_cell_z;
    thrust::device_vector<gca::grid_cell_t> grid_cells_vec(grid_cells_n);

    thrust::device_vector<size_t> grid_cell_indecies_query_points(query_points.size());
    thrust::device_vector<size_t> grid_cell_indecies_reference_points(reference_points.size());
    auto grid_cells_min_bound = make_float3(min_x, min_y, min_z);
    /*
    auto err = cuda_compute_voxel_keys(grid_cell_query_points, query_points, min_bound_grid_cells,
                                       grid_cell_size);
    if (err != ::cudaSuccess)
    {
        // invalid
        return err;
    }
    err = cuda_compute_voxel_keys(grid_cell_reference_points, reference_points,
                                  min_bound_grid_cells, grid_cell_size);
    if (err != ::cudaSuccess)
    {
        // invalid
        return err;
    }*/
}

thrust::device_vector<int> gird_search_radius(const thrust::device_vector<gca::point_t> &points,
                                              float grid_cell_size)
{
    auto min_max_tuple = cuda_compute_min_max_bound(points);
    auto number_of_grid_cell_x =
        (thrust::get<1>(min_max_tuple).x - thrust::get<0>(min_max_tuple).x) / grid_cell_size;
    auto number_of_grid_cell_y =
        (thrust::get<1>(min_max_tuple).y - thrust::get<0>(min_max_tuple).y) / grid_cell_size;
    auto number_of_grid_cell_z =
        (thrust::get<1>(min_max_tuple).z - thrust::get<0>(min_max_tuple).z) / grid_cell_size;

    thrust::device_vector<gca::grid_cell_t> grid_cells_vec(
        number_of_grid_cell_x * number_of_grid_cell_y * number_of_grid_cell_z);
}

} // namespace gca