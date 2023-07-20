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
struct __align__(16) grid_cell
{
    int start_index_in_sorted_points = -1;
    int points_number = -1;
};
/*functor*/
struct grid_knn
{
    /* data */
};

__forceinline__ thrust::device_vector<grid_cell> make_grid_cell_vec(
    const thrust::device_vector<gca::point_t> &points, float grid_cell_size)
{
    auto min_max_tuple = cuda_compute_min_max_bound(points);
    auto number_of_grid_cell_x =
        (thrust::get<1>(min_max_tuple).x - thrust::get<0>(min_max_tuple).x) / grid_cell_size;
    auto number_of_grid_cell_y =
        (thrust::get<1>(min_max_tuple).y - thrust::get<0>(min_max_tuple).y) / grid_cell_size;
    auto number_of_grid_cell_z =
        (thrust::get<1>(min_max_tuple).z - thrust::get<0>(min_max_tuple).z) / grid_cell_size;

    thrust::device_vector<grid_cell> grid_cells_vec(number_of_grid_cell_x * number_of_grid_cell_y *
                                                    number_of_grid_cell_z);
}

} // namespace gca
