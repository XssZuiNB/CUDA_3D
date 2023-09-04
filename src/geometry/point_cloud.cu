#include "geometry/cuda_clustering.cuh"
#include "geometry/cuda_nn_search.cuh"
#include "geometry/cuda_point_cloud_factory.cuh"
#include "geometry/cuda_voxel_grid_down_sample.cuh"
#include "geometry/geometry_util.cuh"
#include "geometry/point_cloud.hpp"
#include "movement_detection/cuda_movement_detection.cuh"
#include "util/console_color.hpp"

#include <memory>
#include <vector>

#include <cuda_runtime_api.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/pair.h>

namespace gca
{
point_cloud::point_cloud(size_t n_points)
    : m_points(n_points)
{
}

point_cloud::point_cloud(const point_cloud &other)
    : m_points(other.m_points)
{
}

point_cloud &point_cloud::operator=(const point_cloud &other)
{
    m_points = other.m_points;
    return *this;
}

std::vector<gca::point_t> point_cloud::download() const
{
    std::vector<gca::point_t> temp(m_points.size());
    thrust::copy(m_points.begin(), m_points.end(), temp.begin());
    return temp;
}
void point_cloud::download(std::vector<gca::point_t> &dst) const
{
    dst.resize(m_points.size());
    thrust::copy(m_points.begin(), m_points.end(), dst.begin());
}

bool point_cloud::compute_min_max_bound()
{
    auto min_max_bound = cuda_compute_min_max_bound(m_points);
    auto err = cudaGetLastError();
    if (err != ::cudaSuccess)
    {
        std::cout << YELLOW << "Compute min and max bound for the point cloud failed!" << std::endl;
        m_has_bound = false;
        return false;
    }
    m_min_bound = min_max_bound.first;
    m_max_bound = min_max_bound.second;
    m_has_bound = true;
    return true;
}

float3 point_cloud::get_min_bound()
{
    if (m_has_bound)
    {
        return m_min_bound;
    }

    if (!compute_min_max_bound())
    {
        std::cout << YELLOW
                  << "Compute bound of point cloud is not possible, invalid bound is returned!"
                  << std::endl;
        return make_float3(0.0, 0.0, 0.0);
    }
    return m_min_bound;
}

float3 point_cloud::get_max_bound()
{
    if (m_has_bound)
    {
        return m_max_bound;
    }

    if (!compute_min_max_bound())
    {
        std::cout << YELLOW
                  << "Compute bound of point cloud is not possible, invalid bound is returned!"
                  << std::endl;
        return make_float3(0.0, 0.0, 0.0);
    }
    return m_max_bound;
}

std::shared_ptr<point_cloud> point_cloud::voxel_grid_down_sample(float voxel_size)
{
    auto output = std::make_shared<point_cloud>(m_points.size());

    if (voxel_size <= 0.0)
    {
        std::cout << YELLOW << "Voxel size is less than 0, a empty point cloud returned!"
                  << std::endl;
        return output;
    }

    if (!m_has_bound)
    {
        if (!compute_min_max_bound())
        {
            std::cout
                << YELLOW
                << "Compute bound of point cloud is not possible, a empty point cloud returned!"
                << std::endl;
            return output;
        }
    }

    const auto voxel_grid_min_bound =
        make_float3(m_min_bound.x - voxel_size * 0.5, m_min_bound.y - voxel_size * 0.5,
                    m_min_bound.z - voxel_size * 0.5);

    const auto voxel_grid_max_bound =
        make_float3(m_max_bound.x + voxel_size * 0.5, m_max_bound.y + voxel_size * 0.5,
                    m_max_bound.z + voxel_size * 0.5);

    if (voxel_size * std::numeric_limits<int>::max() <
        max(max(voxel_grid_max_bound.x - voxel_grid_min_bound.x,
                voxel_grid_max_bound.y - voxel_grid_min_bound.y),
            voxel_grid_max_bound.z - voxel_grid_min_bound.z))
    {
        std::cout << YELLOW << "Voxel size is too small, a empty point cloud returned!"
                  << std::endl;
        return output;
    }

    auto err =
        cuda_voxel_grid_downsample(output->m_points, m_points, voxel_grid_min_bound, voxel_size);
    if (err != ::cudaSuccess)
    {
        std::cout << YELLOW
                  << "Compute voxel grid down sample failed, a invalid point cloud returned! \n"
                  << std::endl;
        return output;
    }

    return output;
}

std::shared_ptr<point_cloud> point_cloud::radius_outlier_removal(
    float radius, gca::counter_t min_neighbors_in_radius)
{
    auto output = std::make_shared<point_cloud>(m_points.size());

    if (radius <= 0.0)
    {
        std::cout << YELLOW << "Radius is less than 0, a empty point cloud returned!" << std::endl;
        return output;
    }

    if (!m_has_bound)
    {
        if (!compute_min_max_bound())
        {
            std::cout
                << YELLOW
                << "Compute bound of point cloud is not possible, a empty point cloud returned!"
                << std::endl;
            return output;
        }
    }

    auto grid_cell_side_len = 2 * radius;
    auto padding = 1.5 * grid_cell_side_len;
    const auto grid_cells_min_bound =
        make_float3(m_min_bound.x - padding, m_min_bound.y - padding, m_min_bound.z - padding);

    const auto grid_cells_max_bound =
        make_float3(m_max_bound.x + padding, m_max_bound.y + padding, m_max_bound.z + padding);

    if (grid_cell_side_len * std::numeric_limits<int>::max() <
        max(max(grid_cells_max_bound.x - grid_cells_min_bound.x,
                grid_cells_max_bound.y - grid_cells_min_bound.y),
            grid_cells_max_bound.z - grid_cells_min_bound.z))
    {
        std::cout << YELLOW << "Radius is too small, a empty point cloud returned!" << std::endl;
        return output;
    }

    auto err = cuda_grid_radius_outliers_removal(output->m_points, m_points, grid_cells_min_bound,
                                                 grid_cells_max_bound, grid_cell_side_len, radius,
                                                 min_neighbors_in_radius);
    if (err != ::cudaSuccess)
    {
        std::cout << YELLOW << "Radius outlier removal failed, a invalid point cloud returned ! \n "
                  << std::endl;
        return output;
    }

    return output;
}

std::pair<std::shared_ptr<std::vector<gca::index_t>>, gca::counter_t> point_cloud::
    euclidean_clustering(const float cluster_tolerance, const gca::counter_t min_cluster_size,
                         const gca::counter_t max_cluster_size)
{
    auto cluster_of_point = std::make_shared<std::vector<gca::index_t>>(m_points.size());

    if (cluster_tolerance <= 0.0)
    {
        std::cout << YELLOW << "Clustering tolerance is less than 0, a empty result returned!"
                  << std::endl;
        return std::make_pair(cluster_of_point, 0);
    }

    if (!m_has_bound)
    {
        if (!compute_min_max_bound())
        {
            std::cout << YELLOW
                      << "Compute bound of point cloud is not possible, a empty result returned!"
                      << std::endl;
            return std::make_pair(cluster_of_point, 0);
        }
    }

    auto grid_cell_side_len = cluster_tolerance;
    auto padding = 1.5 * grid_cell_side_len;
    const auto grid_cells_min_bound =
        make_float3(m_min_bound.x - padding, m_min_bound.y - padding, m_min_bound.z - padding);

    const auto grid_cells_max_bound =
        make_float3(m_max_bound.x + padding, m_max_bound.y + padding, m_max_bound.z + padding);

    if (grid_cell_side_len * std::numeric_limits<int>::max() <
        max(max(grid_cells_max_bound.x - grid_cells_min_bound.x,
                grid_cells_max_bound.y - grid_cells_min_bound.y),
            grid_cells_max_bound.z - grid_cells_min_bound.z))
    {
        std::cout << YELLOW << "Cluster tolerance is too small, a empty result returned!"
                  << std::endl;
        return std::make_pair(cluster_of_point, 0);
    }
    /*
    thrust::device_vector<gca::index_t> cluster_dev(m_points.size());
    gca::counter_t n_clusters;
    auto err = cuda_euclidean_clustering(cluster_dev, n_clusters, m_points, grid_cells_min_bound,
                                         grid_cells_max_bound, cluster_tolerance, min_cluster_size,
                                         max_cluster_size);
    if (err != ::cudaSuccess)
    {
        std::cout << YELLOW << "Radius outlier removal failed, a invalid point cloud returned!\n "
                  << std::endl;
        return std::make_pair(cluster_of_point, 0);
    }

    thrust::copy(cluster_dev.begin(), cluster_dev.end(), cluster_of_point->begin());
    */
    gca::counter_t n_clusters;

    auto err = cuda_euclidean_clustering(*cluster_of_point, n_clusters, m_points,
                                         grid_cells_min_bound, grid_cells_max_bound,
                                         cluster_tolerance, min_cluster_size, max_cluster_size);
    if (err != ::cudaSuccess)
    {
        std::cout << YELLOW << "Clustering failed, a invalid result returned ! \n " << std::endl;
        return std::make_pair(cluster_of_point, 0);
    }

    return std::make_pair(cluster_of_point, n_clusters);
}

std::shared_ptr<point_cloud> point_cloud::movement_detection(point_cloud &last_frame,
                                                             const float geometry_constraint,
                                                             const float color_constraint)
{
    auto output = std::make_shared<point_cloud>();
    if (geometry_constraint <= 0.0 || color_constraint <= 0)
    {
        std::cout << YELLOW << "Constraint is less than 0, FALSE returned!" << std::endl;
        return output;
    }

    if (!m_has_bound)
    {
        if (!compute_min_max_bound())
        {
            std::cout << YELLOW
                      << "Compute bound of this point cloud is not possible, FALSE returned!"
                      << std::endl;
            return output;
        }
    }

    if (!last_frame.m_has_bound)
    {
        if (!(last_frame.compute_min_max_bound()))
        {
            std::cout << YELLOW
                      << "Compute bound of last point cloud is not possible, FALSE result "
                         "returned!"
                      << std::endl;
            return output;
        }
    }

    auto padding = 1.5 * geometry_constraint;
    const auto grid_cells_min_bound =
        make_float3(min(m_min_bound.x, last_frame.m_min_bound.x) - padding,
                    min(m_min_bound.y, last_frame.m_min_bound.y) - padding,
                    min(m_min_bound.z, last_frame.m_min_bound.z) - padding);

    const auto grid_cells_max_bound =
        make_float3(max(m_max_bound.x, last_frame.m_max_bound.x) + padding,
                    max(m_max_bound.y, last_frame.m_max_bound.y) + padding,
                    max(m_max_bound.z, last_frame.m_max_bound.z) + padding);

    if (geometry_constraint * std::numeric_limits<int>::max() <
        max(max(grid_cells_max_bound.x - grid_cells_min_bound.x,
                grid_cells_max_bound.y - grid_cells_min_bound.y),
            grid_cells_max_bound.z - grid_cells_min_bound.z))
    {
        std::cout << YELLOW << "Geometry_constraint is too small, FALSE returned!" << std::endl;
        return output;
    }

    auto err = cuda_movement_detection(output->m_points, m_points, last_frame.m_points,
                                       grid_cells_min_bound, grid_cells_max_bound,
                                       geometry_constraint, color_constraint);
    if (err != ::cudaSuccess)
    {
        std::cout << YELLOW << "Movement detection failed, FALSE returned!\n" << std::endl;
        return output;
    }

    return output;
}

std::shared_ptr<point_cloud> point_cloud::create_from_rgbd(const gca::cuda_depth_frame &depth,
                                                           const gca::cuda_color_frame &color,
                                                           const gca::cuda_camera_param &param,
                                                           float threshold_min_in_meter,
                                                           float threshold_max_in_meter)
{
    auto pc = std::make_shared<point_cloud>();

    auto depth_frame_format = param.get_depth_frame_format();
    auto color_frame_format = param.get_color_frame_format();
    if (depth_frame_format != gca::Z16 || color_frame_format != gca::BGR8)
    {
        std::cout << YELLOW
                  << "Frame format is not supported right now, A empty point cloud returned! \n"
                  << std::endl;
        return pc;
    }

    if (cuda_make_point_cloud(pc->m_points, depth, color, param, threshold_min_in_meter,
                              threshold_max_in_meter) != ::cudaSuccess)
        std::cout << YELLOW << "CUDA can't make a point cloud, A empty point cloud returned! \n"
                  << std::endl;

    return pc;
}

thrust::device_vector<gca::index_t> point_cloud::nn_search(gca::point_cloud &query_pc,
                                                           gca::point_cloud &reference_pc,
                                                           float radius)
{
    thrust::device_vector<gca::index_t> result_nn_idx_in_reference(query_pc.points_number());

    if (radius <= 0.0)
    {
        std::cout << YELLOW << "Radius is less than 0, a empty result returned!" << std::endl;
        return result_nn_idx_in_reference;
    }

    if (!query_pc.m_has_bound)
    {
        if (!(query_pc.compute_min_max_bound()))
        {
            std::cout
                << YELLOW
                << "Compute bound of query point cloud is not possible, a empty result returned!"
                << std::endl;
            return result_nn_idx_in_reference;
        }
    }

    if (!reference_pc.m_has_bound)
    {
        if (!(reference_pc.compute_min_max_bound()))
        {
            std::cout << YELLOW
                      << "Compute bound of reference point cloud is not possible, a empty result "
                         "returned!"
                      << std::endl;
            return result_nn_idx_in_reference;
        }
    }

    auto padding = 1.5 * radius;
    const auto grid_cells_min_bound =
        make_float3(min(query_pc.m_min_bound.x, reference_pc.m_min_bound.x) - padding,
                    min(query_pc.m_min_bound.y, reference_pc.m_min_bound.y) - padding,
                    min(query_pc.m_min_bound.z, reference_pc.m_min_bound.z) - padding);

    const auto grid_cells_max_bound =
        make_float3(max(query_pc.m_max_bound.x, reference_pc.m_max_bound.x) + padding,
                    max(query_pc.m_max_bound.y, reference_pc.m_max_bound.y) + padding,
                    max(query_pc.m_max_bound.z, reference_pc.m_max_bound.z) + padding);

    if (radius * 2 * std::numeric_limits<int>::max() <
        max(max(grid_cells_max_bound.x - grid_cells_min_bound.x,
                grid_cells_max_bound.y - grid_cells_min_bound.y),
            grid_cells_max_bound.z - grid_cells_min_bound.z))
    {
        std::cout << YELLOW << "Radius is too small, a empty result returned!" << std::endl;
        return result_nn_idx_in_reference;
    }

    auto err = cuda_nn_search(result_nn_idx_in_reference, query_pc.m_points, reference_pc.m_points,
                              grid_cells_min_bound, grid_cells_max_bound, radius);
    if (err != ::cudaSuccess)
    {
        std::cout << YELLOW << "Radius outlier removal failed, a invalid result returned! \n"
                  << std::endl;
        return result_nn_idx_in_reference;
    }

    return result_nn_idx_in_reference;
}

void point_cloud::nn_search(std::vector<gca::index_t> &result_nn_idx, gca::point_cloud &query_pc,
                            gca::point_cloud &reference_pc, float radius)
{
    auto start = std::chrono::steady_clock::now();
    auto result_nn_idx_device_vec = point_cloud::nn_search(query_pc, reference_pc, radius);
    auto end = std::chrono::steady_clock::now();
    std::cout << "Cuda nn time in microseconds: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us"
              << std::endl;
    std::cout << "Cuda nn time in milliseconds: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms"
              << std::endl;

    result_nn_idx.resize(result_nn_idx_device_vec.size());
    thrust::copy(result_nn_idx_device_vec.begin(), result_nn_idx_device_vec.end(),
                 result_nn_idx.begin());
}
} // namespace gca
