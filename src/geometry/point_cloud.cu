#include <cuda_runtime_api.h>
#include <memory>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <vector>

#include "geometry/cuda_nn_search.cuh"
#include "geometry/cuda_point_cloud_factory.cuh"
#include "geometry/cuda_voxel_grid_down_sample.cuh"
#include "geometry/geometry_util.cuh"
#include "geometry/point_cloud.hpp"
#include "util/console_color.hpp"

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
    cudaMemcpy(temp.data(), m_points.data().get(), m_points.size() * sizeof(gca::point_t),
               cudaMemcpyDefault);
    return temp;
}
void point_cloud::download(std::vector<gca::point_t> &dst) const
{
    dst.resize(m_points.size());
    cudaMemcpy(dst.data(), m_points.data().get(), m_points.size() * sizeof(gca::point_t),
               cudaMemcpyDefault);
}

bool point_cloud::compute_min_max_bound(::cudaStream_t stream)
{
    auto min_max_bound = cuda_compute_min_max_bound(m_points, stream);
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

std::shared_ptr<point_cloud> point_cloud::voxel_grid_down_sample(float voxel_size,
                                                                 ::cudaStream_t stream)
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
        if (!compute_min_max_bound(stream))
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

    auto err = cuda_voxel_grid_downsample(output->m_points, m_points, voxel_grid_min_bound,
                                          voxel_size, stream);
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

    const auto grid_cells_min_bound =
        make_float3(m_min_bound.x - radius, m_min_bound.y - radius, m_min_bound.z - radius);

    const auto grid_cells_max_bound =
        make_float3(m_max_bound.x + radius, m_max_bound.y + radius, m_max_bound.z + radius);

    if (radius * 2 * std::numeric_limits<int>::max() <
        max(max(grid_cells_max_bound.x - grid_cells_min_bound.x,
                grid_cells_max_bound.y - grid_cells_min_bound.y),
            grid_cells_max_bound.z - grid_cells_min_bound.z))
    {
        std::cout << YELLOW << "Radius is too small, a empty point cloud returned!" << std::endl;
        return output;
    }

    auto err =
        cuda_grid_radius_outliers_removal(output->m_points, m_points, grid_cells_min_bound,
                                          grid_cells_max_bound, radius, min_neighbors_in_radius);
    if (err != ::cudaSuccess)
    {
        std::cout << YELLOW << "Radius outlier removal failed, a invalid point cloud returned! \n"
                  << std::endl;
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

    if (!cuda_make_point_cloud(pc->m_points, depth, color, param, threshold_min_in_meter,
                               threshold_max_in_meter))
        std::cout << YELLOW << "CUDA can't make a point cloud, A empty point cloud returned! \n"
                  << std::endl;

    return pc;
}
} // namespace gca
