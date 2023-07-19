#include <cuda_runtime_api.h>
#include <memory>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <vector>

#include "geometry/cuda_point_cloud_factory.cuh"
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

std::vector<point_t> point_cloud::download() const
{
    std::vector<point_t> temp;
    cudaMemcpy(temp.data(), m_points.data().get(), m_points.size() * sizeof(gca::point_t),
               cudaMemcpyDefault);
    return temp;
}
void point_cloud::download(std::vector<point_t> &dst) const
{
    dst.resize(m_points.size());
    cudaMemcpy(dst.data(), m_points.data().get(), m_points.size() * sizeof(gca::point_t),
               cudaMemcpyDefault);
}

float3 point_cloud::compute_min_bound()
{
    return cuda_compute_min_bound(m_points);
}
float3 point_cloud::compute_max_bound()
{
    return cuda_compute_max_bound(m_points);
}

std::shared_ptr<point_cloud> point_cloud::voxel_grid_down_sample(
    float voxel_size, bool invalid_voxel_removal, uint32_t min_points_num_in_one_voxel)
{
    auto output = std::make_shared<point_cloud>(m_points.size());

    if (voxel_size <= 0.0)
    {
        std::cout << YELLOW << "Voxel size is less than 0, a empty point cloud returned!"
                  << std::endl;
        return output;
    }

    auto min_bound_coordinates = compute_min_bound();
    auto max_bound_coordinates = compute_max_bound();

    const auto voxel_grid_min_bound = make_float3(min_bound_coordinates.x - voxel_size * 0.5,
                                                  min_bound_coordinates.y - voxel_size * 0.5,
                                                  min_bound_coordinates.z - voxel_size * 0.5);

    const auto voxel_grid_max_bound = make_float3(max_bound_coordinates.x + voxel_size * 0.5,
                                                  max_bound_coordinates.y + voxel_size * 0.5,
                                                  max_bound_coordinates.z + voxel_size * 0.5);

    if (voxel_size * std::numeric_limits<int>::max() <
        max(max(voxel_grid_max_bound.x - voxel_grid_min_bound.x,
                voxel_grid_max_bound.y - voxel_grid_min_bound.y),
            voxel_grid_max_bound.z - voxel_grid_min_bound.z))
    {
        std::cout << YELLOW << "Voxel size is too small, a empty point cloud returned!"
                  << std::endl;
        return output;
    }

    auto err = invalid_voxel_removal
                   ? cuda_voxel_grid_downsample(output->m_points, m_points, voxel_grid_min_bound,
                                                voxel_size, min_points_num_in_one_voxel)
                   : cuda_voxel_grid_downsample(output->m_points, m_points, voxel_grid_min_bound,
                                                voxel_size);
    if (err != ::cudaSuccess)
    {
        std::cout << YELLOW
                  << "Compute voxel grid down sample failed, a empty point cloud returned! \n"
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

    cuda_make_point_cloud(pc->m_points, depth, color, param, threshold_min_in_meter,
                          threshold_max_in_meter);
    return pc;
}
} // namespace gca