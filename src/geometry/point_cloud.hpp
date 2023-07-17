#pragma once

#include <memory>
#include <thrust/device_vector.h>
#include <vector>

#include "cuda_container/cuda_container.hpp"
#include "geometry/type.hpp"

namespace gca
{
class point_cloud
{
public:
    point_cloud() = default;
    point_cloud(size_t n_points);
    point_cloud(const point_cloud &other);

    point_cloud &operator=(const point_cloud &other);

    void download(std::vector<point_t> &dst) const;
    std::vector<point_t> download() const;

    float3 compute_min_bound();
    float3 compute_max_bound();

    __host__ __device__ bool has_points() const
    {
        return !m_points.empty();
    }

    __host__ __device__ size_t points_number() const
    {
        return m_points.size();
    }

    std::shared_ptr<point_cloud> voxel_grid_down_sample(float voxel_size,
                                                        bool invalid_voxel_removal = false,
                                                        uint32_t min_points_num_in_one_voxel = 1);

    static std::shared_ptr<point_cloud> create_from_rgbd(const gca::cuda_depth_frame &depth,
                                                         const gca::cuda_color_frame &color,
                                                         const gca::cuda_camera_param &param,
                                                         float threshold_min_in_meter = 0.0,
                                                         float threshold_max_in_meter = 100.0);

    ~point_cloud() = default;

private:
    thrust::device_vector<gca::point_t> m_points;
};

} // namespace gca