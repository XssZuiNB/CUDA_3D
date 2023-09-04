#pragma once

#include "cuda_container/cuda_container.hpp"
#include "geometry/type.hpp"

#include <memory>
#include <vector>

#include <thrust/device_vector.h>

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

    bool compute_min_max_bound();
    float3 get_min_bound();
    float3 get_max_bound();

    bool has_points() const
    {
        return !m_points.empty();
    }

    size_t points_number() const
    {
        return m_points.size();
    }

    std::shared_ptr<point_cloud> voxel_grid_down_sample(float voxel_size);

    std::shared_ptr<point_cloud> radius_outlier_removal(float radius,
                                                        gca::counter_t min_neighbors_in_radius);

    std::pair<std::shared_ptr<std::vector<gca::index_t>>, gca::counter_t> euclidean_clustering(
        const float cluster_tolerance, const gca::counter_t min_cluster_size,
        const gca::counter_t max_cluster_size);

    std::shared_ptr<point_cloud> movement_detection(point_cloud &last_frame,
                                                    const float geometry_constraint,
                                                    const float color_constraint);

    static std::shared_ptr<point_cloud> create_from_rgbd(const gca::cuda_depth_frame &depth,
                                                         const gca::cuda_color_frame &color,
                                                         const gca::cuda_camera_param &param,
                                                         float threshold_min_in_meter = 0.0,
                                                         float threshold_max_in_meter = 100.0);

    static thrust::device_vector<gca::index_t> nn_search(gca::point_cloud &query_pc,
                                                         gca::point_cloud &reference_pc,
                                                         float radius);

    static void nn_search(std::vector<gca::index_t> &result_nn_idx, gca::point_cloud &query_pc,
                          gca::point_cloud &reference_pc, float radius);

    ~point_cloud() = default;

private:
    thrust::device_vector<gca::point_t> m_points;
    bool m_has_bound = false;
    float3 m_min_bound;
    float3 m_max_bound;
};
} // namespace gca
