#pragma once

#include "cuda_container/cuda_container.hpp"
#include "geometry/type.hpp"

#include <memory>
#include <vector>

#include <thrust/device_vector.h>

namespace gca
{
class movement_detection;

class point_cloud
{
public:
    point_cloud() = default;
    point_cloud(size_t n_points);
    point_cloud(const point_cloud &other);

    point_cloud &operator=(const point_cloud &other);

    void download(std::vector<point_t> &dst) const;
    std::vector<point_t> download() const;

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

    bool estimate_normals(float search_radius);

    const thrust::device_vector<gca::point_t> &get_points();
    const thrust::device_vector<float3> &get_normals();

    std::vector<float3> download_normals() const;

    std::shared_ptr<point_cloud> voxel_grid_down_sample(float voxel_size);

    std::shared_ptr<point_cloud> radius_outlier_removal(float radius,
                                                        gca::counter_t min_neighbors_in_radius);

    std::pair<std::shared_ptr<std::vector<gca::index_t>>, gca::counter_t> euclidean_clustering(
        const float cluster_tolerance, const gca::counter_t min_cluster_size,
        const gca::counter_t max_cluster_size);

    static std::shared_ptr<point_cloud> create_from_rgbd(const gca::cuda_depth_frame &depth,
                                                         const gca::cuda_color_frame &color,
                                                         const gca::cuda_camera_param &param,
                                                         float threshold_min_in_meter = 0.0f,
                                                         float threshold_max_in_meter = 10.0f);

    static thrust::device_vector<gca::index_t> nn_search(gca::point_cloud &query_pc,
                                                         gca::point_cloud &reference_pc,
                                                         float radius);

    static auto search_radius(gca::point_cloud &query_pc, gca::point_cloud &reference_pc,
                              float radius)
        -> thrust::pair<thrust::device_vector<gca::index_t>,
                        thrust::device_vector<thrust::pair<gca::index_t, gca::counter_t>>>;

    static void nn_search(std::vector<gca::index_t> &result_nn_idx, gca::point_cloud &query_pc,
                          gca::point_cloud &reference_pc, float radius);

    ~point_cloud() = default;

private:
    bool compute_min_max_bound();

private:
    thrust::device_vector<gca::point_t> m_points;
    thrust::device_vector<float3> m_normals;
    bool m_has_normals = false;
    bool m_has_bound = false;
    float3 m_min_bound;
    float3 m_max_bound;
    friend class movement_detection;
};
} // namespace gca
