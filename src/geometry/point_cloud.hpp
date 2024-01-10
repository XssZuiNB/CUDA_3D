#pragma once

#include "cuda_container/cuda_container.hpp"
#include "geometry/type.hpp"

#include <memory>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

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
    point_cloud &operator+(const point_cloud &other);
    point_cloud &operator+=(const point_cloud &other);
    point_cloud &operator+=(point_cloud &&other);

    void download(std::vector<point_t> &dst) const;
    std::vector<point_t> download() const;

    bool has_points() const
    {
        return !m_points.empty();
    }

    bool has_normals() const
    {
        return m_has_normals;
    }

    size_t points_number() const
    {
        return m_points.size();
    }

    bool estimate_normals(float search_radius);

    const thrust::device_vector<gca::point_t> &get_points() const;
    const thrust::device_vector<float3> &get_normals() const;
    const float3 &get_min_bound() const;
    const float3 &get_max_bound() const;
    const float3 &get_centroid();

    std::vector<float3> download_normals() const;

    void transform(const mat4x4 &trans_mat);

    std::shared_ptr<point_cloud> voxel_grid_down_sample(float voxel_size);

    std::shared_ptr<point_cloud> remove_plane(std::vector<float> &plane_model);

    std::shared_ptr<point_cloud> radius_outlier_removal(float radius,
                                                        gca::counter_t min_neighbors_in_radius);

    std ::vector<thrust::host_vector<gca::index_t>> euclidean_clustering(
        const float cluster_tolerance, const gca::counter_t min_cluster_size,
        const gca::counter_t max_cluster_size);

    std ::vector<thrust::host_vector<gca::index_t>> convex_obj_segmentation(
        const float cluster_tolerance, const gca::counter_t min_cluster_size,
        const gca::counter_t max_cluster_size);

    std ::vector<thrust::host_vector<gca::index_t>> icp_residual_segmentation(
        const float cluster_tolerance, const thrust::device_vector<float> &abs_residual,
        const gca::counter_t min_cluster_size, const gca::counter_t max_cluster_size);

    /*
    auto search_radius(float radius)
        -> thrust::pair<thrust::device_vector<gca::index_t>,
                        thrust::device_vector<thrust::pair<gca::index_t, gca::counter_t>>>;

    static auto search_radius(gca::point_cloud &query_pc, gca::point_cloud &reference_pc,
                              float radius)
        -> thrust::pair<thrust::device_vector<gca::index_t>,
                        thrust::device_vector<thrust::pair<gca::index_t, gca::counter_t>>>;
    */
    std::shared_ptr<point_cloud> create_new_by_index(
        const thrust::device_vector<gca::index_t> &indices) const;

    std::shared_ptr<point_cloud> create_new_by_index(
        const thrust::host_vector<gca::index_t> &indices) const;

    static std::shared_ptr<point_cloud> create_from_rgbd(const gca::cuda_depth_frame &depth,
                                                         const gca::cuda_color_frame &color,
                                                         const gca::cuda_camera_param &param,
                                                         float threshold_min_in_meter = 0.0f,
                                                         float threshold_max_in_meter = 10.0f);

    static std::shared_ptr<point_cloud> create_from_pcl(
        const pcl::PointCloud<pcl::PointXYZRGB> &pcl_pc);

    static thrust::device_vector<gca::index_t> nn_search(const gca::point_cloud &query_pc,
                                                         const gca::point_cloud &reference_pc,
                                                         float radius);

    static void nn_search(std::vector<gca::index_t> &result_nn_idx, gca::point_cloud &query_pc,
                          gca::point_cloud &reference_pc, float radius);

    ~point_cloud() = default;

private:
    void compute_min_max_bound() const;

private:
    thrust::device_vector<gca::point_t> m_points;
    thrust::device_vector<float3> m_normals;
    bool m_has_normals = false;
    mutable bool m_has_bound = false;
    mutable float3 m_min_bound;
    mutable float3 m_max_bound;
    bool m_has_centroid = false;
    float3 m_centroid;
};
} // namespace gca
