#pragma once

#include "geometry/point_cloud.hpp"

#include <memory>

namespace gca
{
class movement_detection
{
public:
    movement_detection();
    void set_source_point_cloud(std::shared_ptr<gca::point_cloud> pc_ptr_src);
    void set_target_point_cloud(std::shared_ptr<gca::point_cloud> pc_ptr_tgt);
    void update_point_cloud(std::shared_ptr<gca::point_cloud> pc_new);
    std::shared_ptr<gca::point_cloud> moving_objects_detection();
    ~movement_detection();

private:
    std::shared_ptr<gca::point_cloud> m_pc_ptr_src;
    std::shared_ptr<gca::point_cloud> m_pc_ptr_tgt;

    // parameters
    static constexpr float m_nn_search_radius = 0.08f;
    static constexpr float m_geometry_weight = 1.0f;
    static constexpr float m_photometry_weight = 40.0f;
    static constexpr float m_clustering_tolerance = 0.04f;

    // color icp
    static constexpr float m_compute_tgt_color_gradient_radius = 0.05f;
    static constexpr float m_color_icp_lambda = 0.968;
    static constexpr float m_color_icp_geometry_weight = 0.983939;  // sqrt(0.968)
    static constexpr float m_color_icp_intensity_weight = 0.178885; // sqrt(1-0.968)
};
} // namespace gca
