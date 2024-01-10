#pragma once

#include "geometry/point_cloud.hpp"
#include "geometry/type.hpp"
#include "util/math.cuh"

// std
#include <memory>

// CUDA
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace gca
{
class color_icp
{
public:
    color_icp() = delete;
    color_icp(size_t maximum_iterations, float max_correspondence_distance,
              float search_radius_color_gradient);

    void set_source_point_cloud(const std::shared_ptr<const gca::point_cloud> new_pc);
    void set_target_point_cloud(const std::shared_ptr<const gca::point_cloud> new_pc);

    bool align();

    std::pair<thrust::device_vector<float>, float> get_abs_residual();

    std::pair<thrust::host_vector<float>, float> get_abs_residual_host();

    const mat4x4 &get_final_transformation_matrix() const;
    float get_RSME() const;

    std::shared_ptr<const gca::point_cloud> get_transformed_source_point_cloud() const;

    ~color_icp() = default;

private:
    bool compute_color_gradient_target();

private:
    // convergence threshold
    // theta = arccos(0.5 * (r_11 + r_22 + r_33 - 1)) from rotation matrix
    // if theta -> 0 => cos(theta) -> 1
    static constexpr float m_color_icp_lambda = 0.968f;             // 0.968 from paper
    static constexpr float m_translations_thres_square = 0.000001f; // square number!
    static constexpr float m_rotation_thres = 1.0f - m_translations_thres_square;
    size_t m_max_iter;
    float m_max_corr_dist;
    bool m_color_icp_done;

    std::shared_ptr<gca::point_cloud> m_source_pc;
    std::shared_ptr<const gca::point_cloud> m_target_pc;
    thrust::device_vector<float3> m_target_pc_color_gradient;

    // parameters for preparing taget_pc
    float m_search_radius_color_gradient;
    bool m_color_gradient_done;

    mat4x4 m_transformation_matrix;
    float m_RMSE;
};
} // namespace gca
