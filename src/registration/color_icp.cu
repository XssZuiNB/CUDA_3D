#include "color_icp.hpp"

#include "geometry/geometry_util.cuh"
#include "registration/cuda_color_icp_build_least_square.cuh"
#include "registration/cuda_compute_color_gradient.cuh"
#include "registration/eigen_solver.hpp"
#include "util/console_color.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace gca
{
color_icp::color_icp(size_t maximum_iterations, float max_correspondence_distance,
                     float search_radius_color_gradient)
    : m_max_iter(maximum_iterations)
    , m_max_corr_dist(max_correspondence_distance)
    , m_color_icp_done(false)
    , m_source_pc(nullptr)
    , m_target_pc(nullptr)
    , m_search_radius_color_gradient(search_radius_color_gradient)
    , m_color_gradient_done(false)
    , m_transformation_matrix(mat4x4::get_identity())
    , m_RMSE(0.0f)
{
}

void color_icp::set_source_point_cloud(const std::shared_ptr<const gca::point_cloud> new_pc)
{
    m_color_icp_done = false;
    m_transformation_matrix.set_identity();
    m_RMSE = 0.0f;

    m_source_pc = std::make_shared<gca::point_cloud>(*new_pc);
}

void color_icp::set_target_point_cloud(const std::shared_ptr<const gca::point_cloud> new_pc)
{
    m_color_icp_done = false;
    m_color_gradient_done = false;
    m_transformation_matrix.set_identity();
    m_RMSE = 0.0f;

    m_target_pc = new_pc;
}

bool color_icp::align()
{
    if (m_color_icp_done)
    {
        return true;
    }

    if (!m_source_pc)
    {
        std::cout << YELLOW << "Need to set a valid source point cloud" << std::endl;
        return false;
    }

    if (!m_target_pc)
    {
        std::cout << YELLOW << "Need to set a valid target point cloud" << std::endl;
        return false;
    }

    if (!m_target_pc->has_normals())
    {
        std::cout << YELLOW << "Need to estimate normals of target point cloud" << std::endl;
        return false;
    }

    auto &pts_tgt = m_target_pc->get_points();
    auto &normals_tgt = m_target_pc->get_normals();

    if (!m_color_gradient_done)
    {
        auto min_bound_tgt_with_padding =
            m_target_pc->get_min_bound() - make_float3(1.5f * m_search_radius_color_gradient,
                                                       1.5f * m_search_radius_color_gradient,
                                                       1.5f * m_search_radius_color_gradient);
        auto max_bound_tgt_with_padding =
            m_target_pc->get_max_bound() + make_float3(1.5f * m_search_radius_color_gradient,
                                                       1.5f * m_search_radius_color_gradient,
                                                       1.5f * m_search_radius_color_gradient);

        if (m_search_radius_color_gradient * std::numeric_limits<int>::max() <
            max(max(max_bound_tgt_with_padding.x - min_bound_tgt_with_padding.x,
                    max_bound_tgt_with_padding.y - min_bound_tgt_with_padding.y),
                max_bound_tgt_with_padding.z - min_bound_tgt_with_padding.z))
        {
            std::cout << YELLOW << "Radius is too small!" << std::endl;
            return false;
        }

        auto err = cuda_compute_color_gradient(
            m_target_pc_color_gradient, pts_tgt, normals_tgt, min_bound_tgt_with_padding,
            max_bound_tgt_with_padding, m_search_radius_color_gradient);

        if (err != ::cudaSuccess)
        {
            return false;
        }

        m_color_gradient_done = true;
    }

    mat6x6 JTJ;
    mat6x1 JTr;
    float RMSE;

    for (size_t i = 0; i < m_max_iter; i++)
    {
        auto nn_src_tgt = gca::point_cloud::nn_search(*m_source_pc, *m_target_pc, m_max_corr_dist);

        auto err = cuda_build_gauss_newton_color_icp(
            JTJ, JTr, RMSE, m_source_pc->get_points(), pts_tgt, normals_tgt,
            m_target_pc_color_gradient, nn_src_tgt, m_color_icp_lambda);
        if (err != ::cudaSuccess)
        {
            std::cout << YELLOW << "Gauss Newton failed!" << std::endl;
            m_transformation_matrix.set_identity();
            return false;
        }

        auto this_transformation_matrix = solve_JTJ_JTr(JTJ, JTr);

        m_source_pc->transform(this_transformation_matrix);

        m_transformation_matrix = this_transformation_matrix * m_transformation_matrix;
        // check convergence
        // Rotation
        float cos_theta =
            0.5 * (this_transformation_matrix(0, 0) + this_transformation_matrix(1, 1) +
                   this_transformation_matrix(2, 2) - 1);
        // Translation square
        float translation_square =
            this_transformation_matrix(0, 3) * this_transformation_matrix(0, 3) +
            this_transformation_matrix(1, 3) * this_transformation_matrix(1, 3) +
            this_transformation_matrix(2, 3) * this_transformation_matrix(2, 3);
        if (cos_theta >= m_rotation_thres && translation_square <= m_translations_thres_square)
        {
            std::cout << GREEN << "Color ICP converged!" << std::endl;
            break;
        }
    }

    m_RMSE = RMSE / m_source_pc->points_number();
    m_color_icp_done = true;
    return true;
}

const mat4x4 &color_icp::get_final_transformation_matrix() const
{
    if (!m_color_icp_done)
    {
        std::cout << YELLOW << "Color ICP is not done yet! Identity Matrix returned!" << std::endl;
    }
    return m_transformation_matrix;
}

float color_icp::get_RSME() const
{
    if (!m_color_icp_done)
    {
        std::cout << YELLOW << "Color ICP is not done yet! 0.0f returned!" << std::endl;
    }
    return m_RMSE;
}

std::shared_ptr<const gca::point_cloud> color_icp::get_transformed_source_point_cloud() const
{
    if (!m_color_icp_done)
    {
        std::cout << YELLOW << "Color ICP is not done yet! Invalid point cloud returned!"
                  << std::endl;
    }
    return m_source_pc;
}
} // namespace gca
