#include "movement_detection.hpp"

#include "geometry/cuda_nn_search.cuh"
#include "geometry/point_cloud.hpp"
#include "movement_detection/cuda_movement_detection.cuh"
#include "registration/cuda_color_icp_build_least_square.cuh"
#include "registration/cuda_compute_color_gradient.cuh"
#include "registration/eigen_solver.hpp"
#include "util/cuda_util.cuh"

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

namespace gca
{
movement_detection::movement_detection()
    : m_pc_ptr_src(nullptr)
    , m_pc_ptr_tgt(nullptr)
{
}

void movement_detection::set_source_point_cloud(std::shared_ptr<gca::point_cloud> pc_ptr_src)
{
    m_pc_ptr_src = pc_ptr_src;
}
void movement_detection::set_target_point_cloud(std::shared_ptr<gca::point_cloud> pc_ptr_tgt)
{
    m_pc_ptr_tgt = pc_ptr_tgt;
}
void movement_detection::update_point_cloud(std::shared_ptr<gca::point_cloud> pc_new)
{
    m_pc_ptr_src = m_pc_ptr_tgt;
    m_pc_ptr_tgt = pc_new;
}

std::shared_ptr<gca::point_cloud> movement_detection::moving_objects_detection()
{
    if (!m_pc_ptr_src || !m_pc_ptr_tgt)
    {
        return nullptr;
    }

    auto &pts_src = m_pc_ptr_src->get_points();
    auto &pts_tgt = m_pc_ptr_tgt->get_points();

    auto &normals_src = m_pc_ptr_src->get_normals();
    auto &normals_tgt = m_pc_ptr_tgt->get_normals();

    auto min_bound_tgt_with_padding =
        m_pc_ptr_tgt->get_min_bound() - make_float3(1.5 * m_compute_tgt_color_gradient_radius,
                                                    1.5 * m_compute_tgt_color_gradient_radius,
                                                    1.5 * m_compute_tgt_color_gradient_radius);
    auto max_bound_tgt_with_padding =
        m_pc_ptr_tgt->get_max_bound() + make_float3(1.5 * m_compute_tgt_color_gradient_radius,
                                                    1.5 * m_compute_tgt_color_gradient_radius,
                                                    1.5 * m_compute_tgt_color_gradient_radius);

    if (m_compute_tgt_color_gradient_radius * std::numeric_limits<int>::max() <
        max(max(max_bound_tgt_with_padding.x - min_bound_tgt_with_padding.x,
                max_bound_tgt_with_padding.y - min_bound_tgt_with_padding.y),
            max_bound_tgt_with_padding.z - min_bound_tgt_with_padding.z))
    {
        std::cout << YELLOW << "Radius is too small!" << std::endl;
        return nullptr;
    }

    // computer color gradient of tgt point cloud.
    auto color_gradient_tgt = thrust::device_vector<float3>(pts_tgt.size());

    auto start = std::chrono::steady_clock::now();
    auto err = cuda_compute_color_gradient(color_gradient_tgt, pts_tgt, normals_tgt,
                                           min_bound_tgt_with_padding, max_bound_tgt_with_padding,
                                           m_compute_tgt_color_gradient_radius);
    auto end = std::chrono::steady_clock::now();
    std::cout << "Idx time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us"
              << std::endl;
    if (err != ::cudaSuccess)
        return nullptr;

    auto nn_src_tgt = gca::point_cloud::nn_search(*m_pc_ptr_src, *m_pc_ptr_tgt, m_nn_search_radius);

    auto output = std::make_shared<gca::point_cloud>(*m_pc_ptr_src);
    thrust::device_vector<float> result_rg_plus_rc(pts_src.size());

    err = cuda_compute_residual_color_icp(result_rg_plus_rc, m_pc_ptr_src->get_points(), pts_tgt,
                                          normals_tgt, color_gradient_tgt, nn_src_tgt,
                                          m_color_icp_lambda);

    thrust::sort(result_rg_plus_rc.begin(), result_rg_plus_rc.end());
    auto thre = result_rg_plus_rc[result_rg_plus_rc.size() / 3];

    /*
    thrust::device_vector<float> residuals;
    float mean_residual_over_all;

    auto nn_s = gca::point_cloud::nn_search(*m_pc_ptr_src, *m_pc_ptr_tgt, m_nn_search_radius);

    auto err =
        cuda_compute_res_and_mean_res(residuals, mean_residual_over_all, pts_src, pts_tgt, nn_s,
                                      m_nn_search_radius, m_geometry_weight,
    m_photometry_weight); if (err != ::cudaSuccess) return nullptr;

    auto clustering_result_pair_host =
        m_pc_ptr_src->euclidean_clustering(m_clustering_tolerance, 80, pts_src.size());
    if (!clustering_result_pair_host.second)
        return nullptr;

    thrust::device_vector<gca::index_t> clusters(pts_src.size());
    thrust::copy(clustering_result_pair_host.first->begin(),
                 clustering_result_pair_host.first->end(), clusters.begin());

    err = cuda_moving_objects_seg(output->m_points, clustering_result_pair_host.second,
    clusters, pts_src, residuals, mean_residual_over_all);
    */
    return output;
}

movement_detection::~movement_detection() = default;
} // namespace gca
