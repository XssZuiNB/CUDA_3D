#include "geometry/point_cloud.hpp"
#include "movement_detection/cuda_movement_detection.cuh"
#include "movement_detection/movement_detection.hpp"
#include "util/cuda_util.cuh"

#include <thrust/copy.h>
#include <thrust/device_vector.h>

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
    m_pc_ptr_tgt = m_pc_ptr_src;
    m_pc_ptr_src = pc_new;
}

std::shared_ptr<gca::point_cloud> movement_detection::moving_objects_detection()
{
    if (!m_pc_ptr_src || !m_pc_ptr_tgt)
    {
        return nullptr;
    }

    auto &pts_src = m_pc_ptr_src->get_points();
    auto &pts_tgt = m_pc_ptr_tgt->get_points();

    auto output = std::make_shared<gca::point_cloud>(pts_src.size());

    thrust::device_vector<float> residuals;
    float mean_residual_over_all;

    auto nn_s = gca::point_cloud::nn_search(*m_pc_ptr_src, *m_pc_ptr_tgt, m_nn_search_radius);

    auto err =
        cuda_compute_res_and_mean_res(residuals, mean_residual_over_all, pts_src, pts_tgt, nn_s,
                                      m_nn_search_radius, m_geometry_weight, m_photometry_weight);
    if (err != ::cudaSuccess)
        return nullptr;

    auto clustering_result_pair_host =
        m_pc_ptr_src->euclidean_clustering(m_clustering_tolerance, 80, pts_src.size());

    thrust::device_vector<gca::index_t> clusters(pts_src.size());
    thrust::copy(clustering_result_pair_host.first->begin(),
                 clustering_result_pair_host.first->end(), clusters.begin());

    err = cuda_moving_objects_seg(output->m_points, clustering_result_pair_host.second, clusters,
                                  pts_src, residuals);

    return output;
}

movement_detection::~movement_detection() = default;
} // namespace gca
