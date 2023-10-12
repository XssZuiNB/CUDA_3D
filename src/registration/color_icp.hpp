#pragma once

#include "geometry/point_cloud.hpp"
#include "geometry/type.hpp"
#include "util/math.cuh"

// Eigen
#include "util/eigen_disable_bad_warnings.cuh"
#include <Eigen/Geometry>

// std
#include <memory>

// CUDA
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

namespace gca
{
class color_icp
{

public:
    color_icp(/* args */);
    ~color_icp();

private:
    void compute_target_color_gradient();

private:
    std::shared_ptr<gca::point_cloud> m_source_pc;
    std::shared_ptr<gca::point_cloud> m_target_pc;
    thrust::device_vector<mat3x1> m_target_pc_color_gradient;
    thrust::device_ptr<mat4x4> m_transformation_ptr;
    Eigen::Matrix4f m_transformation_eigen;

    // parameters for preparing taget_pc
    const float m_search_radius_color_gradient;
    const gca::counter_t min_radius_neighbors;
    bool color_gradient_done = false;

    // parameters for normals estimation if it is not done before
    const float m_search_radius_normal_estimation = m_search_radius_color_gradient;
    bool normal_estimated;
};
} // namespace gca