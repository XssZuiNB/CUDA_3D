#include "color_icp.hpp"

namespace gca
{

// private:
Eigen::Matrix4f color_icp::transform_vec6f_to_mat4f(const Eigen::Matrix<float, 6, 1> &input)
{
    Eigen::Matrix4f output;
    output.setIdentity();
    output.block<3, 3>(0, 0) = (Eigen::AngleAxisf(input(2), Eigen::Vector3f::UnitZ()) *
                                Eigen::AngleAxisf(input(1), Eigen::Vector3f::UnitY()) *
                                Eigen::AngleAxisf(input(0), Eigen::Vector3f::UnitX()))
                                   .matrix();
    output.block<3, 1>(0, 3) = input.block<3, 1>(3, 0);
    return output;
}
} // namespace gca