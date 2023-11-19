#include "eigen_solver.hpp"

#include "util/eigen_disable_bad_warnings.cuh"
#include "util/math.cuh"

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace Eigen
{
using Matrix6f = Eigen::Matrix<float, 6, 6>;
using Vector6f = Eigen::Matrix<float, 6, 1>;
} // namespace Eigen

static Eigen::Matrix4f transform_vec6f_to_mat4f(const Eigen::Vector6f &input)
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

namespace gca
{
mat4x4 solve_JTJ_JTr(const mat6x6 &JTJ, const mat6x1 &JTr)
{
    // JTJ is Symmetric Matrix. Therefore, enenthough Eigen is column-major and mine is Row-major,
    // it can also work.
    Eigen::Matrix6f JTJ_(JTJ.ptr());
    Eigen::Vector6f JTr_(JTr.ptr());

    // equation 21 in paper of color icp
    Eigen::Vector6f se3_vec = JTJ_.ldlt().solve(-JTr_);

    Eigen::Matrix4f SE3_matrix_eigen(transform_vec6f_to_mat4f(se3_vec));

    mat4x4 SE3_matrix;

    for (size_t i = 0; i < 4; i++)
    {
        for (size_t j = 0; j < 4; j++)
        {
            SE3_matrix(i, j) = SE3_matrix_eigen(i, j);
        }
    }

    return SE3_matrix;
}
} // namespace gca