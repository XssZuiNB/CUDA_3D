#include <chrono>
#include <iostream>
#include <memory>
#include <opencv4/opencv2/opencv.hpp>
#include <thread>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

#include "cuda_point_cloud.cuh"
#include "realsense_device.hpp"

/* test function
void transform_point_to_point(float to_point[3], const gca::extrinsics *extrin,
                              const float from_point[3])
{
    to_point[0] = extrin->rotation[0] * from_point[0] + extrin->rotation[3] * from_point[1] +
                  extrin->rotation[6] * from_point[2] + extrin->translation[0];
    to_point[1] = extrin->rotation[1] * from_point[0] + extrin->rotation[4] * from_point[1] +
                  extrin->rotation[7] * from_point[2] + extrin->translation[1];
    to_point[2] = extrin->rotation[2] * from_point[0] + extrin->rotation[5] * from_point[1] +
                  extrin->rotation[8] * from_point[2] + extrin->translation[2];
}

void depth_to_xyz(const int uv[2], uint16_t depth, float xyz[3], float depth_scale,
                  const gca::intrinsics &depth_in)
{
    auto z = depth * depth_scale;
    xyz[2] = z;
    xyz[0] = (uv[0] - depth_in.cx) * z / depth_in.fx;
    xyz[1] = (uv[1] - depth_in.cy) * z / depth_in.fy;
}

void xyz_to_uv(int uv[2], const float xyz[3], const gca::intrinsics &color_in)
{

    uv[0] = (xyz[0] * color_in.fx / xyz[2]) + color_in.cx;
    uv[1] = (xyz[1] * color_in.fy / xyz[2]) + color_in.cy;
}
*/

int main(int argc, char *argv[])
{

    cuda_warm_up_gpu(0);
    std::unique_ptr<gca::device> rs_cam(new gca::realsense_device());
    rs_cam->device_start();

    const gca::intrinsics c_in = rs_cam->get_color_intrinsics();
    const gca::intrinsics d_in = rs_cam->get_depth_intrinsics();
    const gca::extrinsics ex_c_to_d = rs_cam->get_color_to_depth_extrinsics();
    const gca::extrinsics ex_d_to_c = rs_cam->get_depth_to_color_extrinsics();

    auto depth_scale = rs_cam->get_depth_scale();

    typedef pcl::PointXYZRGBA PointT;
    typedef pcl::PointCloud<PointT> PointCloud;
    PointCloud::Ptr cloud(new PointCloud);

    pcl::visualization::CloudViewer viewer("viewer");

    while (true)
    {
        cloud->clear();
        rs_cam->receive_data();

        auto color = rs_cam->get_color_raw_data();
        auto depth = rs_cam->get_depth_raw_data();

        gca::point_t points[640 * 480];
        std::cout << "start" << std::endl;

        auto start = std::chrono::steady_clock::now();

        if (!gpu_make_point_set(points, 640, 480, (uint16_t *)depth, (uint8_t *)color, d_in, c_in,
                                ex_d_to_c, depth_scale))
            std::cout << "fault" << std::endl;
        auto end = std::chrono::steady_clock::now();

        /* this loop is very slow */
        for (auto point : points)
        {
            if (point.z != 0)
            {
                PointT p;
                p.x = point.x;
                p.y = point.y;
                p.z = point.z;
                p.r = point.r;
                p.g = point.g;
                p.b = point.b;
                cloud->points.push_back(p);
            }
        }

        std::cout << "Time in milliseconds: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << "ms" << std::endl;
        std::cout << cloud->size();

        viewer.showCloud(cloud);
    }
    while (!viewer.wasStopped())
    {
    }

    /* test cpu point cloud
    while (true)
    {
        cloud->clear();
        rs_cam->receive_data();
        cv::Mat color = rs_cam->get_color_cv_mat();
        cv::Mat depth = rs_cam->get_depth_cv_mat();

        int pd_uv[2], pc_uv[2];

        float Pdc3[3], Pcc3[3];

        auto start = std::chrono::steady_clock::now();

        for (int row = 0; row < depth.rows; row++)
        {
            for (int col = 0; col < depth.cols; col++)
            {

                uint16_t depth_value = depth.at<uint16_t>(row, col);
                if (depth_value == 0)
                    continue;

                PointT p;
                pd_uv[0] = col;
                pd_uv[1] = row;

                depth_to_xyz(pd_uv, depth_value, Pdc3, depth_scale, d_in);
                p.x = Pdc3[0];
                p.y = Pdc3[1];
                p.z = Pdc3[2];

                transform_point_to_point(Pcc3, &ex_d_to_c, Pdc3);

                xyz_to_uv(pc_uv, Pcc3, c_in);
                auto x = pc_uv[0] + 0.6f;
                auto y = pc_uv[1] + 0.6f;
                if (x < 0 || x > depth.cols - 1 || y < 0 || y > depth.rows - 1)
                {
                    continue;
                }
                p.b = color.at<cv::Vec3b>(y, x)[0];
                p.g = color.at<cv::Vec3b>(y, x)[1];
                p.r = color.at<cv::Vec3b>(y, x)[2];

                cloud->points.push_back(p);
            }
        }
        auto end = std::chrono::steady_clock::now();
        std::cout << "Time in milliseconds: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << "ms " << std::endl;

        std::cout << cloud->size();
        viewer.showCloud(cloud);

    }
    */
    return 0;
}
