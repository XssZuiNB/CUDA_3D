#include <chrono>
#include <iostream>
#include <memory>
#include <opencv4/opencv2/opencv.hpp>
#include <thread>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

#include "cuda_container/cuda_container.hpp"
#include "devices/realsense_device.hpp"
#include "geometry/cuda_point_cloud.cuh"
#include "geometry/type.hpp"
#include "util/gpu_check.hpp"

int main(int argc, char *argv[])
{
    cuda_warm_up_gpu(0);
    auto rs_cam = gca::realsense_device();
    if (!rs_cam.device_start())
        return 1;

    gca::intrinsics c_in = rs_cam.get_color_intrinsics();
    gca::intrinsics d_in = rs_cam.get_depth_intrinsics();
    gca::extrinsics ex_d_to_c = rs_cam.get_depth_to_color_extrinsics();
    auto ex_c_to_d = rs_cam.get_color_to_depth_extrinsics();

    gca::cuda_camera_param cu_param(rs_cam);
    /*
    cu_param.set(d_in, gca::depth_intrinsics);
    cu_param.set(c_in, gca::color_intrinsics);
    cu_param.set(ex_d_to_c, gca::depth2color_extrinsics);
*/
    auto depth_scale = rs_cam.get_depth_scale();

    typedef pcl::PointXYZRGBA PointT;
    typedef pcl::PointCloud<PointT> PointCloud;
    PointCloud::Ptr cloud(new PointCloud);

    pcl::visualization::CloudViewer viewer("viewer");

    // auto gpu_color_intrin = gca::make_cuda_obj(c_in);
    // auto gpu_depth_intrin = gca::make_cuda_obj(d_in);
    // auto gpu_depth_to_color_extrin = gca::make_cuda_obj(ex_d_to_c);

    gca::cuda_color_frame gpu_color(rs_cam.get_width(), rs_cam.get_height());
    gca::cuda_depth_frame gpu_depth(rs_cam.get_width(), rs_cam.get_height());

    while (true)
    {
        cloud->clear();
        rs_cam.receive_data();

        auto color = rs_cam.get_color_raw_data();
        auto depth = rs_cam.get_depth_raw_data();

        gca::point_t points[640 * 480];

        auto start = std::chrono::steady_clock::now();
        gpu_color.upload((uint8_t *)color, 640, 480);
        gpu_depth.upload((uint16_t *)depth, 640, 480);

        if (!gpu_make_point_set(points, gpu_depth, gpu_color, cu_param))
        {
            std::cout << "fault" << std::endl;
            return 1;
        }

        auto end = std::chrono::steady_clock::now();

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
        std::cout << cloud->size() << std::endl;
        std::cout << "__________________________________________________" << std::endl;

        viewer.showCloud(cloud);
    }
    while (!viewer.wasStopped())
    {
    }
    return 0;
}
