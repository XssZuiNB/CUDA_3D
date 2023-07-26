#include <chrono>
#include <iostream>
#include <memory>
#include <opencv4/opencv2/opencv.hpp>
#include <signal.h>
#include <thread>
#include <unistd.h>

#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

#include "camera/realsense_device.hpp"
#include "cuda_container/cuda_container.hpp"
#include "geometry/cuda_point_cloud_factory.cuh"
#include "geometry/point_cloud.hpp"
#include "geometry/type.hpp"
#include "util/gpu_check.hpp"

static bool exit_requested = false; // for ctrl+c exit
static void exit_sig_handler(int param)
{
    exit_requested = true;
}

int main(int argc, char *argv[])
{
    signal(SIGINT, exit_sig_handler); // for ctrl+c exit

    cuda_print_devices();
    cuda_warm_up_gpu(0);
    auto rs_cam = gca::realsense_device();
    if (!rs_cam.device_start())
        return 1;

    gca::intrinsics c_in = rs_cam.get_color_intrinsics();
    gca::intrinsics d_in = rs_cam.get_depth_intrinsics();
    gca::extrinsics ex_d_to_c = rs_cam.get_depth_to_color_extrinsics();
    auto ex_c_to_d = rs_cam.get_color_to_depth_extrinsics();

    gca::cuda_camera_param cu_param(rs_cam);

    auto depth_scale = rs_cam.get_depth_scale();

    typedef pcl::PointXYZRGBA PointT;
    typedef pcl::PointCloud<PointT> PointCloud;
    PointCloud::Ptr cloud(new PointCloud);

    pcl::visualization::CloudViewer viewer("viewer");

    gca::cuda_color_frame gpu_color(rs_cam.get_width(), rs_cam.get_height());
    gca::cuda_depth_frame gpu_depth(rs_cam.get_width(), rs_cam.get_height());

    while (!exit_requested)
    {
        rs_cam.receive_data();

        auto color = rs_cam.get_color_raw_data();
        auto depth = rs_cam.get_depth_raw_data();
        auto start = std::chrono::steady_clock::now();
        gpu_color.upload((uint8_t *)color, 640, 480);
        gpu_depth.upload((uint16_t *)depth, 640, 480);

        auto pc = gca::point_cloud::create_from_rgbd(gpu_depth, gpu_color, cu_param, 0.5, 10.0);

        auto pc_downsampling = pc->voxel_grid_down_sample(0.03f);

        auto pc_remove_noise = pc_downsampling->radius_outlier_removal(0.06, 8);
        auto end = std::chrono::steady_clock::now();
        std::cout << "GPU Voxel : " << pc_downsampling->points_number() << std::endl;
        std::cout << "GPU radius outlier removal : " << pc_remove_noise->points_number()
                  << std::endl;

        auto points = pc_remove_noise->download();

        auto number_of_points = points.size();
        cloud->points.resize(number_of_points);

        for (size_t i = 0; i < number_of_points; i++)
        {
            PointT p;
            p.x = points[i].coordinates.x;
            p.y = -points[i].coordinates.y;
            p.z = -points[i].coordinates.z;
            p.r = points[i].r;
            p.g = points[i].g;
            p.b = points[i].b;
            cloud->points[i] = p;
        }
        /*
                pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered(
                    new pcl::PointCloud<pcl::PointXYZRGBA>);

                pcl::RadiusOutlierRemoval<pcl::PointXYZRGBA> sor_radius;
                sor_radius.setInputCloud(cloud);
                sor_radius.setRadiusSearch(0.06);
                sor_radius.setMinNeighborsInRadius(8);

                sor_radius.filter(*cloud_filtered);


                                                        pcl::StatisticalOutlierRemoval<pcl::PointXYZRGBA>
                           sor_statistical; sor_statistical.setInputCloud(cloud);
                   sor_statistical.setMeanK(10); sor_statistical.setStddevMulThresh(1.0);
                                                        sor_statistical.filter(*cloud_filtered);




                                                pcl::VoxelGrid<pcl::PointXYZRGBA> sor;
                                                sor.setInputCloud(cloud);
                                                sor.setLeafSize(0.02f, 0.02f, 0.02f);
                                                sor.filter(*cloud_filtered);
                                        */
        viewer.showCloud(cloud);
        std::cout << "Time in microseconds: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                  << "us" << std::endl;
        std::cout << "Time in milliseconds: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << "ms" << std::endl;

        std::cout << "Points number PCL filter: " << cloud->size() << std::endl;
        // std::cout << cloud_filtered->size() << std::endl;
        std::cout << "__________________________________________________" << std::endl;
    }

    return 0;
}
