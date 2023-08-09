#include "camera/realsense_device.hpp"
#include "cuda_container/cuda_container.hpp"
#include "geometry/cuda_point_cloud_factory.cuh"
#include "geometry/point_cloud.hpp"
#include "geometry/type.hpp"
#include "util/gpu_check.hpp"

#include <chrono>
#include <iostream>
#include <memory>
#include <opencv4/opencv2/opencv.hpp>
#include <signal.h>
#include <thread>
#include <unistd.h>

#include <omp.h>

#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

static bool exit_requested = false; // for ctrl+c exit
static void exit_sig_handler(int param)
{
    exit_requested = true;
}

static constexpr uint32_t n_cameras = 2;

int main(int argc, char *argv[])
{
    signal(SIGINT, exit_sig_handler); // for ctrl+c exit

    cuda_print_devices();
    cuda_warm_up_gpu(0);

    auto rs_cam_0 = gca::realsense_device(0, 640, 480, 60);
    if (!rs_cam_0.device_start())
        return 1;
    auto rs_cam_1 = gca::realsense_device(1, 640, 480, 60);
    if (!rs_cam_1.device_start())
        return 1;

    gca::cuda_camera_param cu_param_0(rs_cam_0);
    gca::cuda_camera_param cu_param_1(rs_cam_1);

    typedef pcl::PointXYZRGBA PointT;
    typedef pcl::PointCloud<PointT> PointCloud;

    PointCloud::Ptr cloud_0(new PointCloud);
    pcl::visualization::CloudViewer viewer_0("viewer0");
    PointCloud::Ptr cloud_1(new PointCloud);
    // pcl::visualization::CloudViewer viewer_1("viewer1");

    gca::cuda_color_frame gpu_color_0(rs_cam_0.get_width(), rs_cam_0.get_height());
    gca::cuda_depth_frame gpu_depth_0(rs_cam_0.get_width(), rs_cam_0.get_height());
    gca::cuda_color_frame gpu_color_1(rs_cam_1.get_width(), rs_cam_1.get_height());
    gca::cuda_depth_frame gpu_depth_1(rs_cam_1.get_width(), rs_cam_1.get_height());

    // omp_set_num_threads(2);

    while (!exit_requested)
    {
        rs_cam_0.receive_data();
        auto color_0 = rs_cam_0.get_color_raw_data();
        auto depth_0 = rs_cam_0.get_depth_raw_data();
        rs_cam_1.receive_data();
        auto color_1 = rs_cam_1.get_color_raw_data();
        auto depth_1 = rs_cam_1.get_depth_raw_data();

        auto start = std::chrono::steady_clock::now();

        gpu_color_0.upload((uint8_t *)color_0, 640, 480);
        gpu_depth_0.upload((uint16_t *)depth_0, 640, 480);
        gpu_color_1.upload((uint8_t *)color_1, 640, 480);
        gpu_depth_1.upload((uint16_t *)depth_1, 640, 480);

        auto pc_0 =
            gca::point_cloud::create_from_rgbd(gpu_depth_0, gpu_color_0, cu_param_0, 0.0, 10.0);
        auto pc_downsampling_0 = pc_0->voxel_grid_down_sample(0.03f);
        auto pc_remove_noise_0 = pc_downsampling_0->radius_outlier_removal(0.06, 8);

        auto pc_1 =
            gca::point_cloud::create_from_rgbd(gpu_depth_1, gpu_color_1, cu_param_1, 0.0, 10.0);
        auto pc_downsampling_1 = pc_1->voxel_grid_down_sample(0.03f);
        auto pc_remove_noise_1 = pc_downsampling_1->radius_outlier_removal(0.06, 8);

        auto end = std::chrono::steady_clock::now();

        std::cout << "Total cuda time in microseconds: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                  << "us" << std::endl;
        std::cout << "Total cuda time in milliseconds: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << "ms" << std::endl;

        std::cout << "GPU pc1 Voxel number: " << pc_downsampling_0->points_number() << std::endl;
        std::cout << "GPU pc2 Voxel number: " << pc_downsampling_1->points_number() << std::endl;
        std::cout << "GPU pc1 after radius outlier removal points number: "
                  << pc_remove_noise_0->points_number() << std::endl;
        std::cout << "GPU pc2 after radius outlier removal points number: "
                  << pc_remove_noise_1->points_number() << std::endl;

        auto points_0 = pc_downsampling_0->download();

        auto number_of_points = points_0.size();
        cloud_0->points.resize(number_of_points);
        for (size_t i = 0; i < number_of_points; i++)
        {
            PointT p;
            p.x = points_0[i].coordinates.x;
            p.y = -points_0[i].coordinates.y;
            p.z = -points_0[i].coordinates.z;
            p.r = points_0[i].color.r * 255;
            p.g = points_0[i].color.g * 255;
            p.b = points_0[i].color.b * 255;
            cloud_0->points[i] = p;
        }

        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered(
            new pcl::PointCloud<pcl::PointXYZRGBA>);

        pcl::RadiusOutlierRemoval<pcl::PointXYZRGBA> sor_radius;
        sor_radius.setInputCloud(cloud_0);
        sor_radius.setRadiusSearch(0.06);
        sor_radius.setMinNeighborsInRadius(8);

        start = std::chrono::steady_clock::now();
        sor_radius.filter(*cloud_filtered);
        end = std::chrono::steady_clock::now();

        /*
                          pcl::StatisticalOutlierRemoval<pcl::PointXYZRGBA>
                                           sor_statistical; sor_statistical.setInputCloud(cloud);
                                   sor_statistical.setMeanK(10);
           sor_statistical.setStddevMulThresh(1.0); sor_statistical.filter(*cloud_filtered);




                                                                pcl::VoxelGrid<pcl::PointXYZRGBA>
           sor; sor.setInputCloud(cloud); sor.setLeafSize(0.02f, 0.02f, 0.02f);
                                                                sor.filter(*cloud_filtered);
        */
        viewer_0.showCloud(cloud_filtered);
        std::cout << "PCL radius outlier removal time in microseconds: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                  << "us" << std::endl;
        std::cout << "PCL radius outlier removal time in milliseconds: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << "ms" << std::endl;

        std::cout << "Points number after PCL filter: " << cloud_filtered->size() << std::endl;
        // std::cout << cloud_filtered->size() << std::endl;
        std::cout << "__________________________________________________" << std::endl;
    }

    return 0;
}
