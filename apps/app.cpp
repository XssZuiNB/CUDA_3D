#include <chrono>
#include <iostream>
#include <memory>
#include <opencv4/opencv2/opencv.hpp>
#include <thread>

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

int main(int argc, char *argv[])
{
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

    while (true)
    {

        cloud->clear();
        rs_cam.receive_data();

        auto color = rs_cam.get_color_raw_data();
        auto depth = rs_cam.get_depth_raw_data();

        // std::vector<gca::point_t> points(640 * 480);
        auto start = std::chrono::steady_clock::now();
        gpu_color.upload((uint8_t *)color, 640, 480);
        gpu_depth.upload((uint16_t *)depth, 640, 480);
        /*
                if (!cuda_make_point_cloud(points, gpu_depth, gpu_color, cu_param))
                {
                    std::cout << "fault" << std::endl;
                    return 1;
                }
        */

        auto pc = gca::point_cloud::create_from_rgbd(gpu_depth, gpu_color, cu_param, 0.5, 10.0);

        auto pc_downsampling = pc->voxel_grid_down_sample(0.02f);
        auto end = std::chrono::steady_clock::now();
        std::cout << "GPU Voxel : " << pc_downsampling->points_number() << std::endl;

        auto min_b = pc->compute_min_bound();

        auto max_b = pc->compute_max_bound();

        std::cout << "min bound : " << min_b.x << "\n"
                  << min_b.y << "\n"
                  << min_b.z << "\n"
                  << std::endl;
        std::cout << "max bound : " << max_b.x << "\n"
                  << max_b.y << "\n"
                  << max_b.z << "\n"
                  << std::endl;
        auto points = pc_downsampling->download();

        for (auto &point : points)
        {
            // if (point.property != gca::point_property::invalid)
            {
                PointT p;
                p.x = point.coordinates.x;
                p.y = -point.coordinates.y;
                p.z = -point.coordinates.z;
                p.r = point.r;
                p.g = point.g;
                p.b = point.b;
                cloud->points.push_back(p);
            }
        }

        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered(
            new pcl::PointCloud<pcl::PointXYZRGBA>);
        /*
               pcl::RadiusOutlierRemoval<pcl::PointXYZRGBA> sor_radius;
               sor_radius.setInputCloud(cloud);
               sor_radius.setRadiusSearch(0.02);
               sor_radius.setMinNeighborsInRadius(2);
               sor_radius.filter(*cloud_filtered);


                pcl::StatisticalOutlierRemoval<pcl::PointXYZRGBA> sor_statistical;
                sor_statistical.setInputCloud(cloud);
                sor_statistical.setMeanK(10);
                sor_statistical.setStddevMulThresh(1.0);
                sor_statistical.filter(*cloud_filtered);


       */

        pcl::VoxelGrid<pcl::PointXYZRGBA> sor;
        sor.setInputCloud(cloud);
        sor.setLeafSize(0.02f, 0.02f, 0.02f);
        sor.filter(*cloud_filtered);

        viewer.showCloud(cloud);
        std::cout << "Time in microseconds: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                  << "us" << std::endl;
        std::cout << "Time in milliseconds: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << "ms" << std::endl;

        std::cout << "Points number: " << cloud_filtered->size() << std::endl;
        // std::cout << cloud_filtered->size() << std::endl;
        std::cout << "__________________________________________________" << std::endl;
    }
    while (!viewer.wasStopped())
    {
    }
    return 0;
}
