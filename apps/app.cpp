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
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

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

    auto rs_cam_0 = gca::realsense_device(1, 640, 480, 30);
    if (!rs_cam_0.device_start())
        return 1;
    /*
auto rs_cam_1 = gca::realsense_device(1, 640, 480, 60);
if (!rs_cam_1.device_start())
    return 1;*/

    gca::cuda_camera_param cu_param_0(rs_cam_0);
    // gca::cuda_camera_param cu_param_1(rs_cam_1);

    typedef pcl::PointXYZRGBA PointT;
    typedef pcl::PointCloud<PointT> PointCloud;

    PointCloud::Ptr cloud_0(new PointCloud);
    pcl::visualization::CloudViewer viewer_0("viewer0");
    // PointCloud::Ptr cloud_1(new PointCloud);
    // pcl::visualization::CloudViewer viewer_1("viewer1");

    gca::cuda_color_frame gpu_color_0(rs_cam_0.get_width(), rs_cam_0.get_height());
    gca::cuda_depth_frame gpu_depth_0(rs_cam_0.get_width(), rs_cam_0.get_height());
    // gca::cuda_color_frame gpu_color_1(rs_cam_1.get_width(), rs_cam_1.get_height());
    // gca::cuda_depth_frame gpu_depth_1(rs_cam_1.get_width(), rs_cam_1.get_height());

    bool if_first_frame = true;
    std::shared_ptr<gca::point_cloud> last_frame_ptr;

    // test
    /*
    // 上一帧和当前帧的 RGB 和深度图像
    cv::Mat prevRgbFrame;
    cv::Mat currRgbFrame;

    // 帧差图像
    cv::Mat rgbDiff, fgMask, result;

    rs_cam_0.receive_data();
    prevRgbFrame = rs_cam_0.get_color_cv_mat();
    cv::cvtColor(prevRgbFrame, prevRgbFrame, cv::COLOR_BGR2GRAY);

    cv::Ptr<cv::BackgroundSubtractor> pMOG2 = cv::createBackgroundSubtractorMOG2(200, 15.0);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::Mat kernel_close = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10));

    float alpha = 0.6;        // 设置pMOG2结果的权重为0.7
    float beta = 1.0 - alpha; // 设置帧差分算法结果的权重为0.3

    while (true)
    {
        rs_cam_0.receive_data();
        auto start = std::chrono::steady_clock::now();
        currRgbFrame = rs_cam_0.get_color_cv_mat();

        pMOG2->apply(currRgbFrame, fgMask);

        cv::cvtColor(currRgbFrame, currRgbFrame, cv::COLOR_BGR2GRAY);
        cv::absdiff(currRgbFrame, prevRgbFrame, rgbDiff);
        cv::threshold(rgbDiff, rgbDiff, 3, 255, cv::THRESH_BINARY);

        cv::morphologyEx(fgMask, fgMask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(fgMask, fgMask, cv::MORPH_CLOSE, kernel_close);

        cv::morphologyEx(rgbDiff, rgbDiff, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(rgbDiff, rgbDiff, cv::MORPH_CLOSE, kernel_close);

        cv::Mat weightedResult;
        cv::addWeighted(rgbDiff, beta, fgMask, alpha, 0.0, weightedResult);

        cv::bitwise_and(weightedResult, currRgbFrame, result);

        auto end = std::chrono::steady_clock::now();

        std::cout << "opencv: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                  << "us" << std::endl;

        // 显示结果
        cv::imshow("Frame Difference", result);

        // 更新帧
        prevRgbFrame = currRgbFrame.clone();
        if (cv::waitKey(30) == 'q')
        {
            break;
        }
    }
    */

    while (!exit_requested)
    {
        rs_cam_0.receive_data();
        auto color_0 = rs_cam_0.get_color_raw_data();
        auto depth_0 = rs_cam_0.get_depth_raw_data();
        /*
        rs_cam_1.receive_data();
        auto color_1 = rs_cam_1.get_color_raw_data();
        auto depth_1 = rs_cam_1.get_depth_raw_data();*/
        auto start = std::chrono::steady_clock::now();
        gpu_color_0.upload((uint8_t *)color_0, rs_cam_0.get_width(), rs_cam_0.get_height());
        gpu_depth_0.upload((uint16_t *)depth_0, rs_cam_0.get_width(), rs_cam_0.get_height());
        // gpu_color_1.upload((uint8_t *)color_1, 640, 480);
        // gpu_depth_1.upload((uint16_t *)depth_1, 640, 480);

        auto pc_0 =
            gca::point_cloud::create_from_rgbd(gpu_depth_0, gpu_color_0, cu_param_0, 0.3, 4);

        auto pc_remove_noise_0 = pc_0->radius_outlier_removal(0.025f, 6);

        auto pc_downsampling_0 = pc_remove_noise_0->voxel_grid_down_sample(0.03f);
        auto end = std::chrono::steady_clock::now();
        std::shared_ptr<gca::point_cloud> pc_moving;

        if (if_first_frame)
        {
            last_frame_ptr = pc_downsampling_0;
            if_first_frame = false;
            continue;
        }
        else
        {
            pc_moving = pc_downsampling_0->movement_detection(*last_frame_ptr, 0.05f, 0.03f);
            last_frame_ptr = pc_downsampling_0;
        }

        // auto clusters = pc_downsampling_0->euclidean_clustering(0.06f, 100, 25000);

        /*
                auto pc_1 =
                    gca::point_cloud::create_from_rgbd(gpu_depth_1, gpu_color_1, cu_param_1,
           0.2, 6.0); auto pc_downsampling_1 = pc_1->voxel_grid_down_sample(0.045f); auto
           pc_remove_noise_1 = pc_downsampling_1->radius_outlier_removal(0.09, 10);
        */
        std::cout << "Total cuda time in microseconds: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                  << "us" << std::endl;
        std::cout << "Total cuda time in milliseconds: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << "ms" << std::endl;

        std::cout << "GPU pc1 number: " << pc_0->points_number() << std::endl;
        std::cout << "GPU pc1 after remove noise number: " << pc_remove_noise_0->points_number()
                  << std::endl;
        std::cout << "GPU pc1 Voxel number: " << pc_downsampling_0->points_number() << std::endl;
        // std::cout << "GPU cluster num: " << clusters.second << std::endl;
        //  std::cout << "GPU pc2 Voxel number: " << pc_downsampling_1->points_number() <<
        //  std::endl;
        /*std::cout << "GPU pc1 after radius outlier removal points number: "
                  << pc_remove_noise_0->points_number() << std::endl;

std::cout << "GPU pc2 after radius outlier removal points number: "
        << pc_remove_noise_1->points_number() << std::endl;*/

        // std::vector<gca::index_t> result_nn_idx_cuda;
        // gca::point_cloud::nn_search(result_nn_idx_cuda, *pc_remove_noise_1, *pc_remove_noise_0,
        // 1);

        auto points_0 = pc_0->download();
        // auto points_1 = pc_downsampling_1->download();

        auto number_of_points = points_0.size();
        cloud_0->points.resize(number_of_points);

        for (size_t i = 0; i < number_of_points; i++)
        {
            PointT p;
            p.x = points_0[i].coordinates.x;
            p.y = points_0[i].coordinates.y;
            p.z = points_0[i].coordinates.z;
            p.r = points_0[i].color.r * 255;
            p.g = points_0[i].color.g * 255;
            p.b = points_0[i].color.b * 255;
            cloud_0->points[i] = p;
        }

        /*
        number_of_points = points_1.size();
        cloud_1->points.resize(number_of_points);
        for (size_t i = 0; i < number_of_points; i++)
        {
            PointT p;
            p.x = points_1[i].coordinates.x;
            p.y = -points_1[i].coordinates.y;
            p.z = -points_1[i].coordinates.z;
            p.r = points_1[i].color.r * 255;
            p.g = points_1[i].color.g * 255;
            p.b = points_1[i].color.b * 255;
            cloud_1->points[i] = p;
        }
        */

        /* PCL Radius search Test */
        /*
        start = std::chrono::steady_clock::now();
        pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
        kdtree.setInputCloud(cloud_0);

        float radius = 0.06f;
        auto n = 0;

        for (size_t i = 0; i < cloud_0->points.size(); ++i)
        {
            std::vector<int> pointIdxRadiusSearch;
            std::vector<float> pointRadiusSquaredDistance;

            if (kdtree.radiusSearch(cloud_0->points[i], radius, pointIdxRadiusSearch,
                                    pointRadiusSquaredDistance) > 0)
            {
                for (size_t j = 0; j < pointIdxRadiusSearch.size(); ++j)
                    ++n;
            }
        }
        end = std::chrono::steady_clock::now();
        std::cout << "PCL search radius time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << "ms" << std::endl;
        std::cout << "PCL neighbor total number: " << n << std::endl;
        */
        /* PCL Clustering test */
        /*
        start = std::chrono::steady_clock::now();
        pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree(
            new pcl::search::KdTree<pcl::PointXYZRGBA>);
        tree->setInputCloud(cloud_0);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZRGBA> ec;
        ec.setClusterTolerance(0.08);
        ec.setMinClusterSize(100);
        ec.setMaxClusterSize(25000);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud_0);

        ec.extract(cluster_indices);
        end = std::chrono::steady_clock::now();
        int j = 0;
        for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin();
             it != cluster_indices.end(); ++it)
        {
            j++;
        }

        std::cout << "PCL clustering time in microseconds: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << "ms" << std::endl;
        std::cout << "PCL clustering num: " << j << std::endl;
        */
        /* PCL NN Search Test */
        /*
        pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
        kdtree.setInputCloud(cloud_0);

        std::vector<int> nearest_indices_pcl;

        for (size_t i = 0; i < cloud_1->points.size(); ++i)
        {
            std::vector<int> pointIdxNKNSearch(1);
            std::vector<float> pointNKNSquaredDistance(1);

            if (kdtree.nearestKSearch(cloud_1->points[i], 1, pointIdxNKNSearch,
                                      pointNKNSquaredDistance) > 0)
            {
                nearest_indices_pcl.push_back(pointIdxNKNSearch[0]);
            }
            else
            {
                nearest_indices_pcl.push_back(-1);
            }
        }

        if (result_nn_idx_cuda.size() != nearest_indices_pcl.size())
        {
            std::cout << "NN HAS PROBLEM!!!" << std::endl;
        }

        auto different = 0;

        for (size_t i = 0; i < result_nn_idx_cuda.size(); i++)
        {
            if (result_nn_idx_cuda[i] != nearest_indices_pcl[i] && result_nn_idx_cuda[i] != -1)
            {
                // std::cout << "Wrong NN!!! at " << i << std::endl;
                std::cout << "cuda " << result_nn_idx_cuda[i] << std::endl;
                std::cout << "pcl " << nearest_indices_pcl[i] << std::endl;
                different += 1;
            }
        }
        std::cout << "different num " << different << std::endl;
        */

        /* PCL Radius Outlier removal test */

        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered(
            new pcl::PointCloud<pcl::PointXYZRGBA>);

        pcl::RadiusOutlierRemoval<pcl::PointXYZRGBA> sor_radius;
        sor_radius.setInputCloud(cloud_0);
        sor_radius.setRadiusSearch(0.025);
        sor_radius.setMinNeighborsInRadius(6);

        start = std::chrono::steady_clock::now();
        sor_radius.filter(*cloud_filtered);
        end = std::chrono::steady_clock::now();

        std::cout << "PCL radius outlier removal time in microseconds: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                  << "us" << std::endl;
        std::cout << "PCL radius outlier removal time in milliseconds: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << "ms" << std::endl;

        std::cout << "Points number after PCL filter: " << cloud_filtered->size() << std::endl;

        /*
        auto start = std::chrono::steady_clock::now();
     thrust::device_vector<thrust::pair<gca::index_t, gca::counter_t>> idx_and_count;
     thrust::device_vector<gca::index_t> neighbor_indicies;
     cuda_search_radius_neighbors(neighbor_indicies, idx_and_count, m_points,
     grid_cells_min_bound, grid_cells_max_bound, radius); auto end =
     std::chrono::steady_clock::now(); std::cout << "test run time dsearch radius: "
               << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() <<
     "us"
               << std::endl;
     std::cout << "neighbors: " << neighbor_indicies.size() << std::endl;
        */

        viewer_0.showCloud(cloud_0);
        // viewer_1.spinOnce();
        std::cout << "__________________________________________________" << std::endl;
    }

    return 0;
}
