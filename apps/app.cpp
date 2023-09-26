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

#include <pcl/features/normal_3d.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

static bool exit_requested = false; // for ctrl+c exit
static void exit_sig_handler(int param)
{
    exit_requested = true;
}

static constexpr uint32_t n_cameras = 2;

Eigen::Vector3f computeAverageNormal(const pcl::PointCloud<pcl::Normal>::Ptr &normals,
                                     const std::vector<int> &indices)
{
    Eigen::Vector3f avg_normal(0, 0, 0);
    for (int idx : indices)
    {
        avg_normal += normals->points[idx].getNormalVector3fMap();
    }
    avg_normal /= indices.size();
    avg_normal.normalize();
    return avg_normal;
}

Eigen::Vector3i generateRandomColor()
{
    int r = rand() % 256;
    int g = rand() % 256;
    int b = rand() % 256;
    return Eigen::Vector3i(r, g, b);
}

int main(int argc, char *argv[])
{
    signal(SIGINT, exit_sig_handler); // for ctrl+c exit

    cuda_print_devices();
    cuda_warm_up_gpu(0);

    auto rs_cam_0 = gca::realsense_device(0, 640, 480, 30);
    if (!rs_cam_0.device_start())
        return 1;

    gca::cuda_camera_param cu_param_0(rs_cam_0);

    typedef pcl::PointXYZRGBA PointT;
    typedef pcl::PointCloud<PointT> PointCloud;

    PointCloud::Ptr cloud_0(new PointCloud);
    PointCloud::Ptr cloud_1(new PointCloud);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
    pcl::visualization::CloudViewer viewer_0("viewer0");

    gca::cuda_color_frame gpu_color_0(rs_cam_0.get_width(), rs_cam_0.get_height());
    gca::cuda_depth_frame gpu_depth_0(rs_cam_0.get_width(), rs_cam_0.get_height());

    bool if_first_frame = true;
    std::shared_ptr<gca::point_cloud> last_frame_ptr;

    // test
    /*
    cv::Mat frame1, frame2, frame3; // 存储三帧
    cv::Mat gray1, gray2, gray3;    // 存储灰度帧
    cv::Mat diff12, diff23, result; // 差分图像和结果

    rs_cam_0.receive_data();
    frame1 = rs_cam_0.get_color_cv_mat();
    rs_cam_0.receive_data();
    frame2 = rs_cam_0.get_color_cv_mat();

    cv::cvtColor(frame1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frame2, gray2, cv::COLOR_BGR2GRAY);

    cv::Ptr<cv::BackgroundSubtractor> pMOG2 = cv::createBackgroundSubtractorMOG2(80, 14.0);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::Mat kernel_close = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

    while (true)
    {
        rs_cam_0.receive_data();
        auto start = std::chrono::steady_clock::now();
        frame3 = rs_cam_0.get_color_cv_mat();

        // pMOG2->apply(currRgbFrame, fgMask);

        cv::cvtColor(frame3, gray3, cv::COLOR_BGR2GRAY);

        cv::absdiff(gray1, gray2, diff12);
        cv::absdiff(gray2, gray3, diff23);

        cv::threshold(diff12, diff12, 8, 255, cv::THRESH_BINARY);
        cv::threshold(diff23, diff23, 8, 255, cv::THRESH_BINARY);

        cv::bitwise_and(diff12, diff23, result);

        cv::morphologyEx(result, result, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(result, result, cv::MORPH_CLOSE, kernel_close);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(result, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::drawContours(frame2, contours, -1, cv::Scalar(0, 0, 255), 2);

        // cv::bitwise_and(rgbDiff, fgMask, result);

        auto end = std::chrono::steady_clock::now();

        std::cout << "opencv: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                  << "us" << std::endl;

        cv::bitwise_and(result, gray3, result);

        cv::imshow("Frame Difference", frame2);

        frame1 = frame2.clone();
        frame2 = frame3.clone();

        gray1 = gray2.clone();
        gray2 = gray3.clone();
        if (cv::waitKey(30) == 'q')
        {
            break;
        }
    }
    */
    // 初始化背景减除器
    /* GMM
    cv::Ptr<cv::BackgroundSubtractor> pMOG2 = cv::createBackgroundSubtractorMOG2(80, 18.0);

    cv::Mat frame, fgMask, result;

    // 定义形态学操作的核
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat kernel_close = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10));

    while (true)
    {
        auto start = std::chrono::steady_clock::now();
        // 读取摄像头帧
        frame = rs_cam_0.get_color_cv_mat();

        // 应用高斯背景去除算法
        pMOG2->apply(frame, fgMask);

        // 形态学开运算去除噪声
        cv::morphologyEx(fgMask, result, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(result, result, cv::MORPH_CLOSE, kernel_close);
        auto end = std::chrono::steady_clock::now();

        std::cout << "opencv: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                  << "us" << std::endl;

        // 显示原始图像和结果
        // cv::imshow("Original", frame);
        // cv::imshow("FG Mask", fgMask);
        cv::imshow("Result", result);

        // 按 'q' 退出循环
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
        auto start = std::chrono::steady_clock::now();
        gpu_color_0.upload((uint8_t *)color_0, rs_cam_0.get_width(), rs_cam_0.get_height());
        gpu_depth_0.upload((uint16_t *)depth_0, rs_cam_0.get_width(), rs_cam_0.get_height());

        auto pc_0 =
            gca::point_cloud::create_from_rgbd(gpu_depth_0, gpu_color_0, cu_param_0, 0.4, 2);

        auto pc_remove_noise_0 = pc_0->radius_outlier_removal(0.02f, 4);

        auto pc_downsampling_0 = pc_remove_noise_0->voxel_grid_down_sample(0.01f);

        pc_downsampling_0->estimate_normals(0.06f);
        auto end = std::chrono::steady_clock::now();

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

        auto points_0 = pc_downsampling_0->download();
        auto normals_0 = pc_downsampling_0->download_normals();
        auto number_of_points = points_0.size();

        if (if_first_frame)
        {
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
                /*
                            pcl::Normal n;
                            n.normal_x = normals_0[i].x;
                            n.normal_y = -normals_0[i].y;
                            n.normal_z = -normals_0[i].z;
                            normals->points[i] = n;
                            */
            }
            if_first_frame = false;
            continue;
        }

        // normals->points.resize(number_of_points);
        cloud_1->points.resize(number_of_points);
        for (size_t i = 0; i < number_of_points; i++)
        {
            PointT p;
            p.x = points_0[i].coordinates.x;
            p.y = -points_0[i].coordinates.y;
            p.z = -points_0[i].coordinates.z;
            p.r = points_0[i].color.r * 255;
            p.g = points_0[i].color.g * 255;
            p.b = points_0[i].color.b * 255;
            cloud_1->points[i] = p;
            /*
                        pcl::Normal n;
                        n.normal_x = normals_0[i].x;
                        n.normal_y = -normals_0[i].y;
                        n.normal_z = -normals_0[i].z;
                        normals->points[i] = n;
                        */
        }

        pcl::search::Search<pcl::PointXYZRGBA>::Ptr tree =
            std::shared_ptr<pcl::search::Search<pcl::PointXYZRGBA>>(
                new pcl::search::KdTree<pcl::PointXYZRGBA>);
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

        pcl::RegionGrowingRGB<pcl::PointXYZRGBA> reg;
        reg.setInputCloud(cloud_1);
        reg.setSearchMethod(tree);
        reg.setDistanceThreshold(0.02);
        reg.setPointColorThreshold(6);
        reg.setRegionColorThreshold(0);
        reg.setMinClusterSize(60);

        std::vector<pcl::PointIndices> cluster_indices;
        reg.extract(cluster_indices);

        // 2. 对每个聚类进行ICP配准
        for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin();
             it != cluster_indices.end(); ++it)
        {
            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_cluster(
                new pcl::PointCloud<pcl::PointXYZRGBA>);
            for (std::vector<int>::const_iterator pit = it->indices.begin();
                 pit != it->indices.end(); ++pit)
                cloud_cluster->points.push_back(cloud_1->points[*pit]);
            cloud_cluster->width = cloud_cluster->points.size();
            cloud_cluster->height = 1;
            cloud_cluster->is_dense = true;

            pcl::IterativeClosestPoint<pcl::PointXYZRGBA, pcl::PointXYZRGBA> icp;
            icp.setMaximumIterations(0);
            icp.setInputSource(cloud_cluster);
            icp.setInputTarget(cloud_0);
            pcl::PointCloud<pcl::PointXYZRGBA> Final;
            icp.align(Final);

            // 3. 计算每个聚类的平均残差
            double avg_residual = icp.getFitnessScore();

            // 4. 标记残差大的聚类
            if (avg_residual > 0.0005)
            {
                for (std::vector<int>::const_iterator pit = it->indices.begin();
                     pit != it->indices.end(); ++pit)
                {
                    cloud_1->points[*pit].r = 255;
                    cloud_1->points[*pit].g = 0;
                    cloud_1->points[*pit].b = 0;
                }
            }
        }

        /*
        pcl::IterativeClosestPoint<pcl::PointXYZRGBA, pcl::PointXYZRGBA> icp;
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_aligned(
            new pcl::PointCloud<pcl::PointXYZRGBA>);
        icp.setInputSource(cloud_0);
        icp.setInputTarget(cloud_1);
        icp.align(*cloud_aligned);

        if (icp.hasConverged())
        {
            std::cout << "ICP converged with score: " << icp.getFitnessScore() << std::endl;
            pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
            kdtree.setInputCloud(cloud_1);

            for (size_t i = 0; i < cloud_aligned->points.size(); ++i)
            {
                std::vector<int> pointIdxNKNSearch(1);
                std::vector<float> pointNKNSquaredDistance(1);

                if (kdtree.nearestKSearch(cloud_aligned->points[i], 1, pointIdxNKNSearch,
                                          pointNKNSquaredDistance) > 0)
                {
                    double distance = std::sqrt(pointNKNSquaredDistance[0]);
                    if (distance > 0.05)
                    { // 例如，残差阈值为0.05，可以根据需要调整
                        cloud_1->points[i].r = 255;
                        cloud_1->points[i].g = 0;
                        cloud_1->points[i].b = 0;
                    }
                }
            }
        }
        */
        viewer_0.showCloud(cloud_1);
        *cloud_0 = *cloud_1;
        /* PCL Over seg*/
        /*
        pcl::NormalEstimation<pcl::PointXYZRGBA, pcl::Normal> ne;
        ne.setInputCloud(cloud_0);
        pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree(
            new pcl::search::KdTree<pcl::PointXYZRGBA>());
        ne.setSearchMethod(tree);
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        ne.setRadiusSearch(0.05);
        ne.compute(*normals);

        pcl::RegionGrowingRGB<pcl::PointXYZRGBA, pcl::Normal> reg;
        reg.setInputCloud(cloud_0);
        reg.setInputNormals(normals);
        reg.setSearchMethod(tree);
        reg.setDistanceThreshold(0.05);
        reg.setMinClusterSize(200);
        reg.setPointColorThreshold(15);
        reg.setSmoothnessThreshold(1.0 / 180.0 * M_PI);
        reg.setMinClusterSize(0);
        std::vector<pcl::PointIndices> clusters;
        reg.extract(clusters);

        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr surfels(new pcl::PointCloud<pcl::PointXYZRGBA>);
        for (const auto &cluster : clusters)
        {
            auto color = generateRandomColor();
            for (int idx : cluster.indices)
            {
                auto p = cloud_0->points[idx];
                p.r = color(0);
                p.g = color(1);
                p.b = color(2);
                surfels->push_back(p);
            }
        }
        */
        /*
        start = std::chrono::steady_clock::now();
        pcl::NormalEstimation<pcl::PointXYZRGBA, pcl::Normal> ne;
        ne.setInputCloud(cloud_0);

        pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree(
            new pcl::search::KdTree<pcl::PointXYZRGBA>());
        ne.setSearchMethod(tree);

        ne.setRadiusSearch(0.02);
        ne.compute(*normals);
        end = std::chrono::steady_clock::now();
        std::cout << "pcl time in milliseconds: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << "ms" << std::endl;
        */
        /*
        for (size_t i = 0; i < normals->size(); ++i)
        {
            std::cout << "Normal for point " << i << ": " << normals->points[i].normal_x << ", "
                      << normals->points[i].normal_y << ", " << normals->points[i].normal_z
                      << std::endl;
            std::cout << "gpu Normal       " << i << ": " << normals_0[i].x << ", "
                      << normals_0[i].y << ", " << normals_0[i].z << std::endl;
        }
        */
        /*
        start = std::chrono::steady_clock::now();
        pcl::search::Search<pcl::PointXYZRGBA>::Ptr tree(
            new pcl::search::KdTree<pcl::PointXYZRGBA>);

        pcl::RegionGrowingRGB<pcl::PointXYZRGBA> reg;
        reg.setInputCloud(cloud_0);
        reg.setSearchMethod(tree);

        reg.setDistanceThreshold(10);
        reg.setPointColorThreshold(6);
        reg.setRegionColorThreshold(5);

        std::vector<pcl::PointIndices> clusters;
        reg.extract(clusters);
        end = std::chrono::steady_clock::now();
        std::cout << "region: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << std::endl;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud();
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
        ec.setClusterTolerance(0.03);
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
        /*
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
        */
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

        // viewer_0.showCloud(cloud_0);
        //   viewer_1.spinOnce();
        std::cout << "__________________________________________________" << std::endl;
    }

    return 0;
}
