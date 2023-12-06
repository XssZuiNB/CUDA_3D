#include "geometry/cuda_nn_search.cuh"
#include "geometry/point_cloud.hpp"
#include "geometry/type.hpp"
#include "movement_detection/movement_detection.hpp"
#include "registration/color_icp.hpp"
#include "util/eigen_disable_bad_warnings.cuh"
#include "util/gpu_check.hpp"

#include <future>
#include <string>

#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/pcl_visualizer.h>

int loadPointCloud(const std::string &filename, pcl::PointCloud<pcl::PointXYZRGB> &cloud)
{
    std::string extension = filename.substr(filename.size() - 4, 4);
    if (extension == ".pcd")
    {
        if (pcl::io::loadPCDFile(filename, cloud) == -1)
        {
            std::cout << "Was not able to open file " << filename << std::endl;
            return 0;
        }
    }
    else if (extension == ".ply")
    {
        if (pcl::io::loadPLYFile(filename, cloud) == -1)
        {
            std::cout << "Was not able to open file " << filename << std::endl;
            return 0;
        }
    }
    else
    {
        std::cerr << "Was not able to open file " << filename << " (it is neither .pcd nor .ply) "
                  << std::endl;
        return 0;
    }
    return 1;
}

void downSampleVoxelGrids(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
    pcl::VoxelGrid<pcl::PointXYZRGB> sor;
    sor.setLeafSize(0.02, 0.02, 0.02);
    sor.setInputCloud(cloud);
    sor.filter(*cloud);
    std::cout << "Downsampled to " << cloud->size() << " points\n";
}

int main(int argc, char *argv[])
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr src(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tgt(new pcl::PointCloud<pcl::PointXYZRGB>());

    if (!loadPointCloud("../../data/frag_115.ply", *src))
    {
        std::cout << "Cant read source!" << std::endl;
        return 1;
    }

    if (!loadPointCloud("../../data/frag_116.ply", *tgt))
    {
        std::cout << "Cant read target!" << std::endl;
        return 1;
    }

    auto viewer = std::make_shared<pcl::visualization::PCLVisualizer>("point cloud Viewer");
    viewer->addPointCloud(src, "src");
    viewer->addPointCloud(tgt, "tgt");

    while (!viewer->wasStopped())
    {
        viewer->spinOnce();
        pcl_sleep(0.01);
    }

    auto src_device = gca::point_cloud::create_from_pcl(*src);
    auto tgt_device = gca::point_cloud::create_from_pcl(*tgt);

    /* Coarse to fine, good result but seems like not needed */
    /*
    auto down_sample_src = src_device->voxel_grid_down_sample(0.06);
    auto down_sample_tgt = tgt_device->voxel_grid_down_sample(0.06);

    down_sample_tgt->estimate_normals(0.11);

    gca::color_icp color_icp(1000, 0.08, 0.1);
    color_icp.set_source_point_cloud(down_sample_src);
    color_icp.set_target_point_cloud(down_sample_tgt);
    color_icp.align();

    src_device->transform(color_icp.get_final_transformation_matrix());
    down_sample_src = src_device->voxel_grid_down_sample(0.03);
    down_sample_tgt = tgt_device->voxel_grid_down_sample(0.03);

    down_sample_tgt->estimate_normals(0.06);

    gca::color_icp color_icp_2(1000, 0.08, 0.06);
    color_icp_2.set_source_point_cloud(down_sample_src);
    color_icp_2.set_target_point_cloud(down_sample_tgt);
    color_icp_2.align();
    */

    /* for this example best parameters, needs only 15ms to be perfect incl. dowm_sampling and
     * normal estimation */
    auto start = std::chrono::steady_clock::now();

    auto down_sample_src = src_device->voxel_grid_down_sample(0.02f);
    auto down_sample_tgt = tgt_device->voxel_grid_down_sample(0.02f);
    down_sample_tgt->estimate_normals(0.04f);
    gca::color_icp color_icp(50, 0.08f, 0.04f);
    color_icp.set_source_point_cloud(down_sample_src);
    color_icp.set_target_point_cloud(down_sample_tgt);

    color_icp.align();

    auto end = std::chrono::steady_clock::now();
    std::cout << "Total color icp in milliseconds: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms"
              << std::endl;

    std::cout << "RSME " << color_icp.get_RSME() << std::endl;

    // employ the result to original point cloud
    src_device->transform(color_icp.get_final_transformation_matrix());
    auto pc_host = src_device->download();
    auto number_of_points = pc_host.size();
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pcl_pc_result(new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl_pc_result->points.resize(number_of_points);

    auto f = std::async([=] {
        for (size_t i = 0; i < number_of_points / 2; i++)
        {
            pcl::PointXYZRGBA p;
            p.x = pc_host[i].coordinates.x;
            p.y = pc_host[i].coordinates.y;
            p.z = pc_host[i].coordinates.z;
            p.r = pc_host[i].color.r * 255;
            p.g = pc_host[i].color.g * 255;
            p.b = pc_host[i].color.b * 255;
            pcl_pc_result->points[i] = p;
        }
    });
    for (size_t i = number_of_points / 2; i < number_of_points; i++)
    {
        pcl::PointXYZRGBA p;
        p.x = pc_host[i].coordinates.x;
        p.y = pc_host[i].coordinates.y;
        p.z = pc_host[i].coordinates.z;
        p.r = pc_host[i].color.r * 255;
        p.g = pc_host[i].color.g * 255;
        p.b = pc_host[i].color.b * 255;
        pcl_pc_result->points[i] = p;
    }
    f.get();

    auto viewer_2 = std::make_shared<pcl::visualization::PCLVisualizer>("point cloud Viewer");
    viewer_2->addPointCloud(pcl_pc_result, "result");
    viewer_2->addPointCloud(tgt, "tgt");

    while (!viewer_2->wasStopped())
    {
        viewer_2->spinOnce();
        pcl_sleep(0.01);
    }
    return 0;
}
