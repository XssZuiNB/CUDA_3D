#include "visualizer.hpp"

namespace gca
{
visualizer::visualizer()
    : m_viewer(std::make_shared<pcl::visualization::PCLVisualizer>("point cloud Viewer"))
    , m_is_first_frame(true)
{
}

bool visualizer::empty()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_data_queue.empty();
}

void visualizer::threadsafe_push(std::shared_ptr<gca::point_cloud> new_pc)
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_data_queue.push(new_pc);
    }

    m_cond_var.notify_one();
}

void visualizer::threadsafe_pop(std::shared_ptr<gca::point_cloud> &pc)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    m_cond_var.wait(lock, [this] { return !m_data_queue.empty(); });
    pc = m_data_queue.front();
    m_data_queue.pop();
}

void visualizer::visualizer_loop()
{
    while (!m_viewer->wasStopped())
    {
        std::shared_ptr<gca::point_cloud> pc_device;
        threadsafe_pop(pc_device);

        auto pc_host = pc_device->download();
        auto number_of_points = pc_host.size();
        m_pcl_cloud->points.resize(number_of_points);

        for (size_t i = 0; i < number_of_points; i++)
        {
            pcl::PointXYZRGBA p;
            p.x = pc_host[i].coordinates.x;
            p.y = -pc_host[i].coordinates.y;
            p.z = -pc_host[i].coordinates.z;
            p.r = pc_host[i].color.r * 255;
            p.g = pc_host[i].color.g * 255;
            p.b = pc_host[i].color.b * 255;
            m_pcl_cloud->points[i] = p;
        }

        if (m_is_first_frame)
        {
            m_viewer->addPointCloud(m_pcl_cloud, "cloud");
            m_is_first_frame = false;
        }
        else
        {
            m_viewer->updatePointCloud(m_pcl_cloud, "cloud");
        }

        m_viewer->spinOnce();
    }
}

} // namespace gca
