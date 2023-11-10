#include "visualizer.hpp"

namespace gca
{
visualizer::visualizer()
    : m_viewer(std::make_shared<pcl::visualization::PCLVisualizer>("point cloud Viewer"))
    , m_pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>)
{
}

bool visualizer::empty()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_data_queue.empty();
}

void visualizer::threadsafe_push(std::shared_ptr<gca::point_cloud> new_pc)
{
    if (m_data_queue.size() > max_queue_size)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_data_queue.pop();
        m_data_queue.push(new_pc);
    }
    else
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

void visualizer::update(std::shared_ptr<gca::point_cloud> new_pc)
{
    static std::once_flag thread_created_flag;
    std::call_once(thread_created_flag, [this] {
        this->m_thread = std::thread(&gca::visualizer::visualizer_loop, this);
    });

    if (m_viewer->wasStopped())
        return;

    threadsafe_push(new_pc);
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

        if (!m_viewer->contains("cloud"))
        {
            m_viewer->addPointCloud(m_pcl_cloud, "cloud");
        }
        else
        {
            m_viewer->updatePointCloud(m_pcl_cloud, "cloud");
        }

        m_viewer->spinOnce();
    }
}

void visualizer::close()
{
    m_viewer->removeAllPointClouds();
    m_viewer->close();

    if (m_thread.joinable())
    {
        m_thread.join();
    }
}

visualizer::~visualizer()
{
    close();
}

} // namespace gca
