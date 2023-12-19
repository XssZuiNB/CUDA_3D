#include "visualizer.hpp"

#include <future>

namespace gca
{
visualizer::visualizer()
    : m_viewer(nullptr)
    , m_pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>)
{
    m_thread = std::thread(&gca::visualizer::visualizer_loop, this);
    m_running = true;
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
    m_cond_var.wait(lock, [this] { return !m_data_queue.empty() || !m_running; });
    if (m_running) {
     pc = m_data_queue.front();
    m_data_queue.pop();
    }
}

void visualizer::update(std::shared_ptr<gca::point_cloud> new_pc)
{
    if (m_running)
    {
        threadsafe_push(new_pc);
    }
}

void visualizer::visualizer_loop()
{
    using namespace std::chrono_literals;

    auto viewer = std::make_shared<pcl::visualization::PCLVisualizer>("point cloud Viewer");

    while (m_running)
    {
        std::shared_ptr<gca::point_cloud> pc_device;
        threadsafe_pop(pc_device);

        auto pc_host = pc_device->download();
        auto number_of_points = pc_host.size();
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pcl_pc(new pcl::PointCloud<pcl::PointXYZRGBA>);
        pcl_pc->points.resize(number_of_points);

        auto f = std::async([=] {
            for (size_t i = 0; i < number_of_points / 2; i++)
            {
                pcl::PointXYZRGBA p;
                p.x = pc_host[i].coordinates.x;
                p.y = -pc_host[i].coordinates.y;
                p.z = -pc_host[i].coordinates.z;
                p.r = pc_host[i].color.r * 255;
                p.g = pc_host[i].color.g * 255;
                p.b = pc_host[i].color.b * 255;
                pcl_pc->points[i] = p;
            }
        });
        for (size_t i = number_of_points / 2; i < number_of_points; i++)
        {
            pcl::PointXYZRGBA p;
            p.x = pc_host[i].coordinates.x;
            p.y = -pc_host[i].coordinates.y;
            p.z = -pc_host[i].coordinates.z;
            p.r = pc_host[i].color.r * 255;
            p.g = pc_host[i].color.g * 255;
            p.b = pc_host[i].color.b * 255;
            pcl_pc->points[i] = p;
        }
        f.get();

        if (!viewer->contains("cloud"))
        {
            viewer->addPointCloud(pcl_pc, "cloud");
        }
        else
        {
            viewer->updatePointCloud(pcl_pc, "cloud");
        }

        viewer->spinOnce();

        std::this_thread::sleep_for(5ms);
    }

    viewer->removeAllPointClouds();
    viewer->close();
}

void visualizer::close()
{
    m_running = false;

    m_cond_var.notify_one();

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
