#pragma once

#include "geometry/point_cloud.hpp"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <tuple>
#include <utility> // for std::index_sequence and std::make_index_sequence

#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace gca
{
/*
template <size_t N, typename T, typename Seq> struct MakeTupleImpl;

template <size_t N, typename T, size_t... Is> struct MakeTupleImpl<N, T, std::index_sequence<Is...>>
{
using type = std::tuple<decltype(T(Is)))...>;
};

template <size_t N, typename T>
auto make_tuple_of_size() -> typename MakeTupleImpl<N, T, std::make_index_sequence<N>>::type
{
return typename MakeTupleImpl<N, T, std::make_index_sequence<N>>::type();
}
*/
class visualizer
{
public:
    visualizer();
    visualizer(std::shared_ptr<gca::point_cloud> pc);

    visualizer(const visualizer &other) = delete;
    visualizer &operator==(const visualizer &) = delete;

    void update(std::shared_ptr<gca::point_cloud> new_pc);

    ~visualizer();

private:
    void visualizer_loop();
    void threadsafe_push(std::shared_ptr<gca::point_cloud> new_pc);
    void threadsafe_pop(std::shared_ptr<gca::point_cloud> &pc);
    bool empty();

private:
    std::queue<std::shared_ptr<gca::point_cloud>> m_data_queue;
    mutable std::mutex m_mutex;
    std::condition_variable m_cond_var;

    std::shared_ptr<pcl::visualization::PCLVisualizer> m_viewer;
    bool m_is_first_frame;
    std::thread m_thread;

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr m_pcl_cloud;
};
} // namespace gca