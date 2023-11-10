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
template <typename T, size_t N, typename Tuple> struct make_tuple_impl
{
};

template <typename T, size_t N, typename... O> struct make_tuple_impl<T, N, std::tuple<O...>>
{
    using type = typename make_tuple_impl<T, N - 1, std::tuple<T, O...>>::type;
};

template <typename T, typename... O> struct make_tuple_impl<T, 0, std::tuple<O...>>
{
    using type = std::tuple<O...>;
};

template <typename T, size_t N> struct tuple_with_size
{
    using type = typename make_tuple_impl<T, N, std::tuple<>>::type;
};

class visualizer
{
public:
    visualizer();

    visualizer(const visualizer &other) = delete;
    visualizer &operator==(const visualizer &) = delete;

    void update(std::shared_ptr<gca::point_cloud> new_pc);

    void close();

    ~visualizer();

private:
    void visualizer_loop();
    void threadsafe_push(std::shared_ptr<gca::point_cloud> new_pc);
    void threadsafe_pop(std::shared_ptr<gca::point_cloud> &pc);
    bool empty();

private:
    std::queue<std::shared_ptr<gca::point_cloud>> m_data_queue;
    static constexpr size_t max_queue_size = 5;
    mutable std::mutex m_mutex;
    std::condition_variable m_cond_var;
    std::thread m_thread;

    std::shared_ptr<pcl::visualization::PCLVisualizer> m_viewer;
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr m_pcl_cloud;
};
} // namespace gca
