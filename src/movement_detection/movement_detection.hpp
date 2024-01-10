#pragma once

#include "geometry/point_cloud.hpp"
#include "registration/color_icp.hpp"

#include <memory>

namespace gca
{
class movement_detection
{
public:
    movement_detection() = delete;
    movement_detection(std::shared_ptr<gca::color_icp> color_icp);
    void set_source_point_cloud(std::shared_ptr<gca::point_cloud> pc_ptr_src);
    void set_target_point_cloud(std::shared_ptr<gca::point_cloud> pc_ptr_tgt);
    void update_point_cloud(std::shared_ptr<gca::point_cloud> pc_new);
    std::shared_ptr<gca::point_cloud> moving_objects_detection();
    ~movement_detection();

private:
    std::shared_ptr<gca::color_icp> color_icp;
};
} // namespace gca
