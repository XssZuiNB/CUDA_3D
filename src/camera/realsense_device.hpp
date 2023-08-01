#pragma once

#include "camera/camera_device.hpp"

#include <librealsense2/rs.hpp>

namespace gca
{
class realsense_device final : public gca::device
{
public:
    realsense_device();

    realsense_device(size_t device_id);

    realsense_device(size_t device_id, uint32_t width, uint32_t height, uint8_t fps,
                     gca::frame_format color_format = BGR8, gca::frame_format depth_format = Z16);

    realsense_device(const realsense_device &) = delete;

    realsense_device &operator=(const realsense_device &) = delete;

    realsense_device(realsense_device &&) = default;

    realsense_device &operator=(const realsense_device &&other)
    {
        if (this != &other)
        {
            *this = std::move(other);
        }

        return *this;
    }

    ~realsense_device();

private:
    rs2::device m_device;
    rs2::config m_config;
    rs2::pipeline m_pipe_line;
    bool m_pipe_line_active = false;
    rs2::stream_profile m_color_profile;
    rs2::stream_profile m_depth_profile;
    uint32_t m_device_id;

private:
    bool find_device() final;
    bool start_stream() final;
    float read_depth_scale() final;
    gca::intrinsics read_color_intrinsics() final;
    gca::intrinsics read_depth_intrinsics() final;
    gca::extrinsics read_color_to_depth_extrinsics() final;
    gca::extrinsics read_depth_to_color_extrinsics() final;
    void receive_data_from_device() final;
    cv::Mat set_color_to_cv_mat() final;
    cv::Mat set_depth_to_cv_mat() final;
    void stop_stream() final;
};
} // namespace gca
