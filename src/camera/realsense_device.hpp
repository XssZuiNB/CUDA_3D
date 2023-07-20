#pragma once

#include <librealsense2/rs.hpp>

#include "camera/camera_device.hpp"

namespace gca
{
class realsense_device final : public gca::device
{
public:
    realsense_device();

    realsense_device(uint32_t width, uint32_t height, uint8_t fps,
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

    ~realsense_device() = default;

private:
    rs2::device m_device;
    rs2::config m_config;
    rs2::pipeline m_pipe_line;
    rs2::stream_profile m_color_profile;
    rs2::stream_profile m_depth_profile;
    rs2::frame m_rs_color;
    rs2::frame m_rs_depth;

private:
    bool find_device() final;
    bool start_stream() final;
    float read_depth_scale() final;
    gca::intrinsics read_color_intrinsics() final;
    gca::intrinsics read_depth_intrinsics() final;
    gca::extrinsics read_color_to_depth_extrinsics() final;
    gca::extrinsics read_depth_to_color_extrinsics() final;
    void receive_data_from_device() final;
    const void *set_color_raw_data() final;
    const void *set_depth_raw_data() final;
    cv::Mat set_color_to_cv_mat() final;
    cv::Mat set_depth_to_cv_mat() final;
};
} // namespace gca
