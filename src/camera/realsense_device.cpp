#include "camera/realsense_device.hpp"

#include <cstring>
#include <iostream>

namespace gca
{
realsense_device::realsense_device()
    : realsense_device::realsense_device(0, 640, 480, 30)
{
}

realsense_device::realsense_device(size_t device_id)
    : realsense_device::realsense_device(device_id, 640, 480, 30)
{
}

realsense_device::realsense_device(size_t device_id, uint32_t width, uint32_t height, uint8_t fps,
                                   gca::frame_format color_format, gca::frame_format depth_format)
    : m_device_id(device_id)
    , device(width, height, fps, color_format, depth_format)
{
    auto realsense_color_format = RS2_FORMAT_BGR8;
    auto realsense_depth_format = RS2_FORMAT_Z16;

    switch (color_format)
    {
    case BGR8:
        break;

    case RGB8:
        realsense_color_format = RS2_FORMAT_RGB8;

    default:
        break;
    }

    switch (depth_format)
    {
    case Z16:
        break;

    default:
        break;
    }

    m_config.enable_stream(RS2_STREAM_COLOR, width, height, realsense_color_format, fps);
    m_config.enable_stream(RS2_STREAM_DEPTH, width, height, realsense_depth_format, fps);
}

bool realsense_device::find_device()
{
    rs2::context ctx;
    auto list = ctx.query_devices();

    auto n_device = list.size();

    if (n_device == 0 || n_device <= m_device_id)
    {
        return false;
    }

    m_device = list[m_device_id];

    // Set Laser power to max
    for (auto &sensor : m_device.query_sensors())
    {
        if (auto dpt = sensor.as<rs2::depth_sensor>())
        {
            if (dpt.supports(RS2_OPTION_LASER_POWER))
            {
                // Query min and max values:
                auto range = dpt.get_option_range(RS2_OPTION_LASER_POWER);
                dpt.set_option(RS2_OPTION_LASER_POWER, range.max);
            }
        }
    }

    m_config.enable_device(m_device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));

    m_device_name = m_device.get_info(RS2_CAMERA_INFO_NAME);

    return true;
}

bool realsense_device::start_stream()
{
    if (!m_pipe_line_active)
    {
        m_pipe_line.start(m_config);
        m_pipe_line_active = true;

        /* Warm up Camera and get frame profiles of both color and depth frame*/
        while (!m_depth_profile || !m_color_profile)
        {
            auto frames =
                m_pipe_line.wait_for_frames(); // Wait for next set of frames from the camera

            if (frames.size() == 1)
            {
                if (frames.get_profile().stream_type() == RS2_STREAM_COLOR)
                {
                    m_color_profile = frames.get_color_frame().get_profile();
                }
                else
                {
                    m_depth_profile = frames.get_depth_frame().get_profile();
                }
            }
            else
            {
                m_color_profile = frames.get_color_frame().get_profile();
                m_depth_profile = frames.get_depth_frame().get_profile();
            }
        }
    }
    return true;
}

float realsense_device::read_depth_scale()
{
    for (auto &sensor : m_device.query_sensors())
    {
        // Check if the sensor if a depth sensor
        if (auto dpt = sensor.as<rs2::depth_sensor>())
        {
            return dpt.get_depth_scale();
        }
    }

    return 0.0;
}

gca::intrinsics realsense_device::read_color_intrinsics()
{
    auto color_intrinsics = m_color_profile.as<rs2::video_stream_profile>().get_intrinsics();

    gca::intrinsics tmp = {color_intrinsics.fx, color_intrinsics.fy, color_intrinsics.ppx,
                           color_intrinsics.ppy};

    return std::move(tmp);
}

gca::intrinsics realsense_device::read_depth_intrinsics()
{
    auto depth_intrinsics = m_depth_profile.as<rs2::video_stream_profile>().get_intrinsics();

    gca::intrinsics tmp = {depth_intrinsics.fx, depth_intrinsics.fy, depth_intrinsics.ppx,
                           depth_intrinsics.ppy};

    return std::move(tmp);
}

gca::extrinsics realsense_device::read_color_to_depth_extrinsics()
{
    auto color_to_depth_extrinsics =
        m_color_profile.as<rs2::video_stream_profile>().get_extrinsics_to(m_depth_profile);

    gca::extrinsics tmp;

    std::memcpy(&tmp.rotation, &color_to_depth_extrinsics.rotation, sizeof(float) * 9);
    std::memcpy(&tmp.translation, &color_to_depth_extrinsics.translation, sizeof(float) * 3);

    return std::move(tmp);
}

gca::extrinsics realsense_device::read_depth_to_color_extrinsics()
{
    auto depth_to_color_extrinsics =
        m_depth_profile.as<rs2::video_stream_profile>().get_extrinsics_to(m_color_profile);

    gca::extrinsics tmp;

    std::memcpy(&tmp.rotation, &depth_to_color_extrinsics.rotation, sizeof(float) * 9);
    std::memcpy(&tmp.translation, &depth_to_color_extrinsics.translation, sizeof(float) * 3);

    return std::move(tmp);
}

void realsense_device::receive_data_from_device()
{
    auto frames = m_pipe_line.wait_for_frames();
    if (frames.size() == 1)
    {
        std::cout << RED << "BAD FRAMES!" << std::endl;
    }

    m_color_raw_data = frames.get_color_frame().get_data();
    m_depth_raw_data = frames.get_depth_frame().get_data();
}

cv::Mat realsense_device::set_color_to_cv_mat()
{
    cv::Mat tmp;

    switch (m_color_format)
    {
    case BGR8:
        tmp = cv::Mat(cv::Size(m_width, m_height), CV_8UC3, (void *)m_color_raw_data,
                      cv::Mat::AUTO_STEP);
        break;

    case RGB8:
        tmp = cv::Mat(cv::Size(m_width, m_height), CV_8UC3, (void *)m_color_raw_data,
                      cv::Mat::AUTO_STEP);
        cv::cvtColor(tmp, tmp, cv::COLOR_RGB2BGR);
        break;
    }

    return tmp;
}

cv::Mat realsense_device::set_depth_to_cv_mat()
{
    cv::Mat tmp;
    switch (m_depth_format)
    {
    case Z16:
        tmp = cv::Mat(cv::Size(m_width, m_height), CV_16UC1, (void *)m_depth_raw_data,
                      cv::Mat::AUTO_STEP);
        break;
    }

    return tmp;
}

void realsense_device::stop_stream()
{
    if (m_pipe_line_active)
    {
        m_pipe_line.stop();
        m_pipe_line_active = false;
    }
}

realsense_device::~realsense_device()
{
    stop_stream();
}
} // namespace gca
