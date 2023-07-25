#pragma once

#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <string>

#include "camera/camera_param.hpp"
#include "util/console_color.hpp"

namespace gca
{
class device
{
public:
    device() = delete;

    device(uint32_t width, uint32_t height, uint8_t fps, gca::frame_format color_format,
           gca::frame_format depth_format)
        : m_width(width)
        , m_height(height)
        , m_fps(fps)
        , m_color_format(color_format)
        , m_depth_format(depth_format)
    {
    }

    bool device_start()
    {
        if (!m_device_started)
        {
            if (!find_device())
            {
                std::cout << RED << "Didn't find a camera!" << std::endl;
                return false;
            }
            std::cout << D_GREEN << "Found camera! Name: " << m_device_name << std::endl;

            if (!start_stream())
            {
                std::cout << RED << "Couldn't start stream!" << std::endl;
                return false;
            }
            m_depth_scale = read_depth_scale();
            m_color_intrinsics = read_color_intrinsics();
            m_depth_intrinsics = read_depth_intrinsics();
            m_color_to_depth_extrinsics = read_color_to_depth_extrinsics();
            m_depth_to_color_extrinsics = read_depth_to_color_extrinsics();

            m_device_started = true;
        }

        return true;
    }

    void device_stop()
    {
        if (m_device_started)
        {
            stop_stream();
            m_device_started = false;
        }
    }

    uint32_t get_width() const
    {
        return m_width;
    }

    uint32_t get_height() const
    {
        return m_height;
    }

    float get_depth_scale() const
    {
        if (!m_device_started)
        {
            std::cout << YELLOW << "Device not started, invalid depth_scale returned!" << std::endl;
        }

        return m_depth_scale;
    }

    const gca::intrinsics &get_color_intrinsics() const
    {
        if (!m_device_started)
        {
            std::cout << YELLOW << "Device not started, invalid intrinsics returned!" << std::endl;
        }

        return m_color_intrinsics;
    }

    const gca::intrinsics &get_depth_intrinsics() const
    {
        if (!m_device_started)
        {
            std::cout << YELLOW << "Device not started, invalid intrinsics returned!" << std::endl;
        }
        return m_depth_intrinsics;
    }

    const gca::extrinsics &get_color_to_depth_extrinsics() const
    {
        if (!m_device_started)
        {
            std::cout << YELLOW << "Device not started, invalid extrinsics returned!" << std::endl;
        }
        return m_color_to_depth_extrinsics;
    }

    const gca::extrinsics &get_depth_to_color_extrinsics() const
    {
        if (!m_device_started)
        {
            std::cout << YELLOW << "Device not started, invalid extrinsics returned!" << std::endl;
        }
        return m_depth_to_color_extrinsics;
    }

    gca::frame_format get_depth_frame_format() const
    {
        return m_depth_format;
    }

    gca::frame_format get_color_frame_format() const
    {
        return m_color_format;
    }

    void receive_data()
    {
        if (!m_device_started)
        {
            m_device_started = device_start();
        }
        receive_data_from_device();
    }

    cv::Mat get_color_cv_mat(bool if_deep_copy = true)
    {
        cv::Mat tmp = set_color_to_cv_mat();

        if (!if_deep_copy)
        {
            return tmp;
        }

        return tmp.clone();
    }

    cv::Mat get_depth_cv_mat(bool if_deep_copy = true)
    {
        cv::Mat tmp = set_depth_to_cv_mat();

        if (!if_deep_copy)
        {
            return tmp;
        }

        return tmp.clone();
    }

    const void *get_color_raw_data() const
    {
        return m_color_raw_data;
    }

    const void *get_depth_raw_data() const
    {
        return m_depth_raw_data;
    }

    virtual ~device() = default;

protected:
    std::string m_device_name;
    const uint32_t m_width;
    const uint32_t m_height;
    const uint8_t m_fps;
    const gca::frame_format m_color_format;
    const gca::frame_format m_depth_format;

    float m_depth_scale;
    gca::intrinsics m_color_intrinsics;
    gca::intrinsics m_depth_intrinsics;
    gca::extrinsics m_color_to_depth_extrinsics;
    gca::extrinsics m_depth_to_color_extrinsics;
    const void *m_color_raw_data = nullptr;
    const void *m_depth_raw_data = nullptr;

private:
    virtual bool find_device() = 0;
    virtual bool start_stream() = 0;
    virtual void stop_stream() = 0;
    virtual float read_depth_scale() = 0;
    virtual gca::intrinsics read_color_intrinsics() = 0;
    virtual gca::intrinsics read_depth_intrinsics() = 0;
    virtual gca::extrinsics read_color_to_depth_extrinsics() = 0;
    virtual gca::extrinsics read_depth_to_color_extrinsics() = 0;
    virtual void receive_data_from_device() = 0;
    virtual cv::Mat set_color_to_cv_mat() = 0;
    virtual cv::Mat set_depth_to_cv_mat() = 0;

private:
    bool m_device_started = false;
};
} // namespace gca
